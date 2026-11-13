"""Run coding tasks against models through the gateway and turn results into
routing evidence.

Requests go through the gateway itself with each model pinned by name, so the
evaluation exercises the same path, provider settings and metering as real
traffic, and the measured cost per task is what customers would pay.

Output:
* results.jsonl: one row per (model, task): passed, why, tokens, cost, latency
* evidence.json: per model, per task class, passed / n and cost per task, plus
  provenance (task file hash, catalog version, sandbox kind, time); the gateway
  loads it with ROUTE_EVIDENCE.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..config import Catalog, Wire
from ..metering import cost, usage_from_body
from ..policy import Cell, Evidence
from .sandbox import Sandbox
from .tasks import Task, TaskClass

SYSTEM = (
    "You are an expert Python engineer. Answer with exactly one ```python fenced code block containing the "
    "complete code requested, and nothing after it."
)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(answer: str, entry_point: str | None) -> str | None:
    """The fenced block that defines the entry point (else the last block); None if no code."""
    blocks = _FENCE.findall(answer or "")
    if not blocks:
        return None
    if entry_point:
        defining = [b for b in blocks if re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", b, re.M)]
        if defining:
            return defining[-1]
    return blocks[-1]


async def score(task: Task, code: str | None, sandbox: Sandbox, timeout_s: float = 20.0) -> tuple[bool, str]:
    if not code:
        return False, "no code block in the answer"
    if task.task_class is TaskClass.TESTS:
        if task.entry_point and re.search(rf"^\s*def {re.escape(task.entry_point)}\s*\(", code, re.M):
            return False, f"answer redefines {task.entry_point}() instead of testing the given one"
        ref = await sandbox.run({"run.py": task.reference + "\n\n" + code}, ["python", "run.py"], timeout_s)
        if not ref.ok:
            return False, "tests fail on the correct implementation" + (" (timeout)" if ref.timed_out else "")
        for i, mutant in enumerate(task.mutants):
            r = await sandbox.run({"run.py": mutant + "\n\n" + code}, ["python", "run.py"], timeout_s)
            if r.ok:
                return False, f"tests miss planted bug #{i + 1}"
        return True, f"pass on reference, catch all {len(task.mutants)} planted bugs"
    if task.entry_point and not re.search(rf"^\s*def {re.escape(task.entry_point)}\s*\(", code, re.M):
        return False, f"answer does not define {task.entry_point}()"
    r = await sandbox.run(
        {
            "run.py": (task.preamble + "\n\n" if task.preamble else "")
            + code
            + "\n\n# --- hidden tests ---\n"
            + (task.tests or "")
        },
        ["python", "run.py"],
        timeout_s,
    )
    if r.timed_out:
        return False, "timeout"
    return (
        (True, "hidden tests pass")
        if r.ok
        else (
            False,
            "hidden tests fail: " + r.output.strip().splitlines()[-1][:200]
            if r.output.strip()
            else "hidden tests fail",
        )
    )


@dataclass
class Row:
    model: str
    task_id: str
    task_class: str
    passed: bool
    why: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    http_status: int = 0


def _request(task: Task, model: str, wire: Wire, max_tokens: int) -> tuple[str, dict[str, Any]]:
    if wire is Wire.ANTHROPIC:
        return "/v1/messages", {"model": model, "max_tokens": max_tokens, "temperature": 0, "system": SYSTEM,
                                "messages": [{"role": "user", "content": task.prompt}]}  # fmt: skip
    return "/v1/chat/completions", {"model": model, "max_tokens": max_tokens, "temperature": 0,
                                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt}]}  # fmt: skip


def _answer_text(wire: Wire, body: dict[str, Any]) -> str:
    if wire is Wire.ANTHROPIC:
        return "".join(
            b.get("text", "") for b in body.get("content") or [] if isinstance(b, dict) and b.get("type") == "text"
        )
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def estimate_usd(catalog: Catalog, tasks: list[Task], models: list[str], max_tokens: int) -> dict[str, float]:
    """Upper-bound-ish estimate before spending: prompt chars/4 in, half of max_tokens out."""
    out = {}
    for m in models:
        price = catalog.model(m).price
        inp = sum((len(SYSTEM) + len(t.prompt)) // 4 + 20 for t in tasks)
        out[m] = round(price.cost(inp, len(tasks) * max_tokens // 2), 4)
    return out


async def run(
    catalog: Catalog,
    gateway: str,
    key: str,
    tasks: list[Task],
    models: list[str],
    sandbox: Sandbox,
    out_dir: Path,
    concurrency: int = 4,
    max_tokens: int = 2048,
    client: httpx.AsyncClient | None = None,
) -> list[Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    own = client is None
    http = client or httpx.AsyncClient(base_url=gateway.rstrip("/"), timeout=httpx.Timeout(300, connect=15))
    rows: list[Row] = []
    results = (out_dir / "results.jsonl").open("a", encoding="utf-8")

    async def one(model: str, task: Task) -> None:
        m = catalog.model(model)
        wire = Wire.OPENAI_CHAT if Wire.OPENAI_CHAT in m.wires else Wire.ANTHROPIC
        path, body = _request(task, model, wire, max_tokens)
        async with sem:
            t0 = time.monotonic()
            try:
                r = await http.post(
                    path, json=body, headers={"authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
                )
            except httpx.HTTPError as e:
                row = Row(model, task.id, task.task_class.value, False, f"request failed: {type(e).__name__}")
            else:
                data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if r.status_code != 200:
                    row = Row(
                        model, task.id, task.task_class.value, False, f"HTTP {r.status_code}", http_status=r.status_code
                    )
                else:
                    u = usage_from_body(wire, data)
                    ok, why = await score(task, extract_code(_answer_text(wire, data), task.entry_point), sandbox)
                    row = Row(model, task.id, task.task_class.value, ok, why, u.input_tokens + u.cache_read_tokens,
                              u.output_tokens, cost(catalog, model, u).usd, int((time.monotonic() - t0) * 1000), 200)  # fmt: skip
        rows.append(row)
        results.write(json.dumps(asdict(row)) + "\n")
        results.flush()

    try:
        await asyncio.gather(*(one(m, t) for m in models for t in tasks))
    finally:
        results.close()
        if own:
            await http.aclose()
    return rows


def evidence(rows: list[Row], suite: str, provenance: dict[str, Any]) -> tuple[Evidence, dict[str, Any]]:
    """Aggregate rows. Requests that never got an answer (HTTP/transport errors) are
    excluded from n: an outage is not evidence about quality. They are reported."""
    cells: dict[str, dict[TaskClass, list[Row]]] = defaultdict(lambda: defaultdict(list))
    excluded = 0
    for r in rows:
        if r.http_status != 200:
            excluded += 1
            continue
        cells[r.model][TaskClass(r.task_class)].append(r)
    results = {
        m: {t: Cell(n=len(rs), passed=sum(r.passed for r in rs), usd_per_task=round(sum(r.cost_usd for r in rs) / len(rs), 8))
            for t, rs in by.items()}
        for m, by in cells.items()
    }  # fmt: skip
    prov = {**provenance, "rows": len(rows), "excluded_no_answer": excluded}
    ev = Evidence(
        suite=suite, generated_at=datetime.now(UTC).isoformat(timespec="seconds"), results=results, provenance=prov
    )
    return ev, prov
