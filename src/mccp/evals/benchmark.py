"""Benchmark harness: one golden-set protocol, many targets, several test sets.

Answers the commercial questions the single-run L1 report cannot:

* **Generalisation:** score each target on the in-distribution held-out split
  *and* on an out-of-distribution set (unseen vendors, phrasings, threshold
  edge cases), broken down by scenario.
* **Frontier baseline:** score a prompted frontier model through any
  OpenAI-compatible endpoint (LiteLLM, Azure OpenAI, OpenAI) on identical,
  teacher-forced items.
* **Cost and latency:** per-target p50/p95 latency, token usage and cost per
  1,000 decisions.
* **Reproducibility:** `compare_reports` aggregates adapter runs trained with
  different seeds into mean / min / max / spread.

Scoring reuses `mccp.evals.layers` unchanged, so numbers are comparable with
the pipeline's L1 evidence. A *lenient* JSON metric (markdown fences stripped)
is reported alongside the strict one so a frontier model is not penalised only
for wrapping valid JSON in ```json fences; the strict metric remains the
production contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
import jsonschema

from mccp.rules import RULESETS

from . import layers
from .stats import bootstrap_mean_ci, mean, percentile
from .suite import EvalSuite

log = logging.getLogger(__name__)

TargetKind = Literal["adapter", "base", "frontier"]
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


# ------------------------------------------------------------ remote client --
class CircuitOpen(RuntimeError):
    """Raised without calling the endpoint once it has failed repeatedly."""


class OpenAICompatClient:
    """Async client for any OpenAI-compatible `/v1/chat/completions` endpoint.

    * `base_url` is everything before `/v1` (e.g. `http://127.0.0.1:4000`,
      `https://api.openai.com`, `https://<res>.openai.azure.com/openai`).
    * Retries 429/5xx/transport errors with jittered exponential backoff,
      honouring `Retry-After` (capped).
    * Adapts once to providers that reject a parameter: `max_tokens` becomes
      `max_completion_tokens`; `temperature`, `seed` or `parallel_tool_calls`
      are dropped. The adaptation is recorded in `self.adaptations`.
    * A circuit breaker opens after `breaker_threshold` consecutive failures so a
      bad key or dead endpoint fails fast instead of burning the whole run.
    * Account-level failures (bad key, no credit/quota, suspended account) open
      the breaker at once: retrying them cannot succeed. The provider's message
      is kept in `self.fatal_error` for the report.
    * `drop_params` pre-learns adaptations for providers known to reject a
      parameter (e.g. Kimi rejects any `temperature` other than its fixed value).
    """

    # Body markers of errors no retry can fix (OpenAI, Azure, Moonshot, LiteLLM wording).
    _FATAL_MARKERS = (
        "insufficient_quota",
        "credit_balance",
        "no credits",
        "exceeded your current quota",
        "billing",
        "account is suspended",
        "invalid_api_key",
        "incorrect api key",
        "invalid authentication",
    )

    _ADAPTABLE = ("max_tokens", "temperature", "seed", "parallel_tool_calls")

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        auth: Literal["bearer", "api-key"] = "bearer",
        timeout: float = 120.0,
        max_attempts: int = 4,
        breaker_threshold: int = 15,
        retry_after_cap_s: float = 30.0,
        drop_params: tuple[str, ...] = (),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        headers: dict[str, str] = {}
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"} if auth == "bearer" else {"api-key": api_key}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._headers = headers
        self.max_attempts = max_attempts
        self.breaker_threshold = breaker_threshold
        self.retry_after_cap_s = retry_after_cap_s
        unknown = set(drop_params) - set(self._ADAPTABLE)
        if unknown:
            raise ValueError(f"drop_params must be among {self._ADAPTABLE}; got {sorted(unknown)}")
        self.adaptations: dict[str, str] = {p: "drop" for p in drop_params}
        self.fatal_error: str | None = None
        self._consecutive_failures = 0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def circuit_open(self) -> bool:
        return self.fatal_error is not None or self._consecutive_failures >= self.breaker_threshold

    @classmethod
    def is_fatal(cls, status: int, body: str) -> bool:
        if status in (401, 402, 403):
            return True
        text = body.lower()
        return status in (400, 429) and any(m in text for m in cls._FATAL_MARKERS)

    def _apply_adaptations(self, payload: dict[str, Any]) -> dict[str, Any]:
        p = dict(payload)
        for param, action in self.adaptations.items():
            if param in p:
                value = p.pop(param)
                if action.startswith("rename:"):
                    p[action.split(":", 1)[1]] = value
        # Frontier APIs (and LiteLLM->Anthropic) prefer null content beside tool calls.
        p["messages"] = [
            {**m, "content": None}
            if m.get("role") == "assistant" and m.get("tool_calls") and not m.get("content")
            else m
            for m in p["messages"]
        ]
        return p

    async def _learn(self, body: str) -> bool:
        """Record an adaptation for a rejected parameter. True if one was learned."""
        text = body.lower()
        async with self._lock:
            for param in self._ADAPTABLE:
                if param in self.adaptations or param not in text:
                    continue
                if param == "max_tokens":
                    self.adaptations[param] = "rename:max_completion_tokens"
                else:
                    self.adaptations[param] = "drop"
                log.info("endpoint rejected `%s`; adapting (%s)", param, self.adaptations[param])
                return True
        return False

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.circuit_open:
            if self.fatal_error:
                raise CircuitOpen(f"endpoint refused the account: {self.fatal_error}")
            raise CircuitOpen(f"circuit open after {self._consecutive_failures} consecutive failures")
        last_err = "no attempt made"
        attempt = 0
        adapt_budget = len(self._ADAPTABLE)
        while attempt < self.max_attempts:
            attempt += 1
            try:
                r = await self._client.post(self.url, json=self._apply_adaptations(payload), headers=self._headers)
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                await self._backoff(attempt, None)
                continue
            if r.status_code < 400:
                self._consecutive_failures = 0
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            if self.is_fatal(r.status_code, r.text):
                self.fatal_error = last_err
                log.error("frontier endpoint refused the account; skipping further calls: %s", last_err)
                break
            if r.status_code == 400 and adapt_budget > 0 and await self._learn(r.text):
                adapt_budget -= 1
                attempt -= 1  # an adaptation retry does not consume a transport attempt
                continue
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                await self._backoff(attempt, r.headers.get("retry-after"))
                continue
            break  # other 4xx are not retryable
        self._consecutive_failures += 1
        raise RuntimeError(last_err)

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.max_attempts:
            return
        delay = min(self.retry_after_cap_s, 2 ** (attempt - 1)) * (0.5 + random.random())
        if retry_after:
            try:
                delay = min(self.retry_after_cap_s, float(retry_after))
            except ValueError:
                pass
        await asyncio.sleep(delay)


# ------------------------------------------------------------------ targets --
@dataclass
class CallStats:
    latencies_s: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    wall_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies_s)
        return {
            "calls": len(lat) + self.errors,
            "errors": self.errors,
            "latency_p50_s": percentile(lat, 50) if lat else None,
            "latency_p95_s": percentile(lat, 95) if lat else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_s": round(self.wall_s, 2),
        }


@dataclass
class Target:
    """One model under test. `chat` is any async OpenAI-shaped call."""

    name: str
    kind: TargetKind
    model: str
    chat: Any
    extra_body: dict[str, Any] = field(default_factory=dict)
    concurrency: int = 4
    usd_per_m_input: float | None = None  # frontier price per 1M input tokens
    usd_per_m_output: float | None = None
    gpu_usd_per_hour: float | None = None  # self-hosted: price of the serving GPU

    def timed(self, stats: CallStats):
        async def call(payload: dict[str, Any]) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                resp = await self.chat(payload)
            except Exception:
                stats.errors += 1
                raise
            stats.latencies_s.append(time.perf_counter() - start)
            usage = resp.get("usage") or {}
            stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            stats.completion_tokens += int(usage.get("completion_tokens") or 0)
            return resp

        return call


# ------------------------------------------------------------------ scoring --
def strip_fences(text: str) -> str:
    m = _FENCE.match(text or "")
    return m.group(1) if m else (text or "")


def lenient_final_score(content: str, expected: dict[str, Any], schema: dict[str, Any], fields: list[str]) -> float:
    try:
        obj = json.loads(strip_fences(content))
        jsonschema.validate(obj, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError, TypeError):
        return 0.0
    return sum(obj.get(f) == expected.get(f) for f in fields) / len(fields)


def _breakdown(values: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {k: {"n": len(v), "accuracy": round(mean(v), 4)} for k, v in sorted(values.items())}


def _rules_decision_correct(rules: Any, meta: dict[str, Any], sc: dict[str, Any]) -> float:
    """Extract-then-rules: run the policy on what the model extracted (lookups by
    the extracted ids) and compare with the record's label. Invalid output, an
    endpoint error or a record without lookups is wrong, never skipped."""
    if sc.get("error") or not sc.get("json_valid") or "lookups" not in meta or "expected" not in meta:
        return 0.0
    try:
        got = rules(meta, json.loads(sc.get("normalized_content") or sc["content"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0
    exp = meta["expected"]
    return float(got.decision == exp["decision"] and got.reason_code == exp["reason_code"])


def summarize_set(
    items: list[layers.Item],
    scores: list[dict[str, Any]],
    records: list[dict[str, Any]],
    suite: EvalSuite,
) -> dict[str, Any]:
    """L1 summary plus per-scenario / per-OOD-kind accuracy and the lenient metric."""
    out = layers.summarize_l1(scores)
    by_scenario: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    by_field: dict[str, list[float]] = defaultdict(list)
    lenient: list[float] = []
    decision_correct: list[float] = []
    rules = RULESETS[suite.decision_rules] if suite.decision_rules else None
    for it, sc in zip(items, scores, strict=True):
        if it.kind != "final":
            continue
        meta = records[it.record].get("meta") or {}
        expected = json.loads(it.expected["content"])
        for f, hit in (sc.get("fields") or {}).items():
            by_field[f].append(float(hit))
        if rules is not None:
            ok = _rules_decision_correct(rules, meta, sc)
        else:
            ok = float(sc.get("fields", {}).get("reason_code", False) and sc.get("fields", {}).get("decision", False))
        decision_correct.append(ok)
        by_scenario[str(meta.get("scenario", "unknown"))].append(ok)
        if "ood_kind" in meta:
            by_kind[str(meta["ood_kind"])].append(ok)
        lenient.append(
            lenient_final_score(sc.get("content", ""), expected, suite.output_schema, suite.compare_fields)
            if not sc.get("error")
            else 0.0
        )
    out["thinking_leaks"] = sum("<think>" in (sc.get("content") or "") for sc in scores if sc["kind"] == "final")
    out["decision_accuracy"] = bootstrap_mean_ci(decision_correct) if decision_correct else None
    out["field_accuracy_lenient"] = mean(lenient) if lenient else None
    out["by_scenario"] = _breakdown(by_scenario)
    out["field_accuracy_by_field"] = {f: round(mean(v), 4) for f, v in sorted(by_field.items())} or None
    out["decision_source"] = f"rules:{suite.decision_rules}" if rules is not None else "model"
    if by_kind:
        out["by_ood_kind"] = _breakdown(by_kind)
    return out


def _cost(target: Target, stats: CallStats, n_decisions: int) -> dict[str, Any]:
    """USD per 1,000 triage decisions (a decision = the 3 assistant steps of one invoice)."""
    if n_decisions == 0:
        return {"usd_per_1k_decisions": None, "basis": None}
    if target.usd_per_m_input is not None and target.usd_per_m_output is not None:
        if stats.prompt_tokens + stats.completion_tokens == 0:
            return {"usd_per_1k_decisions": None, "basis": "endpoint returned no token usage"}
        usd = (
            stats.prompt_tokens * target.usd_per_m_input + stats.completion_tokens * target.usd_per_m_output
        ) / 1_000_000
        return {"usd_per_1k_decisions": round(usd / n_decisions * 1000, 4), "basis": "token prices"}
    if target.gpu_usd_per_hour is not None and stats.wall_s > 0:
        usd = target.gpu_usd_per_hour * stats.wall_s / 3600
        return {
            "usd_per_1k_decisions": round(usd / n_decisions * 1000, 4),
            "basis": f"GPU ${target.gpu_usd_per_hour}/h x eval wall time (upper bound: eval concurrency, not tuned serving)",
        }
    return {"usd_per_1k_decisions": None, "basis": "no price given"}


async def run_target_on_set(
    target: Target,
    items: list[layers.Item],
    records: list[dict[str, Any]],
    suite: EvalSuite,
) -> dict[str, Any]:
    stats = CallStats()
    start = time.perf_counter()
    scores = await layers.run_items(
        target.timed(stats), target.model, items, suite, target.extra_body, target.concurrency
    )
    stats.wall_s = time.perf_counter() - start
    summary = summarize_set(items, scores, records, suite)
    n_decisions = sum(1 for it in items if it.kind == "final")
    error_rate = stats.errors / max(1, len(items))
    return {
        "target": {"name": target.name, "kind": target.kind, "model": target.model},
        "valid": error_rate <= 0.05,
        "error_rate": round(error_rate, 4),
        "l1": summary,
        "calls": stats.summary(),
        "cost": _cost(target, stats, n_decisions),
        "first_error": next((sc["error"] for sc in scores if sc.get("error")), None),
        "_scores": scores,
    }


async def run_benchmark(
    *,
    sets: dict[str, list[dict[str, Any]]],
    targets: list[Target],
    suite: EvalSuite,
    max_items: int | None = None,
    on_scores: Any = None,
) -> dict[str, Any]:
    """Every target on every set; targets run sequentially so latency numbers
    are not distorted by each other on a shared GPU. `on_scores(set_name,
    target_name, items, records, scores)` receives the per-item scores (raw
    answers included) before they are dropped from the report."""
    report: dict[str, Any] = {"sets": {}, "suite": f"{suite.name}@v{suite.version}"}
    for set_name, records in sets.items():
        items = layers.golden_items(records, max_items or 10**9)
        if not items:
            raise ValueError(f"test set `{set_name}` produced no evaluation items")
        set_out: dict[str, Any] = {
            "records": len(records),
            "items": len(items),
            "decisions": sum(1 for it in items if it.kind == "final"),
            "targets": {},
        }
        for t in targets:
            log.info("benchmark: %s on %s (%d items)", t.name, set_name, len(items))
            set_out["targets"][t.name] = await run_target_on_set(t, items, records, suite)
        names = [t.name for t in targets]
        if "adapter" in set_out["targets"]:
            a = set_out["targets"]["adapter"]["_scores"]
            set_out["adapter_delta"] = {
                n: layers.delta_vs(a, set_out["targets"][n]["_scores"]) for n in names if n != "adapter"
            }
        for name, t in set_out["targets"].items():
            if on_scores is not None:
                on_scores(set_name, name, items, records, t["_scores"])
            t.pop("_scores")
        report["sets"][set_name] = set_out
    return report


# ------------------------------------------------------------ cross-seed --
def _metric(report: dict[str, Any], set_name: str, target: str, path: tuple[str, ...]) -> float | None:
    node: Any = report.get("sets", {}).get(set_name, {}).get("targets", {}).get(target)
    if node is None:
        return None
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


HEADLINE_METRICS: dict[str, tuple[str, ...]] = {
    "task_score": ("l1", "task_score", "mean"),
    "decision_accuracy": ("l1", "decision_accuracy", "mean"),
    "tool_call_accuracy": ("l1", "tool_call_accuracy"),
    "json_validity": ("l1", "json_validity"),
    "field_accuracy_lenient": ("l1", "field_accuracy_lenient"),
    "latency_p50_s": ("calls", "latency_p50_s"),
    "usd_per_1k_decisions": ("cost", "usd_per_1k_decisions"),
}


def compare_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate several benchmark reports (one adapter per seed). Base and
    frontier rows are taken from whichever report contains them."""
    set_names = sorted({s for r in reports.values() for s in r.get("sets", {})})
    out: dict[str, Any] = {"runs": sorted(reports), "sets": {}}
    for s in set_names:
        rows: dict[str, Any] = {}
        for metric, path in HEADLINE_METRICS.items():
            seeds = {run: _metric(r, s, "adapter", path) for run, r in reports.items()}
            vals = [v for v in seeds.values() if v is not None]
            row: dict[str, Any] = {"adapter_by_run": seeds}
            if vals:
                row.update(
                    adapter_mean=round(mean(vals), 4),
                    adapter_min=round(min(vals), 4),
                    adapter_max=round(max(vals), 4),
                    adapter_spread=round(max(vals) - min(vals), 4),
                )
            for other in ("base", "frontier"):
                v = next(
                    (_metric(r, s, other, path) for r in reports.values() if _metric(r, s, other, path) is not None),
                    None,
                )
                row[other] = None if v is None else round(v, 4)
            rows[metric] = row
        out["sets"][s] = rows
    return out


def comparison_markdown(cmp: dict[str, Any]) -> str:
    def fmt(v: Any, metric: str) -> str:
        if v is None:
            return "n/a"
        if metric.startswith("usd"):
            return f"${v:,.3f}"
        if metric.endswith("_s"):
            return f"{v:.2f}s"
        return f"{v * 100:.1f}%"

    lines = [f"Runs: {', '.join(cmp['runs'])}", ""]
    for s, rows in cmp["sets"].items():
        lines += [
            f"### {s}",
            "",
            "| Metric | Adapter mean (min-max) | Base | Frontier |",
            "|---|---|---|---|",
        ]
        for metric, row in rows.items():
            if "adapter_mean" in row:
                a = (
                    f"{fmt(row['adapter_mean'], metric)} "
                    f"({fmt(row['adapter_min'], metric)}-{fmt(row['adapter_max'], metric)})"
                )
            else:
                a = "n/a"
            lines.append(f"| {metric} | {a} | {fmt(row['base'], metric)} | {fmt(row['frontier'], metric)} |")
        lines.append("")
    return "\n".join(lines)
