"""Serving throughput and unit-cost benchmark for one rented GPU (vLLM).

Answers "can one rented GPU serve a realistic customer workload economically?"
For each concurrency level it keeps exactly C requests in flight for a fixed
time (closed loop), streams every response, and records:

  * aggregate output tokens/s (tokens finished in the window / window length)
  * per-request decode tokens/s, time to first token, end-to-end latency (p50/p95)
  * GPU utilization, memory and power (nvidia-smi, 0.5 s samples)
  * vLLM KV-cache usage, running/waiting requests and preemptions (/metrics)
  * hours and cost to generate 100 million output tokens, and cost per 1M
    output tokens:  cost_per_1M = GPU cost per hour / (output tokens per hour) x 1,000,000

Prompts are real invoice-triage requests (system prompt, tool schemas, invoice
text) built by the same generator as the training data, so prefill cost is
realistic. Output length is fixed (`--output-tokens`, sent with vLLM's
`ignore_eos`) so every level generates the same amount of work and the numbers
are comparable; the task's natural answers are shorter, which is reported as a
separate "per decision" figure using the measured tokens per decision.

Runs inside the control-plane venv; vLLM is started and stopped by bench.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

TOKENS_TARGET = 100_000_000


# --------------------------------------------------------------- workload --
def build_prompts(n: int, seed: int) -> list[dict[str, Any]]:
    """First assistant turn of n generated invoices: messages + tools, OpenAI format."""
    from mccp.evals import layers
    from mccp.examples.invoices import generate

    records = generate(n, seed=seed)
    prompts = []
    for ri, rec in enumerate(records):
        msgs = layers.to_openai_messages(rec["messages"], record=ri)
        first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompts.append({"messages": msgs[:first], "tools": rec.get("tools")})
    return prompts


def tokens_per_decision(n: int, seed: int) -> float:
    """Mean assistant output tokens per triage decision in the training data
    (characters / 4, a standard approximation; used only for the per-decision line)."""
    from mccp.examples.invoices import generate

    chars = []
    for rec in generate(n, seed=seed):
        total = 0
        for m in rec["messages"]:
            if m["role"] == "assistant":
                total += len(m.get("content") or "")
                total += sum(len(json.dumps(tc)) for tc in m.get("tool_calls") or [])
        chars.append(total)
    return statistics.mean(chars) / 4


# ---------------------------------------------------------------- requests --
@dataclass
class Result:
    ok: bool
    start: float
    end: float
    ttft_s: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    usage_reported: bool = False
    error: str | None = None

    @property
    def latency_s(self) -> float:
        return self.end - self.start

    @property
    def decode_tps(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        span = self.latency_s - self.ttft_s
        return (self.output_tokens - 1) / span if span > 0 else None


def _has_token(delta: dict[str, Any]) -> bool:
    return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))


async def stream_one(client: httpx.AsyncClient, url: str, headers: dict[str, str], payload: dict[str, Any]) -> Result:
    start = time.perf_counter()
    res = Result(ok=False, start=start, end=start)
    chunks = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")
                res.error = f"HTTP {r.status_code}: {body[:200]}"
                res.end = time.perf_counter()
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    res.prompt_tokens = int(obj["usage"].get("prompt_tokens") or 0)
                    res.output_tokens = int(obj["usage"].get("completion_tokens") or 0)
                    res.usage_reported = True
                for ch in obj.get("choices") or []:
                    if _has_token(ch.get("delta") or {}):
                        chunks += 1
                        if res.ttft_s is None:
                            res.ttft_s = time.perf_counter() - start
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        res.error = f"{type(e).__name__}: {e}"[:200]
        res.end = time.perf_counter()
        return res
    res.end = time.perf_counter()
    if not res.usage_reported:
        res.output_tokens = chunks  # one token per chunk in vLLM streaming; flagged in the report
    res.ok = True
    return res


# --------------------------------------------------------------- samplers --
@dataclass
class Samples:
    gpu: list[dict[str, float]] = field(default_factory=list)
    vllm: list[dict[str, float]] = field(default_factory=list)


async def sample_gpu(samples: Samples, stop: asyncio.Event, interval_ms: int = 500) -> None:
    if not shutil.which("nvidia-smi"):
        return
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
        f"-lms={interval_ms}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    try:
        while not stop.is_set():
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=2)
            except TimeoutError:
                continue
            if not line:
                break
            parts = [p.strip() for p in line.decode().split(",")]
            try:
                util, used, total, power = (float(p) if p not in ("[N/A]", "N/A") else math.nan for p in parts[:4])
            except ValueError:
                continue
            samples.gpu.append(
                {"t": time.perf_counter(), "util": util, "mem_used_mib": used, "mem_total_mib": total, "power_w": power}
            )
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()


_METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")
_WANTED = {
    "vllm:kv_cache_usage_perc": "kv_cache",  # vLLM v1
    "vllm:gpu_cache_usage_perc": "kv_cache",  # older vLLM
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions_total": "preemptions",
}


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC.match(line.strip())
        if m and m.group(1) in _WANTED:
            key = _WANTED[m.group(1)]
            out[key] = out.get(key, 0.0) + float(m.group(2))  # sum across label sets
    return out


async def sample_vllm(
    client: httpx.AsyncClient, base: str, headers: dict[str, str], samples: Samples, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        with contextlib.suppress(httpx.HTTPError):
            r = await client.get(f"{base}/metrics", headers=headers, timeout=5)
            if r.status_code == 200:
                samples.vllm.append({"t": time.perf_counter(), **parse_metrics(r.text)})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=0.5)


# ------------------------------------------------------------------ levels --
def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _window(rows: list[dict[str, float]], t0: float, t1: float, key: str) -> list[float]:
    return [r[key] for r in rows if t0 <= r["t"] <= t1 and key in r and not math.isnan(r[key])]


def summarize_level(
    concurrency: int, results: list[Result], t0: float, t1: float, samples: Samples, rate_per_hour: float
) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    wall = max(t1 - t0, 1e-9)
    tps = out_tokens / wall
    util = _window(samples.gpu, t0, t1, "util")
    mem = _window(samples.gpu, t0, t1, "mem_used_mib")
    power = _window(samples.gpu, t0, t1, "power_w")
    kv = _window(samples.vllm, t0, t1, "kv_cache")
    waiting = _window(samples.vllm, t0, t1, "waiting")
    pre = _window(samples.vllm, t0, t1, "preemptions")
    hours_100m = TOKENS_TARGET / tps / 3600 if tps > 0 else None
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "first_error": next((r.error for r in results if r.error), None),
        "usage_reported": all(r.usage_reported for r in ok) if ok else None,
        "window_s": round(wall, 2),
        "output_tokens": out_tokens,
        "prompt_tokens_mean": round(statistics.mean(r.prompt_tokens for r in ok), 1) if ok else None,
        "output_tokens_per_s": round(tps, 1),
        "requests_per_s": round(len(ok) / wall, 3),
        "per_request_decode_tps_p50": _r(pct([d for r in ok if (d := r.decode_tps)], 50)),
        "ttft_s_p50": _r(pct([r.ttft_s for r in ok if r.ttft_s is not None], 50), 3),
        "ttft_s_p95": _r(pct([r.ttft_s for r in ok if r.ttft_s is not None], 95), 3),
        "latency_s_p50": _r(pct([r.latency_s for r in ok], 50), 2),
        "latency_s_p95": _r(pct([r.latency_s for r in ok], 95), 2),
        "gpu_util_mean_pct": _r(statistics.mean(util) if util else None, 1),
        "gpu_util_max_pct": _r(max(util) if util else None, 1),
        "gpu_mem_used_max_mib": _r(max(mem) if mem else None, 0),
        "gpu_mem_total_mib": _r(samples.gpu[-1]["mem_total_mib"] if samples.gpu else None, 0),
        "gpu_power_mean_w": _r(statistics.mean(power) if power else None, 1),
        "kv_cache_used_max_pct": _r(max(kv) * 100 if kv else None, 1),
        "requests_waiting_max": _r(max(waiting) if waiting else None, 0),
        "preemptions": _r((max(pre) - min(pre)) if pre else None, 0),
        "hours_for_100m_tokens": _r(hours_100m, 1),
        "cost_for_100m_tokens": _r(hours_100m * rate_per_hour if hours_100m else None, 0),
        "cost_per_1m_output_tokens": _r(rate_per_hour / (tps * 3600) * 1_000_000 if tps > 0 else None, 2),
    }


def _r(v: float | None, nd: int = 1) -> float | int | None:
    if v is None:
        return None
    return int(round(v)) if nd == 0 else round(v, nd)


async def run_level(
    concurrency: int,
    duration_s: float,
    prompts: list[dict[str, Any]],
    send: Any,
    make_payload: Any,
) -> tuple[list[Result], float, float]:
    """Keep `concurrency` requests in flight until `duration_s` has passed, then
    let the in-flight ones finish. Warm-up requests (one per slot) are discarded."""
    counter = {"i": 0}

    def next_payload() -> dict[str, Any]:
        p = prompts[counter["i"] % len(prompts)]
        counter["i"] += 1
        return make_payload(p)

    await asyncio.gather(*(send(next_payload()) for _ in range(concurrency)))  # warm-up
    results: list[Result] = []
    t0 = time.perf_counter()
    deadline = t0 + duration_s

    async def worker() -> None:
        while time.perf_counter() < deadline:
            results.append(await send(next_payload()))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = max([r.end for r in results], default=time.perf_counter())
    return results, t0, t1


# --------------------------------------------------------------------- run --
async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.vllm_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.vllm_key}"} if args.vllm_key else {}
    extra = json.loads(args.extra_body) if args.extra_body else {}
    prompts = build_prompts(args.prompts, args.seed)
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    rate = args.gpu_rate_per_hour
    samples = Samples()
    stop = asyncio.Event()
    limits = httpx.Limits(max_connections=max(levels) + 4, max_keepalive_connections=max(levels) + 4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        url = f"{base}/v1/chat/completions"
        model = args.model
        if args.adapter_dir:
            r = await client.post(
                f"{base}/v1/load_lora_adapter",
                json={"lora_name": "bench-adapter", "lora_path": args.adapter_dir},
                headers=headers,
            )
            if r.status_code >= 400 and "already" not in r.text.lower():
                raise SystemExit(f"could not load adapter {args.adapter_dir}: {r.status_code} {r.text[:200]}")
            model = "bench-adapter"

        def make_payload(p: dict[str, Any]) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": model,
                "messages": p["messages"],
                "max_tokens": args.output_tokens,
                "ignore_eos": True,  # vLLM: generate exactly max_tokens, so every level does equal work
                "temperature": 0.7,
                "stream": True,
                "stream_options": {"include_usage": True},
                **extra,
            }
            if p.get("tools"):
                body["tools"] = p["tools"]
                body["tool_choice"] = "none"  # realistic prompt (schemas in context), plain-text decode
            return body

        async def send(payload: dict[str, Any]) -> Result:
            return await stream_one(client, url, headers, payload)

        tasks = [
            asyncio.create_task(sample_gpu(samples, stop)),
            asyncio.create_task(sample_vllm(client, base, headers, samples, stop)),
        ]
        out_levels = []
        try:
            for c in levels:
                print(f"concurrency {c}: {args.duration:.0f} s ...", file=sys.stderr, flush=True)
                results, t0, t1 = await run_level(c, args.duration, prompts, send, make_payload)
                lvl = summarize_level(c, results, t0, t1, samples, rate)
                out_levels.append(lvl)
                print(
                    f"  {lvl['output_tokens_per_s']} output tok/s, GPU {lvl['gpu_util_mean_pct']}%, "
                    f"{lvl['errors']} errors, {rate and lvl['cost_per_1m_output_tokens']} {args.currency}/1M",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            if args.adapter_dir:
                with contextlib.suppress(httpx.HTTPError):
                    await client.post(
                        f"{base}/v1/unload_lora_adapter", json={"lora_name": "bench-adapter"}, headers=headers
                    )
    tpd = tokens_per_decision(50, args.seed)
    best = max(out_levels, key=lambda lv: lv["output_tokens_per_s"]) if out_levels else None
    return {
        "kind": "mccp-throughput-benchmark/v1",
        "model": args.model,
        "served_as": model,
        "adapter_dir": args.adapter_dir,
        "gpu_name": args.gpu_name,
        "gpu_rate_per_hour": rate,
        "currency": args.currency,
        "usd_per_currency": args.usd_per_currency,
        "output_tokens_per_request": args.output_tokens,
        "duration_s_per_level": args.duration,
        "workload": f"{args.prompts} invoice-triage prompts (system + tool schemas + invoice), fixed "
        f"{args.output_tokens}-token outputs (ignore_eos)",
        "task_output_tokens_per_decision_est": round(tpd, 1),
        "serving_settings": args.serving_settings,
        "levels": out_levels,
        "best_level": best["concurrency"] if best else None,
    }


# ------------------------------------------------------------------ report --
def _amount(v: float, symbol: str) -> str:
    if v >= 100:
        return f"{symbol}{v:,.0f}"
    if v >= 1:
        return f"{symbol}{v:,.2f}"
    return f"{symbol}{v:.4f}"


def _money(v: float | None, cur: str, usd: float | None) -> str:
    if v is None:
        return "n/a"
    s = _amount(v, f"{cur} ")
    return f"{s} ({_amount(v * usd, '$')})" if usd else s


def markdown(rep: dict[str, Any], bench_wall_s: float | None = None) -> str:
    cur, usd = rep["currency"], rep["usd_per_currency"]
    lv = rep["levels"]
    lines = [
        f"# Serving throughput: {rep['served_as']} on {rep['gpu_name'] or 'GPU'}",
        "",
        f"Workload: {rep['workload']}. {rep['duration_s_per_level']:.0f} s per level, closed loop "
        f"(C requests always in flight). GPU price: {cur} {rep['gpu_rate_per_hour']}/h"
        + (f" (1 {cur} = ${usd})" if usd else "")
        + ".",
        "",
        "| Metric | " + " | ".join(f"C={x['concurrency']}" for x in lv) + " |",
        "|---|" + "---|" * len(lv),
    ]

    def num(v: Any) -> str:
        return "n/a" if v is None else f"{v:,}"

    def pair(a: str, b: str) -> Any:
        return lambda x: f"{num(x.get(a))} / {num(x.get(b))}"

    def one(key: str, fmt: Any = num) -> Any:
        return lambda x: fmt(x.get(key))

    rows: list[tuple[str, Any]] = [
        ("Output tokens/s (all requests)", one("output_tokens_per_s")),
        ("Decode tokens/s per request (p50)", one("per_request_decode_tps_p50")),
        ("Time to first token p50 / p95 (s)", pair("ttft_s_p50", "ttft_s_p95")),
        ("Latency p50 / p95 (s)", pair("latency_s_p50", "latency_s_p95")),
        ("GPU utilization mean / max (%)", pair("gpu_util_mean_pct", "gpu_util_max_pct")),
        ("GPU memory used (MiB; vLLM pre-allocates)", one("gpu_mem_used_max_mib")),
        ("GPU power mean (W)", one("gpu_power_mean_w")),
        ("KV cache used, max (%)", one("kv_cache_used_max_pct")),
        ("Requests queued (max) / preemptions", pair("requests_waiting_max", "preemptions")),
        ("Requests / errors", pair("requests", "errors")),
        ("Hours to generate 100M output tokens", one("hours_for_100m_tokens")),
        ("Cost to generate 100M output tokens", one("cost_for_100m_tokens", lambda v: _money(v, cur, usd))),
        ("**Cost per 1M output tokens**", one("cost_per_1m_output_tokens", lambda v: _money(v, cur, usd))),
    ]
    for name, fmt in rows:
        lines.append(f"| {name} | " + " | ".join(fmt(x) for x in lv) + " |")
    tpd = rep["task_output_tokens_per_decision_est"]
    best = next((x for x in lv if x["concurrency"] == rep["best_level"]), None)
    lines += ["", "Cost per 1M output tokens = GPU cost per hour / (output tokens per second x 3600) x 1,000,000."]
    if best and best["cost_per_1m_output_tokens"] is not None:
        per_k = best["cost_per_1m_output_tokens"] * tpd / 1000
        lines.append(
            f"At C={best['concurrency']}, one invoice decision (~{tpd:.0f} output tokens, all three steps) costs "
            f"about {_money(per_k, cur, usd)} per 1,000 decisions in output-token terms; real cost is higher "
            "because short answers spend proportionally more time on the prompt (prefill)."
        )
    if any(x["usage_reported"] is False for x in lv):
        lines.append("Token counts for some levels were estimated from stream chunks (server sent no usage).")
    if bench_wall_s:
        lines.append(
            f"Whole benchmark including vLLM start-up: {bench_wall_s / 60:.1f} min, about "
            f"{_money(rep['gpu_rate_per_hour'] * bench_wall_s / 3600, cur, usd)} of GPU rental."
        )
    lines += [
        "",
        "Caveats: the GPU price is the rental rate you set, not measured billing; numbers are for this model, "
        f"this GPU and these serving settings ({rep.get('serving_settings') or 'see vllm.log.txt'}). "
        "Utilization is nvidia-smi's busy-time metric, not compute efficiency.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--vllm-key", default="")
    ap.add_argument("--model", required=True, help="served base model name")
    ap.add_argument("--adapter-dir", default="", help="optional LoRA adapter to serve and measure instead of the base")
    ap.add_argument("--extra-body", default="", help="JSON merged into every request (e.g. chat_template_kwargs)")
    ap.add_argument("--concurrency", default="1,5,10")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds per concurrency level")
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--gpu-rate-per-hour", type=float, default=0.0)
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--usd-per-currency", type=float, default=None, help="e.g. 0.0105 for INR")
    ap.add_argument("--gpu-name", default="")
    ap.add_argument("--serving-settings", default="", help="vLLM settings, for the report caveat")
    ap.add_argument("--bench-started", type=float, default=None, help="epoch seconds when bench.sh started")
    ap.add_argument("--out", required=True, help="directory for throughput.json and throughput.md")
    args = ap.parse_args()
    for c in args.concurrency.split(","):
        if not c.strip().isdigit() or int(c) < 1:
            ap.error("--concurrency must be a comma list of positive integers")
    rep = asyncio.run(run(args))
    wall = time.time() - args.bench_started if args.bench_started else None
    rep["bench_wall_s"] = round(wall, 1) if wall else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "throughput.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    md = markdown(rep, wall)
    (out / "throughput.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if any(x["errors"] for x in rep["levels"]) else 0


if __name__ == "__main__":
    sys.exit(main())
