"""deploy/gpu/throughput.py: streaming accounting, closed-loop levels, metrics, cost math."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys

import httpx
import pytest

from tests.conftest import ROOT

_spec = importlib.util.spec_from_file_location("gpu_throughput", ROOT / "deploy/gpu/throughput.py")
tp = importlib.util.module_from_spec(_spec)
sys.modules["gpu_throughput"] = tp  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(tp)

METRICS = """# HELP vllm:kv_cache_usage_perc KV-cache usage
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:other_metric 9
"""


def _sse(n_tokens: int, usage: bool = True) -> bytes:
    chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]}]
    chunks += [{"choices": [{"delta": {"content": "x"}}]} for _ in range(n_tokens)]
    if usage:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": n_tokens}})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _transport(seen: list[dict], usage: bool = True, fail: bool = False) -> httpx.MockTransport:
    async def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/metrics":
            return httpx.Response(200, text=METRICS)
        if req.url.path.endswith("lora_adapter"):
            return httpx.Response(200, text="ok")
        body = json.loads(req.content)
        seen.append(body)
        if fail:
            return httpx.Response(500, text="boom")
        await asyncio.sleep(0.01)
        return httpx.Response(
            200, content=_sse(body["max_tokens"], usage), headers={"content-type": "text/event-stream"}
        )

    return httpx.MockTransport(handler)


def test_parse_metrics_v1_names_and_sums():
    m = tp.parse_metrics(METRICS)
    assert m == {"kv_cache": 0.25, "running": 3.0, "waiting": 1.0, "preemptions": 2.0}
    assert tp.parse_metrics("vllm:gpu_cache_usage_perc 0.5")["kv_cache"] == 0.5


@pytest.mark.parametrize("usage", [True, False])
async def test_stream_one_counts_tokens_and_ttft(usage):
    async with httpx.AsyncClient(transport=_transport([], usage=usage)) as c:
        r = await tp.stream_one(c, "http://v/v1/chat/completions", {}, {"max_tokens": 32})
    assert r.ok and r.output_tokens == 32 and r.usage_reported is usage
    assert r.ttft_s is not None and r.latency_s >= r.ttft_s


async def test_stream_one_http_error():
    async with httpx.AsyncClient(transport=_transport([], fail=True)) as c:
        r = await tp.stream_one(c, "http://v/v1/chat/completions", {}, {"max_tokens": 4})
    assert not r.ok and "HTTP 500" in r.error


async def test_run_level_keeps_concurrency_and_discards_warmup():
    in_flight = {"now": 0, "max": 0}

    async def send(payload):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.02)
        in_flight["now"] -= 1
        t = tp.time.perf_counter()
        return tp.Result(ok=True, start=t - 0.02, end=t, ttft_s=0.005, output_tokens=10, usage_reported=True)

    results, t0, t1 = await tp.run_level(4, 0.2, [{"messages": []}], send, lambda p: {})
    assert in_flight["max"] == 4
    assert all(r.start >= t0 - 0.03 for r in results)  # warm-up results are not included
    assert len(results) >= 4 * 5


def test_cost_math():
    samples = tp.Samples(
        gpu=[{"t": 1.0, "util": 90.0, "mem_used_mib": 18000, "mem_total_mib": 20475, "power_w": 120.0}]
    )
    res = [
        tp.Result(ok=True, start=0.0, end=10.0, ttft_s=0.1, output_tokens=1000, prompt_tokens=600, usage_reported=True)
    ]
    lvl = tp.summarize_level(1, res, 0.0, 10.0, samples, rate_per_hour=36.0)
    assert lvl["output_tokens_per_s"] == 100.0
    # 36/h over 360,000 tokens/h -> 100 per 1M; 100M tokens -> 277.8 h -> 10,000
    assert lvl["cost_per_1m_output_tokens"] == 100.0
    assert lvl["hours_for_100m_tokens"] == 277.8
    assert lvl["cost_for_100m_tokens"] == 10000
    assert lvl["gpu_util_mean_pct"] == 90.0


async def test_full_run_writes_levels(monkeypatch):
    seen: list[dict] = []
    real = httpx.AsyncClient
    monkeypatch.setattr(tp.httpx, "AsyncClient", lambda **kw: real(transport=_transport(seen), timeout=5))
    args = argparse.Namespace(
        vllm_url="http://v",
        vllm_key="k",
        model="qwen3-8b-base",
        adapter_dir="",
        extra_body='{"chat_template_kwargs": {"enable_thinking": false}}',
        concurrency="1,3",
        duration=1.2,
        output_tokens=16,
        prompts=5,
        seed=1,
        timeout=5,
        gpu_rate_per_hour=30.5,
        currency="INR",
        usd_per_currency=0.0105,
        gpu_name="RTX 4000 Ada",
        serving_settings="test",
    )
    rep = await tp.run(args)
    assert [lv["concurrency"] for lv in rep["levels"]] == [1, 3]
    assert all(lv["errors"] == 0 and lv["output_tokens_per_s"] > 0 for lv in rep["levels"])
    assert rep["levels"][0]["kv_cache_used_max_pct"] == 25.0
    body = seen[0]
    assert body["ignore_eos"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["tool_choice"] == "none"
    assert body["tools"] and body["messages"][0]["role"] == "system"
    md = tp.markdown(rep, 600.0)
    assert "Cost per 1M output tokens" in md and "C=3" in md and "($" in md
