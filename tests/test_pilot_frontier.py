"""deploy/gpu/pilot.py frontier wiring: provider fields, dropped params, preflight skip."""

from __future__ import annotations

import argparse
import importlib.util
import json

import httpx
import pytest

from tests.conftest import ROOT

_spec = importlib.util.spec_from_file_location("gpu_pilot", ROOT / "deploy/gpu/pilot.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

OK = {
    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def _args(**kw) -> argparse.Namespace:
    base = dict(
        baseline_url="https://api.moonshot.ai",
        baseline_model="kimi-k2.6",
        baseline_key_env="MCCP_BASELINE_API_KEY",
        baseline_auth="bearer",
        baseline_concurrency=2,
        baseline_price_in=0.95,
        baseline_price_out=4.0,
        baseline_extra_body='{"thinking": {"type": "disabled"}}',
        baseline_drop="temperature",
        timeout=5.0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def transport(monkeypatch):
    """Route the frontier client's HTTP through a mock; returns (bodies, set_handler)."""
    state = {"bodies": [], "handler": lambda req: httpx.Response(200, json=OK)}

    def handle(req: httpx.Request) -> httpx.Response:
        state["bodies"].append(json.loads(req.content))
        return state["handler"](req)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "mccp.evals.benchmark.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handle), timeout=kw.get("timeout")),
    )
    monkeypatch.setenv("MCCP_BASELINE_API_KEY", "sk-test")
    return state


async def test_kimi_fields_sent_and_temperature_dropped(transport):
    target, client = pilot._frontier_target(_args())
    try:
        assert await pilot._frontier_preflight(target) is None
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        await target.chat({"model": "kimi-k2.6", "messages": [], "temperature": 0, "tools": tools, **target.extra_body})
    finally:
        await client.aclose()
    pre, real = transport["bodies"]
    assert pre["thinking"] == {"type": "disabled"} and "parallel_tool_calls" not in pre  # no tools -> not sent
    assert real["thinking"] == {"type": "disabled"} and "temperature" not in real
    assert real["parallel_tool_calls"] is False


async def test_preflight_reports_no_credit_and_skips(transport):
    transport["handler"] = lambda req: httpx.Response(
        429, json={"error": {"message": "You have no credits remaining.", "code": "credit_balance_exhausted"}}
    )
    target, client = pilot._frontier_target(_args(baseline_extra_body=None, baseline_drop=None))
    try:
        reason = await pilot._frontier_preflight(target)
    finally:
        await client.aclose()
    assert reason and "credit_balance_exhausted" in reason
    assert len(transport["bodies"]) == 1  # not retried


def test_bad_extra_body_rejected(transport):
    with pytest.raises(SystemExit, match="not valid JSON"):
        pilot._frontier_target(_args(baseline_extra_body="{nope"))
