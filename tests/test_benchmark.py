"""Eval kit: OOD generator, OpenAI-compatible client, benchmark scoring, cross-seed compare."""

from __future__ import annotations

import json

import httpx
import pytest

from mccp.evals import layers
from mccp.evals.benchmark import (
    CircuitOpen,
    OpenAICompatClient,
    Target,
    compare_reports,
    comparison_markdown,
    lenient_final_score,
    run_benchmark,
    strip_fences,
    summarize_set,
)
from mccp.evals.suite import EvalSuite
from mccp.examples.invoices import (
    _ITEMS,
    _VENDORS,
    OOD_KINDS,
    OUTPUT_SCHEMA,
    generate,
    generate_ood,
)

SUITE = EvalSuite.model_validate(
    {
        "name": "invoice-triage-suite",
        "version": 1,
        "golden": "finance-ops/invoice-exceptions@v1:test",
        "compare_fields": ["decision", "reason_code"],
        "output_schema": OUTPUT_SCHEMA,
    }
)

EXPECTED_BY_KIND = {
    "boundary_price_over": "PRICE_VARIANCE",
    "boundary_price_under": "WITHIN_TOLERANCE",
    "boundary_bank_recent": "VENDOR_BANK_CHANGE",
    "boundary_bank_ok": "WITHIN_TOLERANCE",
    "boundary_qty_over": "QTY_MISMATCH",
    "partial_billing_ok": "WITHIN_TOLERANCE",
    "precedence_dup_over_price": "DUPLICATE_INVOICE",
    "precedence_bank_over_qty": "VENDOR_BANK_CHANGE",
    "missing_po": "MISSING_PO",
    "clean": "WITHIN_TOLERANCE",
}


# ------------------------------------------------------------- OOD generator --
class TestOodGenerator:
    def test_every_kind_has_its_intended_label(self):
        assert set(EXPECTED_BY_KIND) == set(OOD_KINDS)
        for rec in generate_ood(300, seed=11):
            assert rec["meta"]["scenario"] == EXPECTED_BY_KIND[rec["meta"]["ood_kind"]], rec["meta"]

    def test_final_answers_match_schema_and_decision_table(self):
        import jsonschema

        from mccp.examples.invoices import DECISION, NEXT_ACTION

        for rec in generate_ood(100, seed=3):
            final = json.loads(rec["messages"][-1]["content"])
            jsonschema.validate(final, OUTPUT_SCHEMA)
            assert final["reason_code"] == rec["meta"]["scenario"]
            assert final["decision"] == DECISION[final["reason_code"]]
            assert final["next_action"] == NEXT_ACTION[final["reason_code"]]

    def test_surface_features_are_disjoint_from_training(self):
        ood = generate_ood(200)
        train = generate(500)
        train_text = " ".join(r["messages"][1]["content"] for r in train)
        for rec in ood:
            user = rec["messages"][1]["content"]
            assert not any(v in user for v in _VENDORS), user
            assert not any(f" {i}" in user for i in _ITEMS), user
            assert rec["meta"]["source_id"] not in train_text
        assert all("The matching engine flagged it for review" not in r["messages"][1]["content"] for r in ood)

    def test_deterministic_and_balanced(self):
        a, b = generate_ood(50, seed=5), generate_ood(50, seed=5)
        assert a == b
        kinds = [r["meta"]["ood_kind"] for r in generate_ood(len(OOD_KINDS) * 4)]
        assert all(kinds.count(k) == 4 for k in OOD_KINDS)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            generate_ood(0)


# ------------------------------------------------------- OpenAI-compat client --
def _client(handler, **kw) -> OpenAICompatClient:
    transport = httpx.MockTransport(handler)
    return OpenAICompatClient(
        "https://api.example.test",
        "sk-test",
        client=httpx.AsyncClient(transport=transport),
        retry_after_cap_s=0.0,
        **kw,
    )


OK = {
    "choices": [{"message": {"role": "assistant", "content": "{}"}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


class TestOpenAICompatClient:
    async def test_bearer_header_and_null_content_beside_tool_calls(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["auth"] = req.headers.get("authorization")
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=OK)

        c = _client(handler)
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {}}]}]
        await c.chat({"model": "m", "messages": msgs})
        assert seen["auth"] == "Bearer sk-test"
        assert seen["body"]["messages"][0]["content"] is None
        assert msgs[0]["content"] == ""  # caller's payload is not mutated

    async def test_api_key_auth(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen.update(req.headers)
            return httpx.Response(200, json=OK)

        c = _client(handler, auth="api-key")
        await c.chat({"model": "m", "messages": []})
        assert seen.get("api-key") == "sk-test" and "authorization" not in seen

    async def test_adapts_rejected_parameters_once(self):
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            bodies.append(body)
            if "max_tokens" in body:
                return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'"}})
            if "temperature" in body:
                return httpx.Response(400, json={"error": {"message": "temperature is not supported"}})
            return httpx.Response(200, json=OK)

        c = _client(handler)
        await c.chat({"model": "m", "messages": [], "max_tokens": 9, "temperature": 0})
        assert bodies[-1].get("max_completion_tokens") == 9
        assert "temperature" not in bodies[-1]
        n = len(bodies)
        await c.chat({"model": "m", "messages": [], "max_tokens": 9, "temperature": 0})
        assert len(bodies) == n + 1  # learned adaptations apply first time on later calls
        assert c.adaptations == {"max_tokens": "rename:max_completion_tokens", "temperature": "drop"}

    async def test_retries_429_then_succeeds(self):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, headers={"retry-after": "0"}, json={})
            return httpx.Response(200, json=OK)

        assert (await _client(handler).chat({"model": "m", "messages": []}))["usage"]["prompt_tokens"] == 7
        assert calls["n"] == 3

    async def test_auth_error_not_retried_and_breaker_opens(self):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        c = _client(handler, breaker_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="HTTP 401"):
                await c.chat({"model": "m", "messages": []})
        assert calls["n"] == 3  # one attempt each: 401 is not retryable
        with pytest.raises(CircuitOpen):
            await c.chat({"model": "m", "messages": []})
        assert calls["n"] == 3


# --------------------------------------------------------------- scoring --
class TestScoring:
    def test_strip_fences(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_lenient_accepts_fenced_valid_json_only(self):
        exp = {"decision": "hold", "reason_code": "QTY_MISMATCH"}
        good = {"decision": "hold", "reason_code": "QTY_MISMATCH", "next_action": "Hold it.", "variance_pct": 0.0}
        fenced = "```json\n" + json.dumps(good) + "\n```"
        assert lenient_final_score(fenced, exp, OUTPUT_SCHEMA, ["decision", "reason_code"]) == 1.0
        assert lenient_final_score("not json", exp, OUTPUT_SCHEMA, ["decision", "reason_code"]) == 0.0

    def test_summarize_set_breaks_down_by_scenario_and_kind(self):
        records = generate_ood(20)
        items = layers.golden_items(records, 10**6)
        scores = []
        for it in items:
            if it.kind == "tool_call":
                scores.append({"kind": "tool_call", "score": 1.0, "parsed": True, "args_valid": True})
            else:
                content = it.expected["content"]
                scores.append(
                    layers.score_item(
                        it, {"choices": [{"message": {"content": content}}]}, OUTPUT_SCHEMA, ["decision", "reason_code"]
                    )
                )
        out = summarize_set(items, scores, records, SUITE)
        assert out["decision_accuracy"]["mean"] == 1.0
        assert out["field_accuracy_lenient"] == 1.0
        assert set(out["by_ood_kind"]) == set(OOD_KINDS)
        assert out["thinking_leaks"] == 0


# --------------------------------------------------- end-to-end on fake vLLM --
async def test_benchmark_end_to_end_on_fake_vllm():
    from mccp.serving.clients import VllmClient
    from tests.fake_serving import FakeServing

    fake = FakeServing()
    fake.start()
    try:
        fake.loaded["adapter-x"] = "/tmp/a"
        vllm = VllmClient(fake.url)
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        targets = [
            Target("adapter", "adapter", "adapter-x", vllm.chat, body, 4, gpu_usd_per_hour=0.35),
            Target("base", "base", "qwen3-8b-base", vllm.chat, body, 4, gpu_usd_per_hour=0.35),
        ]
        report = await run_benchmark(
            sets={"in_dist": generate(12, seed=9), "ood": generate_ood(20)}, targets=targets, suite=SUITE
        )
    finally:
        fake.stop()
    for s in ("in_dist", "ood"):
        a = report["sets"][s]["targets"]["adapter"]
        b = report["sets"][s]["targets"]["base"]
        assert a["valid"] and b["valid"]
        assert a["l1"]["decision_accuracy"]["mean"] == 1.0, (s, a["l1"]["by_scenario"])
        assert a["l1"]["tool_call_accuracy"] == 1.0
        assert b["l1"]["task_score"]["mean"] < a["l1"]["task_score"]["mean"]
        assert a["cost"]["usd_per_1k_decisions"] is not None
        assert "_scores" not in a
        assert report["sets"][s]["adapter_delta"]["base"]["delta"] > 0


# -------------------------------------------------------------- compare --
def test_compare_reports_aggregates_seeds_and_takes_baselines_once():
    def rep(task: float, with_base: bool) -> dict:
        t = {"adapter": {"l1": {"task_score": {"mean": task}, "tool_call_accuracy": 1.0}, "calls": {}, "cost": {}}}
        if with_base:
            t["base"] = {"l1": {"task_score": {"mean": 0.01}}, "calls": {}, "cost": {}}
            t["frontier"] = {"l1": {"task_score": {"mean": 0.8}}, "calls": {}, "cost": {}}
        return {"sets": {"ood": {"targets": t}}}

    cmp = compare_reports({"s42": rep(0.9, True), "s7": rep(0.8, False)})
    row = cmp["sets"]["ood"]["task_score"]
    assert row["adapter_mean"] == pytest.approx(0.85)
    assert row["adapter_spread"] == pytest.approx(0.1)
    assert row["base"] == 0.01 and row["frontier"] == 0.8
    md = comparison_markdown(cmp)
    assert "| task_score | 85.0% (80.0%-90.0%) | 1.0% | 80.0% |" in md


def test_cost_is_unknown_not_zero_without_usage():
    from mccp.evals.benchmark import CallStats, _cost

    t = Target("frontier", "frontier", "m", None, usd_per_m_input=2.0, usd_per_m_output=8.0)
    assert _cost(t, CallStats(latencies_s=[0.1]), 10)["usd_per_1k_decisions"] is None
    cs = CallStats(latencies_s=[0.1], prompt_tokens=1_000_000, completion_tokens=0)
    assert _cost(t, cs, 1000)["usd_per_1k_decisions"] == 2.0
