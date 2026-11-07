"""Extract-then-rules task: the model reads the invoice facts in one call, the
rules engine (mccp.rules.invoice_triage) decides.

Guards: the policy refactor left every existing dataset byte-identical; both
tasks are built from exactly the same cases; the extraction label is valid and
the policy run on it reproduces every label; scoring runs the policy on what the
model extracted (a wrong id or price gives a wrong decision); and the pass/fail
gate has no tool-call checks for a task without tool calls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import jsonschema
import pytest
import yaml

from mccp.evals.benchmark import Target, run_benchmark
from mccp.evals.suite import EvalSuite
from mccp.examples.invoices import (
    EXTRACT_FIELDS,
    EXTRACT_MODEL_FIELDS,
    EXTRACT_RAW_SCHEMA,
    EXTRACT_SCHEMA,
    TASKS,
    fixtures,
    generate,
    generate_ood,
)
from mccp.examples.messy import generate_messy
from mccp.rules import RULESETS, invoice_triage
from mccp.rules.extraction import normalize_extraction_v2
from tests.conftest import ROOT

_spec = importlib.util.spec_from_file_location("gpu_pilot_extract", ROOT / "deploy/gpu/pilot.py")
pilot = importlib.util.module_from_spec(_spec)
sys.modules["gpu_pilot_extract"] = pilot
_spec.loader.exec_module(pilot)

EXTRACT_SUITE = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text()))


def _h(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def _sets(task: str) -> dict[str, list[dict]]:
    return {
        "test": generate(300, seed=7777, id_offset=1_000_000, task=task),
        "ood": generate_ood(200, seed=1009, task=task),
        "messy": generate_messy(300, seed=4242, task=task),
    }


# ------------------------------------------------- the refactor changed nothing --
def test_agent_task_datasets_are_byte_identical_to_before_the_rules_module():
    """Hashes taken from the generators before the policy moved to mccp.rules:
    every earlier run's data still reproduces from its seed."""
    assert _h(generate(2000, seed=42)) == "37979150fb3eace5"
    assert _h(generate(2000, seed=42, mix="boundary")) == "51239a8e2ab970c8"
    assert _h(generate(300, seed=7777, id_offset=1_000_000)) == "6d1ad600860722a0"
    assert _h(generate_ood(200, seed=1009)) == "c22179fabf7ba69b"
    assert _h(generate_messy(300, seed=4242)) == "36267de663a8928a"
    assert _h(fixtures(200)) == "6ca2865571cb67d2"


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        generate(1, task="nope")
    assert TASKS == ("agent", "extract")


# ------------------------------------------------------ same cases, both tasks --
@pytest.mark.parametrize("name", ["test", "ood", "messy"])
def test_both_tasks_are_built_from_identical_cases(name):
    agent, extract = _sets("agent")[name], _sets("extract")[name]
    assert len(agent) == len(extract)
    for a, e in zip(agent, extract, strict=True):
        assert a["messages"][1] == e["messages"][1]  # identical invoice text
        for k in ("source_id", "scenario", "ood_kind", "messy_kinds", "template", "split"):
            assert a["meta"].get(k) == e["meta"].get(k)
        facts = json.loads(e["messages"][2]["content"])
        assert facts["po_number"] == a["messages"][2]["tool_calls"][0]["function"]["arguments"]["po_number"]
        assert facts["vendor_id"] == a["messages"][4]["tool_calls"][0]["function"]["arguments"]["vendor_id"]
        assert e["meta"]["expected"] == json.loads(a["messages"][6]["content"])


@pytest.mark.parametrize("name", ["test", "ood", "messy"])
def test_extract_records_are_one_call_and_the_policy_reproduces_every_label(name):
    for r in _sets("extract")[name]:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert "tools" not in r and not any(m.get("tool_calls") for m in r["messages"])
        facts = json.loads(r["messages"][2]["content"])
        jsonschema.validate(facts, EXTRACT_RAW_SCHEMA)  # what the model writes
        assert list(facts) == EXTRACT_MODEL_FIELDS
        canon, _ = normalize_extraction_v2(facts)  # what the rules see
        jsonschema.validate(canon, EXTRACT_SCHEMA)
        assert set(canon) == set(EXTRACT_FIELDS)
        got = invoice_triage.decide_from_record(r["meta"], canon)
        assert got.answer() == r["meta"]["expected"]
        assert got.policy == invoice_triage.POLICY_ID and got.trace[-1]["fired"] in (True, False)


def test_extract_suite_matches_the_code_contract():
    assert EXTRACT_SUITE.output_schema == EXTRACT_SCHEMA
    assert EXTRACT_SUITE.compare_fields == EXTRACT_FIELDS
    assert EXTRACT_SUITE.decision_rules in RULESETS
    raw = yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text())
    with pytest.raises(ValueError, match="unknown decision_rules"):
        EvalSuite.model_validate({**raw, "decision_rules": "made-up@v9"})


# ------------------------------------------------------------- the policy itself --
PO = {"unit_price": 100.0, "received_qty": 10, "paid_invoices": ["INV-9"]}
VENDOR = {"days_since_bank_change": 400}


@pytest.mark.parametrize(
    ("po", "vendor", "inv", "qty", "price", "code"),
    [
        (None, VENDOR, "INV-1", 10, 100.0, "MISSING_PO"),
        ({"error": "not_found"}, VENDOR, "INV-1", 10, 100.0, "MISSING_PO"),
        (PO, None, "INV-1", 10, 100.0, "VENDOR_NOT_FOUND"),
        (PO, VENDOR, "INV-9", 99, 150.0, "DUPLICATE_INVOICE"),  # wins over qty and price
        (PO, {"days_since_bank_change": 29}, "INV-1", 99, 150.0, "VENDOR_BANK_CHANGE"),  # wins over qty
        (PO, {"days_since_bank_change": 30}, "INV-1", 10, 100.0, "WITHIN_TOLERANCE"),  # 30 is not recent
        (PO, VENDOR, "INV-1", 11, 100.0, "QTY_MISMATCH"),
        (PO, VENDOR, "INV-1", 10, 102.01, "PRICE_VARIANCE"),
        (PO, VENDOR, "INV-1", 10, 102.0, "WITHIN_TOLERANCE"),  # exactly 2% is within
        (PO, VENDOR, "INV-1", 10, 97.99, "PRICE_VARIANCE"),
    ],
)
def test_policy_thresholds_and_precedence(po, vendor, inv, qty, price, code):
    r = invoice_triage.evaluate(po, vendor, inv, qty, price)
    assert r.reason_code == code
    assert r.decision == invoice_triage.DECISION[code] and r.next_action == invoice_triage.NEXT_ACTION[code]
    assert r.trace and r.trace[-1]["fired"] == (code != "WITHIN_TOLERANCE")


def test_wrong_extracted_facts_give_the_decision_production_would_make():
    r = generate(1, seed=3, task="extract")[0]
    facts, _ = normalize_extraction_v2(json.loads(r["messages"][2]["content"]))
    assert invoice_triage.decide_from_record(r["meta"], {**facts, "po_number": "PO-1"}).reason_code == "MISSING_PO"
    wrong_vendor = {**facts, "vendor_id": "V-1"}
    assert invoice_triage.decide_from_record(r["meta"], wrong_vendor).reason_code == "VENDOR_NOT_FOUND"


# --------------------------------------------------------------------- scoring --
def _fake_model(mutate):
    """An OpenAI-shaped endpoint answering the gold extraction, changed by `mutate`."""

    async def chat(payload):
        user = payload["messages"][-1]["content"]
        facts = dict(_GOLD[user])
        content = mutate(facts)
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {}}

    return chat


_GOLD: dict[str, dict] = {}


async def _score(records, mutate):
    _GOLD.clear()
    for r in records:
        _GOLD[r["messages"][1]["content"]] = json.loads(r["messages"][2]["content"])
    t = Target("adapter", "adapter", "m", _fake_model(mutate), {}, 4)
    report = await run_benchmark(sets={"in_dist": records}, targets=[t], suite=EXTRACT_SUITE)
    return report["sets"]["in_dist"]["targets"]["adapter"]["l1"]


async def test_perfect_extraction_scores_every_decision_right():
    l1 = await _score(generate_ood(60, task="extract"), lambda f: json.dumps(f))
    assert l1["decision_accuracy"]["mean"] == 1.0
    assert l1["field_accuracy_by_field"] == dict.fromkeys(sorted(EXTRACT_FIELDS), 1.0)
    assert l1["decision_source"] == "rules:invoice-triage-rules@v1"
    assert l1["tool_call_parse_rate"] is None and l1["json_validity"] == 1.0


async def test_a_wrong_id_or_invalid_json_costs_the_decision():
    recs = generate(40, seed=5, task="extract")
    l1 = await _score(recs, lambda f: json.dumps({**f, "po_number": "PO-0"}))
    missing = sum(r["meta"]["expected"]["reason_code"] == "MISSING_PO" for r in recs)
    assert l1["decision_accuracy"]["mean"] == pytest.approx(missing / len(recs))
    assert l1["field_accuracy_by_field"]["po_number"] == 0.0 and l1["field_accuracy_by_field"]["billed_qty"] == 1.0
    l1 = await _score(recs, lambda f: "not json")
    assert l1["decision_accuracy"]["mean"] == 0.0 and l1["json_validity"] == 0.0


async def test_a_wrong_price_changes_only_the_decisions_it_should():
    recs = generate_ood(60, task="extract")
    from mccp.rules.extraction import parse_money

    l1 = await _score(
        recs, lambda f: json.dumps({**f, "unit_price_raw": f"${parse_money(f['unit_price_raw']) * 1.5:,.2f}"})
    )
    assert 0.0 < l1["decision_accuracy"]["mean"] < 1.0  # missing PO / duplicate / bank still decided right
    assert l1["by_ood_kind"]["missing_po"]["accuracy"] == 1.0
    assert l1["by_ood_kind"]["clean"]["accuracy"] == 0.0


def test_gate_has_no_tool_checks_for_a_task_without_tool_calls():
    l1 = {
        "tool_call_parse_rate": None,
        "tool_calls_leaked_into_content": 0,
        "json_validity": 0.99,
        "thinking_leaks": 0,
        "task_score": {"mean": 0.9},
    }
    base = {**l1, "task_score": {"mean": 0.5}}
    t = {"adapter": {"l1": l1, "calls": {"errors": 0}}, "base": {"l1": base}}
    v = pilot._verdict(None, None, {"targets": t})
    assert v["checks"]["tool_calls_parsed_ge_95pct"] is None
    assert v["checks"]["no_tool_calls_leaked_into_content"] is None
    assert v["pass"] is True
    v = pilot._verdict(
        None, None, {"targets": {**t, "adapter": {**t["adapter"], "l1": {**l1, "tool_call_parse_rate": 0.5}}}}
    )
    assert v["pass"] is False  # a tool-call task still fails a low parse rate


# --------------------------------------------------------------------- pipeline --
def _prepare(workdir, *extra):
    cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "prepare", "--workdir", str(workdir), "--ood-n", "20"]
    return subprocess.run([*cmd, *extra], capture_output=True, text=True, cwd=ROOT)


def test_prepare_extract_writes_one_sample_per_record_and_records_the_task(tmp_path):
    r = _prepare(tmp_path / "w", "--n", "60", "--test-n", "30", "--messy-n", "12", "--task", "extract")
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "w/pilot.json").read_text())
    assert meta["task"] == "extract" and meta["decision_rules"] == "invoice-triage-rules@v1"
    assert meta["samples"]["train"] + meta["samples"]["val"] <= 60  # 1 per record (agent: 3)
    for f in ("golden.jsonl", "golden_ood.jsonl", "golden_messy.jsonl"):
        rows = [json.loads(x) for x in (tmp_path / "w/data" / f).read_text().splitlines()]
        assert all(row["meta"]["task"] == "extract" and "lookups" in row["meta"] for row in rows)
    assert pilot.load_suite("extract").decision_rules == "invoice-triage-rules@v1"
    assert pilot.load_suite("agent").decision_rules is None


def test_extract_task_refuses_rgt(tmp_path):
    r = _prepare(tmp_path / "w", "--n", "20", "--task", "extract", "--method", "rgt")
    assert r.returncode == 2 and "extract task has none" in r.stderr


def test_curve_labels_the_extract_task(tmp_path):
    rep = {
        "pilot": {"method": "sft", "task": "extract", "records": 1000, "seed": 42},
        "benchmark": {"sets": {"in_dist": {"targets": {"adapter": {"l1": {"decision_accuracy": {"mean": 0.99}}}}}}},
    }
    d = tmp_path / "sft-extract"
    d.mkdir()
    (d / "report.json").write_text(json.dumps(rep))
    rows = pilot.curve_rows([d / "report.json"])
    assert rows[0]["method"] == "sft+extract" and rows[0]["decision_in_dist"] == 0.99
