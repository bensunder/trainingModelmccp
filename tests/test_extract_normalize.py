"""ID canonicalization (`invoice-extract-ids@v1`) in scoring and serving.

Guards: every house id format the messy set writes maps to the canonical id with
the digits untouched; anything ambiguous is left for the schema to reject (so it
routes to a human); scoring reports the model's raw format validity next to the
canonicalized result; serving applies the identical step; per-answer outputs can
be saved for diagnosis; the agent task's scoring is unchanged."""

from __future__ import annotations

import importlib.util
import json
import random
import re
import sys

import pytest
import yaml

from mccp.agent.extract import ROUTE_DECIDED, ROUTE_HUMAN, triage_extract
from mccp.evals.benchmark import Target, run_benchmark
from mccp.evals.suite import EvalSuite
from mccp.examples.invoices import generate
from mccp.examples.messy import _po_text, _vid_text, generate_messy, make_case
from mccp.rules import NORMALIZERS
from mccp.rules.extraction import NORMALIZER_ID, canonical_id, normalize_extraction
from tests.conftest import ROOT

SUITE = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text()))


# ------------------------------------------------------------ the normalizer --
def test_every_house_format_in_the_messy_generator_maps_to_the_canonical_id():
    rng = random.Random(0)
    seen_po, seen_v = set(), set()
    for i in range(400):
        c = make_case(rng, i + 2_000_000)
        po, vid = _po_text(c, rng, True), _vid_text(c, rng, True)
        seen_po.add(re.sub(r"\d+", "N", po))
        seen_v.add(re.sub(r"\d+", "N", vid))
        assert canonical_id("po_number", po) == c.po_number
        assert canonical_id("vendor_id", vid) == c.vendor_id
    assert len(seen_po) == 4 and len(seen_v) == 3  # all formats exercised


@pytest.mark.parametrize(
    ("field", "value", "want"),
    [
        ("po_number", "PO-4500012345", "PO-4500012345"),
        ("po_number", 4500012345, "PO-4500012345"),
        ("po_number", "purchase order #12", "PO-12"),
        ("po_number", "No. 4500043094", "PO-4500043094"),  # the field says which id it is
        ("vendor_id", "no. 2100064", "V-2100064"),
        ("vendor_id", "supplier id: 77", "V-77"),
        ("invoice_id", "Invoice #200001", "INV-200001"),
        # ambiguous or another field's prefix: never rewritten
        ("po_number", "PO 45 00", None),
        ("po_number", "PO-45O0", None),
        ("po_number", "V-100", None),
        ("po_number", "PO-123 / 456", None),
        ("vendor_id", "PO-100123", None),
        ("invoice_id", "V-200001", None),
        ("po_number", True, None),
        ("po_number", None, None),
        ("po_number", 12.5, None),
    ],
)
def test_canonical_id_rewrites_only_unambiguous_spellings(field, value, want):
    assert canonical_id(field, value) == want


def test_digits_are_never_changed():
    rng = random.Random(1)
    for _ in range(2000):
        digits = str(rng.randrange(1, 10**12))
        noise = rng.choice(["PO", "P.O.", "po#", "PO No.", "", "purchase order"])
        got = canonical_id("po_number", f"{noise} {digits}")
        assert got == f"PO-{digits}"


def test_normalize_records_changes_and_leaves_other_fields_alone():
    obj = {"invoice_id": "INV-1", "po_number": "P.O. 45", "vendor_id": "V 9", "billed_qty": 3, "unit_price_billed": 2.5}
    out, changes = normalize_extraction(obj)
    assert out == {**obj, "po_number": "PO-45", "vendor_id": "V-9"}
    assert [c["field"] for c in changes] == ["po_number", "vendor_id"]
    assert obj["po_number"] == "P.O. 45"  # input not mutated
    assert normalize_extraction(["not", "a", "dict"]) == (["not", "a", "dict"], [])
    assert NORMALIZERS[NORMALIZER_ID] is normalize_extraction
    assert SUITE.output_normalizer == NORMALIZER_ID


def test_unknown_normalizer_is_rejected():
    raw = yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text())
    with pytest.raises(ValueError, match="unknown output_normalizer"):
        EvalSuite.model_validate({**raw, "output_normalizer": "nope@v1"})


# ----------------------------------------------------------------- scoring --
_WRITTEN = re.compile(r"(?:P\.O\. |po#|PO No\. |PO |vendor no\. |V ?)?(?:PO-|V-)?\d+")


def _written(text: str, digits: str, canonical: str) -> str:
    """The id exactly as the invoice text writes it (house format or canonical)."""
    for m in _WRITTEN.finditer(text):
        if m.group(0).endswith(digits) and re.sub(r"\D", "", m.group(0)) == digits:
            return m.group(0)
    return canonical


def _copying_model(records):
    """Answers with the gold facts but the ids exactly as the invoice wrote them,
    the failure mode seen on the GPU (id_format at 66%)."""
    by_text = {}
    for r in records:
        text = r["messages"][1]["content"]
        facts = json.loads(r["messages"][2]["content"])
        by_text[text] = {
            **facts,
            "po_number": _written(text, facts["po_number"].removeprefix("PO-"), facts["po_number"]),
            "vendor_id": _written(text, facts["vendor_id"].removeprefix("V-"), facts["vendor_id"]),
        }

    async def chat(payload):
        content = json.dumps(by_text[payload["messages"][-1]["content"]])
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {}}

    return chat


async def _l1(records, suite):
    t = Target("adapter", "adapter", "m", _copying_model(records), {}, 4)
    rep = await run_benchmark(sets={"messy": records}, targets=[t], suite=suite)
    return rep["sets"]["messy"]["targets"]["adapter"]["l1"]


async def test_copied_house_formats_are_decided_correctly_and_raw_validity_is_still_reported():
    recs = [r for r in generate_messy(120, task="extract") if r["meta"]["ood_kind"] == "id_format"]
    assert len(recs) == 20
    with_norm = await _l1(recs, SUITE)
    assert with_norm["decision_accuracy"]["mean"] == 1.0
    assert with_norm["json_validity"] == 1.0
    assert with_norm["json_validity_raw"] < 0.5  # what the model wrote: mostly not canonical
    assert with_norm["normalized_answers"] >= 10
    without = await _l1(recs, SUITE.model_copy(update={"output_normalizer": None}))
    assert without["decision_accuracy"]["mean"] == with_norm["json_validity_raw"]  # the GPU's failure mode
    assert without["json_validity_raw"] is None


async def test_agent_task_scoring_is_unchanged():
    agent_suite = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-triage.yaml").read_text()))
    assert agent_suite.output_normalizer is None
    recs = generate(10, seed=3)

    async def chat(payload):
        return {"choices": [{"message": {"role": "assistant", "content": "x"}}], "usage": {}}

    t = Target("adapter", "adapter", "m", chat, {}, 4)
    rep = await run_benchmark(sets={"in_dist": recs}, targets=[t], suite=agent_suite)
    l1 = rep["sets"]["in_dist"]["targets"]["adapter"]["l1"]
    assert l1["json_validity_raw"] is None and l1["normalized_answers"] == 0


async def test_on_scores_receives_raw_answers_and_the_saver_writes_them(tmp_path):
    spec = importlib.util.spec_from_file_location("gpu_pilot_norm", ROOT / "deploy/gpu/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules["gpu_pilot_norm"] = pilot
    spec.loader.exec_module(pilot)
    recs = [r for r in generate_messy(60, task="extract") if r["meta"]["ood_kind"] == "id_format"]
    saver = pilot._OutputSaver(tmp_path, "report_messy.json")
    t = Target("adapter", "adapter", "m", _copying_model(recs), {}, 4)
    await run_benchmark(sets={"messy": recs}, targets=[t], suite=SUITE, on_scores=saver)
    path = tmp_path / "report_messy.messy.adapter.outputs.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == len(recs) and saver.written == [path]
    assert any(r["normalized"] for r in rows) and all(r["json_valid"] for r in rows)
    assert all(r["source_id"] and r["content"] and r["ood_kind"] == "id_format" for r in rows)


# ----------------------------------------------------------------- serving --
async def test_serving_canonicalizes_ids_and_records_the_change():
    rec = next(r for r in generate_messy(60, task="extract") if r["meta"]["ood_kind"] == "id_format")
    facts = json.loads(rec["messages"][2]["content"])
    written = {
        **facts,
        "po_number": "P.O. " + facts["po_number"][3:],
        "vendor_id": "vendor no. " + facts["vendor_id"][2:],
    }

    async def chat(payload):
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(written)}}]}

    lk = rec["meta"]["lookups"]
    out = await triage_extract(chat, "x", model="m", lookup_po=lk["po"].get, lookup_vendor=lk["vendor"].get)
    assert out["route"] == ROUTE_DECIDED and out["decision"] == rec["meta"]["expected"]
    assert {c["field"] for c in out["normalized"]} == {"po_number", "vendor_id"}
    assert out["extracted"]["po_number"] == facts["po_number"]


async def test_serving_still_sends_ambiguous_ids_to_a_human():
    bad = {"invoice_id": "INV-1", "po_number": "PO 45 00", "vendor_id": "V-1", "billed_qty": 1, "unit_price_billed": 2}

    async def chat(payload):
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(bad)}}]}

    out = await triage_extract(chat, "x", model="m", lookup_po=lambda k: None, lookup_vendor=lambda k: None)
    assert out["route"] == ROUTE_HUMAN and out["decision"] is None
