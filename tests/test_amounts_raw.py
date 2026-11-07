"""Amounts as printed, parsed by code (invoice-extract@v2, realdoc-amounts@v1).

On real scans the model dropped decimal points when asked to write a number
without symbols or separators ("$1,533.54" -> 153354). The model now copies the
amount as printed and `parse_money` turns it into a number, identically for the
model's answer and the gold label; text it cannot read routes to a human."""

from __future__ import annotations

import json

import pytest
import yaml

from mccp.agent.extract import ROUTE_DECIDED, ROUTE_HUMAN, triage_extract
from mccp.evals.benchmark import Target, run_benchmark
from mccp.evals.suite import EvalSuite
from mccp.examples import invcdip
from mccp.examples.invoices import EXTRACT_RAW_SCHEMA, generate, generate_ood
from mccp.rules.extraction import normalize_extraction_v2, normalize_realdoc, parse_money
from tests.conftest import ROOT

EXTRACT = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text()))
REALDOC = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-realdoc.yaml").read_text()))


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("$1,533.54", 1533.54),
        ("316.77", 316.77),
        ("USD 1,234.50", 1234.5),
        ("1,234.50 USD", 1234.5),
        ("US$1234.50", 1234.5),
        ("$1,234.50/ea", 1234.5),
        ("$857.74 each", 857.74),
        ("$3,000.", 3000.0),
        ("8,333.00~", 8333.0),
        ("****49140.00", 49140.0),
        ("U.S.DOLLARS  62000.00", 62000.0),  # seen on real scans
        ("U.S. DOLLARS 60,000.00", 60000.0),
        ("62000.00 U.S. Dollars", 62000.0),
        # credit markers flip the sign: never accepted, a person reads them
        ("316:77 CR", None),
        ("316.77 CR", None),
        ("(316.77)", None),
        ("-316.77", None),
        (857.74, 857.74),
        # refused: a human must read these
        ("$35.805.60", None),
        ("316:77", None),
        ("2:851475", None),
        ("1,66667", None),
        (".00", None),
        ("12 34", None),
        ("$1,23.45", None),
        ("3l6.77", None),
        (-5, None),
        (True, None),
        (None, None),
    ],
)
def test_parse_money(text, want):
    assert parse_money(text) == want


def test_v2_moves_the_printed_price_and_flags_unreadable_text():
    obj = {
        "invoice_id": "INV-1",
        "po_number": "P.O. 4",
        "vendor_id": "V-2",
        "billed_qty": 3,
        "unit_price_raw": "$1,234.50",
    }
    out, changes = normalize_extraction_v2(obj)
    assert out == {
        "invoice_id": "INV-1",
        "po_number": "PO-4",
        "vendor_id": "V-2",
        "billed_qty": 3,
        "unit_price_billed": 1234.5,
    }
    assert {c["field"] for c in changes} == {"po_number", "unit_price_raw"}
    bad, changes = normalize_extraction_v2({**obj, "unit_price_raw": "316:77"})
    assert "unit_price_billed" not in bad and "unit_price_raw" not in bad
    assert changes[-1]["error"] == "unparseable amount"
    assert normalize_realdoc({"total_amount_raw": None})[0] == {"total_amount": None}


def test_suite_raw_schemas_match_the_code():
    assert EXTRACT.raw_output_schema == EXTRACT_RAW_SCHEMA
    assert REALDOC.raw_output_schema == invcdip.REALDOC_RAW_SCHEMA
    assert EXTRACT.output_normalizer == "invoice-extract@v2" and REALDOC.output_normalizer == "realdoc-amounts@v1"


def _model(answer):
    async def chat(payload):
        return {"choices": [{"message": {"content": json.dumps(answer(payload))}}], "usage": {}}

    return chat


async def _l1(records, suite, answer, set_name="s"):
    rep = await run_benchmark(
        sets={set_name: records}, targets=[Target("adapter", "adapter", "m", _model(answer), {}, 1)], suite=suite
    )
    return rep["sets"][set_name]["targets"]["adapter"]["l1"]


async def test_replay_of_the_real_scan_failure_decimal_dropped_vs_copied():
    exp = {"invoice_number": "10251", "po_number": None, "total_amount_raw": ["GRAND", "TOTAL", "$1,533.54"][-1]}
    rec = invcdip.to_realdoc_record("ffpm0094", "GRAND TOTAL  $1,533.54", exp, {})
    dropped = await _l1(
        [rec], REALDOC, lambda p: {"invoice_number": "10251", "po_number": None, "total_amount_raw": "153354"}
    )
    copied = await _l1(
        [rec], REALDOC, lambda p: {"invoice_number": "10251", "po_number": None, "total_amount_raw": "$1,533.54"}
    )
    assert dropped["field_accuracy_by_field"]["total_amount"] == 0.0
    assert copied["field_accuracy_by_field"]["total_amount"] == 1.0 and copied["json_validity_raw"] == 1.0


async def test_gold_and_answer_are_parsed_by_the_same_code():
    recs = generate_ood(10, task="extract")  # labels print "$1,234.50"; answer another printed form
    l1 = await _l1(
        recs, EXTRACT, lambda p: {**_gold(recs, p), "unit_price_raw": "USD " + _gold(recs, p)["unit_price_raw"][1:]}
    )
    assert l1["decision_accuracy"]["mean"] == 1.0 and l1["field_accuracy_by_field"]["unit_price_billed"] == 1.0


def _gold(recs, payload):
    text = payload["messages"][-1]["content"]
    return json.loads(next(r for r in recs if r["messages"][1]["content"] == text)["messages"][2]["content"])


async def test_an_older_adapter_answering_a_number_is_still_scored_but_not_raw_valid():
    recs = generate(8, seed=4, task="extract")

    def old_style(p):
        g = _gold(recs, p)
        return {
            **{k: v for k, v in g.items() if k != "unit_price_raw"},
            "unit_price_billed": parse_money(g["unit_price_raw"]),
        }

    l1 = await _l1(recs, EXTRACT, old_style)
    assert l1["decision_accuracy"]["mean"] == 1.0 and l1["json_validity_raw"] == 0.0


async def test_unreadable_price_in_scoring_is_wrong_and_counted():
    recs = generate(6, seed=5, task="extract")
    l1 = await _l1(recs, EXTRACT, lambda p: {**_gold(recs, p), "unit_price_raw": "$35.805.60"})
    assert l1["decision_accuracy"]["mean"] == 0.0 and l1["unparseable_amounts"] == 6 and l1["json_validity"] == 0.0


async def test_serving_routes_an_unreadable_price_to_a_human_and_parses_a_good_one():
    rec = generate(1, seed=6, task="extract")[0]
    lk = rec["meta"]["lookups"]
    gold = json.loads(rec["messages"][2]["content"])
    page = rec["messages"][1]["content"]
    out = await triage_extract(
        _model(lambda p: {**gold, "unit_price_raw": "316:77"}),
        page,
        model="m",
        lookup_po=lk["po"].get,
        lookup_vendor=lk["vendor"].get,
    )
    assert out["route"] == ROUTE_HUMAN and "unparseable amount" in out["reason"] and out["decision"] is None
    ok = await triage_extract(
        _model(lambda p: gold), page, model="m", lookup_po=lk["po"].get, lookup_vendor=lk["vendor"].get
    )
    assert ok["route"] == ROUTE_DECIDED and ok["decision"] == rec["meta"]["expected"]
    assert ok["extracted"]["unit_price_billed"] == parse_money(gold["unit_price_raw"])
