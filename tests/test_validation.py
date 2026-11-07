"""Financial-field validation (mccp.rules.validation) and failure classification
(mccp.evals.failures).

The validator never corrects; it confirms a value against the page or blocks the
invoice for a person. Two things must hold: correct answers are never blocked
(checked on every gold answer of every test set), and the failure modes seen on
real scans are blocked (dropped decimals, values not on the page, a line amount
taken as the unit price)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest
import yaml

from mccp.agent.extract import ROUTE_DECIDED, ROUTE_HUMAN, triage_extract
from mccp.evals.benchmark import Target, run_benchmark
from mccp.evals.failures import classify_amount, classify_rows, line_kind
from mccp.evals.suite import EvalSuite
from mccp.examples import invcdip
from mccp.examples.invoices import generate, generate_ood
from mccp.examples.messy import generate_messy
from mccp.rules.extraction import normalize_extraction_v2, normalize_realdoc
from mccp.rules.validation import (
    amount_on_page,
    candidate_totals,
    line_arithmetic,
    validate_extract,
    validate_realdoc,
)
from tests.conftest import ROOT

EXTRACT = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-extract.yaml").read_text()))
REALDOC = EvalSuite.model_validate(yaml.safe_load((ROOT / "examples/suite-invoice-realdoc.yaml").read_text()))

PAGE = (
    "GRAND TOTAL  $1,533.54\nTOTAL CHARGES  316.77\n1875 I ST NW  $22.25\n****TOTAL GROSS  49140.00\n"
    "NET 41,769.00\nSubtotal $35,805.60\npump seals 12 705.55 8,466.60\nTotal 1, 533.54"
)


@pytest.mark.parametrize(
    ("value", "on"),
    [
        (1533.54, True), (153354, False), (316.77, True), (31677, False), (22.25, True), (2225, False),
        (49140, True), (41769, True), (1840, False), (35805.6, True), (1875, True), (705.55, True),
        (8466.6, True), (533.54, False), (1.53, False), (None, False),
    ],
)  # fmt: skip
def test_amount_on_page(value, on):
    assert amount_on_page(value, PAGE) is on


def test_line_arithmetic():
    doc = "E90805  pump seals  12  $705.55  $8,466.60\nFRT Freight 1 $114.99 $114.99"
    assert line_arithmetic(12, 705.55, doc) == "corroborated"
    assert line_arithmetic(12, 8466.60, doc) == "contradicted"  # line amount taken as the unit price
    assert line_arithmetic(13, 705.55, doc) == "contradicted"
    assert line_arithmetic(68, 857.74, "bills 68 x steel pipe at $857.74 each against PO-4500090658.") == "absent"


def test_candidate_totals_skip_subtotal():
    assert candidate_totals(PAGE) == [1533.54, 316.77, 49140.0]


# --------------------------------------------- correct answers are never blocked --
@pytest.mark.parametrize(
    "records",
    [
        generate(300, seed=7777, id_offset=1_000_000, task="extract"),
        generate_ood(200, task="extract"),
        generate_messy(300, task="extract"),
    ],
    ids=["in_dist", "ood", "messy"],
)
def test_no_gold_answer_is_blocked(records):
    blocked = []
    for r in records:
        raw = json.loads(r["messages"][2]["content"])
        v = validate_extract(raw, normalize_extraction_v2(raw)[0], r["messages"][1]["content"])
        if not v.ok:
            blocked.append((r["meta"]["source_id"], v.blocking))
    assert blocked == []


def test_no_gold_answer_is_blocked_on_rendered_documents():
    pytest.importorskip("rapidocr_onnxruntime")
    from mccp.examples.documents import generate_documents

    for r in generate_documents(4, seed=11):
        if r["meta"]["scan"] != "clean":
            continue  # OCR damage can legitimately remove a value from the page
        raw = json.loads(r["messages"][2]["content"])
        v = validate_extract(raw, normalize_extraction_v2(raw)[0], r["messages"][1]["content"])
        assert v.ok, v.blocking
        assert "arithmetic_corroborated" in v.flags


# ------------------------------------------------------ failure modes are blocked --
def _doc_record():
    text = (
        "OCR text of a scanned vendor invoice:\n\nAcme Industrial\nInvoice No. INV-3200008  P.O. Number: PO-4500066947\n"
        "Supplier ID: V-3100008\nD56171 filters 90 $397.84 $35,805.60\nTOTAL DUE $37,595.88"
    )
    facts = {"invoice_id": "INV-3200008", "po_number": "PO-4500066947", "vendor_id": "V-3100008", "billed_qty": 90}
    return text, facts


@pytest.mark.parametrize(
    ("price_raw", "blocked_by"),
    [
        ("$397.84", None),
        ("39784", "not_on_page:unit_price_billed"),  # decimal dropped
        ("$35,805.60", "line_amount_mismatch"),  # line amount taken as the unit price
        ("$397.48", "not_on_page:unit_price_billed"),  # digits swapped
    ],
)
def test_extract_validation_blocks_the_observed_failure_modes(price_raw, blocked_by):
    text, facts = _doc_record()
    raw = {**facts, "unit_price_raw": price_raw}
    v = validate_extract(raw, normalize_extraction_v2(raw)[0], text)
    assert (v.blocking[0] if v.blocking else None) == blocked_by


def test_realdoc_validation_grounds_ids_and_amounts_and_flags_ambiguous_totals():
    canon = normalize_realdoc({"invoice_number": "10251", "po_number": None, "total_amount_raw": "153354"})[0]
    v = validate_realdoc(None, canon, "INVOICE 10251\nGRAND TOTAL $1,533.54")
    assert v.blocking == ["not_on_page:total_amount"]
    ok = validate_realdoc(
        None, {"invoice_number": "10251", "po_number": None, "total_amount": 41769.0}, PAGE + "\n10251"
    )
    assert ok.ok and "total_not_on_total_line" in ok.flags and "several_total_amounts" in ok.flags


async def test_scoring_reports_business_outcomes():
    recs = generate_ood(20, task="extract")
    gold = {r["messages"][1]["content"]: json.loads(r["messages"][2]["content"]) for r in recs}

    def model(transform):
        async def chat(payload):
            g = gold[payload["messages"][-1]["content"]]
            return {"choices": [{"message": {"content": json.dumps(transform(g))}}], "usage": {}}

        return chat

    async def outcomes(transform):
        t = Target("adapter", "adapter", "m", model(transform), {}, 4)
        rep = await run_benchmark(sets={"ood": recs}, targets=[t], suite=EXTRACT)
        return rep["sets"]["ood"]["targets"]["adapter"]["l1"]

    right = await outcomes(lambda g: g)
    assert right["outcomes"]["auto_correct"]["rate"] == 1.0 and right["outcomes"]["auto_wrong"]["n"] == 0
    dropped = await outcomes(lambda g: {**g, "unit_price_raw": g["unit_price_raw"].replace(".", "").replace(",", "")})
    assert dropped["outcomes"]["human_review"]["rate"] == 1.0 and dropped["outcomes"]["auto_wrong"]["n"] == 0
    assert dropped["validation_blocking"]["not_on_page:unit_price_billed"] == 20


async def test_serving_blocks_an_ungrounded_price_before_any_lookup():
    text, facts = _doc_record()
    calls = []

    async def chat(payload):
        return {"choices": [{"message": {"content": json.dumps({**facts, "unit_price_raw": "39784"})}}]}

    def lookup(k):
        calls.append(k)
        return None

    out = await triage_extract(chat, text, model="m", lookup_po=lookup, lookup_vendor=lookup)
    assert out["route"] == ROUTE_HUMAN and "not_on_page:unit_price_billed" in out["reason"]
    assert out["validation"]["ok"] is False and calls == []  # no lookup, no decision

    async def good(payload):
        return {"choices": [{"message": {"content": json.dumps({**facts, "unit_price_raw": "$397.84"})}}]}

    ok = await triage_extract(good, text, model="m", lookup_po=lambda k: None, lookup_vendor=lambda k: None)
    assert ok["route"] == ROUTE_DECIDED and ok["validation"]["ok"] is True


# ---------------------------------------------------------- failure classification --
@pytest.mark.parametrize(
    ("expected", "got", "raw", "cause"),
    [
        (1533.54, 153354.0, "153354", "decimal_dropped"),
        (316.77, 31677.0, "31677", "decimal_dropped"),
        (49140.0, 41769.0, "41,769.00", "wrong_amount_on_page"),
        (1610.0, 1840.0, "1840", "label_not_on_page"),
        (22.25, 22.52, "22.52", "invented_or_misread"),
        (22.25, None, "22:25", "unparseable_copy"),
        (22.25, None, None, "missing"),
    ],
)
def test_classify_amount(expected, got, raw, cause):
    assert classify_amount(expected, got, raw, PAGE) == cause


def test_line_kind():
    assert line_kind(41769.0, PAGE) == "net" and line_kind(35805.6, PAGE) == "subtotal"


def test_classify_rows_counts_causes_and_what_validation_caught():
    def row(sid, exp, got, score):
        return {
            "kind": "final",
            "source_id": sid,
            "expected": json.dumps(exp),
            "content": json.dumps(got),
            "json_valid": True,
            "score": score,
            "input": PAGE + "\nINVOICE 10251",
        }

    e = {"invoice_number": "10251", "po_number": None, "total_amount_raw": "$1,533.54"}
    rows = [
        row("a", e, {**e}, 1.0),
        row("b", e, {**e, "total_amount_raw": "153354"}, 0.5),  # caught: not on page
        row("c", e, {**e, "total_amount_raw": "41,769.00"}, 0.5),  # on page: auto_wrong
    ]
    from mccp.rules.validation import validate_realdoc as v

    res = classify_rows(rows, REALDOC, normalize_realdoc, validator=v)
    assert res["outcomes"] == {"auto_correct": 1, "human_review": 1, "auto_wrong": 1}
    assert res["causes"]["total_amount"] == {"decimal_dropped": 1, "wrong_amount_on_page (net)": 1}
    assert res["caught_by_validation"]["total_amount"] == {"decimal_dropped": 1}


def test_failures_cli_on_a_saved_folder(tmp_path):
    spec = importlib.util.spec_from_file_location("gpu_pilot_fail", ROOT / "deploy/gpu/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules["gpu_pilot_fail"] = pilot
    spec.loader.exec_module(pilot)
    w = tmp_path / "mccp-20990101-000000-x" / "extract-n1000-e1-s42"
    w.mkdir(parents=True)
    (w / "pilot.json").write_text(json.dumps({"task": "extract"}))
    (w / "report_invcdip.json").write_text(json.dumps({"benchmark": {"sets": {"invcdip": {"targets": {}}}}}))
    exp = {"invoice_number": "10251", "po_number": None, "total_amount_raw": "$1,533.54"}
    rec = invcdip.to_realdoc_record("ffpm0094", "INVOICE 10251\nGRAND TOTAL $1,533.54", exp, {})
    rows = [
        {
            "kind": "final", "source_id": "ffpm0094", "expected": rec["messages"][2]["content"],
            "content": json.dumps({**exp, "total_amount_raw": "153354"}), "json_valid": True, "score": 0.5,
            "input": rec["messages"][1]["content"],
        }
    ]  # fmt: skip
    (w / "report_invcdip.invcdip.adapter.outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "failures", str(w.parent), "--examples", "1", "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "decimal_dropped" in r.stdout and "human_review=1" in r.stdout and "page: GRAND TOTAL $1,533.54" in r.stdout
    assert (w / "failures_invcdip_adapter.json").exists()


# The two docs-set answers that got through on the GPU (INV-3200077, INV-3200173):
# the quantity "1" came from the freight line; qty x price = price fooled the check.
_LEAK_PAGES = [
    (
        {"invoice_id": "INV-3200077", "po_number": "PO-4500048190", "vendor_id": "V-3100077"},
        "INV-3200077 PO-4500048190 V-3100077\nItem  Description  Qty 138 Unit Price $418.05  $57,690.90 Amount\n"
        "FRT  Freight and handling  1  $114.99  $114.99",
        138,
        418.05,
    ),
    (  # OCR garbled the line amount ($88.114.15), so only the quantity's line gives it away
        {"invoice_id": "INV-3200173", "po_number": "PO-4500058265", "vendor_id": "V-3100173"},
        "INV-3200173 PO-4500058265 V-3100173\nltem  Description  Qty 193 Unit Price $456.55  $88.114.15 Amount\n"
        "FRT  Freight and handling  1  $85.00  $85.00",
        193,
        456.55,
    ),
]


@pytest.mark.parametrize(("ids", "page", "qty", "price"), _LEAK_PAGES)
def test_quantity_from_another_line_is_blocked(ids, page, qty, price):
    wrong = validate_extract(None, {**ids, "billed_qty": 1, "unit_price_billed": price}, page)
    assert "qty_not_on_price_line" in wrong.blocking
    right = validate_extract(None, {**ids, "billed_qty": qty, "unit_price_billed": price}, page)
    assert right.ok


def test_qty_equal_one_never_corroborates_by_itself():
    assert line_arithmetic(1, 418.05, "Unit Price $418.05  $57,690.90") == "contradicted"
    assert line_arithmetic(1, 418.05, "Unit Price $418.05") == "absent"


def test_invoice_number_label_flag():
    from mccp.rules.validation import validate_realdoc as vr

    # the line under a label counts as near (values often print below their label)
    page = "MAROUIS TELEPHONE 555 FAX 556  INVOICE DATE 951695\nSOLD TO ACME\nSHIP TO ACME\nACCOUNT 0000000500"
    near = vr(None, {"invoice_number": "951695", "po_number": None, "total_amount": None}, page)
    far = vr(None, {"invoice_number": "0000000500", "po_number": None, "total_amount": None}, page)
    assert "invoice_number_not_near_label" not in near.flags and "invoice_number_not_near_label" in far.flags


def test_policy_simulation_uses_the_union_of_flagged_invoices():
    from mccp.evals.failures import simulate_policies

    passed = [
        ("a", True, frozenset()),
        ("b", True, frozenset({"x"})),
        ("c", False, frozenset({"x", "y"})),  # flagged by both: stopped once, not twice
        ("d", False, frozenset({"y"})),
        ("e", False, frozenset()),
    ]
    pols = {
        tuple(p["block_on"]): p
        for p in simulate_policies(passed, {"auto_correct": 2, "auto_wrong": 3, "human_review": 5})
    }
    assert pols[()]["auto_wrong"] == 0.3 and pols[()]["human_review"] == 0.5
    assert (pols[("x",)]["stops_wrong"], pols[("x",)]["extra_reviews_of_correct"]) == (1, 1)
    both = pols[("x", "y")]
    assert (both["stops_wrong"], both["extra_reviews_of_correct"]) == (2, 1)  # c counted once
    assert both["auto_wrong"] == 0.1 and both["human_review"] == 0.8 and both["auto_correct"] == 0.1


def test_classifier_rescores_with_current_code_not_saved_scores():
    """A saved row marked invalid/wrong at evaluation time is judged by today's parser:
    'U.S.DOLLARS 62000.00' was refused then and parses now."""
    exp = {"invoice_number": "77", "po_number": None, "total_amount_raw": "62000.00"}
    page = "INVOICE 77\nTOTAL U.S.DOLLARS 62000.00"
    row = {
        "kind": "final", "source_id": "fklk0189", "expected": json.dumps(exp), "json_valid": False, "score": 0.0,
        "content": json.dumps({**exp, "total_amount_raw": "U.S.DOLLARS  62000.00"}), "input": page,
    }  # fmt: skip
    from mccp.rules.validation import validate_realdoc as v

    res = classify_rows([row], REALDOC, normalize_realdoc, validator=v)
    assert res["outcomes"] == {"auto_correct": 1}
