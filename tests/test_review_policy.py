"""Review policies, the auto-wrong split, label adjudication and the label audit.

Each of the four changes is checked on its own:
  A  realdoc-validation@v2 raises `missing_required_field`; a policy can block on it
  B  audit-labels lists labels the model could not have read; only a person's
     `label_wrong` verdict changes a score
  C  every auto-wrong answer is split by what it got wrong; the split adds up
  D  a policy can block on `several_total_amounts`; other flags stay hints
and the baseline is pinned: v2 with no blocking policy decides exactly what v1
decided, flag for flag."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest
import yaml

from mccp.evals.benchmark import Target, run_benchmark
from mccp.evals.failures import (
    audit_hint,
    classify_rows,
    closest_on_page,
    label_audit_rows,
    load_adjudications,
    wrong_kind,
)
from mccp.evals.suite import EvalSuite
from mccp.examples import invcdip
from mccp.rules.extraction import normalize_realdoc, parse_money
from mccp.rules.review_policy import REVIEW_POLICIES, adhoc_policy
from mccp.rules.validation import (
    MISSING_REQUIRED,
    REALDOC_VALIDATOR_ID,
    REALDOC_VALIDATOR_V2_ID,
    VALIDATOR_FLAGS,
    validate_realdoc,
    validate_realdoc_v2,
)
from tests.conftest import ROOT

SUITE_DOC = yaml.safe_load((ROOT / "examples/suite-invoice-realdoc.yaml").read_text())
REALDOC = EvalSuite.model_validate(SUITE_DOC)
REALDOC_V1 = EvalSuite.model_validate({**SUITE_DOC, "validation": REALDOC_VALIDATOR_ID, "review_policy": None})

# Pages and answers covering every finding: grounded, not on page, several totals,
# total off the total lines, invoice number far from its label, nulls.
PAGE_A = "INVOICE NO 10251\nSUBTOTAL $1,400.00\nTAX $133.54\nGRAND TOTAL $1,533.54"
PAGE_B = "INVOICE 777\nTOTAL $500.00\nBALANCE DUE $250.00\nACCOUNT 0000000500"
PAGE_C = "REMIT TO ACME 4411\nINVOICE DATE 1/2/83\nNET 30 $90.00\nAMOUNT DUE $95.00"
CASES = [
    (PAGE_A, {"invoice_number": "10251", "po_number": None, "total_amount": 1533.54}),
    (PAGE_A, {"invoice_number": "10251", "po_number": None, "total_amount": 153354.0}),
    (PAGE_A, {"invoice_number": None, "po_number": None, "total_amount": 1533.54}),
    (PAGE_A, {"invoice_number": "10251", "po_number": None, "total_amount": None}),
    (PAGE_A, {"invoice_number": "10251", "po_number": None, "total_amount": 1400.0}),
    (PAGE_B, {"invoice_number": "777", "po_number": None, "total_amount": 500.0}),
    (PAGE_B, {"invoice_number": "0000000500", "po_number": None, "total_amount": 250.0}),
    (PAGE_B, {"invoice_number": "", "po_number": "PO-9", "total_amount": None}),
    (PAGE_C, {"invoice_number": "4411", "po_number": None, "total_amount": 90.0}),
    (PAGE_C, {"invoice_number": None, "po_number": None, "total_amount": None}),
]


# -- baseline: v2 + no blocking policy == v1 --------------------------------------


@pytest.mark.parametrize(("page", "canon"), CASES)
def test_v2_reports_v1_findings_unchanged_plus_missing(page, canon):
    v1, v2 = validate_realdoc(None, canon, page), validate_realdoc_v2(None, canon, page)
    assert v2.blocking == v1.blocking and v2.grounded == v1.grounded
    assert [f for f in v2.flags if f != MISSING_REQUIRED] == v1.flags
    assert v2.ok == v1.ok  # the new finding never blocks by itself


def test_v1_output_shape_is_unchanged():
    d = validate_realdoc(None, CASES[2][1], PAGE_A).to_dict()
    assert set(d) == {"validator", "blocking", "flags", "grounded", "ok"} and d["validator"] == REALDOC_VALIDATOR_ID


def _row(sid, exp, got, page):
    return {
        "kind": "final",
        "source_id": sid,
        "expected": json.dumps(exp),
        "content": json.dumps(got),
        "json_valid": True,
        "input": page,
    }


def _raw(c):
    t = c["total_amount"]
    return {"invoice_number": c["invoice_number"], "po_number": c["po_number"], "total_amount_raw": None if t is None else f"${t:,.2f}"}  # fmt: skip


GOLD = {"invoice_number": "10251", "po_number": None, "total_amount_raw": "$1,533.54"}
GOLD_B = {"invoice_number": "777", "po_number": None, "total_amount_raw": "$500.00"}
GOLD_C = {"invoice_number": "4411", "po_number": None, "total_amount_raw": "$95.00"}
ROWS = [
    _row(f"r{i}", {PAGE_A: GOLD, PAGE_B: GOLD_B, PAGE_C: GOLD_C}[page], _raw(c), page)
    for i, (page, c) in enumerate(CASES)
]


def test_baseline_v2_with_no_blocking_policy_decides_exactly_like_v1():
    v1 = classify_rows(ROWS, REALDOC_V1, normalize_realdoc, validator=validate_realdoc)
    none = REVIEW_POLICIES["realdoc-review-flags-none@v1"]
    v2 = classify_rows(ROWS, REALDOC, normalize_realdoc, validator=validate_realdoc_v2, policy=none)
    assert v2["outcomes"] == v1["outcomes"] and v2["causes"] == v1["causes"]
    assert {k: v for k, v in v2["flags"].items() if k != MISSING_REQUIRED} == v1["flags"]
    # every simulation row of the v1 flags is reproduced number for number
    rows1 = {tuple(p["block_on"]): p for p in v1["policies"]}
    rows2 = {tuple(p["block_on"]): p for p in v2["policies"]}
    for k, p in rows1.items():
        assert rows2[k] == p


def test_suite_default_is_the_baseline():
    assert REALDOC.validation == REALDOC_VALIDATOR_V2_ID and REALDOC.review_policy == "realdoc-review-flags-none@v1"
    assert REVIEW_POLICIES[REALDOC.review_policy].block_on == frozenset()


# -- A: missing required field -----------------------------------------------------


def test_missing_required_field_is_a_finding_for_invoice_number_and_total_only():
    v = validate_realdoc_v2(None, {"invoice_number": None, "po_number": None, "total_amount": 1533.54}, PAGE_A)
    assert MISSING_REQUIRED in v.flags and v.missing == ["invoice_number"] and v.ok
    assert v.to_dict()["missing"] == ["invoice_number"]
    po_only = validate_realdoc_v2(None, {"invoice_number": "10251", "po_number": None, "total_amount": 1533.54}, PAGE_A)
    assert MISSING_REQUIRED not in po_only.flags  # most invoices print no PO
    blank = validate_realdoc_v2(None, {"invoice_number": " - ", "po_number": None, "total_amount": None}, PAGE_A)
    assert blank.missing == ["invoice_number", "total_amount"]


def test_missing_policy_routes_a_null_field_to_a_person():
    pol = REVIEW_POLICIES["realdoc-review-missing@v1"]
    res = classify_rows(ROWS, REALDOC, normalize_realdoc, validator=validate_realdoc_v2, policy=pol)
    base = classify_rows(ROWS, REALDOC, normalize_realdoc, validator=validate_realdoc_v2)
    hits = base["flags"][MISSING_REQUIRED]
    assert hits.get("on_wrong", 0) > 0  # the fixture has null fields that would be auto-wrong
    assert res["review_policy_blocked"] == {MISSING_REQUIRED: hits.get("on_correct", 0) + hits.get("on_wrong", 0)}
    assert res["outcomes"]["auto_wrong"] == base["outcomes"]["auto_wrong"] - hits["on_wrong"]


# -- D and the policy machinery ----------------------------------------------------


def test_policy_blocks_only_its_flags_and_passes_the_rest_as_hints():
    pol = REVIEW_POLICIES["realdoc-review-several-totals@v1"]
    by, hints = pol.decide({"flags": ["several_total_amounts", "invoice_number_not_near_label"]})
    assert by == ["several_total_amounts"] and hints == ["invoice_number_not_near_label"]
    assert pol.decide(None) == ([], [])


@pytest.mark.parametrize("pid", sorted(REVIEW_POLICIES))
def test_reported_outcomes_equal_the_simulation_row_of_the_same_policy(pid):
    """The policy actually applied and the simulated policy are the same thing."""
    pol = REVIEW_POLICIES[pid]
    res = classify_rows(ROWS, REALDOC, normalize_realdoc, validator=validate_realdoc_v2, policy=pol)
    sim = {tuple(p["block_on"]): p for p in res["policies"]}
    present = tuple(sorted(f for f in pol.block_on if f in res["flags"]))
    row = sim[present]
    n = res["rows"]
    assert round(res["outcomes"].get("auto_wrong", 0) / n, 4) == row["auto_wrong"]
    assert round(res["outcomes"].get("human_review", 0) / n, 4) == row["human_review"]
    assert res["auto_wrong_by_kind"] == row["auto_wrong_by_kind"]


def test_every_registered_policy_names_only_flags_its_validator_raises():
    for p in REVIEW_POLICIES.values():
        assert p.block_on <= VALIDATOR_FLAGS[p.validator]
    with pytest.raises(ValueError, match="never raises"):
        adhoc_policy(REALDOC_VALIDATOR_ID, [MISSING_REQUIRED])  # v1 has no such finding
    assert adhoc_policy(REALDOC_VALIDATOR_V2_ID, ["several_total_amounts"]).block_on == {"several_total_amounts"}


def test_suite_rejects_unknown_or_mismatched_policy():
    with pytest.raises(ValueError, match="unknown review_policy"):
        EvalSuite.model_validate({**SUITE_DOC, "review_policy": "nope@v1"})
    with pytest.raises(ValueError, match="reads realdoc-validation@v2"):
        EvalSuite.model_validate({**SUITE_DOC, "validation": REALDOC_VALIDATOR_ID})


async def test_live_benchmark_applies_the_suite_policy():
    rec = invcdip.to_realdoc_record("ffpm0094", PAGE_A, GOLD, {})
    suite = EvalSuite.model_validate({**SUITE_DOC, "review_policy": "realdoc-review-missing@v1"})

    async def chat(payload):
        ans = {**GOLD, "invoice_number": None}
        return {"choices": [{"message": {"content": json.dumps(ans)}}], "usage": {}}

    rep = await run_benchmark(sets={"s": [rec]}, targets=[Target("adapter", "adapter", "m", chat, {}, 1)], suite=suite)
    l1 = rep["sets"]["s"]["targets"]["adapter"]["l1"]
    assert l1["outcomes"]["human_review"]["n"] == 1 and l1["outcomes"]["auto_wrong"]["n"] == 0
    assert l1["outcomes"]["review_policy"] == "realdoc-review-missing@v1"
    assert l1["review_policy_blocking"] == {MISSING_REQUIRED: 1}


# -- C: what the auto-wrong got wrong ----------------------------------------------


def test_wrong_kind():
    assert wrong_kind({"total_amount"}) == "amount_only"
    assert wrong_kind({"invoice_number", "po_number"}) == "id_only"
    assert wrong_kind({"total_amount", "invoice_number"}) == "amount_and_id"
    assert wrong_kind({"billed_qty"}) == "other"


def test_split_adds_up_to_the_auto_wrong_count_in_every_simulation_row():
    res = classify_rows(ROWS, REALDOC, normalize_realdoc, validator=validate_realdoc_v2)
    assert sum(res["auto_wrong_by_kind"].values()) == res["outcomes"].get("auto_wrong", 0)
    for p in res["policies"]:
        assert sum(p["auto_wrong_by_kind"].values()) == p["auto_wrong_n"]
        assert sum(p["auto_wrong_by_field"].values()) >= p["auto_wrong_n"]


# -- B: label audit and adjudication -----------------------------------------------

LOST = {"invoice_number": "98765", "po_number": None, "total_amount_raw": "$1,533.54"}  # 98765 not on PAGE_A


def test_audit_lists_only_labels_not_in_the_text():
    rows = [_row("lost", LOST, {**LOST, "invoice_number": "10251"}, PAGE_A), _row("ok", GOLD, GOLD, PAGE_A)]
    items = label_audit_rows(rows, REALDOC, normalize_realdoc, validator=validate_realdoc_v2)
    assert [(i["source_id"], i["field"]) for i in items] == [("lost", "invoice_number")]
    it = items[0]
    assert it["outcome"] == "auto_wrong" and it["answer_on_page"] and it["hint"] == "answer_on_page_label_absent"


def test_closest_on_page_and_hint():
    assert closest_on_page("10521", "INVOICE NO 10251 x") == ("10251", 2)
    assert audit_hint("8100005268", False, 1) == "ocr_near_miss"
    assert audit_hint("10251", True, None) == "answer_on_page_label_absent"
    assert audit_hint("10251", False, 4) == "absent"


def _sheet(rows):
    return [{"source_id": s, "field": f, "verdict": v, "corrected_label": c} for s, f, v, c in rows]


def test_only_a_verified_label_error_changes_the_score():
    rows = [_row("lost", LOST, {**LOST, "invoice_number": "10251"}, PAGE_A)]
    kw = {"validator": validate_realdoc_v2}
    measured = classify_rows(rows, REALDOC, normalize_realdoc, **kw)
    assert measured["outcomes"] == {"auto_wrong": 1}
    for verdict in ("ocr_lost", "model_wrong"):
        same = classify_rows(rows, REALDOC, normalize_realdoc, overrides=load_adjudications(_sheet([("lost", "invoice_number", verdict, "")])), **kw)  # fmt: skip
        assert same["outcomes"] == {"auto_wrong": 1}  # OCR is part of the pipeline: stays scored
    unscored = load_adjudications(_sheet([("lost", "invoice_number", "label_wrong", "")]))
    assert classify_rows(rows, REALDOC, normalize_realdoc, overrides=unscored, **kw)["outcomes"] == {"auto_correct": 1}
    corrected = load_adjudications(_sheet([("lost", "invoice_number", "label_wrong", "10251")]))
    assert classify_rows(rows, REALDOC, normalize_realdoc, overrides=corrected, **kw)["outcomes"] == {"auto_correct": 1}
    wrong_fix = load_adjudications(_sheet([("lost", "invoice_number", "label_wrong", "55555")]))
    assert classify_rows(rows, REALDOC, normalize_realdoc, overrides=wrong_fix, **kw)["outcomes"] == {"auto_wrong": 1}


def test_corrected_total_goes_through_the_same_parser():
    rows = [_row("t", {**GOLD, "total_amount_raw": "$1,999.00"}, GOLD, PAGE_A)]
    fix = load_adjudications(_sheet([("t", "total_amount", "label_wrong", "$1,533.54")]))
    res = classify_rows(rows, REALDOC, normalize_realdoc, validator=validate_realdoc_v2, overrides=fix)
    assert res["outcomes"] == {"auto_correct": 1} and res["label_overrides_applied"] == 1


@pytest.mark.parametrize(
    ("rows", "msg"),
    [
        ([("a", "invoice_number", "typo", "")], "not one of"),
        ([("a", "invoice_number", "ocr_lost", "123")], "only goes with label_wrong"),
        ([("a", "invoice_number", "", "123")], "without a verdict"),
        ([("a", "total_amount", "label_wrong", "$9.999.99")], "not readable"),  # never guessed
        ([("a", "total_amount", "label_wrong", "99.999.99")], "not readable"),
        ([("a", "invoice_number", "ocr_lost", ""), ("a", "invoice_number", "model_wrong", "")], "twice"),
    ],
)
def test_adjudication_sheet_errors_are_never_guessed_around(rows, msg):
    with pytest.raises(ValueError, match=msg):
        load_adjudications(_sheet(rows))


def test_blank_verdicts_are_skipped():
    assert load_adjudications(_sheet([("a", "invoice_number", "", "")])) == {}


@pytest.mark.parametrize("text", ["$9.999.99", "99.999.99", "9.999.99", "$1.234.56"])
def test_ambiguous_punctuation_is_still_refused(text):
    assert parse_money(text) is None


def test_label_box_scales_annotation_pixels_to_the_page():
    ann = {
        "image_dims": "[3000, 2000, 1]",
        "Fields": [
            {"key": {}, "value": {"label": "amount_due", "tag": "$5.00", "bbox": {"xmin": 200, "ymin": 300, "xmax": 400, "ymax": 600}}},
            {"key": {}, "value": {"label": "Invoice_number", "tag": "7", "bbox": {"xmin": 1000, "ymin": 1500, "xmax": 2000, "ymax": 3000}}},
        ],
    }  # fmt: skip
    assert invcdip.label_box(ann, "invoice_number") == (0.5, 0.5, 1.0, 1.0)
    assert invcdip.label_box(ann, "total_amount") == (0.1, 0.1, 0.2, 0.2)  # falls back to amount_due
    assert invcdip.label_box(ann, "po_number") is None
    assert invcdip.label_box({"Fields": []}, "invoice_number") is None


# -- CLI: read-only on the run folder ----------------------------------------------


def _folder(tmp_path):
    w = tmp_path / "mccp-20990101-000000-x" / "extract-n1000-e1-s42"
    w.mkdir(parents=True)
    (w / "pilot.json").write_text(json.dumps({"task": "extract"}))
    (w / "report_invcdip.json").write_text(json.dumps({"benchmark": {"sets": {"invcdip": {"targets": {}}}}}))
    rows = [
        _row("ffpm0094", LOST, {**LOST, "invoice_number": "10251"}, PAGE_A),
        _row("ffpm0095", GOLD, {**GOLD, "invoice_number": None}, PAGE_A),
        _row("ffpm0096", GOLD, GOLD, PAGE_A),
    ]
    (w / "report_invcdip.invcdip.adapter.outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return w.parent


def _digest(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}  # fmt: skip


def _pilot(*a):
    cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), *a, "--cache", "/nonexistent-cache"]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_audit_and_failures_write_outside_the_run_folder_and_keep_verdicts(tmp_path):
    run = _folder(tmp_path)
    before = _digest(run)
    r = _pilot("audit-labels", str(run))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "mccp-audit" / run.name / "invcdip"
    sheet = out / "audit.csv"
    with sheet.open() as fh:
        items = list(csv.DictReader(fh))
    assert [(i["source_id"], i["field"]) for i in items] == [("ffpm0094", "invoice_number")]
    assert (out / "index.html").exists() and _digest(run) == before
    items[0]["verdict"], items[0]["corrected_label"] = "label_wrong", "10251"
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    assert _pilot("audit-labels", str(run)).returncode == 0  # a re-run keeps what a person typed
    with sheet.open() as fh:
        assert next(csv.DictReader(fh))["verdict"] == "label_wrong"
    r = _pilot("failures", str(run), "--set", "invcdip", "--policy", "realdoc-review-missing@v1",
               "--adjudications", str(sheet), "--json")  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "auto_wrong=1 (33.3%)" in r.stdout  # measured: the lost label
    assert "excluding 1 verified label errors: auto_correct=2 (66.7%)" in r.stdout
    assert "missing_required_field" in r.stdout and "wrong amt/id/both" in r.stdout
    assert (out / "failures_extract-n1000-e1-s42_adapter.json").exists()
    assert _digest(run) == before  # nothing in the run folder changed


def test_out_inside_the_run_folder_is_refused(tmp_path):
    run = _folder(tmp_path)
    r = _pilot("audit-labels", str(run), "--out", str(run / "audit"))
    assert r.returncode != 0 and "inside the run folder" in (r.stderr + r.stdout)


def test_unknown_flag_in_block_on_is_refused(tmp_path):
    run = _folder(tmp_path)
    r = _pilot("failures", str(run), "--set", "invcdip", "--block-on", "made_up_flag")
    assert r.returncode != 0 and "never raises" in r.stderr


def _load_pilot():
    spec = importlib.util.spec_from_file_location("gpu_pilot_rp", ROOT / "deploy/gpu/pilot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_pilot_rp"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_policy_rows_are_baseline_singles_and_registered_combinations():
    pilot = _load_pilot()
    pols = [{"block_on": list(c)} for c in [(), ("a",), ("b",), ("a", "b"), (MISSING_REQUIRED, "several_total_amounts"), (MISSING_REQUIRED, "a")]]  # fmt: skip
    got = [tuple(p["block_on"]) for p in pilot._pick_policies(pols, "default", REALDOC_VALIDATOR_V2_ID, None)]
    assert got == [(), ("a",), ("b",), (MISSING_REQUIRED, "several_total_amounts")]
    assert len(pilot._pick_policies(pols, "all", REALDOC_VALIDATOR_V2_ID, None)) == len(pols)
    assert [tuple(p["block_on"]) for p in pilot._pick_policies(pols, "a+b", None, None)] == [(), ("a", "b")]


def test_audit_images_box_the_label_on_a_cached_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((300, 100), "INVOICE NO 10251")
    pdf = doc.tobytes()
    full, crop = invcdip.audit_images(pdf, (0.5, 0.1, 0.7, 0.14))
    assert full.mode == "RGB" and crop is not None and crop.size[0] < full.size[0]
    assert full.getpixel((int(0.5 * full.size[0]) - 3, int(0.12 * full.size[1])))[0] > 200  # the red box
    assert invcdip.audit_images(pdf, None)[1] is None
    run = _folder(tmp_path)
    cache = tmp_path / "cache"
    (cache / "invcdip" / "pdf").mkdir(parents=True)
    (cache / "invcdip" / "pdf" / "ffpm0094.pdf").write_bytes(pdf)
    cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "audit-labels", str(run), "--cache", str(cache)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mccp-audit" / run.name / "invcdip" / "images" / "ffpm0094_invoice_number_page.png").exists()
