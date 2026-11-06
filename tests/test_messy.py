"""Messy-invoice evaluation set: labels and facts preserved, surface changed."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter

from mccp.examples.invoices import DECISION, decide
from mccp.examples.messy import MESSY_ID_OFFSET, MESSY_KINDS, generate_messy, ocr_noise
from tests.conftest import ROOT

RECS = generate_messy(240, seed=4242)


def _facts(rec):
    msgs = rec["messages"]
    po = json.loads(msgs[3]["content"])
    vendor = json.loads(msgs[5]["content"])
    final = json.loads(msgs[-1]["content"])
    return po, vendor, final


def test_deterministic_and_balanced():
    assert generate_messy(30, seed=1) == generate_messy(30, seed=1)
    assert generate_messy(30, seed=1) != generate_messy(30, seed=2)
    kinds = Counter(r["meta"]["ood_kind"] for r in RECS)
    assert set(kinds) == set(MESSY_KINDS) and len(set(kinds.values())) == 1  # round-robin
    multi = sum(len(r["meta"]["messy_kinds"]) == 2 for r in RECS)
    assert 0.15 * len(RECS) < multi < 0.45 * len(RECS)


def test_labels_follow_the_records_and_cover_every_rule():
    codes = Counter()
    for r in RECS:
        po, vendor, final = _facts(r)
        assert final["decision"] == DECISION[final["reason_code"]]
        codes[final["reason_code"]] += 1
        assert final["variance_pct"] is None or isinstance(final["variance_pct"], float)
    assert len(codes) == 6


def test_numbers_and_ids_survive_and_tool_calls_stay_canonical():
    for r in RECS:
        text = r["messages"][1]["content"]
        inv = r["meta"]["source_id"]
        assert inv in text  # invoice id never damaged
        n = int(inv.removeprefix("INV-")) - 200000
        assert n >= MESSY_ID_OFFSET
        calls = [m["tool_calls"][0]["function"] for m in r["messages"] if m.get("tool_calls")]
        assert calls[0]["name"] == "lookup_po" and re.fullmatch(r"PO-\d{10}", calls[0]["arguments"]["po_number"])
        assert calls[1]["name"] == "lookup_vendor" and re.fullmatch(r"V-\d+", calls[1]["arguments"]["vendor_id"])
        po_digits = calls[0]["arguments"]["po_number"].removeprefix("PO-")
        assert po_digits in text  # the PO is in the text, perhaps in a house format
        assert calls[1]["arguments"]["vendor_id"].removeprefix("V-") in text


def test_billed_price_and_qty_in_text_reproduce_the_label():
    """Re-derive the decision from the messy text + tool results: the label must match."""
    for r in RECS:
        po, vendor, final = _facts(r)
        if "error" in po:
            assert final["reason_code"] == "MISSING_PO"
            continue
        text = r["messages"][1]["content"].replace(",", "")
        prices = re.findall(r"\d+\.\d{2}", text)  # the unit price is the only decimal number in the text
        assert len(prices) == 1, text
        m = re.search(r"(\d+)\s*[xX]\s|\|\s*(\d+)\s*\|", text)
        assert m, text
        qty = int(m.group(1) or m.group(2))
        prices = [float(prices[0])]
        inv = r["meta"]["source_id"]
        assert decide(po, vendor, inv, qty, prices[0])[0] == final["reason_code"], text


def test_distractor_notes_argue_against_the_label():
    seen = 0
    for r in RECS:
        if "distractor_note" in r["meta"]["messy_kinds"]:
            seen += 1
            assert r["messages"][1]["content"].count("\n") >= 1
    assert seen >= 40


def test_ocr_noise_never_touches_digits():
    import random

    rng = random.Random(0)
    text = "Invoice INV-2200001 bills 47 x laptops at $878.18 against PO-4500072682 from Acme Industrial"
    for _ in range(200):
        noisy = ocr_noise(text, rng, rate=0.9)
        assert re.findall(r"\d+", noisy) == re.findall(r"\d+", text)


def test_add_messy_and_curve_messy_column(tmp_path):
    spec = importlib.util.spec_from_file_location("gpu_pilot_messy", ROOT / "deploy/gpu/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules["gpu_pilot_messy"] = pilot
    spec.loader.exec_module(pilot)
    w = tmp_path / "rgt-n100"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/gpu/pilot.py"),
            "prepare",
            "--workdir",
            str(w),
            "--n",
            "60",
            "--test-n",
            "20",
            "--ood-n",
            "12",
            "--messy-n",
            "18",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (w / "data/golden_messy.jsonl").read_text().splitlines()]
    assert len(rows) == 18 and "<EMAIL_" in rows[1]["messages"][1]["content"]  # redacted like other golden sets
    assert json.loads((w / "pilot.json").read_text())["messy"]["records"] == 18
    # add-messy to a workdir prepared without it
    (w / "data/golden_messy.jsonl").unlink()
    r = subprocess.run(
        [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "add-messy", "--workdir", str(w), "--n", "24"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0 and len((w / "data/golden_messy.jsonl").read_text().splitlines()) == 24
    # evaluate refuses a main report without in_dist
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/gpu/pilot.py"),
            "evaluate",
            "--workdir",
            str(w),
            "--vllm-api-key",
            "k",
            "--sets",
            "messy",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 2 and "--report-name" in r.stderr
    # curve picks up report_messy.json
    l1 = {"decision_accuracy": {"mean": 0.9}, "task_score": {"mean": 0.9}, "tool_call_accuracy": 1.0}
    rep = {
        "pilot": {"method": "rgt", "records": 100, "samples": {"train": 300}, "test_set": {"fixed": True}},
        "benchmark": {"sets": {"in_dist": {"targets": {"adapter": {"l1": l1}}}}},
    }
    (w / "report.json").write_text(json.dumps(rep))
    messy_l1 = {**l1, "decision_accuracy": {"mean": 0.7}, "tool_call_accuracy": 0.5}
    (w / "report_messy.json").write_text(
        json.dumps(
            {"benchmark": {"sets": {"messy": {"targets": {"adapter": {"l1": messy_l1}, "base": {"l1": messy_l1}}}}}}
        )
    )
    rows = pilot.curve_rows([w / "report.json"])
    assert rows[0]["decision_messy"] == 0.7 and rows[0]["tool_call_messy"] == 0.5
    md = pilot.curve_markdown(rows, pilot.curve_summary(rows, 0.95))
    assert "Decision acc (messy)" in md and "70.0%" in md and "(messy)." in md
