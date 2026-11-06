"""Boundary training mix: near-threshold density, correct labels, standard mix unchanged."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys

import pytest

from mccp.examples.invoices import TRAIN_MIXES, decide, generate, make_case
from tests.conftest import ROOT

# Fingerprints of the standard mix taken before the boundary mix existed: old datasets
# (and the Nov 6 2026 learning-curve run) must reproduce byte for byte from their seeds.
STANDARD_SHA = {
    (1000, 42, 0): "f7e16879511eeeb28734f515ff5e610e31df95ab382c565bac8651bc839d7939",
    (300, 7777, 1_000_000): "6d1ad600860722a0da4b7eb63056b2b394fd074ff9cc13cff59002a4503143e7",
    (50, 7, 0): "60dd9cea2067640ef5b936d74880c8c74852d73a164d75f2088bc27c0400c182",
}


@pytest.mark.parametrize(("n", "seed", "offset"), list(STANDARD_SHA))
def test_standard_mix_is_byte_identical_to_before(n, seed, offset):
    blob = json.dumps(generate(n, seed=seed, id_offset=offset), sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest() == STANDARD_SHA[(n, seed, offset)]
    assert generate(n, seed=seed, id_offset=offset) == generate(n, seed=seed, id_offset=offset, mix="standard")


def _cases(mix: str, n: int = 2000):
    rng = random.Random(42)
    return [make_case(rng, i, mix) for i in range(n)]


def test_boundary_mix_is_dense_at_both_thresholds():
    b = _cases("boundary")
    s = _cases("standard")
    bank = [c for c in b if 25 <= c.vendor["days_since_bank_change"] <= 35]
    var = [c for c in b if c.variance_pct is not None and 1.5 <= abs(c.variance_pct) <= 3.0]
    assert len(bank) > 0.12 * len(b) and len(var) > 0.10 * len(b)
    # both sides of each threshold, including the exact edge values
    days = {c.vendor["days_since_bank_change"] for c in bank}
    assert {29, 30} <= days and min(days) == 25 and max(days) == 35
    assert any(abs(c.variance_pct) > 2.0 for c in var) and any(abs(c.variance_pct) <= 2.0 for c in var)
    # the standard mix has (almost) none of these: that was the gap
    assert sum(1 for c in s if c.variance_pct is not None and 1.5 <= abs(c.variance_pct) <= 3.0) < 0.01 * len(s)


def test_labels_are_the_policy_and_thresholds_are_exact():
    for c in _cases("boundary"):
        assert decide(c.po, c.vendor, c.invoice_id, c.billed_qty, c.unit_price_billed) == (
            c.reason_code,
            c.variance_pct,
        )
        d = c.vendor["days_since_bank_change"]
        if c.po and c.invoice_id not in c.po["paid_invoices"]:
            assert (c.reason_code == "VENDOR_BANK_CHANGE") == (d < 30)


def test_scenario_coverage_includes_collisions_and_partial_billing():
    b = _cases("boundary")
    bank_qty = [
        c for c in b if c.vendor["days_since_bank_change"] < 30 and c.po and c.billed_qty > c.po["received_qty"]
    ]
    assert bank_qty and all(c.reason_code == "VENDOR_BANK_CHANGE" for c in bank_qty)  # precedence: bank over qty
    partial = [c for c in b if c.po and c.billed_qty == c.po["received_qty"] < c.po["ordered_qty"]]
    assert any(c.reason_code == "WITHIN_TOLERANCE" for c in partial)
    dup_price = [c for c in b if c.po and c.invoice_id in c.po["paid_invoices"] and abs(c.variance_pct) > 2]
    assert dup_price and all(c.reason_code == "DUPLICATE_INVOICE" for c in dup_price)
    codes = {c.reason_code for c in b}
    assert len(codes) == 6


def test_unknown_mix_rejected():
    assert TRAIN_MIXES == ("standard", "boundary")
    with pytest.raises(ValueError, match="unknown training mix"):
        make_case(random.Random(0), 0, "everything")


def test_prepare_boundary_mix_keeps_every_evaluation_set_identical(tmp_path):
    def prep(name, *extra):
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "deploy/gpu/pilot.py"),
                "prepare",
                "--workdir",
                str(tmp_path / name),
                "--n",
                "200",
                "--test-n",
                "60",
                "--ood-n",
                "20",
                "--messy-n",
                "20",
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert r.returncode == 0, r.stderr
        return tmp_path / name

    a, b = prep("std"), prep("bnd", "--train-mix", "boundary")
    for f in ("golden.jsonl", "golden_ood.jsonl", "golden_messy.jsonl"):
        assert (a / "data" / f).read_bytes() == (b / "data" / f).read_bytes(), f
    assert (a / "data/train.jsonl").read_bytes() != (b / "data/train.jsonl").read_bytes()
    meta = json.loads((b / "pilot.json").read_text())
    assert meta["train_mix"] == "boundary" and json.loads((a / "pilot.json").read_text())["train_mix"] == "standard"
