"""deploy/gpu/pilot.py: fixed test set, RGT prepare (with cached on-policy attempts), learning curve."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from tests.conftest import ROOT
from tests.fake_serving import BASE, FakeServing

_spec = importlib.util.spec_from_file_location("gpu_pilot_rgt", ROOT / "deploy/gpu/pilot.py")
pilot = importlib.util.module_from_spec(_spec)
sys.modules["gpu_pilot_rgt"] = pilot
_spec.loader.exec_module(pilot)


def _prepare(workdir, *extra):
    cmd = [
        sys.executable,
        str(ROOT / "deploy/gpu/pilot.py"),
        "prepare",
        "--workdir",
        str(workdir),
        "--ood-n",
        "20",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_fixed_test_set_is_identical_across_sizes_and_disjoint_from_training(tmp_path):
    for n in (60, 120):
        r = _prepare(tmp_path / f"n{n}", "--n", str(n), "--test-n", "40")
        assert r.returncode == 0, r.stderr
    g60 = (tmp_path / "n60/data/golden.jsonl").read_bytes()
    assert g60 == (tmp_path / "n120/data/golden.jsonl").read_bytes()
    test_ids = {r["meta"]["source_id"] for r in _jsonl(tmp_path / "n60/data/golden.jsonl")}
    assert len(test_ids) == 40
    meta = json.loads((tmp_path / "n120/pilot.json").read_text())
    assert {k: meta["test_set"][k] for k in ("fixed", "records", "seed")} == {
        "fixed": True,
        "records": 40,
        "seed": 7777,
    }
    assert meta["test_set"]["dropped_train_val_samples"] >= 0
    train_text = (tmp_path / "n120/data/train.jsonl").read_text()
    assert not any(i in train_text for i in test_ids)


def test_rgt_prepare_adds_strategy_rows_and_records_provenance(tmp_path):
    sft = _prepare(tmp_path / "sft", "--n", "80", "--test-n", "30")
    rgt = _prepare(tmp_path / "rgt", "--n", "80", "--test-n", "30", "--method", "rgt")
    assert sft.returncode == 0 and rgt.returncode == 0, rgt.stderr
    a = _jsonl(tmp_path / "sft/data/train.jsonl")
    b = _jsonl(tmp_path / "rgt/data/train.jsonl")
    meta = json.loads((tmp_path / "rgt/pilot.json").read_text())
    assert meta["method"] == "rgt" and meta["rgt"]["b_rows"] == len(b) - len(a) == len(a) // 3
    assert meta["rgt"]["grader"] == "invoice-rule-grader/v1"
    strategies = _jsonl(tmp_path / "rgt/data/rgt_strategies.jsonl")
    assert len(strategies) == meta["rgt"]["b_rows"] and all(s["pair"] == "B" for s in strategies)
    # validation rows stay plain (val loss measures the served behaviour)
    assert "<think>" not in (tmp_path / "rgt/data/train_and_val.jsonl").read_text().split("\n", len(b))[-1]


def test_rgt_attempts_are_on_policy_and_cached(tmp_path):
    fake = FakeServing()
    fake.start()
    try:
        cache = tmp_path / "attempts.jsonl"
        args = [
            "--n",
            "60",
            "--test-n",
            "20",
            "--method",
            "rgt",
            "--attempts-url",
            fake.url,
            "--attempts-key",
            "k",
            "--attempts-model",
            BASE,
            "--attempts-cache",
            str(cache),
        ]
        r1 = _prepare(tmp_path / "a", *args)
        n_requests = len(fake.requests)
        r2 = _prepare(tmp_path / "b", *args)
    finally:
        fake.stop()
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    assert n_requests > 0 and len(fake.requests) == n_requests  # second build served from cache
    meta = json.loads((tmp_path / "a/pilot.json").read_text())
    assert meta["rgt"]["attempts"]["graded"] == meta["rgt"]["b_rows"]
    assert meta["rgt"]["attempts_model"] == BASE
    assert meta["rgt"]["b_rows_with_mistake"] == meta["rgt"]["attempts"]["graded"] - meta["rgt"]["attempts"]["passed"]


def test_chance_overlap_with_fixed_test_set_is_dropped_not_fatal(tmp_path):
    # Production sizes: random PO numbers make a few chance 13-gram matches likely.
    r = _prepare(tmp_path / "n300", "--n", "300", "--test-n", "300")
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "n300/pilot.json").read_text())
    dropped = meta["test_set"]["dropped_train_val_samples"]
    assert 0 < dropped <= 0.02 * (meta["samples"]["train"] + meta["samples"]["val"] + dropped)


def test_attempts_flag_needs_rgt(tmp_path):
    r = _prepare(tmp_path / "x", "--n", "40", "--attempts-url", "http://x", "--attempts-model", "m")
    assert r.returncode == 2 and "needs --method rgt" in r.stderr


def _report(dirpath, method, records, acc, base=None, train_s=600, prepare_s=30):
    dirpath.mkdir(parents=True)
    l1 = {
        "decision_accuracy": {"mean": acc},
        "task_score": {"mean": acc},
        "tool_call_accuracy": 1.0,
        "json_validity": 1.0,
    }
    targets = {"adapter": {"l1": l1}}
    if base is not None:
        targets["base"] = {"l1": {**l1, "decision_accuracy": {"mean": base}}}
    rep = {
        "pilot": {
            "method": method,
            "records": records,
            "seed": 42,
            "samples": {"train": records * 3},
            "test_set": {"fixed": True, "records": 300},
            "rgt": {"attempts": {"pass_rate": 0.4}} if method == "rgt" else None,
        },
        "benchmark": {"sets": {"in_dist": {"targets": targets}, "ood": {"targets": {"adapter": {"l1": l1}}}}},
    }
    (dirpath / "report.json").write_text(json.dumps(rep))
    (dirpath / "timing.jsonl").write_text(
        json.dumps({"stage": "prepare", "seconds": prepare_s})
        + "\n"
        + json.dumps({"stage": "train", "seconds": train_s})
    )
    return dirpath / "report.json"


def test_curve_reports_threshold_crossing_and_ratio(tmp_path):
    paths = [
        _report(tmp_path / "sft100", "sft", 100, 0.60, base=0.10),
        _report(tmp_path / "sft300", "sft", 300, 0.90),
        _report(tmp_path / "sft1000", "sft", 1000, 0.97, train_s=2000),
        _report(tmp_path / "rgt100", "rgt", 100, 0.85),
        _report(tmp_path / "rgt300", "rgt", 300, 0.96, train_s=800),
        _report(tmp_path / "rgt1000", "rgt", 1000, 0.99),
    ]
    rows = pilot.curve_rows(paths)
    assert [(r["method"], r["records"]) for r in rows][:3] == [("rgt", 100), ("rgt", 300), ("rgt", 1000)]
    s = pilot.curve_summary(rows, 0.95)
    assert s["methods"]["sft"] == {"records": 1000, "minutes_to_adapter": 33.8}
    assert s["methods"]["rgt"] == {"records": 300, "minutes_to_adapter": 13.8}
    assert s["sft_over_rgt_records"] == pytest.approx(3.33)
    md = pilot.curve_markdown(rows, s)
    assert "SFT needs 3.33x the records" in md and "Base model, no fine-tuning: 10.0%" in md
    assert "same fixed test set" in md


def test_curve_cli_writes_files(tmp_path):
    p = _report(tmp_path / "r", "sft", 100, 0.5, base=0.1)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "curve", str(p), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "curve.md").exists() and json.loads((out / "curve.json").read_text())["rows"][0]["records"] == 100
