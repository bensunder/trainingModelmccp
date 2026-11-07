"""Checkpoint safety controls in the trainer entrypoint: pruning, mirroring to
persistent storage, and restoring from the mirror on a new machine.

The end-to-end tests run the real `train()` loop against a fake `soup` that
writes checkpoints over time the way the training library does (weights first,
trainer_state.json last), so the polling, pruning, mirroring and resume wiring
is exercised without a GPU."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.conftest import ROOT


def _load_entry():
    spec = importlib.util.spec_from_file_location("ckpt_trainer_entry", ROOT / "trainer/mccp_trainer_entry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ckpt_trainer_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load_entry()


class _Ev:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, type_: str, **kw) -> None:
        self.events.append({"type": type_, **kw})

    def of(self, t: str) -> list[dict]:
        return [e for e in self.events if e["type"] == t]


def _ckpt(output: Path, step: int, complete: bool = True, size: int = 1000) -> Path:
    d = output / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"w" * size)
    if complete:
        (d / "trainer_state.json").write_text(json.dumps({"log_history": [{"step": step, "loss": 1.0 / step}]}))
    return d


# ------------------------------------------------------------------ pruning --
def test_prune_keeps_two_newest_complete(tmp_path):
    for s in (100, 200, 300, 400):
        _ckpt(tmp_path, s)
    removed = E.prune_checkpoints(str(tmp_path), keep=2)
    assert [s for s, _ in removed] == [100, 200]
    assert all(b == 1000 + len(json.dumps({"log_history": [{"step": s, "loss": 1.0 / s}]})) for s, b in removed)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-300", "checkpoint-400"]


def test_prune_never_touches_a_checkpoint_being_written(tmp_path):
    for s in (100, 200):
        _ckpt(tmp_path, s)
    _ckpt(tmp_path, 300, complete=False)  # still being saved
    assert E.prune_checkpoints(str(tmp_path), keep=2) == []  # the half one does not count toward 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-100", "checkpoint-200", "checkpoint-300"]


def test_prune_ignores_other_files_and_rejects_keep_zero(tmp_path):
    _ckpt(tmp_path, 100)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"final")
    (tmp_path / "checkpoint-notanumber").mkdir()
    assert E.prune_checkpoints(str(tmp_path), keep=1) == []
    assert (tmp_path / "adapter_model.safetensors").exists()
    with pytest.raises(ValueError):
        E.prune_checkpoints(str(tmp_path), keep=0)


# ------------------------------------------------------------------ mirror --
def test_mirror_copies_newest_atomically_and_keeps_only_one(tmp_path):
    out, root, ev = tmp_path / "out", tmp_path / "ws", _Ev()
    run_json = tmp_path / "run.json"
    run_json.write_text("{}")
    m = E.CheckpointMirror(str(root), "abc123", ev)
    for s in (100, 200):
        m.submit(s, str(_ckpt(out, s)), str(run_json))
        m.wait()
    assert m.latest() == (200, str(root / "abc123" / "ckpt-200"))
    names = sorted(p.name for p in (root / "abc123").iterdir())
    assert names == ["IDENTITY.json", "LATEST", "ckpt-200"]  # no temp dirs, one checkpoint
    assert [e["step"] for e in ev.of("checkpoint_mirrored")] == [100, 200]


def test_mirror_latest_ignores_a_pointer_to_an_incomplete_copy(tmp_path):
    root = tmp_path / "ws" / "id1"
    (root / "ckpt-300").mkdir(parents=True)  # copy without trainer_state.json
    (root / "LATEST").write_text("300")
    assert E.CheckpointMirror(str(tmp_path / "ws"), "id1", _Ev()).latest() is None


def test_mirror_failure_is_reported_not_raised(tmp_path):
    ev = _Ev()
    run_json = tmp_path / "run.json"
    run_json.write_text("{}")
    m = E.CheckpointMirror(str(tmp_path / "ws"), "id1", ev)
    m.submit(100, str(tmp_path / "does-not-exist"), str(run_json))
    m.wait()
    assert ev.of("checkpoint_mirror_failed") and not ev.of("checkpoint_mirrored")
    assert m.latest() is None
    assert not list((tmp_path / "ws" / "id1").glob(".tmp-*"))


def test_restore_only_into_an_empty_output(tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text("{}")
    m = E.CheckpointMirror(str(tmp_path / "ws"), "id1", _Ev())
    m.submit(500, str(_ckpt(tmp_path / "old", 500)), str(run_json))
    m.wait()
    out = tmp_path / "new"
    assert m.restore_into(str(out)) == 500
    assert E._complete(str(out / "checkpoint-500"))
    assert m.restore_into(str(out)) is None  # local checkpoints win; never overwritten


def test_identity_tracks_run_contract_and_training_data(tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.jsonl").write_text('{"a": 1}\n')
    run = {"soup_yaml": {"data": {"train": "data/train.jsonl"}}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    a = E.run_identity("run.json", run)
    assert a == E.run_identity(str(tmp_path / "run.json"), run)  # path-independent
    (tmp_path / "data/train.jsonl").write_text('{"a": 2}\n')
    b = E.run_identity("run.json", run)
    assert b != a  # different data never matches
    (tmp_path / "run.json").write_text(json.dumps({**run, "x": 1}))
    assert E.run_identity("run.json", run) not in (a, b)  # different contract never matches


# --------------------------------------------- end to end with a fake soup --
FAKE_SOUP = textwrap.dedent(
    """\
    #!{py}
    import json, os, sys, time
    out = "output"
    argv_log = os.environ["FAKE_SOUP_ARGV"]
    with open(argv_log, "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\\n")
    os.makedirs(out, exist_ok=True)
    have = [int(d.split("-")[1]) for d in os.listdir(out) if d.startswith("checkpoint-")]
    start = max(have) if "--resume" in sys.argv and have else 0
    stop_at = int(os.environ.get("FAKE_SOUP_CRASH_AT", "0"))
    for step in range(start + 10, 61, 10):
        d = os.path.join(out, f"checkpoint-{{step}}")
        os.makedirs(d)
        with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
            f.write(b"w" * 4000)
        time.sleep(0.05)
        with open(os.path.join(d, "trainer_state.json"), "w") as f:  # written last
            json.dump({{"log_history": [{{"step": step, "loss": 1.0 / step}}]}}, f)
        if stop_at and step == stop_at:
            sys.exit(9)  # the host "drops"
        time.sleep(0.15)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with open(os.path.join(out, name), "w") as f:
            f.write("final")
    """
)


def _workdir(tmp_path: Path, name: str) -> Path:
    w = tmp_path / name
    (w / "data").mkdir(parents=True)
    (w / "data/train.jsonl").write_text('{"messages": []}\n')
    run = {"soup_yaml": {"base": "x", "output": "./output", "data": {"train": "data/train.jsonl"}}}
    (w / "run.json").write_text(json.dumps(run))
    return w


@pytest.fixture
def fake_soup(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    soup = bindir / "soup"
    soup.write_text(FAKE_SOUP.format(py=sys.executable))
    soup.chmod(soup.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SOUP_ARGV", str(tmp_path / "argv.jsonl"))
    real_sleep = time.sleep
    monkeypatch.setattr(E.time, "sleep", lambda s: real_sleep(min(s, 0.02)))  # poll fast
    monkeypatch.setattr(E, "KEEP_CHECKPOINTS", 2)
    return tmp_path / "argv.jsonl"


def _train(w: Path, root: Path | None, monkeypatch) -> tuple[int, _Ev]:
    monkeypatch.chdir(w)
    monkeypatch.setattr(E, "MIRROR_ROOT", str(root) if root else "")
    run = json.loads((w / "run.json").read_text())
    run["_path"] = str(w / "run.json")
    ev = _Ev()
    rc = E.train(run, ev, {"trained_tokens": 1})
    return rc, ev


def test_train_prunes_mirrors_and_keeps_mirror_until_saved(tmp_path, fake_soup, monkeypatch):
    w, root = _workdir(tmp_path, "w1"), tmp_path / "workspace" / "mccp-ckpt"
    rc, ev = _train(w, root, monkeypatch)
    assert rc == E.EXIT_OK
    assert sorted(p.name for p in (w / "output").glob("checkpoint-*")) == ["checkpoint-50", "checkpoint-60"]
    pruned = [e["step"] for e in ev.of("checkpoint_pruned")]
    assert pruned == [10, 20, 30, 40] and all(e["bytes_freed"] > 0 for e in ev.of("checkpoint_pruned"))
    assert [e["step"] for e in ev.of("checkpoint")] == [10, 20, 30, 40, 50, 60]
    mirror_dir = Path(ev.of("checkpoint_mirror")[0]["dir"])
    assert (mirror_dir / "LATEST").read_text() == "60"
    assert ev.of("done")[0]["checkpoint_mirror"] == str(mirror_dir)
    assert (w / ".ckpt-mirror").read_text().strip() == str(mirror_dir)  # batch deletes it after save_run
    assert sorted(p.name for p in mirror_dir.iterdir()) == ["IDENTITY.json", "LATEST", "ckpt-60"]


def test_lost_host_resumes_from_mirror_on_a_new_machine(tmp_path, fake_soup, monkeypatch):
    root = tmp_path / "workspace" / "mccp-ckpt"
    monkeypatch.setenv("FAKE_SOUP_CRASH_AT", "30")
    w1 = _workdir(tmp_path, "machine1")
    rc, _ = _train(w1, root, monkeypatch)
    assert rc == E.EXIT_TRAIN
    monkeypatch.delenv("FAKE_SOUP_CRASH_AT")
    # New machine: same run contract and data, nothing local.
    w2 = _workdir(tmp_path, "machine2")
    rc, ev = _train(w2, root, monkeypatch)
    assert rc == E.EXIT_OK
    assert ev.of("checkpoint_restored")[0]["step"] == 30
    argv = [json.loads(line) for line in fake_soup.read_text().splitlines()]
    assert "--resume" not in argv[0] and argv[1][-2:] == ["--resume", "auto"]
    assert [e["step"] for e in ev.of("checkpoint")] == [40, 50, 60]  # 30 not re-reported or re-mirrored


def test_different_run_never_picks_up_another_runs_mirror(tmp_path, fake_soup, monkeypatch):
    root = tmp_path / "workspace" / "mccp-ckpt"
    monkeypatch.setenv("FAKE_SOUP_CRASH_AT", "30")
    _train(_workdir(tmp_path, "a"), root, monkeypatch)
    monkeypatch.delenv("FAKE_SOUP_CRASH_AT")
    w = _workdir(tmp_path, "b")
    (w / "data/train.jsonl").write_text('{"messages": ["other data"]}\n')
    rc, ev = _train(w, root, monkeypatch)
    assert rc == E.EXIT_OK and not ev.of("checkpoint_restored")
    assert [e["step"] for e in ev.of("checkpoint")][0] == 10  # started from scratch


def test_without_mirror_root_only_pruning_runs(tmp_path, fake_soup, monkeypatch):
    w = _workdir(tmp_path, "w")
    rc, ev = _train(w, None, monkeypatch)
    assert rc == E.EXIT_OK and not ev.of("checkpoint_mirrored") and not (w / ".ckpt-mirror").exists()
    assert len(list((w / "output").glob("checkpoint-*"))) == 2


# ------------------------- resume on a new machine needs a deterministic prepare --
def test_reprepared_workdir_has_the_same_identity(tmp_path):
    """A lost host loses its workdir; the batch re-prepares it on the new machine.
    The mirror is found only if that rebuild is byte-identical (same run.json and
    training data), so this guards the whole resume path."""
    import subprocess

    ids = []
    for name in ("machine1", "machine2"):
        w = tmp_path / name
        cmd = [sys.executable, str(ROOT / "deploy/gpu/pilot.py"), "prepare", "--workdir", str(w)]
        cmd += ["--n", "80", "--test-n", "40", "--ood-n", "20", "--train-mix", "boundary"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        run = json.loads((w / "run.json").read_text())
        cwd = os.getcwd()
        os.chdir(w)
        try:
            ids.append(E.run_identity("run.json", run))
        finally:
            os.chdir(cwd)
    assert ids[0] == ids[1]
