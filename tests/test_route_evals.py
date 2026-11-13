"""Coding evaluation: sandboxed scoring, evidence aggregation, and a full run
through the gateway against fake providers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from mccp.route.app import Settings, create_app
from mccp.route.cache import MemoryCache
from mccp.route.config import TaskClass, load_catalog
from mccp.route.evals import runner, tasks
from mccp.route.evals.sandbox import DockerSandbox, LocalSandbox
from mccp.route.policy import Evidence, choose
from mccp.route.state import MemoryStore
from mccp.route.upstream import Upstream
from tests.conftest import ROOT

EXAMPLE = ROOT / "deploy/route/route.example.yaml"
SB = LocalSandbox()
TASK = {t.id: t for t in tasks.SAMPLE}


def fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}\n```"


GOOD = {
    "qe-chunk": "def chunk(items, size):\n    if size < 1:\n        raise ValueError('size')\n    return [items[i:i + size] for i in range(0, len(items), size)]",
    "dbg-dedupe": "def dedupe(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
    "tst-is-leap": "assert is_leap(2000)\nassert not is_leap(1900)\nassert is_leap(2024)\nassert not is_leap(2023)",
}


def test_sample_tasks_are_valid_and_reference_tests_pass():
    assert len({t.id for t in tasks.SAMPLE}) == len(tasks.SAMPLE)
    assert {t.task_class for t in tasks.SAMPLE} == {
        TaskClass.QUICK_EDIT,
        TaskClass.DEBUG,
        TaskClass.REFACTOR,
        TaskClass.TESTS,
    }
    with pytest.raises(ValueError, match="no mechanical score"):
        tasks.Task(id="x", task_class=TaskClass.EXPLAIN, prompt="explain this code please", tests="assert 1")


def test_extract_code_prefers_the_block_defining_the_entry_point():
    answer = "```python\nimport os\n```\ntext\n```python\ndef chunk(a, b):\n    return []\n```\n```\nprint(1)\n```"
    assert "def chunk" in runner.extract_code(answer, "chunk")
    assert runner.extract_code("no code here", "chunk") is None


@pytest.mark.parametrize("tid", list(GOOD))
async def test_correct_answers_pass(tid):
    ok, why = await runner.score(TASK[tid], GOOD[tid], SB)
    assert ok, why


async def test_wrong_answers_fail_with_reasons():
    t = TASK["dbg-dedupe"]
    ok, why = await runner.score(t, "def dedupe(items):\n    return list(set(items))", SB)
    assert not ok and why.startswith("hidden tests fail")  # the original bug is still there
    ok, why = await runner.score(t, "def other():\n    pass", SB)
    assert not ok and "does not define" in why
    ok, why = await runner.score(t, None, SB)
    assert not ok and "no code" in why
    # weak tests: pass on the reference but miss a planted bug
    ok, why = await runner.score(TASK["tst-is-leap"], "assert is_leap(2024)\nassert not is_leap(2023)", SB)
    assert not ok and "planted bug" in why
    # tests that fail on the correct implementation
    ok, why = await runner.score(TASK["tst-is-leap"], "assert is_leap(1900)", SB)
    assert not ok and "correct implementation" in why


async def test_sandbox_enforces_timeout_and_no_network():
    r = await SB.run({"run.py": "while True:\n    pass"}, ["python", "run.py"], timeout_s=1.5)
    assert r.timed_out and not r.ok
    if not shutil.which("unshare"):
        pytest.skip("unshare not available")
    code = "import socket\ns = socket.create_connection(('1.1.1.1', 53), timeout=2)\nprint('connected')"
    r = await SB.run({"run.py": code}, ["python", "run.py"], timeout_s=10)
    assert not r.ok and "connected" not in r.output


def test_docker_sandbox_flags_lock_the_container_down(tmp_path):
    argv = DockerSandbox(image="python:3.12-slim@sha256:abc").argv(tmp_path, "n", ["python", "run.py"])
    joined = " ".join(argv)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--user 65534:65534",
        "--pids-limit",
        ":ro",
    ):
        assert flag in joined
    assert argv[-3:] == ["python:3.12-slim@sha256:abc", "python", "run.py"]


def test_evidence_excludes_outages_and_drives_routing():
    rows = [runner.Row("glm", f"t{i}", "quick_edit", i < 190, "x", cost_usd=0.001, http_status=200) for i in range(200)]
    rows += [runner.Row("glm", "down", "quick_edit", False, "HTTP 503", http_status=503)]
    ev, prov = runner.evidence(rows, "suite@1", {"tasks_sha256": "abc"})
    c = ev.results["glm"][TaskClass.QUICK_EDIT]
    assert (c.n, c.passed) == (200, 190) and prov["excluded_no_answer"] == 1 and ev.provenance["tasks_sha256"] == "abc"
    cat = load_catalog(EXAMPLE)
    back = Evidence.model_validate_json(ev.model_dump_json())  # round-trips through the file format
    assert choose(cat, cat.profile("auto-cost"), TaskClass.QUICK_EDIT, "openai_chat", back).model == "glm"


def test_estimate_is_positive_and_per_model():
    cat = load_catalog(EXAMPLE)
    est = runner.estimate_usd(cat, tasks.SAMPLE, ["deepseek-flash", "claude-sonnet"], 1024)
    assert 0 < est["deepseek-flash"] < est["claude-sonnet"]


def test_full_run_through_the_gateway(tmp_path):
    cat = load_catalog(EXAMPLE)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = json.dumps(body["messages"])
        good = body["model"] == "glm-5.3"  # glm answers correctly, deepseek returns broken code
        code = (
            next((GOOD[k] for k in GOOD if TASK[k].prompt[:40] in prompt), "def nothing(): pass")
            if good
            else "def broken(:"
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": fenced(code)}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": 80}})  # fmt: skip

    keys = {"DEEPSEEK_API_KEY": "d", "MOONSHOT_API_KEY": "m", "ZAI_API_KEY": "z", "ANTHROPIC_API_KEY": "a"}
    up = Upstream(cat, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), env=keys)
    admin_token = "adm"
    settings = Settings(
        catalog_path=str(EXAMPLE),
        key_pepper=b"p" * 32,
        admin_token_sha256=hashlib.sha256(admin_token.encode()).hexdigest(),
    )
    app = create_app(settings, catalog=cat, store=MemoryStore(), cache=MemoryCache(), upstream=up)
    picked = [TASK[k] for k in GOOD]
    with TestClient(app) as c:
        h = {"authorization": f"Bearer {admin_token}"}
        org = c.post("/admin/v1/orgs", headers=h, json={"name": "eval"}).json()
        team = c.post(f"/admin/v1/orgs/{org['id']}/teams", headers=h, json={"name": "eval"}).json()
        key = c.post(f"/admin/v1/orgs/{org['id']}/teams/{team['id']}/keys", headers=h, json={"label": "eval"}).json()[
            "key"
        ]

        async def go():
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
            try:
                return await runner.run(
                    cat, "http://gw", key, picked, ["glm", "deepseek-flash"], SB, tmp_path, 2, 512, client=client
                )
            finally:
                await client.aclose()

        rows = c.portal.call(go)
    by = {(r.model, r.task_id): r for r in rows}
    assert all(by[("glm", t.id)].passed for t in picked), [by[("glm", t.id)].why for t in picked]
    assert not any(by[("deepseek-flash", t.id)].passed for t in picked)
    assert all(r.cost_usd > 0 and r.http_status == 200 for r in rows)
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(lines) == len(rows) == 2 * len(picked)


@pytest.mark.skipif(
    not shutil.which("docker") or os.system("docker info >/dev/null 2>&1") != 0, reason="no docker daemon"
)
async def test_docker_sandbox_runs_code():
    r = await DockerSandbox().run({"run.py": "print(sum(range(10)))"}, ["python", "run.py"], 60)
    assert r.ok and "45" in r.output
