"""Benchmark importer: pinning, planted bugs that really change behaviour, and
tasks that are solvable by the canonical answer and unsolvable by the bug."""

from __future__ import annotations

import gzip
import json

import pytest

from mccp.route.config import TaskClass
from mccp.route.evals import importers, runner
from mccp.route.evals.sandbox import LocalSandbox

SB = LocalSandbox()

HE = [
    {
        "task_id": "HumanEval/0",
        "entry_point": "count_over",
        "prompt": 'from typing import List\n\n\ndef count_over(xs: List[int], k: int) -> int:\n    """How many xs are > k."""\n',
        "canonical_solution": "    n = 0\n    for x in xs:\n        if x > k:\n            n += 1\n    return n\n",
        "test": "def check(candidate):\n    assert candidate([1, 2, 3], 2) == 1\n    assert candidate([5, 5], 4) == 2\n"
        "    assert candidate([], 0) == 0\n    assert candidate([3, 3], 3) == 0\n",
    },
    {  # canonical solution fails its own tests: must be dropped
        "task_id": "HumanEval/1",
        "entry_point": "broken",
        "prompt": 'def broken(x):\n    """Return x."""\n',
        "canonical_solution": "    return x + 1\n",
        "test": "def check(candidate):\n    assert candidate(1) == 1\n",
    },
]
MBPP = [
    {
        "task_id": 7,
        "prompt": "Write a function to return the larger of two numbers.",
        "code": "def bigger(a, b):\n\tif a >= b:\n\t\treturn a\n\treturn b",
        "test_imports": [],
        "test_list": ["assert bigger(1, 2) == 2", "assert bigger(3, 1) == 3", "assert bigger(-1, -5) == -1"],
    }
]


def _he() -> bytes:
    return gzip.compress("".join(json.dumps(d) + "\n" for d in HE).encode())


def test_unpinned_file_is_refused_unless_allowed():
    with pytest.raises(importers.SourceError, match="does not match the pinned release"):
        importers.verify("humaneval", _he(), allow_unpinned=False)
    with pytest.raises(importers.SourceError, match="unknown source"):
        importers.verify("leetcode", b"", allow_unpinned=True)
    assert len(importers.verify("humaneval", _he(), allow_unpinned=True)) == 64


def test_mutants_are_single_site_and_distinct():
    ref = "def f(a, b):\n    return a + b if a < 1 else a\n"
    ms = importers.mutants(ref, "f")
    assert len(ms) == len(set(ms)) >= 3
    assert all(m != ref and "def f" in m for m in ms)
    assert importers.mutants(ref, "missing") == []


async def test_humaneval_tasks_are_solvable_and_bugs_fail():
    ts, rep = await importers.build("humaneval", _he(), SB, allow_unpinned=True)
    assert rep.dropped_reference_fails == ["he-1"]
    by = {t.task_class: t for t in ts}
    assert set(by) == {TaskClass.QUICK_EDIT, TaskClass.DEBUG, TaskClass.TESTS}
    assert all(t.source.endswith("UNPINNED") and "MIT" in t.source for t in ts)
    canonical = "def count_over(xs: List[int], k: int) -> int:\n    return sum(1 for x in xs if x > k)"
    # the answer omits `from typing import List`; the preamble supplies it, as the spec showed it
    assert (await runner.score(by[TaskClass.QUICK_EDIT], canonical, SB))[0]
    assert (await runner.score(by[TaskClass.DEBUG], canonical, SB))[0]
    buggy = by[TaskClass.DEBUG].prompt.split("Buggy code:\n```python\n")[1].rsplit("```", 1)[0]
    assert not (await runner.score(by[TaskClass.DEBUG], buggy, SB))[0]
    t = by[TaskClass.TESTS]
    strong = "assert count_over([1, 2, 3], 2) == 1\nassert count_over([3, 3], 3) == 0\nassert count_over([5, 5], 4) == 2\nassert count_over([], 0) == 0"
    ok, why = await runner.score(t, strong, SB)
    assert ok, why
    ok, why = await runner.score(t, "def count_over(xs, k):\n    return 0\nassert count_over([], 0) == 0", SB)
    assert not ok and "redefines" in why  # cannot replace the implementation under test


async def test_mbpp_import_and_determinism():
    raw = json.dumps(MBPP).encode()
    a, _ = await importers.build("mbpp", raw, SB, allow_unpinned=True, seed=1)
    b, _ = await importers.build("mbpp", raw, SB, allow_unpinned=True, seed=1)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]
    qe = next(t for t in a if t.task_class is TaskClass.QUICK_EDIT)
    assert qe.entry_point == "bigger" and "assert bigger(1, 2) == 2" in qe.prompt
    assert (await runner.score(qe, "def bigger(a, b):\n    return max(a, b)", SB))[0]
    only_qe, rep = await importers.build("mbpp", raw, SB, (TaskClass.QUICK_EDIT,), allow_unpinned=True)
    assert rep.tasks_by_class == {"quick_edit": 1} and len(only_qe) == 1
