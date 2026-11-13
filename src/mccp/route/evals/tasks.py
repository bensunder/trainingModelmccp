"""Executable coding tasks, scored by running code, never by judging prose.

Each task belongs to one routing task class and is scored mechanically:
* quick_edit / debug / refactor: the model returns code; it must pass the task's
  hidden tests (a refactor must keep behaviour; a debug fix must make the failing
  test pass).
* tests: the model writes tests; they must pass on the reference implementation
  AND fail on every planted bug (mutant). A test suite that passes everything
  proves nothing.

`explain` and `agentic` have no mechanical score here; without evidence the
router keeps those classes on the strong model.

Task files are JSONL, one object per line (schema: `Task`). `SAMPLE` is a small,
original starter set for plumbing checks; it is far too small to move routing
decisions on its own (the Wilson bound keeps a handful of passes below any bar).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import TaskClass

SCORED = (TaskClass.QUICK_EDIT, TaskClass.DEBUG, TaskClass.REFACTOR, TaskClass.TESTS)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,120}$")
    task_class: TaskClass
    language: Literal["python"] = "python"
    prompt: str = Field(min_length=10)
    entry_point: str | None = None  # function the answer must define
    preamble: str | None = None  # imports the spec relies on, prepended to the answer when scoring
    tests: str | None = None  # hidden asserts run against the answer (quick_edit/debug/refactor)
    reference: str | None = None  # correct implementation (tests class)
    mutants: list[str] = Field(default_factory=list)  # buggy implementations (tests class)
    source: str = "original"  # provenance and licence of the task

    @model_validator(mode="after")
    def _shape(self) -> Task:
        if self.task_class not in SCORED:
            raise ValueError(f"{self.id}: class {self.task_class.value} has no mechanical score")
        if self.task_class is TaskClass.TESTS:
            if not self.reference or not self.mutants:
                raise ValueError(f"{self.id}: a tests task needs a reference and at least one mutant")
        elif not self.tests:
            raise ValueError(f"{self.id}: needs hidden tests")
        return self


def load(path: str | Path) -> tuple[list[Task], str]:
    """(tasks, sha256 of the file) — the hash goes into the evidence provenance."""
    raw = Path(path).read_bytes()
    tasks = [Task.model_validate(json.loads(line)) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids")
    return tasks, hashlib.sha256(raw).hexdigest()


def write(tasks: list[Task], path: str | Path) -> None:
    Path(path).write_text("".join(t.model_dump_json(exclude_defaults=False) + "\n" for t in tasks), encoding="utf-8")


SAMPLE: list[Task] = [
    Task(
        id="qe-slugify",
        task_class=TaskClass.QUICK_EDIT,
        entry_point="slugify",
        prompt="Write a Python function `slugify(title: str) -> str` that lowercases the title, replaces every run of "
        "characters that are not ASCII letters or digits with a single hyphen, and strips leading/trailing hyphens.",
        tests="assert slugify('Hello, World!') == 'hello-world'\nassert slugify('  A--B  ') == 'a-b'\n"
        "assert slugify('Ünïcode 2026') == 'n-code-2026'\nassert slugify('---') == ''\n",
    ),  # fmt: skip
    Task(
        id="qe-chunk",
        task_class=TaskClass.QUICK_EDIT,
        entry_point="chunk",
        prompt="Write `chunk(items: list, size: int) -> list[list]` that splits a list into consecutive chunks of "
        "`size` (the last may be shorter). Raise ValueError if size < 1.",
        tests="assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunk([], 3) == []\n"
        "try:\n    chunk([1], 0)\n    raise SystemExit(1)\nexcept ValueError:\n    pass\n",
    ),  # fmt: skip
    Task(
        id="qe-merge-intervals",
        task_class=TaskClass.QUICK_EDIT,
        entry_point="merge",
        prompt="Write `merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges overlapping or "
        "touching closed intervals and returns them sorted by start.",
        tests="assert merge([(5,6),(1,3),(2,4)]) == [(1,4),(5,6)]\nassert merge([(1,2),(2,3)]) == [(1,3)]\nassert merge([]) == []\n",
    ),  # fmt: skip
    Task(
        id="dbg-median",
        task_class=TaskClass.DEBUG,
        entry_point="median",
        prompt="This function returns the wrong value for even-length lists and crashes on unsorted input orders in "
        "some callers' expectations. Fix it so it returns the true median of any non-empty list of numbers.\n\n"
        "```python\ndef median(xs):\n    n = len(xs)\n    return xs[n // 2]\n```",
        tests="assert median([3, 1, 2]) == 2\nassert median([4, 1, 3, 2]) == 2.5\nassert median([7]) == 7\n",
    ),  # fmt: skip
    Task(
        id="dbg-dedupe",
        task_class=TaskClass.DEBUG,
        entry_point="dedupe",
        prompt="`dedupe` should remove duplicates while keeping the first occurrence order, but it loses order. Fix it.\n\n"
        "```python\ndef dedupe(items):\n    return list(set(items))\n```",
        tests="assert dedupe([3,1,3,2,1]) == [3,1,2]\nassert dedupe([]) == []\nassert dedupe(['b','a','b']) == ['b','a']\n",
    ),  # fmt: skip
    Task(
        id="dbg-parse-duration",
        task_class=TaskClass.DEBUG,
        entry_point="parse_duration",
        prompt="`parse_duration('1h30m')` should return seconds (5400) and accept any of h/m/s in that order, each "
        "optional, but it raises on '45s' and miscounts minutes. Fix it.\n\n```python\nimport re\n"
        "def parse_duration(s):\n    h, m, sec = re.match(r'(\\d+)h(\\d+)m(\\d+)s', s).groups()\n"
        "    return int(h) * 3600 + int(m) * 6 + int(sec)\n```",
        tests="assert parse_duration('1h30m') == 5400\nassert parse_duration('45s') == 45\nassert parse_duration('2h') == 7200\n"
        "assert parse_duration('1h1m1s') == 3661\n",
    ),  # fmt: skip
    Task(
        id="rf-price-rules",
        task_class=TaskClass.REFACTOR,
        entry_point="shipping_cost",
        prompt="Refactor this function to remove the duplicated branches without changing its behaviour.\n\n"
        "```python\ndef shipping_cost(weight, express):\n    if express:\n        if weight <= 1:\n            return 10\n"
        "        elif weight <= 5:\n            return 15\n        else:\n            return 15 + (weight - 5) * 2\n"
        "    else:\n        if weight <= 1:\n            return 5\n        elif weight <= 5:\n            return 8\n"
        "        else:\n            return 8 + (weight - 5) * 1\n```",
        tests="for w in (0.5, 1, 3, 5, 6, 10):\n    exp_e = 10 if w <= 1 else 15 if w <= 5 else 15 + (w - 5) * 2\n"
        "    exp_s = 5 if w <= 1 else 8 if w <= 5 else 8 + (w - 5)\n"
        "    assert shipping_cost(w, True) == exp_e and shipping_cost(w, False) == exp_s, w\n",
    ),  # fmt: skip
    Task(
        id="rf-word-count",
        task_class=TaskClass.REFACTOR,
        entry_point="top_words",
        prompt="Refactor for readability (keep the exact behaviour: case-insensitive, ties broken alphabetically).\n\n"
        "```python\ndef top_words(text, n):\n    d = {}\n    for w in text.lower().split():\n        w = w.strip('.,!?')\n"
        "        if w:\n            if w in d:\n                d[w] = d[w] + 1\n            else:\n                d[w] = 1\n"
        "    items = list(d.items())\n    items.sort(key=lambda x: (-x[1], x[0]))\n    return [w for w, c in items[:n]]\n```",
        tests="assert top_words('b a B, c a b!', 2) == ['b', 'a']\nassert top_words('', 3) == []\nassert top_words('x y z', 2) == ['x', 'y']\n",
    ),  # fmt: skip
    Task(
        id="tst-is-leap",
        task_class=TaskClass.TESTS,
        entry_point="is_leap",
        prompt="Write plain-assert Python tests (no framework) for `is_leap(year: int) -> bool` (Gregorian rules). "
        "Assume `is_leap` is already defined; do not redefine it.",
        reference="def is_leap(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n",
        mutants=[
            "def is_leap(y):\n    return y % 4 == 0\n",
            "def is_leap(y):\n    return y % 4 == 0 and y % 100 != 0\n",
            "def is_leap(y):\n    return y % 400 == 0\n",
        ],
    ),  # fmt: skip
]
