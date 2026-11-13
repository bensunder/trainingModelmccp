"""Build routing evaluation tasks from public coding benchmarks.

Sources are pinned: the importer reads a local copy of the benchmark file and
refuses it unless its SHA-256 matches a pinned release (override only with
`allow_unpinned`, which is recorded in every task's `source`). Nothing is
downloaded here; `docs/route.md` gives the exact clone commands.

| source    | file                 | upstream (commit)                          | licence |
|-----------|----------------------|--------------------------------------------|---------|
| humaneval | HumanEval.jsonl.gz   | github.com/openai/human-eval (b9453e8)     | MIT (LICENSE file in the repo) |
| mbpp      | sanitized-mbpp.json  | github.com/google-research/google-research/mbpp (f8d6d14) | repo: Apache-2.0; dataset card: CC-BY-4.0 |

From each problem (spec + canonical solution + tests) up to three scored tasks:

* quick_edit: implement the function from its spec; hidden tests = the
  benchmark's tests.
* debug: the canonical solution with ONE planted bug (a single AST mutation:
  comparison, arithmetic, boolean operator, constant off-by-one); the model must
  return the fixed function; hidden tests = the benchmark's tests.
* tests: write assert-based tests from the spec; they must pass on the canonical
  solution and fail on every planted bug.

A planted bug is kept only if the benchmark's own tests fail on it (it really
changes behaviour) and the canonical solution passes them. Problems whose
canonical solution does not pass its own tests in the sandbox are dropped and
reported. `refactor`, `explain` and `agentic` get no tasks from these sources.

Known limitation, stated up front: these benchmarks are public and old, so
models may have seen them in training. Absolute pass rates are optimistic; the
comparison between models on the same tasks is what routing uses.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import gzip
import hashlib
import json
import random
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TaskClass
from .sandbox import Sandbox
from .tasks import Task


@dataclass(frozen=True)
class Pin:
    sha256: str
    upstream: str
    licence: str


PINS: dict[str, Pin] = {
    "humaneval": Pin(
        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        "github.com/openai/human-eval@b9453e8f1865df0a0057a1383767caa400acfbcf data/HumanEval.jsonl.gz",
        "MIT",
    ),
    "mbpp": Pin(
        "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
        "github.com/google-research/google-research@f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81 mbpp/sanitized-mbpp.json",
        "Apache-2.0 (repo); CC-BY-4.0 (dataset card)",
    ),
}


def _parse(src: str) -> ast.Module:
    with warnings.catch_warnings():  # some benchmark solutions use invalid escapes in regex strings
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src)


class SourceError(ValueError):
    pass


@dataclass
class Problem:
    """One benchmark problem, normalised."""

    id: str
    spec: str  # what the model sees: signature + docstring, or text + example asserts
    entry_point: str
    preamble: str  # imports the spec relies on (prepended when scoring)
    reference: str  # complete correct module source (preamble included)
    tests: str  # asserts / check() call that must pass on the reference


@dataclass
class Report:
    source: str
    problems: int = 0
    dropped_reference_fails: list[str] = field(default_factory=list)
    no_killed_mutant: list[str] = field(default_factory=list)
    tasks_by_class: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- sources


def verify(source: str, raw: bytes, allow_unpinned: bool) -> str:
    if source not in PINS:
        raise SourceError(f"unknown source {source!r}; known: {sorted(PINS)}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINS[source].sha256 and not allow_unpinned:
        raise SourceError(
            f"{source}: sha256 {digest[:16]}... does not match the pinned release "
            f"{PINS[source].sha256[:16]}... ({PINS[source].upstream}); refusing (use --allow-unpinned to override)"
        )
    return digest


def _imports(src: str) -> str:
    return "\n".join(ast.unparse(n) for n in _parse(src).body if isinstance(n, ast.Import | ast.ImportFrom))


def _context(prompt: str, entry_point: str) -> str:
    """Everything in a HumanEval prompt except the function to write: imports and the
    helper functions some problems provide (the answer may call them)."""
    return "\n\n".join(
        ast.unparse(n) for n in _parse(prompt).body if not (isinstance(n, ast.FunctionDef) and n.name == entry_point)
    )


def humaneval(raw: bytes) -> Iterator[Problem]:
    text = gzip.decompress(raw).decode("utf-8") if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ep = d["entry_point"]
        yield Problem(
            id="he-" + d["task_id"].split("/")[-1],
            spec=d["prompt"].strip("\n"),
            entry_point=ep,
            preamble=_context(d["prompt"], ep),
            reference=d["prompt"] + d["canonical_solution"],
            tests=d["test"] + f"\n\ncheck({ep})\n",
        )


_CALLED = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def mbpp(raw: bytes) -> Iterator[Problem]:
    for d in json.loads(raw.decode("utf-8")):
        code = d["code"].replace("\t", "    ")
        try:
            tree = _parse(code)
        except SyntaxError:
            continue
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        called = [m for t in d["test_list"] for m in _CALLED.findall(t)]
        ep = next((c for c in called if c in defined), None)
        if not ep:
            continue
        imports = "\n".join(d.get("test_imports") or [])
        asserts = "\n".join(d["test_list"])
        yield Problem(
            id=f"mbpp-{d['task_id']}",
            spec=f"{d['prompt'].strip()}\nYour function must be named `{ep}` and pass these tests:\n{asserts}",
            entry_point=ep,
            preamble=_imports(code),
            reference=code,
            tests=(imports + "\n" if imports else "") + asserts + "\n",
        )


READERS = {"humaneval": humaneval, "mbpp": mbpp}


# ---------------------------------------------------------------- planted bugs

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def _sites(fn: ast.FunctionDef) -> list[tuple[str, int]]:
    """(kind, index) of every mutable site inside the function, in source order."""
    out: list[tuple[str, int]] = []
    for i, node in enumerate(ast.walk(fn)):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP_SWAP:
            out.append(("cmp", i))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
            out.append(("bin", i))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
            out.append(("bool", i))
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 2):
            out.append(("const", i))
    return out


def mutants(reference: str, entry_point: str) -> list[str]:
    """Every single-site mutant of the entry-point function, as full module source."""
    tree = _parse(reference)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point]
    if not fns:
        return []
    fn_index = tree.body.index(fns[0])
    out = []
    for kind, site in _sites(fns[0]):
        t = copy.deepcopy(tree)
        node = list(ast.walk(t.body[fn_index]))[site]
        if kind == "cmp":
            node.ops[0] = _CMP_SWAP[type(node.ops[0])]()
        elif kind == "bin":
            node.op = _BIN_SWAP[type(node.op)]()
        elif kind == "bool":
            node.op = _BOOL_SWAP[type(node.op)]()
        else:
            node.value = node.value + 1
        out.append(ast.unparse(t))
    return list(dict.fromkeys(out))  # dedupe, keep order


def _strip_docstrings(src: str, entry_point: str) -> str:
    """Debug prompts show the function body; the spec is given separately, so drop
    the docstring from the shown code to avoid printing it twice."""
    tree = _parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == entry_point and ast.get_docstring(n) is not None:
            n.body = n.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------- tasks


def _quick_edit(p: Problem, src: str) -> Task:
    return Task(
        id=f"{p.id}-qe",
        task_class=TaskClass.QUICK_EDIT,
        entry_point=p.entry_point,
        preamble=p.preamble or None,
        prompt=f"Implement this Python function. Return the complete function.\n\n```python\n{p.spec}\n```",
        tests=p.tests,
        source=src,
    )


def _debug(p: Problem, bug: str, src: str) -> Task:
    return Task(
        id=f"{p.id}-dbg",
        task_class=TaskClass.DEBUG,
        entry_point=p.entry_point,
        preamble=p.preamble or None,
        prompt=(
            f"This function is supposed to satisfy the specification below, but it has a bug. Find and fix it; "
            f"return the complete corrected function.\n\nSpecification:\n```python\n{p.spec}\n```\n\n"
            f"Buggy code:\n```python\n{_strip_docstrings(bug, p.entry_point)}\n```"
        ),
        tests=p.tests,
        source=src,
    )


def _tests(p: Problem, bugs: list[str], src: str) -> Task:
    return Task(
        id=f"{p.id}-tst",
        task_class=TaskClass.TESTS,
        entry_point=p.entry_point,
        prompt=(
            f"Write Python tests for `{p.entry_point}` as plain top-level `assert` statements (no test framework, "
            f"no imports of the function, do NOT define `{p.entry_point}` yourself: it is already in scope). The "
            f"tests must pass on a correct implementation and catch incorrect ones.\n\nSpecification:\n"
            f"```python\n{p.spec}\n```"
        ),
        reference=p.reference,
        mutants=bugs,
        source=src,
    )


async def _passes(sandbox: Sandbox, module: str, tests: str, timeout_s: float) -> bool:
    r = await sandbox.run({"run.py": module + "\n\n" + tests}, ["python", "run.py"], timeout_s)
    return r.ok


async def build(
    source: str,
    raw: bytes,
    sandbox: Sandbox,
    classes: tuple[TaskClass, ...] = (TaskClass.QUICK_EDIT, TaskClass.DEBUG, TaskClass.TESTS),
    allow_unpinned: bool = False,
    max_mutants_tried: int = 12,
    mutants_per_tests_task: int = 3,
    limit: int | None = None,
    concurrency: int = 8,
    timeout_s: float = 10.0,
    seed: int = 0,
) -> tuple[list[Task], Report]:
    """Validate every problem in the sandbox and emit tasks. Deterministic for a given
    file, seed and options (mutant choice uses a seeded RNG per problem id)."""
    digest = verify(source, raw, allow_unpinned)
    pin = PINS[source]
    src = f"{source}@{digest[:12]} ({pin.licence})" + ("" if digest == pin.sha256 else " UNPINNED")
    problems = list(READERS[source](raw))
    if limit:
        problems = problems[:limit]
    report = Report(source=source, problems=len(problems))
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, list[Task]] = {}

    async def one(p: Problem) -> None:
        async with sem:
            if not await _passes(sandbox, p.reference, p.tests, timeout_s):
                report.dropped_reference_fails.append(p.id)
                return
            tasks: list[Task] = []
            if TaskClass.QUICK_EDIT in classes:
                tasks.append(_quick_edit(p, src))
            if TaskClass.DEBUG in classes or TaskClass.TESTS in classes:
                cands = mutants(p.reference, p.entry_point)
                random.Random(f"{seed}:{p.id}").shuffle(cands)
                killed = []
                for m in cands[:max_mutants_tried]:
                    if not await _passes(sandbox, m, p.tests, timeout_s):
                        killed.append(m)
                        if len(killed) >= mutants_per_tests_task:
                            break
                if not killed:
                    report.no_killed_mutant.append(p.id)
                else:
                    if TaskClass.DEBUG in classes:
                        tasks.append(_debug(p, killed[0], src))
                    if TaskClass.TESTS in classes:
                        tasks.append(_tests(p, killed, src))
            out[p.id] = tasks

    await asyncio.gather(*(one(p) for p in problems))
    tasks = [t for p in problems for t in out.get(p.id, [])]  # file order, not completion order
    for t in tasks:
        report.tasks_by_class[t.task_class.value] = report.tasks_by_class.get(t.task_class.value, 0) + 1
    report.dropped_reference_fails.sort()
    report.no_killed_mutant.sort()
    return tasks, report


def read(path: str | Path) -> bytes:
    return Path(path).read_bytes()
