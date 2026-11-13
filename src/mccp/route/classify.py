"""Classify a coding request into a task class, cheaply and explainably.

No model call on the hot path: deterministic rules over request features, each
decision returning the reasons it fired. Ordered from structural signals (what a
harness is doing: a tool loop, a side request) to text signals (what the user
asked for). Misclassification costs money or quality only through the policy,
which falls back to the strong model when evidence for a class is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import TaskClass
from .wire import Features

BACKGROUND_SUFFIX = ":background"

_PATTERNS: list[tuple[TaskClass, re.Pattern[str]]] = [
    (
        TaskClass.DEBUG,
        re.compile(
            r"traceback|stack ?trace|exception|segfault|\berror\b|\bbug\b|doesn'?t work|not working|fails?\b|failing|crash",
            re.I,
        ),
    ),
    (
        TaskClass.TESTS,
        re.compile(
            r"\bunit tests?\b|\btests? for\b|\bwrite (a )?tests?\b|pytest|jest|vitest|test cases?|coverage", re.I
        ),
    ),
    (
        TaskClass.REFACTOR,
        re.compile(
            r"refactor|clean ?up|restructure|simplify|\brename\b|extract (a |the )?(function|method|class)|dedupe|modernize",
            re.I,
        ),
    ),
    (
        TaskClass.EXPLAIN,
        re.compile(r"^\s*(explain|what (does|is)|why (does|is)|how (does|do|is)|walk me through|describe)\b", re.I),
    ),
]
_BACKGROUND_SYSTEM = re.compile(
    r"(write|generate) (a )?(concise |short )?(title|summary)|summari[sz]e (this|the) conversation|"
    r"5-10 word|topic of (this|the) conversation|is this a new (topic|conversation)",
    re.I,
)


@dataclass(frozen=True)
class Classification:
    task: TaskClass
    reasons: tuple[str, ...]


def classify(f: Features) -> Classification:
    model = f.requested_model.lower()
    if model.endswith(BACKGROUND_SUFFIX) or "haiku" in model:
        return Classification(TaskClass.BACKGROUND, ("requested background model",))
    if not f.has_tools and _BACKGROUND_SYSTEM.search(f.system_text or f.last_user_text):
        return Classification(TaskClass.BACKGROUND, ("side request: title/summary prompt",))
    if not f.has_tools and f.max_tokens is not None and f.max_tokens <= 512 and f.n_messages <= 2:
        return Classification(TaskClass.BACKGROUND, (f"short side request (max_tokens={f.max_tokens})",))
    if f.has_tools and f.n_assistant_turns >= 1:
        return Classification(TaskClass.AGENTIC, (f"tool loop ({f.n_assistant_turns} assistant turns)",))
    for task, pat in _PATTERNS:
        m = pat.search(f.last_user_text)
        if m:
            return Classification(task, (f"user text matches '{m.group(0).strip()[:40]}'",))
    if f.has_tools:
        return Classification(TaskClass.AGENTIC, ("tools offered",))
    return Classification(TaskClass.QUICK_EDIT, ("default: short single-shot request",))
