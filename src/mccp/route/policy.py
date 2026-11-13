"""Choose a model for a request from measured evidence.

Rule: no evidence, no downgrade. A cheaper model is only chosen for a task class
when the evaluation shows its pass rate clears the profile's quality bar, judged
by the Wilson lower bound (so 9/10 does not beat a bar of 0.85 the way 90/100
does). Without evidence the request goes to the profile's fallback (strong) model,
except harness background requests, which go to the profile's background model.

Sessions are sticky: once an agentic conversation is on a model it stays there
(switching models mid-session breaks tool context and throws away the provider's
prompt cache, which is most of the saving on long sessions).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import Catalog, Model, Profile, Strategy, TaskClass, Wire


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n: int = Field(ge=0)
    passed: int = Field(ge=0)
    usd_per_task: float | None = Field(default=None, ge=0)

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def lower(self) -> float:
        return wilson_lower(self.passed, self.n)


class Evidence(BaseModel):
    """Per model, per task class: how many evaluation tasks it passed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    suite: str
    generated_at: str
    results: dict[str, dict[TaskClass, Cell]] = Field(default_factory=dict)

    def cell(self, model_id: str, task: TaskClass) -> Cell | None:
        return self.results.get(model_id, {}).get(task)

    @classmethod
    def empty(cls) -> Evidence:
        return cls(suite="none", generated_at="never")

    @classmethod
    def load(cls, path: str | Path | None) -> Evidence:
        if not path or not Path(path).exists():
            return cls.empty()
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def wilson_lower(passed: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


@dataclass(frozen=True)
class Choice:
    model: str
    fallbacks: tuple[str, ...]
    reason: str


class NoEligibleModel(Exception):
    pass


def _blended(m: Model) -> float:
    # Coding traffic is input-heavy; 3:1 input:output when no measured per-task cost.
    return 0.75 * m.price.input_per_m + 0.25 * m.price.output_per_m


def eligible(catalog: Catalog, ids: list[str], wire: Wire, approved: set[str] | None, needs_tools: bool) -> list[str]:
    out = []
    for mid in ids:
        m = catalog.model(mid)
        if wire not in m.wires or (needs_tools and not m.supports_tools):
            continue
        if approved is not None and mid not in approved:
            continue
        out.append(mid)
    return list(dict.fromkeys(out))


def choose(
    catalog: Catalog,
    profile: Profile,
    task: TaskClass,
    wire: Wire,
    evidence: Evidence,
    approved: set[str] | None = None,
    needs_tools: bool = False,
    sticky: str | None = None,
) -> Choice:
    cands = eligible(catalog, profile.candidates, wire, approved, needs_tools)
    chain = eligible(catalog, profile.escalation, wire, approved, needs_tools)

    def done(model: str, reason: str) -> Choice:
        return Choice(model, tuple(m for m in chain if m != model), reason)

    if sticky and sticky in cands:
        return done(sticky, "session pinned to this model")
    if task is TaskClass.BACKGROUND and profile.background_model:
        bg = eligible(catalog, [profile.background_model], wire, approved, needs_tools)
        if bg:
            return done(bg[0], "background request -> background model")
    if not cands:
        raise NoEligibleModel(
            f"profile {profile.id}: no approved model serves {wire.value}" + (" with tools" if needs_tools else "")
        )
    bar = profile.bar(task)
    measured = [(mid, evidence.cell(mid, task)) for mid in cands]
    measured = [(mid, c) for mid, c in measured if c is not None and c.n > 0]
    qualifying = [(mid, c) for mid, c in measured if c.lower >= bar]
    fallback = profile.fallback if profile.fallback in cands else cands[0]
    if not qualifying:
        why = "no evidence for this task" if not measured else f"no model clears the bar {bar:.2f}"
        return done(fallback, f"{why} -> fallback")

    def cost(mid: str, c: Cell) -> float:
        return c.usd_per_task if c.usd_per_task is not None else _blended(catalog.model(mid))

    if profile.strategy is Strategy.INTELLIGENCE:
        mid, c = max(qualifying, key=lambda x: (x[1].lower, -cost(*x)))
        return done(mid, f"best measured quality ({c.passed}/{c.n})")
    if profile.strategy is Strategy.BALANCED:
        best = max(c.lower for _, c in qualifying)
        near = [(mid, c) for mid, c in qualifying if c.lower >= best - profile.balanced_margin]
        mid, c = min(near, key=lambda x: cost(*x))
        return done(mid, f"cheapest within {profile.balanced_margin:.0%} of best ({c.passed}/{c.n})")
    mid, c = min(qualifying, key=lambda x: cost(*x))
    return done(mid, f"cheapest clearing bar {bar:.2f} ({c.passed}/{c.n})")


def explain(catalog: Catalog, profile: Profile, evidence: Evidence) -> dict[str, Any]:
    """For the admin API: what each task class routes to today, and why."""
    out: dict[str, Any] = {}
    for task in TaskClass:
        try:
            c = choose(catalog, profile, task, Wire.ANTHROPIC, evidence)
            out[task.value] = {"model": c.model, "reason": c.reason}
        except NoEligibleModel as e:
            out[task.value] = {"model": None, "reason": str(e)}
    return out
