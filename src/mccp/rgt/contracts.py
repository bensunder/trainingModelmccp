"""RGT contracts: what a grader returns, what a distiller returns.

Kept small and serialisable: every strategy that ends up in training data is
recorded with the grader/distiller identity and the feedback it came from, so
the audit trail can answer "why did the model learn this rule?".
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Issue(_Strict):
    """One thing the attempt got wrong, with the rule that makes it wrong."""

    field: str = Field(description="Output field or aspect, e.g. decision, reason_code, format")
    expected: Any = None
    got: Any = None
    rule_id: str | None = Field(default=None, description="Stable id of the violated rule, e.g. PRICE_VARIANCE")
    rationale: str = Field(min_length=1)


class Feedback(_Strict):
    """A grader's verdict on one attempt."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: tuple[Issue, ...] = ()
    rationale: str = Field(min_length=1, description="Why the reference answer is correct")
    grader: str = Field(min_length=1, description="Grader id and version")


class Attempt(_Strict):
    """What the current model produced for a hinge prompt (on-policy sample)."""

    key: str = Field(min_length=8, description="Hash of the prompt the attempt answers")
    content: str | None
    model: str
    error: str | None = None


class Strategy(_Strict):
    """The compact rule placed in the reasoning block of pair B."""

    text: str = Field(min_length=1, max_length=1200)
    rule_ids: tuple[str, ...] = ()
    from_mistake: bool = False
    distiller: str = Field(min_length=1)


@runtime_checkable
class Grader(Protocol):
    id: str

    def grade(self, sample: dict[str, Any], attempt: str | None) -> Feedback:
        """Grade `attempt` (the final assistant content, or None when no attempt
        exists) against the reference answer carried by `sample`."""
        ...


@runtime_checkable
class Distiller(Protocol):
    id: str

    def distill(self, sample: dict[str, Any], feedback: Feedback) -> Strategy:
        """Turn grader feedback into a compact, reusable rule."""
        ...
