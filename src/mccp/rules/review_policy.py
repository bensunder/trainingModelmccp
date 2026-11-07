"""Review policies: which validation findings route an invoice to a person.

The validator (mccp.rules.validation) reports findings: blocking ones (a value is
not on the page) always route to a person; flags are evidence that something may
be wrong. A review policy names the flags that also block. The flags it does not
name are passed to the reviewer as hints. Keeping the two apart lets each blocking
rule be measured on its own (`pilot.py failures --policy ...`) before it is
adopted, and a suite names the policy it decides with (`review_policy`).

A policy only ever adds human reviews; it never changes, corrects or accepts a
value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .validation import MISSING_REQUIRED, REALDOC_VALIDATOR_V2_ID, VALIDATOR_FLAGS


@dataclass(frozen=True)
class ReviewPolicy:
    id: str
    validator: str  # the validator whose flags this policy reads
    block_on: frozenset[str]
    description: str

    def decide(self, validation: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        """(flags that block under this policy, flags left as reviewer hints)."""
        flags = list((validation or {}).get("flags") or [])
        return [f for f in flags if f in self.block_on], [f for f in flags if f not in self.block_on]


def _policy(pid: str, block_on: set[str], description: str) -> ReviewPolicy:
    unknown = block_on - VALIDATOR_FLAGS[REALDOC_VALIDATOR_V2_ID]
    if unknown:
        raise ValueError(f"{pid}: {REALDOC_VALIDATOR_V2_ID} never raises {sorted(unknown)}")
    return ReviewPolicy(pid, REALDOC_VALIDATOR_V2_ID, frozenset(block_on), description)


REVIEW_POLICIES: dict[str, ReviewPolicy] = {
    p.id: p
    for p in (
        _policy("realdoc-review-flags-none@v1", set(), "no flag blocks (validation grounding only)"),
        _policy("realdoc-review-missing@v1", {MISSING_REQUIRED}, "a null invoice number or total goes to a person"),
        _policy(
            "realdoc-review-several-totals@v1",
            {"several_total_amounts"},
            "several distinct amounts on total lines go to a person",
        ),
        _policy(
            "realdoc-review-missing-several-totals@v1",
            {MISSING_REQUIRED, "several_total_amounts"},
            "both of the above",
        ),
    )
}


def adhoc_policy(validator: str, block_on: list[str]) -> ReviewPolicy:
    """An unregistered policy for offline measurement (`failures --block-on`)."""
    unknown = set(block_on) - VALIDATOR_FLAGS.get(validator, frozenset())
    if unknown:
        known = ", ".join(sorted(VALIDATOR_FLAGS.get(validator, ())))
        raise ValueError(f"{validator} never raises {sorted(unknown)} (it raises: {known})")
    return ReviewPolicy("adhoc:" + "+".join(sorted(block_on)), validator, frozenset(block_on), "measurement only")
