"""Deterministic decision policies that run next to a model, never inside it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import invoice_triage

# Policy id -> (record meta, extracted facts) -> RuleResult. An eval suite names
# its policy in `decision_rules`; scoring then runs the policy on what the model
# extracted and compares the resulting decision with the record's label.
RULESETS: dict[str, Callable[[dict[str, Any], dict[str, Any]], invoice_triage.RuleResult]] = {
    invoice_triage.POLICY_ID: invoice_triage.decide_from_record,
}
