"""Deterministic RGT grader and distiller for invoice-exception triage.

The task's policy is an ordered rule list (see `mccp.examples.invoices`), so
the grader needs no model: it reads the facts the assistant saw (the PO and
vendor tool results), the reference decision, and the attempt, and explains
both why the reference is right and what the attempt got wrong. The
distiller turns that into the compact strategy for the reasoning block:
the rule chain evaluated on this invoice, plus the mistake when there was one.

Costs nothing to run and is exactly reproducible, which is what makes it a
fair first test of RGT before paying for an LLM grader on open-ended tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mccp.examples.invoices import DECISION

from .contracts import Feedback, Issue, Strategy

GRADER_ID = "invoice-rule-grader/v1"
DISTILLER_ID = "invoice-template-distiller/v1"
ORDER = (
    "MISSING_PO",
    "DUPLICATE_INVOICE",
    "VENDOR_BANK_CHANGE",
    "QTY_MISMATCH",
    "PRICE_VARIANCE",
    "WITHIN_TOLERANCE",
)
TOLERANCE_PCT = 2.0
BANK_DAYS = 30
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class FactsError(ValueError):
    """The sample does not carry what the invoice grader needs."""


def facts(sample: dict[str, Any]) -> dict[str, Any]:
    """PO, vendor and reference answer as seen in the conversation."""
    msgs = sample["messages"]
    po: dict[str, Any] | None = None
    po_seen = False
    vendor: dict[str, Any] | None = None
    pending: list[str] = []
    for m in msgs[:-1]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = [tc["function"]["name"] for tc in m["tool_calls"]]
        elif m.get("role") == "tool" and pending:
            name = pending.pop(0)
            try:
                body = json.loads(m.get("content") or "")
            except json.JSONDecodeError as e:
                raise FactsError(f"tool result for {name} is not JSON") from e
            if name == "lookup_po":
                po_seen = True
                po = None if isinstance(body, dict) and body.get("error") else body
            elif name == "lookup_vendor":
                vendor = body
    if not po_seen or vendor is None:
        raise FactsError("sample lacks lookup_po / lookup_vendor results")
    try:
        ref = json.loads(msgs[-1]["content"])
    except json.JSONDecodeError as e:
        raise FactsError("reference answer is not JSON") from e
    if ref.get("reason_code") not in ORDER:
        raise FactsError(f"unknown reason_code {ref.get('reason_code')!r}")
    return {"po": po, "vendor": vendor, "ref": ref}


def _check_text(rule: str, f: dict[str, Any], fires: bool) -> str:
    po, vendor, v = f["po"], f["vendor"], f["ref"].get("variance_pct")
    if rule == "MISSING_PO":
        return "PO lookup returned not_found" if fires else f"{po['po_number']} found"
    if rule == "DUPLICATE_INVOICE":
        return "invoice already in the PO's paid invoices" if fires else "invoice not yet paid on this PO"
    if rule == "VENDOR_BANK_CHANGE":
        d = vendor.get("days_since_bank_change")
        return f"bank details changed {d} days ago ({'<' if fires else '>='} {BANK_DAYS})"
    if rule == "QTY_MISMATCH":
        r = po.get("received_qty") if po else None
        return f"billed qty {'>' if fires else '<='} received qty {r}"
    if rule == "PRICE_VARIANCE":
        return f"price variance {v:+.2f}% {'exceeds' if fires else 'within'} +/-{TOLERANCE_PCT:g}%"
    return "no exception applies"


def rule_chain(f: dict[str, Any]) -> str:
    """The ordered checks on this invoice, ending at the rule that fires."""
    code = f["ref"]["reason_code"]
    k = ORDER.index(code)
    parts = [f"{_check_text(r, f, fires=False)} ok" for r in ORDER[:k] if r != "WITHIN_TOLERANCE"]
    if code != "WITHIN_TOLERANCE":
        parts.append(_check_text(code, f, fires=True))
    v = f["ref"].get("variance_pct")
    var = (
        "variance_pct = null (no PO price)"
        if v is None
        else f"variance_pct = (billed - PO unit price) / PO unit price x 100 = {v:+.2f}"
    )
    return f"Check in order: {'; '.join(parts)} -> {DECISION[code]} / {code}. {var}."


def _parse(attempt: str) -> dict[str, Any] | None:
    text = attempt.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class InvoiceRuleGrader:
    id = GRADER_ID

    def grade(self, sample: dict[str, Any], attempt: str | None) -> Feedback:
        f = facts(sample)
        ref = f["ref"]
        why = rule_chain(f)
        if attempt is None:  # no on-policy sample: the reference is graded against itself
            return Feedback(passed=True, score=1.0, rationale=why, grader=self.id)
        got = _parse(attempt)
        if got is None:
            return Feedback(
                passed=False,
                score=0.0,
                rationale=why,
                grader=self.id,
                issues=(
                    Issue(
                        field="format",
                        got=attempt[:200],
                        rule_id="OUTPUT_FORMAT",
                        rationale="output was not a single JSON object with decision, reason_code, "
                        "next_action and variance_pct",
                    ),
                ),
            )
        issues: list[Issue] = []
        code, got_code = ref["reason_code"], got.get("reason_code")
        if got_code != code:
            if got_code in ORDER and ORDER.index(got_code) < ORDER.index(code):
                because = f"{_check_text(got_code, f, fires=False)}, so {got_code} does not apply"
            elif got_code in ORDER and code != "WITHIN_TOLERANCE":
                because = f"{_check_text(code, f, fires=True)} and {code} is checked before {got_code}"
            elif got_code in ORDER:
                because = f"{_check_text(got_code, f, fires=False)}, so no exception applies"
            else:
                because = f"{got_code!r} is not a valid reason code"
            issues.append(
                Issue(
                    field="reason_code",
                    expected=code,
                    got=got_code,
                    rule_id=code,
                    rationale=f"chose {got_code}; {because}",
                )
            )
        if got.get("decision") != ref["decision"]:
            issues.append(
                Issue(
                    field="decision",
                    expected=ref["decision"],
                    got=got.get("decision"),
                    rule_id=code,
                    rationale=f"{code} always maps to {DECISION[code]}",
                )
            )
        v, gv = ref.get("variance_pct"), got.get("variance_pct")
        v_ok = (v is None and gv is None) or (
            isinstance(v, (int, float)) and isinstance(gv, (int, float)) and abs(v - gv) <= 0.011
        )
        if not v_ok:
            issues.append(
                Issue(
                    field="variance_pct",
                    expected=v,
                    got=gv,
                    rule_id="VARIANCE_FORMULA",
                    rationale="variance_pct must be null when the PO is missing"
                    if v is None
                    else f"variance_pct must be {v:+.2f} = (billed - PO unit price) / PO unit price x 100, "
                    "rounded to 2 decimals",
                )
            )
        score = 1.0 - len({i.field for i in issues}) / 3
        return Feedback(passed=not issues, score=max(0.0, score), issues=tuple(issues), rationale=why, grader=self.id)


class InvoiceTemplateDistiller:
    """Strategy = the rule chain for this invoice (+ the mistake, if any).

    Short on purpose: the paper's strategies are one-line rules; this one is
    the decision procedure evaluated on the invoice, about 60-90 tokens."""

    id = DISTILLER_ID

    def distill(self, sample: dict[str, Any], feedback: Feedback) -> Strategy:
        f = facts(sample)
        text = feedback.rationale
        rule_ids = [f["ref"]["reason_code"]]
        if feedback.issues:
            lead = feedback.issues[0]
            text += f" Mistake to avoid: {lead.rationale}."
            rule_ids.extend(i.rule_id for i in feedback.issues if i.rule_id and i.rule_id not in rule_ids)
        return Strategy(text=text, rule_ids=tuple(rule_ids), from_mistake=bool(feedback.issues), distiller=self.id)
