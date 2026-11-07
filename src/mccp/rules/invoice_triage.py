"""Accounts-payable invoice-exception policy, as code.

This is the single source of truth for the triage decision. The synthetic data
generator labels with it, the extraction evaluation scores with it, and the
serving path (`mccp.agent.graph.triage_extract`) decides with it, so the
policy a model is measured against is exactly the policy production applies.

Rules, first match wins:

1. PO not found                              -> hold      MISSING_PO
2. vendor not found (lookup failed)          -> hold      VENDOR_NOT_FOUND
3. invoice already paid against the PO       -> reject    DUPLICATE_INVOICE
4. vendor bank details changed < 30 days ago -> escalate  VENDOR_BANK_CHANGE
5. billed qty > received qty                 -> hold      QTY_MISMATCH
6. |price variance| > 2%                     -> hold      PRICE_VARIANCE
7. otherwise                                 -> approve   WITHIN_TOLERANCE

Rule 2 is unreachable from the generators (every case has a vendor record); it
exists for production, where an extracted vendor id can fail to resolve.
Every result carries a trace of the rules evaluated, for the audit record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, cast

POLICY_ID = "invoice-triage-rules@v1"
BANK_CHANGE_DAYS = 30
PRICE_TOLERANCE_PCT = 2.0

NEXT_ACTION = {
    "WITHIN_TOLERANCE": "Approve for payment in the next payment run.",
    "PRICE_VARIANCE": "Hold and request a price confirmation or credit memo from the vendor.",
    "QTY_MISMATCH": "Hold until the remaining quantity is received or a credit memo is issued.",
    "MISSING_PO": "Hold and ask the requester for a valid purchase order number.",
    "DUPLICATE_INVOICE": "Reject as a duplicate and notify the vendor.",
    "VENDOR_BANK_CHANGE": "Escalate to vendor-master security for call-back verification of bank details.",
    "VENDOR_NOT_FOUND": "Hold and verify the vendor id against the vendor master.",
}

DECISION = {
    "WITHIN_TOLERANCE": "approve",
    "PRICE_VARIANCE": "hold",
    "QTY_MISMATCH": "hold",
    "MISSING_PO": "hold",
    "DUPLICATE_INVOICE": "reject",
    "VENDOR_BANK_CHANGE": "escalate",
    "VENDOR_NOT_FOUND": "hold",
}


def round2(x: float) -> float:
    return round(x + 0.0, 2)


def found(lookup: dict[str, Any] | None) -> bool:
    """A lookup result is usable unless it is missing or an error payload."""
    return isinstance(lookup, dict) and "error" not in lookup


@dataclass(frozen=True)
class RuleResult:
    decision: str
    reason_code: str
    next_action: str
    variance_pct: float | None
    policy: str = POLICY_ID
    trace: list[dict[str, Any]] = field(default_factory=list)

    def answer(self) -> dict[str, Any]:
        """The four fields of the triage answer (same shape the agent task emits)."""
        return {
            "decision": self.decision,
            "reason_code": self.reason_code,
            "next_action": self.next_action,
            "variance_pct": self.variance_pct,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    po: dict[str, Any] | None,
    vendor: dict[str, Any] | None,
    invoice_id: str,
    billed_qty: int,
    unit_price_billed: float,
) -> RuleResult:
    """Apply the policy to looked-up records and the facts read from the invoice.

    `po` / `vendor` are lookup results: None or an `{"error": ...}` payload means
    not found. Variance is computed only when a PO exists (it needs the PO price).
    """
    trace: list[dict[str, Any]] = []

    def done(code: str, variance: float | None) -> RuleResult:
        return RuleResult(DECISION[code], code, NEXT_ACTION[code], variance, POLICY_ID, trace)

    def check(rule: str, fired: bool, **facts: Any) -> bool:
        trace.append({"rule": rule, "fired": fired, **facts})
        return fired

    if check("po_found", not found(po), po_found=found(po)):
        return done("MISSING_PO", None)
    po = cast(dict[str, Any], po)
    variance = round2((unit_price_billed - po["unit_price"]) / po["unit_price"] * 100)
    if check("vendor_found", not found(vendor), vendor_found=found(vendor)):
        return done("VENDOR_NOT_FOUND", variance)
    vendor = cast(dict[str, Any], vendor)
    if check("duplicate", invoice_id in po["paid_invoices"], invoice_id=invoice_id):
        return done("DUPLICATE_INVOICE", variance)
    days = vendor["days_since_bank_change"]
    if check("bank_change_recent", days < BANK_CHANGE_DAYS, days_since_bank_change=days, limit=BANK_CHANGE_DAYS):
        return done("VENDOR_BANK_CHANGE", variance)
    if check(
        "qty_over_received",
        billed_qty > po["received_qty"],
        billed_qty=billed_qty,
        received_qty=po["received_qty"],
    ):
        return done("QTY_MISMATCH", variance)
    if check(
        "price_variance",
        abs(variance) > PRICE_TOLERANCE_PCT,
        variance_pct=variance,
        limit=PRICE_TOLERANCE_PCT,
    ):
        return done("PRICE_VARIANCE", variance)
    return done("WITHIN_TOLERANCE", variance)


Lookup = Callable[[str], "dict[str, Any] | None"]


def decide_extracted(extracted: dict[str, Any], lookup_po: Lookup, lookup_vendor: Lookup) -> RuleResult:
    """The extract-then-rules pipeline after the model call: look up the PO and the
    vendor by the ids the model read, then apply the policy. `extracted` must
    already be valid against the extraction schema."""
    return evaluate(
        lookup_po(extracted["po_number"]),
        lookup_vendor(extracted["vendor_id"]),
        extracted["invoice_id"],
        int(extracted["billed_qty"]),
        float(extracted["unit_price_billed"]),
    )


def decide_from_record(meta: dict[str, Any], extracted: dict[str, Any]) -> RuleResult:
    """Evaluation: the record's `meta.lookups` stands in for the PO and vendor
    services. A wrong extracted id finds nothing, as it would in production."""
    lookups = meta["lookups"]
    return decide_extracted(extracted, lookups["po"].get, lookups["vendor"].get)
