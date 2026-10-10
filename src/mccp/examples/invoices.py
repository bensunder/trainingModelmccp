"""Deterministic synthetic dataset for the demo use case: accounts-payable
invoice-exception triage with `lookup_po` / `lookup_vendor` tool calls and a
JSON decision. Every field is verifiable, so the same rules power the golden
set (L1 eval) and, in Phase 2, GRPO reward functions.

Decision rules (the "policy" the adapter must learn):
* PO not found                          -> hold      MISSING_PO
* invoice already paid against the PO   -> reject    DUPLICATE_INVOICE
* vendor bank details changed < 30 days -> escalate  VENDOR_BANK_CHANGE
* billed qty > received qty             -> hold      QTY_MISMATCH
* |price variance| > 2%                 -> hold      PRICE_VARIANCE
* otherwise                             -> approve   WITHIN_TOLERANCE
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "You are the accounts-payable exception triage assistant. Use the tools to look up the purchase "
    "order and the vendor before deciding. Reply with only a JSON object with keys decision, "
    "reason_code, next_action and variance_pct."
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason_code", "next_action", "variance_pct"],
    "properties": {
        "decision": {"enum": ["approve", "hold", "reject", "escalate"]},
        "reason_code": {
            "enum": [
                "WITHIN_TOLERANCE",
                "PRICE_VARIANCE",
                "QTY_MISMATCH",
                "MISSING_PO",
                "DUPLICATE_INVOICE",
                "VENDOR_BANK_CHANGE",
            ]
        },
        "next_action": {"type": "string", "minLength": 3, "maxLength": 300},
        "variance_pct": {"type": ["number", "null"]},
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_po",
            "description": "Fetch a purchase order with ordered/received quantities, unit price and paid invoices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "po_number": {"type": "string", "description": "Purchase order number, e.g. PO-4500012345"}
                },
                "required": ["po_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_vendor",
            "description": "Fetch vendor master data including days since the last bank-detail change.",
            "parameters": {
                "type": "object",
                "properties": {"vendor_id": {"type": "string", "description": "Vendor id, e.g. V-10023"}},
                "required": ["vendor_id"],
            },
        },
    },
]

NEXT_ACTION = {
    "WITHIN_TOLERANCE": "Approve for payment in the next payment run.",
    "PRICE_VARIANCE": "Hold and request a price confirmation or credit memo from the vendor.",
    "QTY_MISMATCH": "Hold until the remaining quantity is received or a credit memo is issued.",
    "MISSING_PO": "Hold and ask the requester for a valid purchase order number.",
    "DUPLICATE_INVOICE": "Reject as a duplicate and notify the vendor.",
    "VENDOR_BANK_CHANGE": "Escalate to vendor-master security for call-back verification of bank details.",
}

DECISION = {
    "WITHIN_TOLERANCE": "approve",
    "PRICE_VARIANCE": "hold",
    "QTY_MISMATCH": "hold",
    "MISSING_PO": "hold",
    "DUPLICATE_INVOICE": "reject",
    "VENDOR_BANK_CHANGE": "escalate",
}

_ITEMS = [
    "safety valves",
    "drill bits",
    "hydraulic hose",
    "pump seals",
    "laptops",
    "office chairs",
    "steel pipe",
    "filters",
]
_VENDORS = [
    "Acme Industrial",
    "Northwind Supply",
    "Contoso Parts",
    "Fabrikam Metals",
    "Summit Fluid Systems",
    "Tailspin Tools",
]


@dataclass(frozen=True)
class Case:
    invoice_id: str
    vendor_id: str
    vendor_name: str
    contact_email: str
    po_number: str
    item: str
    billed_qty: int
    unit_price_billed: float
    po: dict[str, Any] | None
    vendor: dict[str, Any]
    reason_code: str
    variance_pct: float | None


def _round(x: float) -> float:
    return round(x + 0.0, 2)


def make_case(rng: random.Random, idx: int) -> Case:
    vendor_idx = rng.randrange(len(_VENDORS))
    # One vendor record per case keeps tool fixtures consistent with labels.
    vendor_id = f"V-{100000 + idx}"
    vendor_name = _VENDORS[vendor_idx]
    invoice_id = f"INV-{200000 + idx}"
    po_number = f"PO-45000{rng.randrange(10000, 99999)}"
    item = rng.choice(_ITEMS)
    ordered = rng.randrange(5, 200)
    unit_price = _round(rng.uniform(5, 900))
    bank_days = rng.choice(
        [rng.randrange(1, 29), rng.randrange(30, 2000), rng.randrange(30, 2000), rng.randrange(30, 2000)]
    )
    vendor = {"vendor_id": vendor_id, "name": vendor_name, "status": "active", "days_since_bank_change": bank_days}

    scenario = rng.choices(["ok", "price", "qty", "missing", "dup", "bank"], weights=[30, 20, 15, 10, 10, 15])[0]
    received = ordered
    billed_qty = ordered
    billed_price = unit_price
    paid: list[str] = []
    po: dict[str, Any] | None
    if scenario == "price":
        billed_price = _round(unit_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.25)))
    elif scenario == "ok":
        billed_price = _round(unit_price * (1 + rng.uniform(-0.015, 0.015)))
    elif scenario == "qty":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
    elif scenario == "dup":
        paid = [invoice_id]
    if scenario == "bank":
        vendor["days_since_bank_change"] = rng.randrange(1, 29)
    elif bank_days < 30:
        vendor["days_since_bank_change"] = rng.randrange(30, 2000)

    po = (
        None
        if scenario == "missing"
        else {
            "po_number": po_number,
            "vendor_id": vendor_id,
            "item": item,
            "ordered_qty": ordered,
            "received_qty": received,
            "unit_price": unit_price,
            "currency": "USD",
            "paid_invoices": paid,
        }
    )
    reason, variance = decide(po, vendor, invoice_id, billed_qty, billed_price)
    return Case(
        invoice_id=invoice_id,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        contact_email=f"ap.{vendor_name.split()[0].lower()}{rng.randrange(100)}@example.com",
        po_number=po_number,
        item=item,
        billed_qty=billed_qty,
        unit_price_billed=billed_price,
        po=po,
        vendor=vendor,
        reason_code=reason,
        variance_pct=variance,
    )


def decide(
    po: dict[str, Any] | None, vendor: dict[str, Any], invoice_id: str, billed_qty: int, billed_price: float
) -> tuple[str, float | None]:
    """The ground-truth policy. Also used as a verifiable reward in Phase 2."""
    if po is None:
        return "MISSING_PO", None
    variance = _round((billed_price - po["unit_price"]) / po["unit_price"] * 100)
    if invoice_id in po["paid_invoices"]:
        return "DUPLICATE_INVOICE", variance
    if vendor["days_since_bank_change"] < 30:
        return "VENDOR_BANK_CHANGE", variance
    if billed_qty > po["received_qty"]:
        return "QTY_MISMATCH", variance
    if abs(variance) > 2.0:
        return "PRICE_VARIANCE", variance
    return "WITHIN_TOLERANCE", variance


def user_message(c: Case) -> str:
    return (
        f"Invoice {c.invoice_id} from {c.vendor_name} ({c.vendor_id}) bills {c.billed_qty} x {c.item} at "
        f"${c.unit_price_billed:,.2f} each against {c.po_number}. The matching engine flagged it for review. "
        f"Vendor AP contact: {c.contact_email}. Please triage."
    )


def final_answer(c: Case) -> dict[str, Any]:
    return {
        "decision": DECISION[c.reason_code],
        "reason_code": c.reason_code,
        "next_action": NEXT_ACTION[c.reason_code],
        "variance_pct": c.variance_pct,
    }


def to_record(c: Case) -> dict[str, Any]:
    po_result = c.po if c.po is not None else {"error": "not_found", "po_number": c.po_number}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(c)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "lookup_po", "arguments": {"po_number": c.po_number}}}
                ],
            },
            {"role": "tool", "content": json.dumps(po_result)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "lookup_vendor", "arguments": {"vendor_id": c.vendor_id}}}
                ],
            },
            {"role": "tool", "content": json.dumps(c.vendor)},
            {"role": "assistant", "content": json.dumps(final_answer(c))},
        ],
        "tools": TOOLS,
        "meta": {"source_id": c.invoice_id, "license": "internal", "scenario": c.reason_code},
    }


def generate(n: int, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [to_record(make_case(rng, i)) for i in range(n)]


def fixtures(n: int, seed: int = 7) -> dict[str, Any]:
    """Backing data for the agent's tools, consistent with `generate`."""
    rng = random.Random(seed)
    pos: dict[str, Any] = {}
    vendors: dict[str, Any] = {}
    invoices = []
    for i in range(n):
        c = make_case(rng, i)
        if c.po is not None:
            pos[c.po_number] = c.po
        vendors[c.vendor_id] = c.vendor
        invoices.append({"invoice_id": c.invoice_id, "prompt": user_message(c), "expected": final_answer(c)})
    return {"purchase_orders": pos, "vendors": vendors, "invoices": invoices}
