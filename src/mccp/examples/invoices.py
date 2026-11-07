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

from mccp.rules import invoice_triage as _rules

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

# The policy lives in mccp.rules.invoice_triage (single source of truth); these
# names stay importable from here for the generators, RGT and tests.
NEXT_ACTION = _rules.NEXT_ACTION
DECISION = _rules.DECISION

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


_round = _rules.round2


# Training mixes. `standard` is the original distribution and must stay bit-identical
# (old datasets reproduce from their seed). `boundary` keeps the standard scenarios at
# lower weight and adds the cases the Nov 2026 OOD breakdown showed the model never
# saw: values right at the 30-day and 2% thresholds, clean and partial-billing
# approvals, and rule collisions where precedence decides the outcome.
_MIXES: dict[str, tuple[list[str], list[int]]] = {
    "standard": (["ok", "price", "qty", "missing", "dup", "bank"], [30, 20, 15, 10, 10, 15]),
    "boundary": (
        [
            "ok",
            "price",
            "qty",
            "missing",
            "dup",
            "bank",
            "bank_edge_recent",  # bank change 25-29 days ago        -> VENDOR_BANK_CHANGE
            "bank_edge_ok",  # bank change 30-35 days ago            -> not a bank flag
            "price_edge_over",  # |variance| 2.05-3.0%               -> PRICE_VARIANCE
            "price_edge_under",  # |variance| 1.5-1.95%              -> WITHIN_TOLERANCE
            "clean",  # nothing wrong, variance within +/-0.5%       -> WITHIN_TOLERANCE
            "partial_billing",  # billed = received < ordered        -> WITHIN_TOLERANCE
            "collision_bank_qty",  # recent bank change + billed > received -> VENDOR_BANK_CHANGE
            "collision_dup_price",  # already paid + price variance  -> DUPLICATE_INVOICE
        ],
        [10, 8, 8, 6, 6, 6, 8, 8, 8, 8, 6, 8, 6, 4],
    ),
}
TRAIN_MIXES = tuple(_MIXES)


def _signed(rng: random.Random, lo: float, hi: float) -> float:
    return rng.choice([-1, 1]) * rng.uniform(lo, hi)


def make_case(rng: random.Random, idx: int, mix: str = "standard") -> Case:
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

    if mix not in _MIXES:
        raise ValueError(f"unknown training mix {mix!r}; choose from {', '.join(_MIXES)}")
    names, weights = _MIXES[mix]
    scenario = rng.choices(names, weights=weights)[0]
    received = ordered
    billed_qty = ordered
    billed_price = unit_price
    paid: list[str] = []
    po: dict[str, Any] | None
    # Standard scenarios: unchanged code path and random-draw order (reproducibility).
    if scenario == "price":
        billed_price = _round(unit_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.25)))
    elif scenario == "ok":
        billed_price = _round(unit_price * (1 + rng.uniform(-0.015, 0.015)))
    elif scenario == "qty":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
    elif scenario == "dup":
        paid = [invoice_id]
    # Boundary scenarios (only drawn by the `boundary` mix).
    elif scenario == "price_edge_over":
        billed_price = _round(unit_price * (1 + _signed(rng, 0.0205, 0.030)))
    elif scenario == "price_edge_under":
        billed_price = _round(unit_price * (1 + _signed(rng, 0.015, 0.0195)))
    elif scenario in ("clean", "bank_edge_ok", "bank_edge_recent"):
        billed_price = _round(unit_price * (1 + rng.uniform(-0.005, 0.005)))
    elif scenario == "partial_billing":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
        billed_qty = received
        billed_price = _round(unit_price * (1 + rng.uniform(-0.01, 0.01)))
    elif scenario == "collision_bank_qty":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
    elif scenario == "collision_dup_price":
        paid = [invoice_id]
        billed_price = _round(unit_price * (1 + _signed(rng, 0.04, 0.15)))
    if scenario in ("bank", "collision_bank_qty"):
        vendor["days_since_bank_change"] = rng.randrange(1, 29)
    elif scenario == "bank_edge_recent":
        vendor["days_since_bank_change"] = rng.randrange(25, 30)
    elif scenario == "bank_edge_ok":
        vendor["days_since_bank_change"] = rng.randrange(30, 36)
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
    """The ground-truth label: `mccp.rules.invoice_triage.evaluate`. Also used as a
    verifiable reward in Phase 2."""
    r = _rules.evaluate(po, vendor, invoice_id, billed_qty, billed_price)
    return r.reason_code, r.variance_pct


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


def to_record(c: Case, user_text: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    po_result = c.po if c.po is not None else {"error": "not_found", "po_number": c.po_number}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(c) if user_text is None else user_text},
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
        "meta": {"source_id": c.invoice_id, "license": "internal", "scenario": c.reason_code, **(meta or {})},
    }


# ------------------------------------------------------------ extract task --
# The model reads the invoice and returns its facts; code does the lookups and
# applies `mccp.rules.invoice_triage`. One model call per invoice, no tool calls.
TASKS = ("agent", "extract")
# Canonical facts (what the rules and the scoring use) ...
EXTRACT_FIELDS = ["invoice_id", "po_number", "vendor_id", "billed_qty", "unit_price_billed"]
# ... and what the model writes: the unit price exactly as printed. Code parses it
# (mccp.rules.extraction, invoice-extract@v2); asking the model to strip symbols and
# separators itself made it drop decimal points on real scans.
EXTRACT_MODEL_FIELDS = ["invoice_id", "po_number", "vendor_id", "billed_qty", "unit_price_raw"]
EXTRACT_SYSTEM_PROMPT = (
    "You read accounts-payable invoice exceptions and extract their facts. Reply with only a JSON object "
    "with keys invoice_id, po_number, vendor_id, billed_qty and unit_price_raw. Write ids in canonical "
    "form (INV-<digits>, PO-<digits>, V-<digits>) and billed_qty as an integer. For unit_price_raw copy "
    "the unit price exactly as printed on the invoice, as text, including any currency symbol; do not "
    "reformat it."
)
EXTRACT_RAW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": EXTRACT_MODEL_FIELDS,
    "properties": {
        "invoice_id": {"type": "string", "pattern": "^INV-[0-9]+$"},
        "po_number": {"type": "string", "pattern": "^PO-[0-9]+$"},
        "vendor_id": {"type": "string", "pattern": "^V-[0-9]+$"},
        "billed_qty": {"type": "integer", "minimum": 0},
        "unit_price_raw": {"type": "string", "minLength": 1},
    },
}
EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": EXTRACT_FIELDS,
    "properties": {
        "invoice_id": {"type": "string", "pattern": "^INV-[0-9]+$"},
        "po_number": {"type": "string", "pattern": "^PO-[0-9]+$"},
        "vendor_id": {"type": "string", "pattern": "^V-[0-9]+$"},
        "billed_qty": {"type": "integer", "minimum": 0},
        "unit_price_billed": {"type": "number", "minimum": 0},
    },
}


def extraction(c: Case) -> dict[str, Any]:
    """What the model must write for this invoice (the extract task's label). The
    price is in the form every generator prints it; scoring compares the parsed
    value, so other printed forms of the same amount ("USD 1,234.50") match too."""
    return {
        "invoice_id": c.invoice_id,
        "po_number": c.po_number,
        "vendor_id": c.vendor_id,
        "billed_qty": c.billed_qty,
        "unit_price_raw": f"${c.unit_price_billed:,.2f}",
    }


def to_extract_record(c: Case, user_text: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Same case and user text as `to_record`, extract-task supervision. `meta.lookups`
    holds the system-of-record data keyed by id (what the PO and vendor services
    would return), and `meta.expected` the policy answer, so the evaluation can run
    the real pipeline on the *extracted* ids: a wrong id finds nothing, as it would
    in production."""
    return {
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message(c) if user_text is None else user_text},
            {"role": "assistant", "content": json.dumps(extraction(c))},
        ],
        "meta": {
            "source_id": c.invoice_id,
            "license": "internal",
            "scenario": c.reason_code,
            **(meta or {}),
            "task": "extract",
            "lookups": {
                "po": {c.po_number: c.po} if c.po is not None else {},
                "vendor": {c.vendor_id: c.vendor},
            },
            "expected": final_answer(c),
        },
    }


def record_builder(task: str):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    return to_record if task == "agent" else to_extract_record


def generate(
    n: int, seed: int = 7, id_offset: int = 0, mix: str = "standard", task: str = "agent"
) -> list[dict[str, Any]]:
    """`id_offset` moves invoice/vendor ids into a disjoint range, so a separately
    generated test set can never share a `source_id` with training data.
    `mix` picks the scenario distribution (`standard` or `boundary`, see `_MIXES`);
    labels always come from `decide`, so every mix is correct by construction.
    `task` picks the supervision format; the cases are identical for every task."""
    build = record_builder(task)
    rng = random.Random(seed)
    return [build(make_case(rng, i + id_offset, mix)) for i in range(n)]


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


# ---------------------------------------------------------------------------
# Out-of-distribution (OOD) evaluation set.
#
# The training generator above is one distribution: six vendors, eight items,
# one sentence template, and scenario values well away from the policy
# thresholds. A held-out split of it measures memorisation of that
# distribution as much as generalisation. The OOD set changes every surface
# feature while keeping the same policy (`decide`), so a high score means the
# adapter learned the rules rather than the generator:
#
# * disjoint vendor names, items and id ranges (nothing shared with training);
# * eight new request phrasings (forwarded emails, terse queue entries, ...);
# * cases within a fraction of a percent / a day / one unit of each threshold;
# * rule-precedence collisions (duplicate + price variance, bank change + qty).
#
# Labels always come from `decide`, so they are correct by construction.
# ---------------------------------------------------------------------------

OOD_VENDORS = [
    "Granite Peak Industrial",
    "Bluewater Process Controls",
    "Ironclad Fastener Co",
    "Sterling Instrumentation",
    "Cascade Electrical Supply",
    "Redline Safety Products",
    "Meridian Pipe & Valve",
    "Northstar Lab Equipment",
]
OOD_ITEMS = [
    "pressure gauges",
    "torque wrenches",
    "control valves",
    "fiber patch cables",
    "welding rods",
    "server rack rails",
    "gas detectors",
    "bearing assemblies",
]
OOD_KINDS = [
    "boundary_price_over",  # |variance| just above 2%      -> PRICE_VARIANCE
    "boundary_price_under",  # |variance| just below 2%     -> WITHIN_TOLERANCE
    "boundary_bank_recent",  # bank change 25-29 days ago   -> VENDOR_BANK_CHANGE
    "boundary_bank_ok",  # bank change 30-34 days ago       -> not a bank flag
    "boundary_qty_over",  # billed exactly one more than received
    "partial_billing_ok",  # partial receipt, billed == received -> WITHIN_TOLERANCE
    "precedence_dup_over_price",  # duplicate wins over price variance
    "precedence_bank_over_qty",  # bank change wins over quantity mismatch
    "missing_po",
    "clean",
]
_OOD_TEMPLATES = [
    "FW: Vendor invoice {inv} | {vendor} ({vid}) | {qty} units of {item} @ ${price} per unit | PO ref {po}. "
    "Exception raised by 3-way match. Contact {email}. Please advise.",
    "AP queue item: {vendor} [{vid}] submitted {inv} for {qty} {item}, unit price ${price}, citing {po}. "
    "Needs a triage decision.",
    "Can you check {inv}? It's from {vendor}, vendor id {vid}. They billed {qty} {item} at ${price} each "
    "and referenced purchase order {po}.",
    "Exception {inv}: supplier={vendor} ({vid}); qty={qty}; item={item}; unit_price=${price}; po={po}. "
    "Reply with the decision.",
    "{vendor} ({vid}) sent invoice {inv} referencing {po}: {qty} x {item} billed at ${price}/ea. "
    "Their AP mailbox is {email}. Triage please.",
    "Hi team - invoice {inv} came in from {vendor}, id {vid}, for {item} ({qty} units, ${price} each) "
    "against {po}. The match failed; what should we do?",
    "Triage request. Vendor: {vendor} / {vid}. Invoice: {inv}. PO: {po}. Line: {item}, quantity {qty}, "
    "unit cost ${price}.",
    "Please review invoice {inv} ({vendor}, {vid}) - {qty} {item} invoiced at ${price} each on {po}. "
    "Questions to {email}.",
]


def _ood_variance(rng: random.Random, lo: float, hi: float) -> float:
    return rng.choice([-1, 1]) * rng.uniform(lo, hi) / 100.0


def make_ood_case(rng: random.Random, idx: int, kind: str) -> Case:
    vendor_id = f"V-{900000 + idx}"
    vendor_name = OOD_VENDORS[rng.randrange(len(OOD_VENDORS))]
    invoice_id = f"INV-{900000 + idx}"
    po_number = f"PO-46{rng.randrange(10000000, 99999999)}"
    item = rng.choice(OOD_ITEMS)
    ordered = rng.randrange(5, 400)
    unit_price = _round(rng.uniform(5, 2500))
    received = ordered
    billed_qty = ordered
    billed_price = unit_price
    paid: list[str] = []
    bank_days = rng.randrange(60, 3000)
    missing = False

    if kind == "boundary_price_over":
        billed_price = _round(unit_price * (1 + _ood_variance(rng, 2.15, 2.9)))
    elif kind == "boundary_price_under":
        billed_price = _round(unit_price * (1 + _ood_variance(rng, 1.1, 1.85)))
    elif kind == "boundary_bank_recent":
        bank_days = rng.randrange(25, 30)
    elif kind == "boundary_bank_ok":
        bank_days = rng.randrange(30, 35)
    elif kind == "boundary_qty_over":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 3)))
        billed_qty = received + 1
    elif kind == "partial_billing_ok":
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
        billed_qty = received
    elif kind == "precedence_dup_over_price":
        paid = [invoice_id]
        billed_price = _round(unit_price * (1 + _ood_variance(rng, 4.0, 15.0)))
    elif kind == "precedence_bank_over_qty":
        bank_days = rng.randrange(1, 29)
        received = max(1, ordered - rng.randrange(1, max(2, ordered // 2)))
    elif kind == "missing_po":
        missing = True
    elif kind != "clean":
        raise ValueError(f"unknown OOD kind {kind!r}")

    vendor = {"vendor_id": vendor_id, "name": vendor_name, "status": "active", "days_since_bank_change": bank_days}
    po = (
        None
        if missing
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
    slug = vendor_name.split()[0].lower()
    return Case(
        invoice_id=invoice_id,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        contact_email=f"payables.{slug}{rng.randrange(1000)}@example.org",
        po_number=po_number,
        item=item,
        billed_qty=billed_qty,
        unit_price_billed=billed_price,
        po=po,
        vendor=vendor,
        reason_code=reason,
        variance_pct=variance,
    )


def ood_user_message(c: Case, template_idx: int) -> str:
    return _OOD_TEMPLATES[template_idx % len(_OOD_TEMPLATES)].format(
        inv=c.invoice_id,
        vendor=c.vendor_name,
        vid=c.vendor_id,
        qty=c.billed_qty,
        item=c.item,
        price=f"{c.unit_price_billed:,.2f}",
        po=c.po_number,
        email=c.contact_email,
    )


def generate_ood(n: int, seed: int = 1009, task: str = "agent") -> list[dict[str, Any]]:
    """`n` OOD records, kinds balanced round-robin, phrasings rotated
    independently so kind and wording are not confounded."""
    if n < 1:
        raise ValueError("n must be >= 1")
    build = record_builder(task)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = OOD_KINDS[i % len(OOD_KINDS)]
        c = make_ood_case(rng, i, kind)
        tmpl = (i // len(OOD_KINDS) + i) % len(_OOD_TEMPLATES)
        out.append(build(c, ood_user_message(c, tmpl), {"split": "ood", "ood_kind": kind, "template": tmpl}))
    return out
