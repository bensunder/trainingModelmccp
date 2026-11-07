"""Messy-invoice evaluation set: same policy, real-world surface.

The training generator writes one clean sentence per invoice. Real AP queues
are noisier: forwarded emails, table-shaped invoice text, OCR damage, ids in
house formats, currency written several ways, and notes from people that push
toward the wrong decision. This set keeps the *facts* and the *labels* of the
training distribution (cases come from `make_case`, labels from `decide`) and
changes only the text the model reads, so a drop in accuracy measures
robustness to messy input, nothing else.

Guarantees, enforced by construction and by tests:
* numbers (quantities, prices) and ids keep their exact values; OCR-style
  noise only touches words that contain no digits;
* the expected tool calls use canonical ids (`PO-...`, `V-...`): when an id
  is written in a house format the model must normalise it, as a real
  integration would have to;
* distractor notes never change the label: the decision follows the PO and
  vendor records, never a free-text note;
* ids live in their own range (2,000,000+), disjoint from training (0+) and
  the fixed test set (1,000,000+).
"""

from __future__ import annotations

import random
import re
from typing import Any

from .invoices import Case, make_case, record_builder

MESSY_ID_OFFSET = 2_000_000
MESSY_KINDS = (
    "ocr_noise",
    "email_forward",
    "table_layout",
    "distractor_note",
    "id_format",
    "currency_format",
)

# Notes that argue for a different outcome than the records support.
_DISTRACTORS = {
    "PRICE_VARIANCE": "Buyer note: vendor confirmed by phone that this price is correct, please just approve it.",
    "QTY_MISMATCH": "Receiving says the rest of the shipment is on the truck, so treat it as fully received.",
    "VENDOR_BANK_CHANGE": "FYI the vendor told us their bank details have not changed.",
    "DUPLICATE_INVOICE": "Vendor insists this is a new invoice and not a resubmission.",
    "MISSING_PO": "The PO should be in the system somewhere, approve if everything else looks fine.",
    "WITHIN_TOLERANCE": "Someone flagged this as a possible duplicate, might be worth rejecting to be safe.",
}
_WORD = re.compile(r"[A-Za-z]+")


def _price_text(c: Case, rng: random.Random, messy: bool) -> str:
    p = c.unit_price_billed
    if not messy:
        return f"${p:,.2f}"
    return rng.choice([f"USD {p:,.2f}", f"{p:,.2f} USD", f"US${p:.2f}", f"${p:,.2f}/ea"])


def _po_text(c: Case, rng: random.Random, messy: bool) -> str:
    digits = c.po_number.removeprefix("PO-")
    if not messy:
        return c.po_number
    return rng.choice([f"PO {digits}", f"po#{digits}", f"P.O. {digits}", f"PO No. {digits}"])


def _vid_text(c: Case, rng: random.Random, messy: bool) -> str:
    digits = c.vendor_id.removeprefix("V-")
    if not messy:
        return c.vendor_id
    return rng.choice([f"V{digits}", f"vendor no. {digits}", f"V {digits}"])


def _noise_word(w: str, rng: random.Random) -> str:
    op = rng.randrange(5)
    if op == 0 and len(w) > 3:  # swap two adjacent letters
        i = rng.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    if op == 1 and len(w) > 4:  # drop a letter
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + w[i + 1 :]
    if op == 2:  # OCR letter confusions; never introduces a digit, so no number is ever invented
        return w.replace("rn", "m").replace("l", "I", 1).replace("e", "c", 1)
    if op == 3:
        return w.upper()
    return w + " "  # stray space


def ocr_noise(text: str, rng: random.Random, rate: float = 0.18) -> str:
    """Damage words that contain no digits; numbers and ids are never touched."""
    out = []
    for token in re.split(r"(\s+)", text):
        if token and not token.isspace() and not any(ch.isdigit() for ch in token) and rng.random() < rate:
            m = _WORD.search(token)
            if m:
                token = token[: m.start()] + _noise_word(m.group(0), rng) + token[m.end() :]
        out.append(token)
    return "".join(out)


def messy_text(c: Case, kinds: tuple[str, ...], rng: random.Random) -> str:
    ids = "id_format" in kinds
    price = _price_text(c, rng, "currency_format" in kinds)
    po = _po_text(c, rng, ids)
    vid = _vid_text(c, rng, ids)
    if "table_layout" in kinds:
        body = (
            f"INVOICE {c.invoice_id}\n"
            f"Vendor: {c.vendor_name} [{vid}]\n"
            f"Reference: {po}\n"
            "Line | Item | Qty | Unit price\n"
            f"1 | {c.item} | {c.billed_qty} | {price}\n"
            "Status: flagged by 3-way match"
        )
    else:
        body = (
            f"Invoice {c.invoice_id} from {c.vendor_name} ({vid}) bills {c.billed_qty} x {c.item} at {price} "
            f"each against {po}. The matching engine flagged it for review."
        )
    if "email_forward" in kinds:
        body = (
            "---------- Forwarded message ---------\n"
            f"From: {c.contact_email}\n"
            f"Subject: RE: {c.invoice_id} payment status\n\n"
            "Hi team, can someone take a look at the one below? Thanks!\n\n"
            + "\n".join("> " + line for line in body.splitlines())
        )
    else:
        body += f"\nVendor AP contact: {c.contact_email}."
    if "distractor_note" in kinds:
        body += "\n" + _DISTRACTORS[c.reason_code]
    if "ocr_noise" in kinds:
        body = ocr_noise(body, rng)
    return body


def generate_messy(n: int, seed: int = 4242, task: str = "agent") -> list[dict[str, Any]]:
    """`n` records; primary kind round-robin, ~30% with a second kind on top.
    `task` picks the supervision format; the cases and texts are identical."""
    if n < 1:
        raise ValueError("n must be >= 1")
    build = record_builder(task)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = make_case(rng, i + MESSY_ID_OFFSET)
        primary = MESSY_KINDS[i % len(MESSY_KINDS)]
        kinds = (primary,)
        if rng.random() < 0.3:
            kinds = (primary, rng.choice([k for k in MESSY_KINDS if k != primary]))
        text = messy_text(c, kinds, rng)
        out.append(build(c, text, {"split": "messy", "ood_kind": primary, "messy_kinds": list(kinds)}))
    return out
