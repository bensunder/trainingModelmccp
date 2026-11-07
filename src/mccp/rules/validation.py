"""Financial-field validation: between extraction and the rules engine.

The model reads; code parses (mccp.rules.extraction); this module checks the
parsed facts against the document the model was given, before any lookup or
decision. It never corrects a value. It only confirms one, flags it, or blocks
the invoice for human review:

* grounding (blocking): every extracted value must be printed on the page. An
  amount is grounded when one of its printed forms (1,533.54 / 1533.54, and
  1,534 / 1534 for whole amounts) occurs in the text with no digit touching it;
  an id when its digits (or, for free-form real-document ids, its letters and
  digits) occur. `153354` for a page that says `$1,533.54` is not on the page,
  so it is blocked rather than paid.
* line arithmetic (blocking when contradicted): on the line that prints the unit
  price, if other amounts with cents are printed, one of them must equal
  qty x price. Taking the line amount (or another column) as the unit price
  contradicts it. A page that prints no line amount is not a contradiction
  (corroboration is reported).
* total ambiguity (flag, not blocking): amounts printed on total-labelled lines
  (TOTAL, AMOUNT DUE, BALANCE DUE, GRAND TOTAL; not SUBTOTAL). Several distinct
  ones, or an answer that is on the page but not on such a line, is reported so
  the policy for those cases can be chosen from measured rates.

`VALIDATORS` maps a validator id to `(raw answer, canonical answer, source
text) -> Validation`; an eval suite names one in `validation`, the serving path
calls the same function.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .extraction import parse_money

EXTRACT_VALIDATOR_ID = "invoice-extract-validation@v1"
REALDOC_VALIDATOR_ID = "realdoc-validation@v1"

_MONEY_TOKEN = re.compile(r"(?<![\d.,])\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?![\d,]|\.\d)")
_TOTAL_LINE = re.compile(r"\b(grand\s*total|total|amount\s*due|balance\s*due|pay\s*this\s*amount)\b", re.IGNORECASE)
_SUBTOTAL = re.compile(r"sub\s*-?\s*total", re.IGNORECASE)


_INVOICE_LABEL = re.compile(r"\binv(?:oice)?\b|\binv\.|invoice", re.IGNORECASE)


def _near_label(value: str, text: str, label: re.Pattern[str]) -> bool:
    """`value` printed on a line carrying the label, or on the line right below it."""
    lines = text.splitlines()
    want = _squash(value)
    for i, ln in enumerate(lines):
        if label.search(ln):
            window = ln + (" " + lines[i + 1] if i + 1 < len(lines) else "")
            if want in _squash(window):
                return True
    return False


@dataclass
class Validation:
    validator: str
    blocking: list[str] = field(default_factory=list)  # reasons that route the invoice to a human
    flags: list[str] = field(default_factory=list)  # reported, not blocking
    grounded: dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def amount_forms(value: float) -> list[str]:
    """Printed forms of an amount, most specific first."""
    forms = [f"{value:,.2f}", f"{value:.2f}"]
    if round(value) == value:
        forms += [f"{value:,.0f}", f"{value:.0f}"]
    return list(dict.fromkeys(forms))


_GAP_IN_NUMBER = re.compile(r"(?<=\d)[ \t]*([.,])[ \t]*(?=\d)")


def amount_on_page(value: float | None, text: str) -> bool:
    """True when a printed form of `value` occurs as a whole number in the text.
    Spaces and line breaks stay boundaries; only gaps OCR puts inside a number
    around a separator ("1, 533.54") are closed first."""
    if value is None:
        return False
    flat = _GAP_IN_NUMBER.sub(r"\1", text)
    for form in amount_forms(value):
        for m in re.finditer(re.escape(form), flat):
            before = flat[m.start() - 1] if m.start() > 0 else ""
            after = flat[m.end()] if m.end() < len(flat) else ""
            # not part of a longer number: no digit touching, no ",d"/".d" continuing it
            if before.isdigit() or after.isdigit() or re.match(r"[.,]\d", flat[m.end() : m.end() + 2]):
                continue
            if re.search(r"\d[.,]$", flat[max(0, m.start() - 2) : m.start()]):
                continue
            return True
    return False


def digits_on_page(digits: str, text: str) -> bool:
    return bool(digits) and re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", text) is not None


def _squash(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def money_tokens(line: str, cents_only: bool = False) -> list[float]:
    """Amounts printed on a line. `cents_only`: only amounts with two decimals, so
    SKUs, PO numbers and quantities are never mistaken for line amounts."""
    out = []
    for m in _MONEY_TOKEN.finditer(_GAP_IN_NUMBER.sub(r"\1", line)):  # "1, 533.54" is one amount
        if cents_only and (m.group(2) is None or len(m.group(2)) != 2):
            continue
        v = parse_money(m.group(0).replace("$", "").replace(" ", ""))
        if v is not None:
            out.append(v)
    return out


def candidate_totals(text: str) -> list[float]:
    """Distinct amounts printed on total-labelled lines (not SUBTOTAL), in page order."""
    seen: list[float] = []
    for line in text.splitlines():
        if _TOTAL_LINE.search(line) and not _SUBTOTAL.search(line):
            for v in money_tokens(line):
                if v not in seen and v > 0:
                    seen.append(v)
    return seen


def line_arithmetic(qty: int, price: float, text: str) -> str:
    """'corroborated' (a line prints qty x price next to the price), 'contradicted'
    (the price line prints other amounts, none equal to qty x price), or 'absent'."""
    product = round(qty * price, 2)
    contradicted = False
    for line in text.splitlines():
        if not amount_on_page(price, line):
            continue
        others = [v for v in money_tokens(line, cents_only=True) if abs(v - price) >= 0.005]
        if any(abs(v - product) < 0.005 for v in others):
            return "corroborated"
        if others:  # the price line prints other amounts, and none is qty x price
            contradicted = True
    if contradicted:  # a printed line amount that is not qty x price beats any match elsewhere
        return "contradicted"
    # qty 1 makes the product equal the price: finding it proves nothing
    if abs(product - price) >= 0.005 and amount_on_page(product, text):
        return "corroborated"
    return "absent"


def int_on_line(n: int, line: str) -> bool:
    """`n` printed as a standalone integer: not part of an amount ("$1,234.50"
    holds no 1), a decimal or a longer number."""
    return re.search(rf"(?<![\d.,$]){n}(?!\d|[.,]\d)", line) is not None


def qty_on_price_line(qty: int, price: float, text: str) -> bool | None:
    """True/False when the page has a line printing the unit price (is the quantity
    on such a line?); None when no line prints the price."""
    lines = [ln for ln in text.splitlines() if amount_on_page(price, ln)]
    if not lines:
        return None
    for ln in lines:
        flat = _GAP_IN_NUMBER.sub(r"\1", ln)
        for form in amount_forms(price):  # the price's own digits are not the quantity
            flat = flat.replace(form, " ")
        if int_on_line(qty, flat):
            return True
    return False


def validate_extract(raw: dict[str, Any] | None, canon: dict[str, Any], text: str) -> Validation:
    """invoice-extract-validation@v1: grounding of every fact, line arithmetic."""
    v = Validation(EXTRACT_VALIDATOR_ID)
    for key in ("invoice_id", "po_number", "vendor_id"):
        val = canon.get(key)
        v.grounded[key] = isinstance(val, str) and digits_on_page(val.split("-", 1)[-1], text)
    qty, price = canon.get("billed_qty"), canon.get("unit_price_billed")
    v.grounded["billed_qty"] = isinstance(qty, int) and digits_on_page(str(qty), text)
    v.grounded["unit_price_billed"] = isinstance(price, (int, float)) and amount_on_page(float(price), text)
    for key, ok in v.grounded.items():
        if not ok:
            v.blocking.append(f"not_on_page:{key}")
    if v.grounded["billed_qty"] and v.grounded["unit_price_billed"]:
        arith = line_arithmetic(int(qty), float(price), text)  # type: ignore[arg-type]
        if arith == "contradicted":
            v.blocking.append("line_amount_mismatch")
        else:
            v.flags.append(f"arithmetic_{arith}")
        # The quantity belongs to the line that prices it (a "1" from the freight line
        # next to the goods price is the failure this stops).
        if qty_on_price_line(int(qty), float(price), text) is False:  # type: ignore[arg-type]
            v.blocking.append("qty_not_on_price_line")
    return v


def validate_realdoc(raw: dict[str, Any] | None, canon: dict[str, Any], text: str) -> Validation:
    """realdoc-validation@v1: grounding of what was extracted (nulls are not checked),
    total ambiguity flags."""
    v = Validation(REALDOC_VALIDATOR_ID)
    flat = _squash(text)
    inv = canon.get("invoice_number")
    if isinstance(inv, str) and _squash(inv) and not _near_label(inv, text, _INVOICE_LABEL):
        v.flags.append("invoice_number_not_near_label")
    for key in ("invoice_number", "po_number"):
        val = canon.get(key)
        if val is not None:
            v.grounded[key] = isinstance(val, str) and bool(_squash(val)) and _squash(val) in flat
    total = canon.get("total_amount")
    if total is not None:
        v.grounded["total_amount"] = isinstance(total, (int, float)) and amount_on_page(float(total), text)
        cands = candidate_totals(text)
        if v.grounded["total_amount"] and cands and not any(abs(c - total) < 0.005 for c in cands):
            v.flags.append("total_not_on_total_line")
        if len(cands) > 1:
            v.flags.append("several_total_amounts")
    for key, ok in v.grounded.items():
        if not ok:
            v.blocking.append(f"not_on_page:{key}")
    return v


VALIDATORS = {EXTRACT_VALIDATOR_ID: validate_extract, REALDOC_VALIDATOR_ID: validate_realdoc}
