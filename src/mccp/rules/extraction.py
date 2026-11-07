"""Deterministic canonicalization of extracted ids (`invoice-extract-ids@v1`).

The model reads ids written in house formats (`P.O. 4500012345`, `po#4500012345`,
`vendor no. 100123`, `V 100123`) and sometimes copies the format instead of the
canonical form. Canonical form is a formatting rule, not a judgement, so code
applies it: the extraction schema, the lookups and the rules then see
`PO-4500012345` / `V-100123` / `INV-...`.

Safety, by construction and by test:
* digits are never changed, added, dropped or reordered; only the prefix and
  separators are rewritten;
* a value is rewritten only when it is exactly one run of digits with an
  optional, recognised prefix for *that* field (`PO`/`P.O.`/`purchase order` for
  po_number, `V`/`vendor`/`supplier` for vendor_id, `INV`/`invoice` for
  invoice_id) and/or `no.`, `number`, `id`, `#`, `:` or `-` (the field already
  says which id it is, so `No. 4500012345` in po_number is that PO). Another
  field's prefix, split digits or letters among the digits are left as written,
  so the schema rejects them and the invoice goes to a human;
* a canonical id that does not exist still finds nothing in the lookup and the
  policy holds the invoice: normalization can never turn a misread into an
  approval.
"""

from __future__ import annotations

import re
from typing import Any

NORMALIZER_ID = "invoice-extract-ids@v1"

_SUFFIX = r"(?:\s*(?:no\.?|number|num\.?|id|#)\s*)?"
_SEP = r"\s*[#:\-]?\s*"
_PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
    "po_number": (
        "PO",
        re.compile(rf"^\s*(?:p\s*\.?\s*o\s*\.?|purchase\s+order)?{_SUFFIX}{_SEP}(\d+)\s*$", re.IGNORECASE),
    ),
    "vendor_id": (
        "V",
        re.compile(rf"^\s*(?:v|vendor|supplier)?{_SUFFIX}{_SEP}(\d+)\s*$", re.IGNORECASE),
    ),
    "invoice_id": (
        "INV",
        re.compile(rf"^\s*(?:inv\.?|invoice)?{_SUFFIX}{_SEP}(\d+)\s*$", re.IGNORECASE),
    ),
}


def canonical_id(field: str, value: Any) -> str | None:
    """The canonical form of `value` for `field`, or None when it is not a
    recognised spelling of one id (left for the schema to reject)."""
    prefix, pattern = _PATTERNS[field]
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return None
    m = pattern.match(value)
    return f"{prefix}-{m.group(1)}" if m else None


def normalize_extraction(obj: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Return (a copy of `obj` with recognised id spellings made canonical, the
    changes made). Non-dict input and unknown fields pass through untouched."""
    if not isinstance(obj, dict):
        return obj, []
    out = dict(obj)
    changes: list[dict[str, Any]] = []
    for field in _PATTERNS:
        if field not in out:
            continue
        canon = canonical_id(field, out[field])
        if canon is not None and canon != out[field]:
            changes.append({"field": field, "from": out[field], "to": canon})
            out[field] = canon
    return out, changes


# --------------------------------------------------------------- amounts (v2) --
# The model copies an amount exactly as printed ("$1,533.54", "USD 1,234.50",
# "$857.74/ea"); code turns it into a number. Asking the model to strip symbols
# and separators itself made it drop the decimal point ("316.77" -> 31677) on
# real scans, so number formatting is code's job, like id formatting.
EXTRACT_NORMALIZER_ID = "invoice-extract@v2"  # ids (v1 rules) + unit_price_raw -> unit_price_billed
REALDOC_NORMALIZER_ID = "realdoc-amounts@v1"  # total_amount_raw -> total_amount

_CURRENCY = re.compile(r"^(?:usd|us\$|\$|us\s*\$)\s*|\s*(?:usd|dollars?)$", re.IGNORECASE)
_PER_UNIT = re.compile(r"\s*(?:/\s*(?:ea|each|unit|pc|pcs)\.?|each|per\s+unit|ea\.?)$", re.IGNORECASE)
_MONEY = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{0,2}))?$")


def parse_money(text: Any) -> float | None:
    """US-format amount as printed -> number, or None when it is not unambiguous.

    Accepts a currency symbol or code before/after, a per-unit suffix, thousands
    commas in the right places and up to two decimals ("$3,000." -> 3000.0).
    Rejects anything a human would have to interpret: "$35.805.60", "316:77",
    "1,66667", ".00", two numbers, letters inside the digits. A number (int/float)
    passes through, so an answer already numeric is still scored."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return round(float(text), 2) if text >= 0 else None
    if not isinstance(text, str):
        return None
    s = text.strip().strip("~*_=").strip()  # OCR marks around an amount; never digits or separators
    for _ in range(2):  # symbol and suffix in either order
        s = _PER_UNIT.sub("", _CURRENCY.sub("", s)).strip()
    m = _MONEY.match(s)
    if not m:
        return None
    cents = (m.group(2) or "").ljust(2, "0")
    return round(float(f"{m.group(1).replace(',', '')}.{cents}"), 2)


def _move_amount(obj: dict[str, Any], raw_key: str, key: str, changes: list[dict[str, Any]]) -> None:
    if raw_key not in obj:
        return
    raw = obj.pop(raw_key)
    if raw is None:
        obj[key] = None
        return
    value = parse_money(raw)
    if value is None:
        # Leave the canonical field absent: the schema rejects the answer, so
        # serving routes it to a human instead of guessing a number.
        changes.append({"field": raw_key, "from": raw, "to": None, "error": "unparseable amount"})
        return
    obj[key] = value
    changes.append({"field": raw_key, "from": raw, "to": value})


def normalize_extraction_v2(obj: Any) -> tuple[Any, list[dict[str, Any]]]:
    """invoice-extract@v2: canonical ids (as v1) and the printed unit price parsed."""
    out, changes = normalize_extraction(obj)
    if isinstance(out, dict):
        _move_amount(out, "unit_price_raw", "unit_price_billed", changes)
    return out, changes


def normalize_realdoc(obj: Any) -> tuple[Any, list[dict[str, Any]]]:
    """realdoc-amounts@v1: the printed total parsed; ids are left exactly as read."""
    if not isinstance(obj, dict):
        return obj, []
    out = dict(obj)
    changes: list[dict[str, Any]] = []
    _move_amount(out, "total_amount_raw", "total_amount", changes)
    return out, changes
