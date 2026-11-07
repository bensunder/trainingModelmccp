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
