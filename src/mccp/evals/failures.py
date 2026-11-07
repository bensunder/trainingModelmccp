"""Failure classification: why each wrong field was wrong, and whether validation caught it.

For every scored answer that got a field wrong, compare three things: the label,
what the model wrote, and the exact text the model read. Each wrong field gets
one cause:

amounts
  decimal_dropped          same digits as the label, decimal point gone (153354 for 1,533.54)
  unparseable_copy         the model copied text the parser refuses (316:77)
  missing                  the model returned null / nothing
  wrong_amount_on_page     another amount that is printed on the page (net vs gross, tax, ...)
  invented_or_misread      the label is on the page, the answer is not
  label_not_on_page        the label's value is not in the text the model read (OCR lost it,
                           or the label disagrees with the page); the model cannot be blamed
ids
  missing / wrong_value_on_page / invented_or_misread / label_not_on_page  (same meaning)

and the row's outcome: `human_review` when validation (or invalid output) stopped
it, `auto_wrong` when it would have been decided wrongly without anyone noticing.
Pure functions over saved answers; no model calls.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from mccp.rules.validation import amount_on_page, digits_on_page

AMOUNT_FIELDS = {"unit_price_billed": "unit_price_raw", "total_amount": "total_amount_raw"}
_LINE_KINDS = [
    ("subtotal", r"sub\s*-?\s*total"),
    ("tax", r"\btax\b|\bvat\b|\bgst\b"),
    ("freight", r"freight|shipping|handling|postage"),
    ("net", r"\bnet\b"),
    ("gross", r"gross"),
    ("discount", r"discount|credit|\bcr\b|allowance"),
    ("paid", r"paid|payment|deposit"),
    ("balance_or_due", r"balance|amount\s*due|due"),
    ("total", r"total"),
    ("quantity_or_unit", r"\bqty\b|quantity|unit|each|@"),
]


def _squash(s: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).casefold())


def _cents_digits(v: float) -> str:
    return f"{v:.2f}".replace(".", "")


def line_kind(value: float, text: str) -> str:
    """What the line printing `value` is about (first matching keyword), or 'other'."""
    for line in text.splitlines():
        if amount_on_page(value, line):
            low = line.casefold()
            for kind, pat in _LINE_KINDS:
                if re.search(pat, low):
                    return kind
            return "other"
    return "not_found"


def classify_amount(expected: float, got: Any, raw_text: Any, text: str) -> str:
    if raw_text is not None and got is None:
        return "unparseable_copy"
    if got is None:
        return "missing"
    if not amount_on_page(expected, text):
        return "label_not_on_page"
    exp_digits = _cents_digits(expected)
    got_digits = re.sub(r"\D", "", str(raw_text if raw_text is not None else got))
    if got_digits.lstrip("0") in (exp_digits.lstrip("0"), exp_digits.rstrip("0").lstrip("0")) and "." not in str(
        raw_text if raw_text is not None else got
    ):
        return "decimal_dropped"
    if amount_on_page(float(got), text):
        return "wrong_amount_on_page"
    return "invented_or_misread"


def classify_id(expected: Any, got: Any, text: str, digits_only: bool) -> str:
    if got is None or got == "":
        return "missing"
    if digits_only:
        exp_on = digits_on_page(str(expected).split("-", 1)[-1], text)
        got_on = digits_on_page(str(got).split("-", 1)[-1], text)
    else:
        flat = _squash(text)
        exp_on = _squash(expected) in flat
        got_on = bool(_squash(got)) and _squash(got) in flat
    if not exp_on:
        return "label_not_on_page"
    return "wrong_value_on_page" if got_on else "invented_or_misread"


def _field_ok(suite: Any, e: Any, g: Any) -> bool:
    from mccp.examples.invcdip import normalized_equal

    return normalized_equal(e, g) if suite.field_match == "normalized" else g == e


def classify_rows(
    rows: list[dict[str, Any]],
    suite: Any,
    normalize: Any,
    texts: dict[str, str] | None = None,
    validator: Any = None,
) -> dict[str, Any]:
    """Re-score saved answers (`<report>.<set>.<target>.outputs.jsonl`) under the
    CURRENT parser, schema and validator (the model's answers do not change when
    that code does, so this is exact and needs no GPU), classify every wrong field,
    and simulate flag-based blocking policies.

    `texts` ({source_id: model input}) fills in rows saved before inputs were recorded."""
    import jsonschema

    texts = texts or {}
    causes: dict[str, Counter] = defaultdict(Counter)
    caught: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    outcomes: Counter = Counter()
    flag_hits: dict[str, Counter] = defaultdict(Counter)  # flag -> {correct, wrong} among auto-decided
    refused_shapes: Counter = Counter()  # what the parser refused, digits as 9
    passed: list[tuple[str, bool, frozenset[str]]] = []  # (source_id, correct, flags) of auto-decided rows
    missing_text = 0
    for r in rows:
        if r.get("kind") != "final":
            continue
        text = r.get("input") or texts.get(str(r.get("source_id"))) or ""
        if not text:
            missing_text += 1
        exp_raw = json.loads(r["expected"]) if isinstance(r.get("expected"), str) else (r.get("expected") or {})
        exp = normalize(exp_raw)[0] if normalize else exp_raw
        try:
            got_raw = json.loads(r.get("content") or "")
        except json.JSONDecodeError:
            got_raw = None
        if not isinstance(got_raw, dict) or r.get("error"):
            outcomes["human_review"] += 1
            causes["(whole answer)"]["invalid_json_or_error"] += 1
            caught["(whole answer)"]["invalid_json_or_error"] += 1
            continue
        got = normalize(got_raw)[0] if normalize else got_raw
        try:
            jsonschema.validate(got, suite.output_schema)
            valid = True
        except jsonschema.ValidationError:
            valid = False
        scored = [f for f in suite.compare_fields if not (exp.get(f) is None and suite.field_match == "normalized")]
        wrong = [f for f in scored if not (valid and _field_ok(suite, exp.get(f), got.get(f)))]
        correct = not wrong
        verdict = validator(got_raw, got, text).to_dict() if (validator is not None and valid and text) else None
        blocked = (not valid) or bool((verdict or {}).get("blocking"))
        outcome = "human_review" if blocked else ("auto_correct" if correct else "auto_wrong")
        outcomes[outcome] += 1
        flags = frozenset(f for f in (verdict or {}).get("flags", []) if not f.startswith("arithmetic_"))
        if not blocked:
            passed.append((str(r.get("source_id")), correct, flags))
            for f in flags:
                flag_hits[f]["on_correct" if correct else "on_wrong"] += 1
        for f in wrong:
            e, g = exp.get(f), got.get(f)
            if f in AMOUNT_FIELDS:
                cause = classify_amount(float(e), g, got_raw.get(AMOUNT_FIELDS[f]), text)
                if cause == "wrong_amount_on_page":
                    cause += f" ({line_kind(float(g), text)})"
                if cause == "unparseable_copy":
                    refused_shapes[re.sub(r"\d", "9", str(got_raw.get(AMOUNT_FIELDS[f])))[:40]] += 1
            elif f == "billed_qty":
                cause = (
                    "missing"
                    if g is None
                    else (
                        "label_not_on_page"
                        if not digits_on_page(str(e), text)
                        else ("wrong_value_on_page" if digits_on_page(str(g), text) else "invented_or_misread")
                    )
                )
            else:
                digits = f in ("invoice_id", "po_number", "vendor_id") and suite.field_match == "exact"
                cause = classify_id(e, g, text, digits_only=digits)
            causes[f][cause] += 1
            if outcome == "human_review":
                caught[f][cause] += 1
            if len(examples[(f, cause)]) < 3:
                examples[(f, cause)].append(
                    {
                        "source_id": r.get("source_id"),
                        "expected": e,
                        "got": got_raw.get(AMOUNT_FIELDS.get(f, f), g),
                        "outcome": outcome,
                        "blocking": (verdict or {}).get("blocking", []) or ([] if valid else ["schema"]),
                        "flags": sorted(flags),
                        "line": _line_with(e, text),
                        "answer": got_raw,
                    }
                )
    return {
        "rows": sum(outcomes.values()),
        "outcomes": dict(outcomes),
        "causes": {f: dict(c.most_common()) for f, c in causes.items()},
        "caught_by_validation": {f: dict(c) for f, c in caught.items()},
        "examples": {f"{f} | {c}": v for (f, c), v in examples.items()},
        "rows_without_input_text": missing_text,
        # Hits on correct answers = extra human reviews if the flag blocked; on wrong
        # answers = errors it would stop. Counted among auto-decided answers only.
        "flags": {f: dict(c) for f, c in flag_hits.items()},
        "refused_amount_shapes": dict(refused_shapes.most_common(15)),
        "policies": simulate_policies(passed, dict(outcomes)),
    }


def simulate_policies(passed: list[tuple[str, bool, frozenset[str]]], outcomes: dict[str, int]) -> list[dict[str, Any]]:
    """For every combination of flags (as one blocking rule: block when ANY flag in
    the combination is raised), on the union of flagged invoices (never summed):
    wrong invoices stopped, correct ones sent to a person, and the resulting rates."""
    from itertools import combinations

    n = sum(outcomes.values()) or 1
    base_wrong = sum(1 for _, ok, _ in passed if not ok)
    base_human = outcomes.get("human_review", 0)
    names = sorted({f for _, _, fl in passed for f in fl})
    out = [
        {
            "block_on": [],
            "stops_wrong": 0,
            "extra_reviews_of_correct": 0,
            "auto_correct": round((len(passed) - base_wrong) / n, 4),
            "human_review": round(base_human / n, 4),
            "auto_wrong": round(base_wrong / n, 4),
        }
    ]
    for k in range(1, len(names) + 1):
        for combo in combinations(names, k):
            hit = [(sid, ok) for sid, ok, fl in passed if fl & set(combo)]  # union of flagged invoices
            stops = sum(1 for _, ok in hit if not ok)
            extra = sum(1 for _, ok in hit if ok)
            out.append(
                {
                    "block_on": list(combo),
                    "stops_wrong": stops,
                    "extra_reviews_of_correct": extra,
                    "auto_correct": round((len(passed) - base_wrong - extra) / n, 4),
                    "human_review": round((base_human + len(hit)) / n, 4),
                    "auto_wrong": round((base_wrong - stops) / n, 4),
                }
            )
    return out


def _line_with(expected: Any, text: str) -> str | None:
    """The OCR line that prints the expected value, for the reviewer."""
    for line in text.splitlines():
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if amount_on_page(float(expected), line):
                return line.strip()[:100]
        elif expected is not None and _squash(expected) and _squash(expected) in _squash(line):
            return line.strip()[:100]
    return None
