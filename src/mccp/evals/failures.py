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

from mccp.rules.validation import amount_on_page as _amount_on_page
from mccp.rules.validation import digits_on_page


def amount_on_page(value: float | None, text: str) -> bool:
    """Classification reads the page like realdoc-validation@v3: 48.1 is 48.10."""
    return _amount_on_page(value, text, one_decimal=True)


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


ID_FIELDS = {"invoice_id", "po_number", "vendor_id", "invoice_number"}
# A PII placeholder (mccp.datasets.pii: "<CREDIT_CARD_1>") is never a real label: the
# redactor rewrote the label along with the text. Sets built before the builder
# dropped such labels still carry them; scoring treats them as unlabelled.
REDACTION_PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*_\d+>")
VERDICTS = ("label_wrong", "ocr_lost", "model_wrong")


def wrong_kind(fields: frozenset[str] | set[str]) -> str:
    """amount_only | id_only | amount_and_id (| other... for e.g. a quantity): what a
    wrong answer got wrong. A wrong amount costs money directly; a wrong id mostly
    breaks duplicate detection and matching."""
    groups = {"amount" if f in AMOUNT_FIELDS else "id" if f in ID_FIELDS else "other" for f in fields}
    if groups == {"amount"}:
        return "amount_only"
    if groups == {"id"}:
        return "id_only"
    if groups == {"amount", "id"}:
        return "amount_and_id"
    return "+".join(sorted(groups)) or "none"


def _split(wrong_rows: list[frozenset[str]]) -> tuple[dict[str, int], dict[str, int]]:
    kinds: Counter = Counter(wrong_kind(w) for w in wrong_rows)
    fields: Counter = Counter(f for w in wrong_rows for f in w)
    return dict(sorted(kinds.items())), dict(sorted(fields.items()))


def _label_on_page(suite: Any, f: str, e: Any, text: str) -> bool:
    if f in AMOUNT_FIELDS:
        return amount_on_page(float(e), text)
    if f == "billed_qty" or (f in ID_FIELDS and suite.field_match == "exact"):
        return digits_on_page(str(e).split("-", 1)[-1] if f != "billed_qty" else str(e), text)
    return bool(_squash(e)) and _squash(e) in _squash(text)


def _rescore(
    r: dict[str, Any],
    suite: Any,
    normalize: Any,
    texts: dict[str, str],
    validator: Any,
    overrides: dict[tuple[str, str], dict[str, Any]] | None,
    keep_redacted: bool = False,
) -> dict[str, Any] | None:
    """One saved final answer judged by today's code. None for non-final rows.
    `keep_redacted` scores redaction-placeholder labels anyway (reproduces reports
    made before that was fixed)."""
    import jsonschema

    if r.get("kind") != "final":
        return None
    sid = str(r.get("source_id"))
    text = r.get("input") or texts.get(sid) or ""
    exp_raw = json.loads(r["expected"]) if isinstance(r.get("expected"), str) else dict(r.get("expected") or {})
    unscored: set[str] = set()
    redacted = (
        []
        if keep_redacted
        else [k for k, v in exp_raw.items() if isinstance(v, str) and REDACTION_PLACEHOLDER.search(v)]
    )
    for k in redacted:
        unscored.add({v: k for k, v in AMOUNT_FIELDS.items()}.get(k, k))
    for (osid, f), o in (overrides or {}).items():  # verified label errors only (load_adjudications)
        if osid != sid or o["verdict"] != "label_wrong":
            continue
        if o.get("corrected") is None:
            unscored.add(f)
        else:
            exp_raw[AMOUNT_FIELDS.get(f, f)] = o["corrected"]
    exp = normalize(exp_raw)[0] if normalize else exp_raw
    out: dict[str, Any] = {
        "sid": sid, "text": text, "exp_raw": exp_raw, "exp": exp, "invalid": False, "redacted_labels": redacted,
    }  # fmt: skip
    try:
        got_raw = json.loads(r.get("content") or "")
    except json.JSONDecodeError:
        got_raw = None
    if not isinstance(got_raw, dict) or r.get("error"):
        out["invalid"] = True
        return out
    got = normalize(got_raw)[0] if normalize else got_raw
    try:
        jsonschema.validate(got, suite.output_schema)
        valid = True
    except jsonschema.ValidationError:
        valid = False
    scored = [
        f
        for f in suite.compare_fields
        if f not in unscored and not (exp.get(f) is None and suite.field_match == "normalized")
    ]
    wrong = [f for f in scored if not (valid and _field_ok(suite, exp.get(f), got.get(f)))]
    verdict = validator(got_raw, got, text).to_dict() if (validator is not None and valid and text) else None
    out.update(
        got_raw=got_raw,
        got=got,
        valid=valid,
        scored=scored,
        wrong=wrong,
        verdict=verdict,
        blocked=(not valid) or bool((verdict or {}).get("blocking")),
    )
    return out


def classify_rows(
    rows: list[dict[str, Any]],
    suite: Any,
    normalize: Any,
    texts: dict[str, str] | None = None,
    validator: Any = None,
    policy: Any = None,
    overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
    keep_redacted: bool = False,
) -> dict[str, Any]:
    """Re-score saved answers (`<report>.<set>.<target>.outputs.jsonl`) under the
    CURRENT parser, schema and validator (the model's answers do not change when
    that code does, so this is exact and needs no GPU), classify every wrong field,
    and simulate flag-based blocking policies.

    `policy` (mccp.rules.review_policy.ReviewPolicy) decides the reported outcomes;
    the simulation always starts from validation alone, so its rows are comparable
    whatever policy is active. `overrides` ({(source_id, field): adjudication}) are
    a person's verdicts: a verified label error is re-scored against the corrected
    label, or not scored when none is given; every other verdict changes nothing.
    `texts` ({source_id: model input}) fills in rows saved before inputs were recorded."""
    texts = texts or {}
    causes: dict[str, Counter] = defaultdict(Counter)
    caught: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    outcomes: Counter = Counter()
    flag_hits: dict[str, Counter] = defaultdict(Counter)  # flag -> {correct, wrong} among auto-decided
    refused_shapes: Counter = Counter()  # what the parser refused, digits as 9
    hints: Counter = Counter()  # flags the active policy passed to the reviewer, on auto-decided rows
    policy_blocked: Counter = Counter()
    # (source_id, correct, flags, wrong fields) of every answer validation lets through
    passed: list[tuple[str, bool, frozenset[str], frozenset[str]]] = []
    auto_wrong_fields: list[frozenset[str]] = []
    auto_wrong_ids: list[tuple[str, dict[str, tuple[Any, Any]], frozenset[str]]] = []
    labels_by_field: dict[str, set[str]] = defaultdict(set)
    missing_text = 0
    redacted_labels = 0
    for r in rows:
        x = _rescore(r, suite, normalize, texts, validator, overrides, keep_redacted)
        if x is None:
            continue
        redacted_labels += len(x["redacted_labels"])
        text = x["text"]
        if not text:
            missing_text += 1
        if x["invalid"]:
            outcomes["human_review"] += 1
            causes["(whole answer)"]["invalid_json_or_error"] += 1
            caught["(whole answer)"]["invalid_json_or_error"] += 1
            continue
        exp, got, got_raw, wrong, verdict = x["exp"], x["got"], x["got_raw"], x["wrong"], x["verdict"]
        correct = not wrong
        flags = frozenset(f for f in (verdict or {}).get("flags", []) if not f.startswith("arithmetic_"))
        blocked = x["blocked"]
        if not blocked:
            passed.append((x["sid"], correct, flags, frozenset(wrong)))
            for f in flags:
                flag_hits[f]["on_correct" if correct else "on_wrong"] += 1
            if policy is not None:
                by, hint = policy.decide(verdict)
                if by:
                    blocked = True
                    policy_blocked.update(by)
                else:
                    hints.update(h for h in hint if not h.startswith("arithmetic_"))
        outcome = "human_review" if blocked else ("auto_correct" if correct else "auto_wrong")
        outcomes[outcome] += 1
        if outcome == "auto_wrong":
            auto_wrong_fields.append(frozenset(wrong))
            auto_wrong_ids.append(
                (x["sid"], {f: (exp.get(f), got.get(f)) for f in wrong if f in ID_FIELDS}, frozenset(wrong))
            )
        for f in ID_FIELDS & set(x["scored"]):
            labels_by_field[f].add(str(exp.get(f)))
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
                        "source_id": x["sid"],
                        "expected": e,
                        "got": got_raw.get(AMOUNT_FIELDS.get(f, f), g),
                        "outcome": outcome,
                        "blocking": (verdict or {}).get("blocking", []) or ([] if x["valid"] else ["schema"]),
                        "flags": sorted(flags),
                        "line": _line_with(e, text),
                        "answer": got_raw,
                    }
                )
    kinds, by_field = _split(auto_wrong_fields)
    return {
        "confusable_ids": confusable_report(auto_wrong_ids, labels_by_field),
        "rows": sum(outcomes.values()),
        "outcomes": dict(outcomes),
        "review_policy": getattr(policy, "id", None),
        "review_policy_blocked": dict(policy_blocked),
        "review_hints": dict(hints.most_common()),
        "auto_wrong_by_kind": kinds,
        "auto_wrong_by_field": by_field,
        "causes": {f: dict(c.most_common()) for f, c in causes.items()},
        "caught_by_validation": {f: dict(c) for f, c in caught.items()},
        "examples": {f"{f} | {c}": v for (f, c), v in examples.items()},
        "rows_without_input_text": missing_text,
        "redacted_labels_unscored": redacted_labels,
        "label_overrides_applied": sum(1 for o in (overrides or {}).values() if o["verdict"] == "label_wrong"),
        # Hits on correct answers = extra human reviews if the flag blocked; on wrong
        # answers = errors it would stop. Counted among answers validation lets through.
        "flags": {f: dict(c) for f, c in flag_hits.items()},
        "refused_amount_shapes": dict(refused_shapes.most_common(15)),
        "policies": simulate_policies(passed, outcomes_before_policy(outcomes, policy_blocked, passed)),
    }


def outcomes_before_policy(outcomes: Counter, policy_blocked: Counter, passed: list[Any]) -> dict[str, int]:
    """Outcome counts with validation alone (the simulation's starting point)."""
    n = sum(outcomes.values())
    wrong = sum(1 for p in passed if not p[1])
    return {"auto_correct": len(passed) - wrong, "auto_wrong": wrong, "human_review": n - len(passed)}


# OCR letter/digit confusions seen on the INV-CDIP scans (329S1/329Si, #36106/#36l06,
# 04T44636/04744636, G7866655/67866655, 010535/O10535, 80750B/807508), plus Z/2.
CONFUSABLES = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "b": "8", "g": "6", "t": "7", "z": "2"})
CONFUSABLE_RULE = "confusables-v1: o=0 i/l=1 s=5 b=8 g=6 t=7 z=2 (case, spaces, punctuation ignored)"


def fold_confusables(value: Any) -> str:
    """An id as a confusable-tolerant lookup key: ignore case, spaces and punctuation,
    and map letters OCR mistakes for digits onto the digit. Two ids with the same key
    cannot be told apart by such a lookup."""
    return _squash(value).translate(CONFUSABLES)


def confusable_report(
    auto_wrong: list[tuple[str, dict[str, tuple[Any, Any]], frozenset[str]]], labels: dict[str, set[str]]
) -> dict[str, Any]:
    """What a confusable-tolerant id lookup would do to the auto-wrong answers. A
    measurement, never a scoring change: an id counts as recoverable only when the
    model's id and the label have the same folded key. The risk side is reported
    too: labelled ids that become indistinguishable from another labelled id after
    folding (in production: two vendor-master or PO records one key would match, so
    the lookup must accept only a single match)."""
    by_field: dict[str, Counter] = defaultdict(Counter)
    id_only = id_only_fixed = mixed = mixed_ids_fixed = 0
    examples: list[dict[str, Any]] = []
    for sid, ids, wrong in auto_wrong:
        if not ids:
            continue
        fixed = {f for f, (e, g) in ids.items() if g not in (None, "") and fold_confusables(e) == fold_confusables(g)}
        for f in ids:
            by_field[f]["wrong"] += 1
            by_field[f]["recoverable"] += f in fixed
        if set(ids) == set(wrong):  # only ids are wrong
            id_only += 1
            if fixed == set(ids):
                id_only_fixed += 1
                if len(examples) < 8:
                    examples.append({"source_id": sid, **{f: {"label": e, "answer": g} for f, (e, g) in ids.items()}})
        else:
            mixed += 1
            mixed_ids_fixed += fixed == set(ids)
    collisions: dict[str, int] = {}
    for f, vals in labels.items():
        keys: dict[str, set[str]] = defaultdict(set)
        for v in vals:
            if _squash(v):
                keys[fold_confusables(v)].add(_squash(v))
        collisions[f] = sum(len(s) for s in keys.values() if len(s) > 1)
    return {
        "rule": CONFUSABLE_RULE,
        "id_only_auto_wrong": id_only,
        "id_only_recoverable": id_only_fixed,
        "amount_and_id_auto_wrong": mixed,
        "amount_and_id_ids_recoverable": mixed_ids_fixed,
        "by_field": {f: dict(c) for f, c in sorted(by_field.items())},
        "labelled_ids_colliding_after_folding": collisions,
        "examples": examples,
    }


def simulate_policies(passed: list[tuple[Any, ...]], outcomes: dict[str, int]) -> list[dict[str, Any]]:
    """For every combination of flags (as one blocking rule: block when ANY flag in
    the combination is raised), on the union of flagged invoices (never summed):
    wrong invoices stopped, correct ones sent to a person, the resulting rates, and
    what the auto-wrong that remains got wrong (`auto_wrong_by_kind` / `_by_field`).

    `passed`: (source_id, correct, flags[, wrong fields]) of every answer validation
    lets through; `outcomes`: counts with validation alone."""
    from itertools import combinations

    rows = [(p[0], p[1], p[2], p[3] if len(p) > 3 else frozenset()) for p in passed]
    n = sum(outcomes.values()) or 1
    base_wrong = sum(1 for _, ok, _, _ in rows if not ok)
    base_human = outcomes.get("human_review", 0)
    names = sorted({f for _, _, fl, _ in rows for f in fl})

    def row(combo: tuple[str, ...]) -> dict[str, Any]:
        hit = [(ok, w) for _, ok, fl, w in rows if fl & set(combo)]  # union of flagged invoices
        stops = sum(1 for ok, _ in hit if not ok)
        extra = sum(1 for ok, _ in hit if ok)
        left = [w for _, ok, fl, w in rows if not ok and not (fl & set(combo))]
        kinds, by_field = _split(left)
        return {
            "block_on": list(combo),
            "stops_wrong": stops,
            "extra_reviews_of_correct": extra,
            "auto_correct": round((len(rows) - base_wrong - extra) / n, 4),
            "human_review": round((base_human + len(hit)) / n, 4),
            "auto_wrong": round((base_wrong - stops) / n, 4),
            "auto_wrong_n": base_wrong - stops,
            "auto_wrong_by_kind": kinds,
            "auto_wrong_by_field": by_field,
        }

    return [row(())] + [row(c) for k in range(1, len(names) + 1) for c in combinations(names, k)]


# -- label audit (labels the model could not have read) ---------------------------


def _levenshtein(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def closest_on_page(label: str, text: str, max_tokens: int = 3) -> tuple[str | None, int | None]:
    """The run of 1..3 OCR tokens closest to the label (ignoring case, spaces and
    punctuation), and its edit distance; (None, None) when nothing is within reach."""
    want = _squash(label)
    if not want:
        return None, None
    toks = text.split()
    cap = max(2, len(want) // 2)
    best: tuple[int, str] | None = None
    for i in range(len(toks)):
        for k in range(1, max_tokens + 1):
            win = " ".join(toks[i : i + k])
            d = _levenshtein(want, _squash(win), cap)
            if d <= cap and (best is None or d < best[0] or (d == best[0] and len(win) < len(best[1]))):
                best = (d, win)
            if i + k >= len(toks):
                break
    return (best[1], best[0]) if best else (None, None)


def audit_hint(label: str, answer_on_page: bool, distance: int | None) -> str:
    """A first guess for the reviewer, never a verdict: `ocr_near_miss` (the OCR
    printed something within a character or two of the label: OCR damage, or a typo
    in the label), `answer_on_page_label_absent` (the model's value is printed and
    the label is not: possibly a wrong label), `absent`."""
    if distance is not None and distance <= max(1, min(2, len(_squash(label)) // 5)):
        return "ocr_near_miss"
    return "answer_on_page_label_absent" if answer_on_page else "absent"


def label_audit_rows(
    rows: list[dict[str, Any]],
    suite: Any,
    normalize: Any,
    texts: dict[str, str] | None = None,
    validator: Any = None,
    policy: Any = None,
) -> list[dict[str, Any]]:
    """Every scored field whose label is not in the text the model read, with what a
    person needs to judge it. Read-only; one entry per (source_id, field)."""
    texts = texts or {}
    out: list[dict[str, Any]] = []
    for r in rows:
        x = _rescore(r, suite, normalize, texts, validator, None)
        if x is None or x["invalid"] or not x["text"]:
            continue
        blocked = x["blocked"]
        if not blocked and policy is not None:
            blocked = bool(policy.decide(x["verdict"])[0])
        correct = not x["wrong"]
        outcome = "human_review" if blocked else ("auto_correct" if correct else "auto_wrong")
        for f in x["scored"]:
            e = x["exp"].get(f)
            if _label_on_page(suite, f, e, x["text"]):
                continue
            label_text = str(x["exp_raw"].get(AMOUNT_FIELDS.get(f, f)))
            g = x["got"].get(f)
            ans_on = g is not None and (
                (isinstance(g, (int, float)) and amount_on_page(float(g), x["text"]))
                if f in AMOUNT_FIELDS
                else _label_on_page(suite, f, g, x["text"])
            )
            snippet, dist = closest_on_page(label_text, x["text"])
            out.append(
                {
                    "source_id": x["sid"],
                    "field": f,
                    "outcome": outcome,
                    "field_correct": f not in x["wrong"],
                    "label": label_text,
                    "answer": x["got_raw"].get(AMOUNT_FIELDS.get(f, f)),
                    "answer_on_page": bool(ans_on),
                    "closest_on_page": snippet,
                    "distance": dist,
                    "hint": audit_hint(label_text, bool(ans_on), dist),
                }
            )
    return out


def load_adjudications(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """{(source_id, field): {"verdict", "corrected"}} from the filled-in audit sheet.
    Blank verdicts are skipped; anything else unexpected is an error, never a guess:
    an unknown verdict, a corrected label without `label_wrong`, a corrected total
    the amount parser refuses, or the same (source_id, field) twice."""
    from mccp.rules.extraction import parse_money

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for i, r in enumerate(rows, 2):  # line numbers as a spreadsheet shows them
        verdict = (r.get("verdict") or "").strip()
        corrected = (r.get("corrected_label") or "").strip() or None
        key = (str(r.get("source_id") or "").strip(), str(r.get("field") or "").strip())
        if not verdict:
            if corrected:
                raise ValueError(f"line {i} {key}: corrected_label given without a verdict")
            continue
        if verdict not in VERDICTS:
            raise ValueError(f"line {i} {key}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        if corrected and verdict != "label_wrong":
            raise ValueError(f"line {i} {key}: corrected_label only goes with label_wrong")
        if corrected and key[1] in AMOUNT_FIELDS and parse_money(corrected) is None:
            raise ValueError(f"line {i} {key}: corrected amount {corrected!r} is not readable; type it as printed")
        if key in out:
            raise ValueError(f"line {i} {key}: adjudicated twice")
        out[key] = {"verdict": verdict, "corrected": corrected}
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
