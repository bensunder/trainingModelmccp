"""PII redaction to typed placeholders (`<EMAIL_1>`).

Two redactors share one interface:
* `PatternRedactor` – deterministic, dependency-free recognizers for the
  structured identifiers that dominate enterprise text (email, phone, US SSN,
  payment cards with Luhn check, IBAN with mod-97 check, IPv4).
* `PresidioRedactor` – Microsoft Presidio (NER + recognizers) for names,
  locations etc. Optional extra: `pip install mccp[presidio]` plus a spaCy model.

Placeholders are numbered per record and consistent within it (the same
email maps to the same placeholder), which preserves conversational
coherence. The original-to-placeholder mapping is never persisted; only
counts by entity type are reported.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not (15 <= len(s) <= 34):
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


@dataclass(frozen=True)
class _Rule:
    entity: str
    pattern: re.Pattern[str]
    check: object = None  # callable(str) -> bool


_RULES: tuple[_Rule, ...] = (
    _Rule("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    _Rule(
        "IBAN",
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
        _iban_ok,
    ),
    _Rule(
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"),
        lambda s: _luhn_ok(re.sub(r"\D", "", s)) and 13 <= len(re.sub(r"\D", "", s)) <= 19,
    ),
    _Rule("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    _Rule(
        "PHONE",
        re.compile(r"(?<![\w-])(?:\+?1[ .\-]?)?(?:\(\d{3}\)|\d{3})[ .\-]\d{3}[ .\-]\d{4}(?![\w-])"),
    ),
    _Rule(
        "IP_ADDRESS",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ),
)


@dataclass
class RedactionScope:
    """Per-record placeholder table so repeated values map consistently."""

    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    counters: Counter = field(default_factory=Counter)

    def placeholder(self, entity: str, value: str) -> str:
        key = (entity, value)
        if key not in self.mapping:
            self.counters[entity] += 1
            self.mapping[key] = f"<{entity}_{self.counters[entity]}>"
        return self.mapping[key]


class Redactor(Protocol):
    name: str

    def redact(self, text: str, scope: RedactionScope) -> str: ...


class PatternRedactor:
    name = "pattern-v1"

    def redact(self, text: str, scope: RedactionScope) -> str:
        if not text:
            return text
        for rule in _RULES:

            def repl(m: re.Match[str], rule: _Rule = rule) -> str:
                value = m.group(0)
                if rule.check is not None and not rule.check(value):  # type: ignore[operator]
                    return value
                return scope.placeholder(rule.entity, value)

            text = rule.pattern.sub(repl, text)
        return text


class PresidioRedactor:
    """Presidio NER redaction layered on top of the pattern rules."""

    name = "presidio"

    def __init__(self, entities: list[str] | None = None, language: str = "en") -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - optional extra
            raise RuntimeError("PresidioRedactor requires `pip install mccp[presidio]` and a spaCy model") from e
        self._analyzer = AnalyzerEngine()
        self._entities = entities or [
            "PERSON",
            "LOCATION",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
        ]
        self._language = language
        self._patterns = PatternRedactor()

    def redact(self, text: str, scope: RedactionScope) -> str:  # pragma: no cover - optional extra
        text = self._patterns.redact(text, scope)
        results = self._analyzer.analyze(text=text, entities=self._entities, language=self._language)
        for r in sorted(results, key=lambda r: r.start, reverse=True):
            value = text[r.start : r.end]
            if value.startswith("<") and value.endswith(">"):
                continue
            text = text[: r.start] + scope.placeholder(r.entity_type, value) + text[r.end :]
        return text


def make_redactor(kind: str) -> Redactor:
    if kind == "pattern":
        return PatternRedactor()
    if kind == "presidio":
        return PresidioRedactor()
    raise ValueError(f"unknown redactor {kind!r}")


def redact_value(value: object, redactor: Redactor, scope: RedactionScope) -> object:
    """Redact strings inside nested JSON values (tool-call arguments) while
    keeping structure and types intact."""
    if isinstance(value, str):
        return redactor.redact(value, scope)
    if isinstance(value, list):
        return [redact_value(v, redactor, scope) for v in value]
    if isinstance(value, dict):
        return {k: redact_value(v, redactor, scope) for k, v in value.items()}
    return value
