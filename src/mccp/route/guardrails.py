"""Guardrails on what developers send to model providers.

Three detectors, each with its own action (off | flag | redact | block):
* secrets: cloud keys, tokens, private keys, credentials in connection strings
* pii: e-mail, phone, card and national-id patterns (mccp.datasets.pii rules)
* injection: instructions planted in content that try to override the system
  prompt (flag only: blocking text a developer pasted from a web page is wrong
  more often than right)

Findings record the rule and a count, never the matched text: the audit trail
must not become a second copy of the secret.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from mccp.datasets.pii import PatternRedactor, RedactionScope


class Action(StrEnum):
    OFF = "off"
    FLAG = "flag"
    REDACT = "redact"
    BLOCK = "block"


class GuardrailPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    secrets: Action = Action.REDACT
    pii: Action = Action.FLAG
    injection: Action = Action.FLAG


SECRET_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
        ),
    ),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_access_key", re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_live_key", re.compile(r"\b(?:sk|rk)_live_[0-9a-zA-Z]{24,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s:/@'\"]{1,64}:[^\s@/'\"]{3,128}@[^\s/'\"]+")),
    (
        "assigned_password",
        re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api_?key|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
    ),
]
INJECTION = re.compile(
    r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules)|"
    r"\byou are now (?:in )?(?:developer|dan|jailbreak)|reveal (?:your|the) system prompt|"
    r"\bexfiltrate\b|send (?:the |all )?(?:secrets|credentials|env(?:ironment)? variables) to"
)
_PII = PatternRedactor()


@dataclass
class Verdict:
    blocked: bool = False
    redacted: bool = False
    findings: Counter = field(default_factory=Counter)  # "secrets:github_token" -> count
    flags: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"blocked": self.blocked, "redacted": self.redacted, "findings": dict(self.findings)}


def _secret_sub(text: str, v: Verdict, redact: bool) -> str:
    for name, pat in SECRET_RULES:
        hits = len(pat.findall(text))
        if hits:
            v.findings[f"secrets:{name}"] += hits
            if redact:
                text = pat.sub(f"[REDACTED:{name}]", text)
    return text


def _pii_sub(text: str, v: Verdict, redact: bool) -> str:
    scope = RedactionScope()
    out = _PII.redact(text, scope)
    n = sum(scope.counters.values())
    if n:
        for ent, c in scope.counters.items():
            v.findings[f"pii:{ent.lower()}"] += c
        if redact:
            return out
    return text


_SCAN_KEYS = frozenset({"content", "text", "system", "messages", "function"})
_DEEP_KEYS = frozenset({"input", "arguments"})  # tool arguments: any key may hold a secret


def _walk(node: Any, fn, deep: bool = False) -> Any:
    """Apply fn to every string in message/system content and tool arguments.
    Structural fields (role, type, id, name) are never touched."""
    if isinstance(node, str):
        return fn(node)
    if isinstance(node, list):
        return [_walk(x, fn, deep) for x in node]
    if isinstance(node, dict):
        return {
            k: (_walk(v, fn, deep or k in _DEEP_KEYS) if (deep or k in _SCAN_KEYS or k in _DEEP_KEYS) else v)
            for k, v in node.items()
        }
    return node


def apply(body: dict[str, Any], policy: GuardrailPolicy) -> tuple[dict[str, Any], Verdict]:
    """Scan (and if the policy says so, redact) a request body. Returns the body to
    forward and the verdict; a blocked verdict means: do not forward."""
    v = Verdict()
    scoped = {k: body[k] for k in ("messages", "system") if k in body}

    def scan(text: str) -> str:
        if policy.secrets is not Action.OFF:
            text = _secret_sub(text, v, policy.secrets is Action.REDACT)
        if policy.pii is not Action.OFF:
            text = _pii_sub(text, v, policy.pii is Action.REDACT)
        if policy.injection is not Action.OFF and INJECTION.search(text):
            v.findings["injection:override_attempt"] += 1
        return text

    new_scoped = _walk(scoped, scan)
    kinds = {k.split(":")[0] for k in v.findings}
    for kind, action in (("secrets", policy.secrets), ("pii", policy.pii), ("injection", policy.injection)):
        if kind in kinds:
            if action is Action.BLOCK:
                v.blocked = True
            elif action is Action.REDACT:
                v.redacted = True
            elif action is Action.FLAG:
                v.flags.append(kind)
    if v.redacted and not v.blocked:
        return {**body, **new_scoped}, v
    return body, v
