"""Token usage and cost, from JSON responses and from SSE streams.

Usage is read from what the provider reports, never estimated from text: a
request that ends without a usage report is recorded with `usage_reported=False`
so missing data is visible instead of silently billed as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Catalog, Wire


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reported: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else 0


def usage_from_anthropic(u: dict[str, Any] | None, into: Usage | None = None) -> Usage:
    out = into or Usage()
    if not isinstance(u, dict):
        return out
    if "input_tokens" in u:
        out.input_tokens = _int(u.get("input_tokens"))
    if "cache_read_input_tokens" in u:
        out.cache_read_tokens = _int(u.get("cache_read_input_tokens"))
    if "cache_creation_input_tokens" in u:
        out.cache_write_tokens = _int(u.get("cache_creation_input_tokens"))
    if "output_tokens" in u:  # message_delta reports the cumulative output count
        out.output_tokens = _int(u.get("output_tokens"))
    out.reported = True
    return out


def usage_from_openai(u: dict[str, Any] | None) -> Usage:
    out = Usage()
    if not isinstance(u, dict):
        return out
    prompt = _int(u.get("prompt_tokens"))
    details = u.get("prompt_tokens_details") or {}
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    cached = cached or _int(u.get("prompt_cache_hit_tokens"))  # DeepSeek reports cache hits here
    out.cache_read_tokens = min(cached, prompt)
    out.input_tokens = prompt - out.cache_read_tokens
    out.output_tokens = _int(u.get("completion_tokens"))
    out.reported = True
    return out


def usage_from_body(wire: Wire, body: dict[str, Any]) -> Usage:
    u = body.get("usage") if isinstance(body, dict) else None
    return usage_from_anthropic(u) if wire is Wire.ANTHROPIC else usage_from_openai(u)


@dataclass
class StreamMeter:
    """Feed it the raw SSE bytes as they pass through; read `.usage` at the end.
    Parses whole lines only, so chunk boundaries anywhere are safe."""

    wire: Wire
    usage: Usage = field(default_factory=Usage)
    first_byte: bool = False
    _buf: bytes = b""
    error: str | None = None

    def feed(self, chunk: bytes) -> None:
        self.first_byte = True
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.strip())

    def close(self) -> None:
        if self._buf.strip():
            self._line(self._buf.strip())
        self._buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(ev, dict):
            return
        if self.wire is Wire.ANTHROPIC:
            t = ev.get("type")
            if t == "message_start":
                usage_from_anthropic((ev.get("message") or {}).get("usage"), self.usage)
            elif t == "message_delta":
                usage_from_anthropic(ev.get("usage"), self.usage)
            elif t == "error":
                self.error = str((ev.get("error") or {}).get("type") or "error")
        elif isinstance(ev.get("usage"), dict):
            self.usage = usage_from_openai(ev["usage"])


@dataclass(frozen=True)
class Cost:
    usd: float
    baseline_usd: float  # same tokens on the catalog's baseline model

    @property
    def saved_usd(self) -> float:
        return max(0.0, self.baseline_usd - self.usd)


def cost(catalog: Catalog, model_id: str, u: Usage) -> Cost:
    m = catalog.model(model_id)
    b = catalog.model(catalog.baseline_model)
    args = (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens)
    return Cost(round(m.price.cost(*args), 8), round(b.price.cost(*args), 8))
