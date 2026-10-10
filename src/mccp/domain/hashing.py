"""Canonical JSON and SHA-256 helpers.

Canonical form: object keys sorted, no insignificant whitespace, UTF-8, no
NaN/Infinity. Used for idempotency request hashes, config hashes, evidence
bundle digests and the audit hash chain, so it must never change silently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS = "0" * 64


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    ).encode("utf-8")


def _default(obj: Any) -> Any:
    # pydantic models and dataclass-like objects
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not canonically serializable: {type(obj).__name__}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(value: Any) -> str:
    return "sha256:" + sha256_hex(canonical_json(value))


def chain_link(prev_event_sha: str, canonical_body: bytes) -> str:
    """event_sha = sha256(prev_event_sha || 0x00 || canonical(body))."""
    h = hashlib.sha256()
    h.update(prev_event_sha.encode("ascii"))
    h.update(b"\x00")
    h.update(canonical_body)
    return h.hexdigest()


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
