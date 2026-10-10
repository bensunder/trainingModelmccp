"""Deduplication, decontamination and leakage-free group splitting."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from datasketch import MinHash, MinHashLSH

from mccp.domain.hashing import canonical_json

_WORD = re.compile(r"\w+", re.UNICODE)


def record_text(messages: Sequence[dict[str, Any]]) -> str:
    """Full example text (used for dedup): everything but the system prompt."""
    parts = []
    for m in messages:
        if m.get("role") == "system":
            continue
        parts.append(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            parts.append(fn["name"])
            parts.append(canonical_json(fn["arguments"]).decode())
    return "\n".join(parts)


def input_text(messages: Sequence[dict[str, Any]]) -> str:
    """Example *inputs* (user turns), used for decontamination.

    Leakage means the same input appears in train and test. Structured
    outputs (JSON decisions) legitimately repeat across examples of a class,
    and their numeric fields collide by chance, so they are excluded."""
    return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def exact_key(messages: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(list(messages))).hexdigest()


def _minhash(toks: list[str], num_perm: int, shingle: int) -> MinHash:
    mh = MinHash(num_perm=num_perm, seed=1)
    if len(toks) < shingle:
        grams = [" ".join(toks)]
    else:
        grams = [" ".join(toks[i : i + shingle]) for i in range(len(toks) - shingle + 1)]
    for g in grams:
        mh.update(g.encode("utf-8"))
    return mh


def dedup(
    texts: Sequence[str], keys: Sequence[str], *, near_threshold: float = 0.8, num_perm: int = 128, shingle: int = 5
) -> tuple[list[int], dict[str, int]]:
    """Return indices to keep (first occurrence wins) and stats.

    Exact duplicates by canonical-message SHA-256, then near-duplicates by
    MinHash LSH (Jaccard >= `near_threshold` over word 5-gram shingles).
    """
    seen: set[str] = set()
    lsh = MinHashLSH(threshold=near_threshold, num_perm=num_perm)
    keep: list[int] = []
    exact = near = 0
    for i, (t, k) in enumerate(zip(texts, keys, strict=True)):
        if k in seen:
            exact += 1
            continue
        seen.add(k)
        mh = _minhash(tokens(t), num_perm, shingle)
        if lsh.query(mh):
            near += 1
            continue
        lsh.insert(str(i), mh)
        keep.append(i)
    return keep, {"exact_duplicates": exact, "near_duplicates": near, "kept": len(keep)}


def ngrams(toks: list[str], n: int) -> Iterable[tuple[str, ...]]:
    return (tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def contaminated(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    n: int = 13,
    *,
    boilerplate_doc_fraction: float = 0.01,
    boilerplate_min_docs: int = 3,
) -> list[bool]:
    """True for each train text that shares a *non-boilerplate* n-gram with
    the test set.

    N-grams that occur in many test documents are shared instructions or
    templates, not leaked examples; flagging them would discard the whole
    training set for templated enterprise data. An n-gram counts as
    boilerplate when it appears in more than
    max(boilerplate_min_docs, boilerplate_doc_fraction * len(test)) test docs.
    """
    doc_freq: Counter[tuple[str, ...]] = Counter()
    for t in test_texts:
        doc_freq.update(set(ngrams(tokens(t), n)))
    limit = max(boilerplate_min_docs, int(boilerplate_doc_fraction * len(test_texts)))
    bank = {g for g, c in doc_freq.items() if c <= limit}
    return [any(g in bank for g in ngrams(tokens(t), n)) for t in train_texts]


def group_split(
    group_ids: Sequence[str], *, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> list[str]:
    """Deterministic group split: all records of a source land in one split.

    Assignment hashes `seed:group` into [0,1), so adding records never moves
    an existing group to a different split.
    """
    if abs(sum(ratios) - 1.0) > 1e-9 or min(ratios) < 0:
        raise ValueError("ratios must be non-negative and sum to 1")
    tr, va, _ = ratios
    out = []
    for g in group_ids:
        h = int.from_bytes(hashlib.sha256(f"{seed}:{g}".encode()).digest()[:8], "big") / 2**64
        out.append("train" if h < tr else "val" if h < tr + va else "test")
    return out
