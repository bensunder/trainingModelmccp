"""Dataset build pipeline (spec FR-1/FR-2).

Fixed stage order; every stage reports what it did, and every output file is
content-hashed into a manifest whose digest identifies the dataset version.

    parse+validate → redact PII → dedup → group split → decontaminate
    (train/val vs. test 13-grams) → thinking/format checks → export

Outputs:
* `train.jsonl`, `val.jsonl` – Soup `chatml` format (tools pre-rendered,
  one sample per assistant turn when `split_turns`).
* `test.jsonl` – the frozen golden set in native format (messages + tools),
  consumed by the evaluation suite through the vLLM OpenAI API.
* `card.json` – dataset card (counts, hashes, PII report, licenses, stats).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from mccp.domain.hashing import canonical_json, digest_of, sha256_hex

from .pii import RedactionScope, Redactor, redact_value
from .quality import contaminated, dedup, exact_key, group_split, input_text, record_text
from .render import contains_think_tags, prerender_tools, split_per_assistant_turn
from .schema import Record


@dataclass(frozen=True)
class BuildParams:
    seed: int = 1234
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    near_dup_threshold: float = 0.8
    decontam_ngram: int = 13
    split_turns: bool = True
    enable_thinking: bool = False
    max_reject_fraction: float = 0.05
    min_records: dict[str, int] = field(default_factory=lambda: {"train": 1, "val": 1, "test": 1})


@dataclass
class BuildResult:
    files: dict[str, bytes]
    card: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return digest_of(self.manifest).removeprefix("sha256:")


class DatasetBuildError(ValueError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(r) + b"\n" for r in rows)


def build_dataset(raw: bytes, redactor: Redactor, params: BuildParams = BuildParams()) -> BuildResult:
    # ---- 1. parse + validate ------------------------------------------------
    records: list[Record] = []
    rejects: list[dict[str, Any]] = []
    lines = raw.decode("utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(Record.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            msg = e.errors()[0]["msg"] if isinstance(e, ValidationError) else str(e)
            rejects.append({"line": lineno, "reason": msg[:300]})
    total = len(records) + len(rejects)
    if total == 0:
        raise DatasetBuildError("dataset is empty", {"rejects": rejects})
    if len(rejects) / total > params.max_reject_fraction:
        raise DatasetBuildError(
            f"{len(rejects)}/{total} records failed validation (limit {params.max_reject_fraction:.0%})",
            {"rejects": rejects[:50]},
        )

    # ---- 2. PII redaction ----------------------------------------------------
    pii_counts: Counter = Counter()
    redacted: list[dict[str, Any]] = []
    for rec in records:
        scope = RedactionScope()
        d = rec.to_json()
        for m in d["messages"]:
            m["content"] = redactor.redact(m.get("content", ""), scope)
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = redact_value(tc["function"]["arguments"], redactor, scope)
        redacted.append(d)
        pii_counts.update(scope.counters)

    # Residual scan: the deterministic recognizers must find nothing left.
    from .pii import PatternRedactor

    residual = 0
    probe = PatternRedactor()
    for d in redacted:
        s = RedactionScope()
        for m in d["messages"]:
            probe.redact(m.get("content", ""), s)
            for tc in m.get("tool_calls") or []:
                redact_value(tc["function"]["arguments"], probe, s)
        residual += sum(s.counters.values())

    # ---- 3. dedup ------------------------------------------------------------
    texts = [record_text(d["messages"]) for d in redacted]
    keep, dedup_stats = dedup(
        texts, [exact_key(d["messages"]) for d in redacted], near_threshold=params.near_dup_threshold
    )
    redacted = [redacted[i] for i in keep]
    texts = [texts[i] for i in keep]

    # ---- 4. group split --------------------------------------------------------
    assign = group_split([d["meta"]["source_id"] for d in redacted], seed=params.seed, ratios=params.ratios)
    by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, s in enumerate(assign):
        by_split[s].append(i)

    # ---- 5. decontaminate train/val against the frozen test set ----------------
    inputs = [input_text(d["messages"]) for d in redacted]
    test_texts = [inputs[i] for i in by_split["test"]]
    decontam: dict[str, int] = {}
    for s in ("train", "val"):
        flags = contaminated([inputs[i] for i in by_split[s]], test_texts, n=params.decontam_ngram)
        decontam[s] = sum(flags)
        by_split[s] = [i for i, bad in zip(by_split[s], flags, strict=True) if not bad]

    # ---- 6. format checks ---------------------------------------------------------
    if not params.enable_thinking:
        thinking = [i for i, d in enumerate(redacted) if contains_think_tags(d["messages"])]
        if thinking:
            raise DatasetBuildError(
                f"{len(thinking)} records contain <think> blocks but the recipe trains with enable_thinking=false",
                {"records": thinking[:50]},
            )
    for s, need in params.min_records.items():
        if len(by_split[s]) < need:
            raise DatasetBuildError(
                f"split `{s}` has {len(by_split[s])} records, need >= {need}",
                {"split_counts": {k: len(v) for k, v in by_split.items()}},
            )

    # ---- 7. export -------------------------------------------------------------------
    def soup_rows(indices: list[int]) -> list[dict[str, Any]]:
        rows = []
        for i in indices:
            d = redacted[i]
            msgs = prerender_tools(d["messages"], d.get("tools"))
            samples = split_per_assistant_turn(msgs) if params.split_turns else [msgs]
            rows.extend({"messages": s} for s in samples)
        return rows

    train_rows, val_rows = soup_rows(by_split["train"]), soup_rows(by_split["val"])
    test_rows = [{k: v for k, v in redacted[i].items() if k in ("messages", "tools", "meta")} for i in by_split["test"]]
    files = {"train.jsonl": _jsonl(train_rows), "val.jsonl": _jsonl(val_rows), "test.jsonl": _jsonl(test_rows)}

    licenses = sorted({d["meta"].get("license", "unknown") for d in redacted})
    manifest = {
        "files": {name: {"sha256": sha256_hex(b), "bytes": len(b)} for name, b in sorted(files.items())},
        "params": {
            "seed": params.seed,
            "ratios": list(params.ratios),
            "near_dup_threshold": params.near_dup_threshold,
            "decontam_ngram": params.decontam_ngram,
            "split_turns": params.split_turns,
            "enable_thinking": params.enable_thinking,
            "redactor": redactor.name,
            "format": "soup-chatml/qwen3-hermes-prerendered",
        },
        "raw_sha256": sha256_hex(raw),
    }
    card = {
        "records": {"input": total, "rejected": len(rejects), "after_dedup": len(redacted)},
        "split": {
            "method": "group_by_meta.source_id",
            "seed": params.seed,
            "ratios": list(params.ratios),
            "records": {k: len(v) for k, v in by_split.items()},
            "samples": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        },
        "dedup": dedup_stats,
        "decontamination": {"ngram": params.decontam_ngram, "dropped": decontam},
        "pii": {
            "redactor": redactor.name,
            "placeholders_by_type": dict(sorted(pii_counts.items())),
            "residual_matches": residual,
            "passed": residual == 0,
        },
        "licenses": licenses,
        "tools_present": any(d.get("tools") for d in redacted),
        "rejects": rejects[:50],
        "manifest": manifest,
    }
    return BuildResult(files=files, card=card, manifest=manifest)
