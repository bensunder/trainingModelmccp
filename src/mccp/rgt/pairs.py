"""Build RGT training rows from per-turn SFT rows.

Input rows are what the dataset pipeline exports with `split_turns`: one sample
per assistant turn, each ending with that turn. The *hinge* samples are the
ones whose final assistant turn is the decision (content, no tool calls); the
tool-lookup turns follow a fixed procedure and carry no strategic choice.

For every hinge sample the output keeps pair A unchanged (the chat template
renders an empty reasoning block for it, which is exactly how the model is
served) and adds pair B, whose final assistant content is prefixed with the
reasoning block holding the distilled strategy. Non-hinge rows pass through.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .contracts import Attempt, Distiller, Grader, Strategy


def is_hinge(sample: dict[str, Any]) -> bool:
    msgs = sample.get("messages") or []
    if not msgs:
        return False
    last = msgs[-1]
    return last.get("role") == "assistant" and not last.get("tool_calls") and bool((last.get("content") or "").strip())


def hinge_rows(rows: list[dict[str, Any]]) -> list[int]:
    return [i for i, r in enumerate(rows) if is_hinge(r)]


def final_turn_prompt(sample: dict[str, Any]) -> dict[str, Any]:
    """The request that produces the hinge turn: everything before it, plus tools."""
    out: dict[str, Any] = {"messages": copy.deepcopy(sample["messages"][:-1])}
    if sample.get("tools"):
        out["tools"] = sample["tools"]
    return out


def prompt_key(sample: dict[str, Any]) -> str:
    """Stable id of a hinge prompt (attempt cache key)."""
    p = final_turn_prompt(sample)
    blob = json.dumps(p, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_reasoning_block(text: str, reasoning: dict[str, Any]) -> str:
    """Reasoning block in the model's own format, placed before the answer.

    Qwen3's template, for example, splits assistant content on the close tag and
    renders the part before it as the turn's reasoning. Whether a template keeps
    it is verified by the trainer against the real tokenizer before training."""
    open_tag, close_tag = reasoning["open_tag"], reasoning["close_tag"]
    if open_tag in text or close_tag in text:
        raise ValueError("strategy text must not contain the reasoning tags")
    return f"{open_tag}\n{text.strip()}\n{close_tag}\n\n"


@dataclass
class RGTBuild:
    rows: list[dict[str, Any]]
    provenance: dict[str, Any]
    strategies: list[Strategy] = field(default_factory=list)


def build_rgt_rows(
    rows: list[dict[str, Any]],
    *,
    grader: Grader,
    distiller: Distiller,
    reasoning: dict[str, Any] | None,
    attempts: dict[str, Attempt] | None = None,
) -> RGTBuild:
    """Return A rows (unchanged) + one B row per hinge sample, with provenance."""
    if not reasoning or not reasoning.get("open_tag") or not reasoning.get("close_tag"):
        raise ValueError("RGT needs a model profile with a reasoning block (open_tag/close_tag)")
    out: list[dict[str, Any]] = []
    strategies: list[Strategy] = []
    rule_counts: Counter[str] = Counter()
    graded = passed = mistakes = missing_attempts = attempt_errors = 0
    strategy_chars = 0
    for row in rows:
        if not is_hinge(row):
            out.append(row)
            continue
        final = row["messages"][-1]["content"]
        if reasoning["open_tag"] in final:
            raise ValueError("a training answer already contains a reasoning block; RGT expects plain answers")
        attempt_text: str | None = None
        if attempts is not None:
            a = attempts.get(prompt_key(row))
            if a is None:
                missing_attempts += 1
            elif a.error:
                attempt_errors += 1
            else:
                attempt_text = a.content
        fb = grader.grade(row, attempt_text)
        if attempt_text is not None:
            graded += 1
            passed += int(fb.passed)
        strat = distiller.distill(row, fb)
        mistakes += int(strat.from_mistake)
        rule_counts.update(strat.rule_ids)
        strategy_chars += len(strat.text)
        strategies.append(strat)
        b = copy.deepcopy(row)
        b["messages"][-1]["content"] = render_reasoning_block(strat.text, reasoning) + final
        b["rgt"] = {"pair": "B", "strategy": strat.text, "rule_ids": list(strat.rule_ids)}
        a_row = dict(row)
        a_row["rgt"] = {"pair": "A"}
        out.extend([a_row, b])
    n_b = len(strategies)
    provenance = {
        "method": "rgt",
        "reference": "O'Neill, Partridge, Jayasekara. Rationale-Guided Training. Baseten Research, 2025-10-29",
        "grader": grader.id,
        "distiller": distiller.id,
        "input_rows": len(rows),
        "output_rows": len(out),
        "hinge_samples": n_b,
        "b_rows": n_b,
        "attempts": {
            "provided": attempts is not None,
            "graded": graded,
            "passed": passed,
            "pass_rate": round(passed / graded, 4) if graded else None,
            "missing": missing_attempts,
            "errors": attempt_errors,
        },
        "b_rows_with_mistake": mistakes,
        "rules": dict(sorted(rule_counts.items())),
        "strategy_chars_mean": round(strategy_chars / n_b, 1) if n_b else 0,
        "strategies_sha256": hashlib.sha256("\n".join(s.text for s in strategies).encode("utf-8")).hexdigest(),
    }
    return RGTBuild(rows=out, provenance=provenance, strategies=strategies)
