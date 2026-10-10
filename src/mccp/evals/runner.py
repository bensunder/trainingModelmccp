"""Evaluation orchestrator: runs every configured layer on the *served*
adapter (spec D6, "evaluate what you ship") and emits one machine-readable
report. A layer that cannot run is reported as `error`, never silently
omitted; the governance policy treats missing required evidence as a deny.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mccp.domain.hashing import digest_of
from mccp.serving.clients import VllmClient

from . import layers
from .suite import EvalSuite

log = logging.getLogger(__name__)


@dataclass
class EvalTargets:
    adapter: str  # version-specific serving name
    base: str  # served name of the base model
    champion: str | None = None  # current champion's serving name, if any


async def _layer(name: str, coro) -> dict[str, Any]:
    try:
        return {"status": "ok", **(await coro)}
    except Exception as e:  # noqa: BLE001 - recorded as evidence of failure
        log.warning("eval layer %s failed: %s", name, e)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}


async def run_suite(
    *,
    suite: EvalSuite,
    golden: list[dict[str, Any]],
    targets: EvalTargets,
    vllm: VllmClient,
    vllm_base_url: str,
    extra_body: dict[str, Any],
    concurrency: int,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> dict[str, Any]:
    items = layers.golden_items(golden, suite.max_items)
    if not items:
        raise ValueError("golden set produced no evaluation items")
    report: dict[str, Any] = {
        "suite": f"{suite.name}@v{suite.version}",
        "suite_sha256": digest_of(suite.model_dump(mode="json")),
        "golden_sha256": digest_of(golden),
        "targets": {"adapter": targets.adapter, "base": targets.base, "champion": targets.champion},
        "sampling": suite.sampling.model_dump(),
    }

    # L1 (always): adapter, base and champion on identical items.
    adapter_scores = await layers.run_items(vllm.chat, targets.adapter, items, suite, extra_body, concurrency)
    base_scores = await layers.run_items(vllm.chat, targets.base, items, suite, extra_body, concurrency)
    l1: dict[str, Any] = {
        "status": "ok",
        "adapter": layers.summarize_l1(adapter_scores),
        "base": layers.summarize_l1(base_scores),
        "delta_vs_base": layers.delta_vs(adapter_scores, base_scores),
    }
    if targets.champion:
        champ_scores = await layers.run_items(vllm.chat, targets.champion, items, suite, extra_body, concurrency)
        l1["champion"] = layers.summarize_l1(champ_scores)
        l1["delta_vs_champion"] = layers.delta_vs(adapter_scores, champ_scores)
    else:
        l1["delta_vs_champion"] = None
    report["l1"] = l1

    # L2 judge (optional).
    if suite.judge and judge_base_url:
        judge_client = VllmClient(judge_base_url, judge_api_key)
        report["l2"] = await _layer(
            "l2",
            layers.run_l2(
                judge_client.chat,
                suite,
                items,
                [s.get("content", "") for s in adapter_scores],
                [s.get("content", "") for s in base_scores],
            ),
        )
    else:
        report["l2"] = {"status": "skipped", "reason": "no judge configured"}

    # L3 general-capability regression (optional external harness).
    if suite.regression:
        report["l3"] = await _layer(
            "l3", layers.run_l3(suite.regression, vllm_base_url, {"adapter": targets.adapter, "base": targets.base})
        )
    else:
        report["l3"] = {"status": "skipped", "reason": "no regression harness configured"}

    # L4 safety (required by policy).
    guard = VllmClient(suite.safety.guard_base_url) if suite.safety.guard_base_url else vllm
    report["l4"] = await _layer(
        "l4",
        layers.run_l4(vllm.chat, guard.chat, suite, {"adapter": targets.adapter, "base": targets.base}, extra_body),
    )

    # L5 groundedness applies only to RAG-coupled adapters; not part of this vertical.
    report["l5"] = {"status": "skipped", "reason": "adapter is not RAG-coupled"}

    # L6 performance.
    report["l6"] = await _layer(
        "l6", layers.run_l6(vllm, suite, {"adapter": targets.adapter, "base": targets.base}, items, extra_body)
    )
    report["summary"] = {
        "task_score": l1["adapter"]["task_score"]["mean"],
        "json_validity": l1["adapter"]["json_validity"],
        "tool_call_parse_rate": l1["adapter"]["tool_call_parse_rate"],
        "layers": {k: report[k]["status"] for k in ("l1", "l2", "l3", "l4", "l5", "l6")},
    }
    return report


__all__ = ["EvalTargets", "run_suite"]
