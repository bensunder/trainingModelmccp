"""L0 publish guard (spec risk: vLLM silently skipping adapter modules).

Static checks on the artifact, then a live load on the pinned vLLM:
1. Every LoRA module in the safetensors is in the fleet's allowlist for the
   base architecture (anything else would be skipped *silently*).
2. Rank <= fleet max_lora_rank; adapter base model == recipe base model.
3. vLLM loads it and lists it; a smoke completion succeeds.
4. Observable-effect probe: greedy outputs on N golden prompts are compared
   with the base. If every output is identical the adapter "loaded and did
   nothing", which is exactly the failure mode of #38085 (or a no-op
   rehearsal adapter). The gate blocks production promotion on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from mccp.serving.clients import FleetProfile, VllmClient


def adapter_modules(adapter_dir: Path) -> tuple[set[str], dict[str, Any]]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    modules: set[str] = set()
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="np") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
            if ".lora_" not in key:
                continue
            stem = key.split(".lora_", 1)[0]
            modules.add(stem.rsplit(".", 1)[-1])
    return modules, config


def static_guard(adapter_dir: Path, *, fleet: FleetProfile, base_model_id: str) -> dict[str, Any]:
    modules, config = adapter_modules(adapter_dir)
    base = fleet.base_for(base_model_id)
    allowed = set(fleet.architectures[base.architecture].lora_modules)
    skipped = sorted(modules - allowed)
    problems = []
    if not modules:
        problems.append("adapter contains no LoRA tensors")
    if skipped:
        problems.append(f"modules not applied by vLLM {fleet.vllm_version}: {skipped}")
    rank = int(config.get("r", 0))
    if rank > fleet.max_lora_rank:
        problems.append(f"rank {rank} exceeds fleet max_lora_rank {fleet.max_lora_rank}")
    declared_base = config.get("base_model_name_or_path")
    if declared_base and declared_base not in (base_model_id, "./model") and not str(declared_base).endswith("/model"):
        problems.append(f"adapter declares base {declared_base}, recipe base is {base_model_id}")
    return {
        "vllm_version": fleet.vllm_version,
        "architecture": base.architecture,
        "trained_modules": sorted(modules),
        "skipped_modules": skipped,
        "rank": rank,
        "fleet_max_lora_rank": fleet.max_lora_rank,
        "problems": problems,
    }


def _signature(resp: dict[str, Any]) -> str:
    msg = resp["choices"][0]["message"]
    calls = [(tc["function"]["name"], tc["function"].get("arguments")) for tc in (msg.get("tool_calls") or [])]
    return json.dumps([msg.get("content") or "", calls], sort_keys=True)


async def live_guard(
    vllm: VllmClient,
    *,
    serving_name: str,
    lora_path: str,
    base_served_name: str,
    probes: list[dict[str, Any]],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    await vllm.load_lora(serving_name, lora_path)
    identical = 0
    for p in probes:
        payload = {**p, **extra_body, "temperature": 0.0, "max_tokens": 256, "seed": 0}
        a = await vllm.chat({**payload, "model": serving_name})
        b = await vllm.chat({**payload, "model": base_served_name})
        identical += _signature(a) == _signature(b)
    return {
        "loaded": True,
        "effect_probes": len(probes),
        "identical_to_base": identical,
        "no_observable_effect": bool(probes) and identical == len(probes),
    }
