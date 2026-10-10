"""Evidence bundle, CycloneDX ML-BOM, model card, and the OPA decision call."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from mccp.domain.hashing import digest_of


def ml_bom(*, adapter: dict[str, Any], resolved: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """CycloneDX 1.6 ML-BOM describing the adapter and everything it was built from."""
    base = resolved["base_model"]
    adapter_ref = f"adapter:{adapter['serving_name']}"
    components: list[dict[str, Any]] = [
        {
            "type": "machine-learning-model",
            "bom-ref": f"model:{base['id']}@{base['revision']}",
            "name": base["id"],
            "version": base["revision"],
            "licenses": [{"license": {"id": base["license"]}}],
        },
        {
            "type": "container",
            "bom-ref": "trainer-image",
            "name": "mccp-trainer",
            "version": resolved["trainer_image"],
        },
        {
            "type": "library",
            "bom-ref": "lib:soup-cli",
            "name": "soup-cli",
            "version": resolved["soup"]["version"],
            "purl": f"pkg:pypi/soup-cli@{resolved['soup']['version']}",
        },
    ]
    for name, ver in sorted(resolved.get("engine", {}).items()):
        components.append(
            {
                "type": "library",
                "bom-ref": f"lib:{name}",
                "name": name,
                "version": ver,
                "purl": f"pkg:pypi/{name}@{ver}",
            }
        )
    for d in datasets:
        components.append(
            {
                "type": "data",
                "bom-ref": f"data:{d['ref']}",
                "name": d["ref"],
                "hashes": [{"alg": "SHA-256", "content": d["manifest_sha256"]}],
                "licenses": [{"license": {"name": lic}} for lic in d["licenses"]],
            }
        )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "component": {
                "type": "machine-learning-model",
                "bom-ref": adapter_ref,
                "name": adapter["serving_name"],
                "version": str(adapter["version"]),
                "hashes": [{"alg": "SHA-256", "content": adapter["artifact_sha256"]}],
                "modelCard": {
                    "modelParameters": {
                        "approach": {"type": "supervised"},
                        "task": "text-generation",
                        "architectureFamily": "qwen3",
                        "modelArchitecture": f"LoRA r={adapter['peft']['r']} on {base['id']}",
                    }
                },
            },
        },
        "components": components,
        "dependencies": [{"ref": adapter_ref, "dependsOn": [c["bom-ref"] for c in components]}],
    }


def model_card(
    *, job: dict[str, Any], adapter: dict[str, Any], resolved: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "name": adapter["serving_name"],
        "base_model": f"{resolved['base_model']['id']}@{resolved['base_model']['revision']}",
        "method": f"{resolved['method']} / {resolved['finetuning_type']} (engine: soup-cli {resolved['soup']['version']})",
        "intended_use": job["request"]["intended_use"],
        "risk_tier": job["risk_tier"],
        "training_data": [str(resolved["training_dataset"]), str(resolved["validation_dataset"])],
        "evaluation": {
            "suite": report.get("suite"),
            "task_score": summary.get("task_score"),
            "json_validity": summary.get("json_validity"),
            "tool_call_parse_rate": summary.get("tool_call_parse_rate"),
            "layers": summary.get("layers"),
        },
        "serving": {"thinking": resolved["enable_thinking"], "tools_rendering": resolved["tools_rendering"]},
        "limitations": [
            "Trained with thinking disabled; serve with chat_template_kwargs.enable_thinking=false.",
            "QLoRA adapter trained against an NF4 base and served on a bf16 base; scores are from the served artifact.",
            "Not a source of factual knowledge; document facts belong in retrieval.",
        ],
    }


def evidence_bundle(
    *,
    job: dict[str, Any],
    resolved: dict[str, Any],
    adapter: dict[str, Any],
    datasets: list[dict[str, Any]],
    report: dict[str, Any],
    guard: dict[str, Any],
    compute: dict[str, Any],
) -> dict[str, Any]:
    """Everything the gate decides on. Content-addressed; approvals and target
    are added per decision so one bundle backs staging and production."""
    eval_section = {k: report[k] for k in ("l1", "l2", "l3", "l4", "l5", "l6") if k in report}
    eval_section["publish_guard"] = guard
    eval_section["report_sha256"] = digest_of(report)
    return {
        "schema": "mccp.evidence/v1",
        "lineage": {
            "job_id": job["id"],
            "config_hash": job["config_hash"],
            "recipe": resolved["recipe"],
            "trainer_image": resolved["trainer_image"],
            "datasets": [d["ref"] for d in datasets],
        },
        "intended_use": job["request"]["intended_use"],
        "risk_tier": job["risk_tier"],
        "submitter": job["submitted_by"],
        "base_model": resolved["base_model"],
        "artifact": {
            "sha256": adapter["artifact_sha256"],
            "uri": adapter["artifact_uri"],
            "serving_name": adapter["serving_name"],
        },
        "adapter": {"peft": adapter["peft"], "version": adapter["version"]},
        "signatures": adapter["signatures"],
        "datasets": datasets,
        "eval": eval_section,
        "compute": compute,
        "ml_bom": ml_bom(adapter=adapter, resolved=resolved, datasets=datasets),
        "model_card": model_card(job=job, adapter=adapter, resolved=resolved, report=report),
    }


class PolicyError(RuntimeError):
    pass


class OpaClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=15.0)

    async def decide(self, bundle: dict[str, Any], *, target: str, approvals: list[dict[str, Any]]) -> dict[str, Any]:
        opa_input = {
            **{k: v for k, v in bundle.items() if k not in ("ml_bom", "model_card")},
            "target": target,
            "approvals": approvals,
        }
        r = await self._client.post(f"{self.base}/v1/data/promotion/decision", json={"input": opa_input})
        if r.status_code >= 400:
            raise PolicyError(f"OPA returned {r.status_code}: {r.text[:300]}")
        result = r.json().get("result")
        if not isinstance(result, dict) or "allow" not in result:
            raise PolicyError("OPA has no promotion policy loaded (empty result)")
        return {**result, "policy_input_sha256": digest_of(opa_input)}
