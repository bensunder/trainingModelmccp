"""Serving-plane clients: the fleet profile, the private vLLM admin/inference
API, and the LiteLLM gateway that gives agents a stable logical model name."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArchitectureProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lora_modules: list[str] = Field(min_length=1, description="Leaf module names the pinned vLLM applies LoRA to")


class BaseServing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    served_name: str
    architecture: str


class FleetProfile(BaseModel):
    """What the serving fleet actually supports. Pinned per vLLM version; the
    publish guard checks every adapter against it (vLLM #38085: unsupported
    module names are skipped silently, so this must be explicit)."""

    model_config = ConfigDict(extra="forbid")
    vllm_version: str
    max_lora_rank: int = Field(ge=1)
    max_loras: int = Field(ge=1)
    base_models: dict[str, BaseServing]
    architectures: dict[str, ArchitectureProfile]

    @classmethod
    def load(cls, path: Path) -> "FleetProfile":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def base_for(self, model_id: str) -> BaseServing:
        try:
            return self.base_models[model_id]
        except KeyError as e:
            raise KeyError(f"base model {model_id} is not served by this fleet") from e


class ServingError(RuntimeError):
    pass


class VllmClient:
    """Private-network client. The runtime LoRA endpoints must never be
    exposed publicly (vLLM documents them as unsafe outside a trusted network)."""

    def __init__(
        self, base_url: str, api_key: str | None = None, client: httpx.AsyncClient | None = None, timeout: float = 120.0
    ) -> None:
        self.base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers)
        self._headers = headers

    async def models(self) -> list[dict[str, Any]]:
        r = await self._client.get(f"{self.base}/v1/models", headers=self._headers)
        r.raise_for_status()
        return r.json().get("data", [])

    async def has_model(self, name: str) -> bool:
        return any(m.get("id") == name for m in await self.models())

    async def load_lora(self, name: str, path: str) -> None:
        if await self.has_model(name):
            return
        r = await self._client.post(
            f"{self.base}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}, headers=self._headers
        )
        if r.status_code >= 400 and "already" not in r.text.lower():
            raise ServingError(f"vLLM load_lora_adapter({name}) failed: {r.status_code} {r.text[:300]}")
        if not await self.has_model(name):
            raise ServingError(f"vLLM accepted {name} but does not list it in /v1/models")

    async def unload_lora(self, name: str) -> None:
        r = await self._client.post(
            f"{self.base}/v1/unload_lora_adapter", json={"lora_name": name}, headers=self._headers
        )
        if r.status_code >= 400 and r.status_code != 404 and "not found" not in r.text.lower():
            raise ServingError(f"vLLM unload_lora_adapter({name}) failed: {r.status_code} {r.text[:300]}")

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(f"{self.base}/v1/chat/completions", json=payload, headers=self._headers)
        if r.status_code >= 400:
            raise ServingError(f"chat completion failed: {r.status_code} {r.text[:300]}")
        return r.json()

    async def chat_stream_ttft(self, payload: dict[str, Any]) -> tuple[float, float]:
        """Return (time-to-first-token, total latency) in seconds."""
        body = {**payload, "stream": True}
        start = time.perf_counter()
        ttft: float | None = None
        async with self._client.stream(
            "POST", f"{self.base}/v1/chat/completions", json=body, headers=self._headers
        ) as r:
            if r.status_code >= 400:
                raise ServingError(f"streaming chat failed: {r.status_code}")
            async for line in r.aiter_lines():
                if ttft is None and line.startswith("data:") and line.strip() != "data: [DONE]":
                    ttft = time.perf_counter() - start
        total = time.perf_counter() - start
        return (ttft if ttft is not None else total), total


class GatewayClient:
    """LiteLLM proxy model management. Each adapter version is a deployment
    under the logical model name; `litellm_params.weight` drives the canary
    split. Deployment ids are deterministic so calls are idempotent."""

    def __init__(
        self, base_url: str, master_key: str | None, vllm_api_base: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self.base = base_url.rstrip("/")
        self.vllm_api_base = vllm_api_base.rstrip("/") + "/v1"
        self._headers = {"Authorization": f"Bearer {master_key}"} if master_key else {}
        self._client = client or httpx.AsyncClient(timeout=30.0)

    @staticmethod
    def deployment_id(logical_name: str, serving_name: str) -> str:
        # Keyed by (model name, adapter version) so the same adapter can back
        # both the logical name and its canary probe name.
        return f"mccp::{logical_name}::{serving_name}"

    @staticmethod
    def probe_name(logical_name: str) -> str:
        """Gateway name that routes only to the canary adapter (smoke tests)."""
        return f"{logical_name}~probe"

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the gateway exactly as an agent would (OpenAI-compatible)."""
        r = await self._client.post(f"{self.base}/v1/chat/completions", json=payload, headers=self._headers, timeout=120.0)
        if r.status_code >= 400:
            raise ServingError(f"gateway chat failed: {r.status_code} {r.text[:300]}")
        return r.json()

    @staticmethod
    def access_group(tenant: str) -> str:
        return f"mccp-tenant-{tenant}"

    async def tenant_key(self, tenant: str, *, alias: str, duration: str | None = None) -> str:
        """Create a LiteLLM virtual key limited to the tenant's access group.
        Untested against a real LiteLLM; verify with /key/info after creation."""
        body: dict[str, Any] = {"models": [self.access_group(tenant)], "key_alias": alias, "metadata": {"mccp_tenant": tenant}}
        if duration:
            body["duration"] = duration
        r = await self._client.post(f"{self.base}/key/generate", json=body, headers=self._headers)
        if r.status_code >= 400:
            raise ServingError(f"LiteLLM /key/generate failed: {r.status_code} {r.text[:300]}")
        return r.json()["key"]

    async def _deployments(self) -> dict[str, dict[str, Any]]:
        r = await self._client.get(f"{self.base}/model/info", headers=self._headers)
        if r.status_code >= 400:
            raise ServingError(f"LiteLLM /model/info failed: {r.status_code} {r.text[:300]}")
        out = {}
        for d in r.json().get("data", []):
            mid = (d.get("model_info") or {}).get("id")
            if mid:
                out[mid] = d
        return out

    async def set_weights(
        self, logical_name: str, weights: dict[str, int], extra_body: dict[str, Any] | None = None
    ) -> None:
        """Make the deployments under `logical_name` exactly `weights`
        (serving_name -> weight). Weight 0 removes the deployment."""
        tenant = logical_name.split(".", 1)[0]
        existing = await self._deployments()
        wanted = {self.deployment_id(logical_name, n): (n, w) for n, w in weights.items() if w > 0}
        for dep_id, (serving_name, weight) in wanted.items():
            body = {
                "model_name": logical_name,
                "litellm_params": {
                    "model": f"hosted_vllm/{serving_name}",
                    "api_base": self.vllm_api_base,
                    "weight": weight,
                    **({"extra_body": extra_body} if extra_body else {}),
                },
                "model_info": {
                    "id": dep_id,
                    "mccp_serving_name": serving_name,
                    "mccp_tenant": tenant,
                    # LiteLLM model access group: a tenant-scoped virtual key
                    # (see tenant_key) can call only its own tenant's models.
                    "access_groups": [self.access_group(tenant)],
                },
            }
            if dep_id in existing:
                if (existing[dep_id].get("litellm_params") or {}).get("weight") == weight:
                    continue
                await self._delete(dep_id)
            r = await self._client.post(f"{self.base}/model/new", json=body, headers=self._headers)
            if r.status_code >= 400:
                raise ServingError(f"LiteLLM /model/new failed: {r.status_code} {r.text[:300]}")
        for dep_id, d in existing.items():
            if (
                dep_id.startswith("mccp::")
                and d.get("model_name") == logical_name
                and (d.get("model_info") or {}).get("mccp_tenant") in (tenant, None)
                and dep_id not in wanted
            ):
                await self._delete(dep_id)

    async def _delete(self, dep_id: str) -> None:
        r = await self._client.post(f"{self.base}/model/delete", json={"id": dep_id}, headers=self._headers)
        if r.status_code >= 400 and r.status_code != 404:
            raise ServingError(f"LiteLLM /model/delete failed: {r.status_code} {r.text[:300]}")


class NullGateway:
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ServingError("no gateway configured")

    async def set_weights(
        self, logical_name: str, weights: dict[str, int], extra_body: dict[str, Any] | None = None
    ) -> None:
        return None
