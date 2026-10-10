"""MLflow mirror of the adapter registry (spec D7).

The control plane's Postgres tables are the transactional source of truth
for governance state (they commit together with the audit chain). Every
registration and alias move is mirrored to MLflow 3.x so ML teams see the
same versions, aliases (@candidate/@challenger/@champion) and lineage tags
in their usual tool. Mirroring is idempotent and retried by the pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class RegistryMirror(Protocol):
    async def register_version(self, model_name: str, source: str, tags: dict[str, str], description: str) -> str: ...
    async def set_alias(self, model_name: str, alias: str, version: str) -> None: ...
    async def delete_alias(self, model_name: str, alias: str) -> None: ...


class NullMirror:
    async def register_version(self, model_name: str, source: str, tags: dict[str, str], description: str) -> str:
        return ""

    async def set_alias(self, model_name: str, alias: str, version: str) -> None:
        return None

    async def delete_alias(self, model_name: str, alias: str) -> None:
        return None


class MlflowError(RuntimeError):
    pass


class MlflowMirror:
    def __init__(self, tracking_uri: str, client: httpx.AsyncClient | None = None, token: str | None = None) -> None:
        self.base = tracking_uri.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.AsyncClient(timeout=30.0, headers=headers)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(f"{self.base}/api/2.0/mlflow/{path}", json=body)
        if r.status_code >= 400:
            raise MlflowError(f"MLflow {path} failed: {r.status_code} {r.text[:300]}", r.status_code, r.text)
        return r.json() if r.content else {}

    async def _ensure_registered_model(self, name: str) -> None:
        r = await self._client.post(f"{self.base}/api/2.0/mlflow/registered-models/create", json={"name": name})
        if r.status_code >= 400 and "RESOURCE_ALREADY_EXISTS" not in r.text:
            raise MlflowError(f"create registered model failed: {r.status_code} {r.text[:300]}")

    async def register_version(self, model_name: str, source: str, tags: dict[str, str], description: str) -> str:
        await self._ensure_registered_model(model_name)
        # Idempotency: reuse a version that already points at this source.
        r = await self._client.get(
            f"{self.base}/api/2.0/mlflow/model-versions/search",
            params={"filter": f"name='{model_name}'", "max_results": 1000},
        )
        if r.status_code < 400:
            for mv in r.json().get("model_versions", []):
                if mv.get("source") == source:
                    return str(mv["version"])
        body = {
            "name": model_name,
            "source": source,
            "description": description,
            "tags": [{"key": k, "value": v[:5000]} for k, v in sorted(tags.items())],
        }
        out = await self._post("model-versions/create", body)
        return str(out["model_version"]["version"])

    async def set_alias(self, model_name: str, alias: str, version: str) -> None:
        await self._post("registered-models/alias", {"name": model_name, "alias": alias, "version": version})

    async def delete_alias(self, model_name: str, alias: str) -> None:
        r = await self._client.request(
            "DELETE", f"{self.base}/api/2.0/mlflow/registered-models/alias", json={"name": model_name, "alias": alias}
        )
        if r.status_code >= 400 and r.status_code != 404:
            raise MlflowError(f"delete alias failed: {r.status_code} {r.text[:300]}")
