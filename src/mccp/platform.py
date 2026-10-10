"""Dependency container shared by the API and the pipeline worker."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from mccp.db.engine import make_engine
from mccp.governance.evidence import OpaClient
from mccp.registry.artifacts import Signer
from mccp.registry.mlflow import MlflowMirror, NullMirror, RegistryMirror
from mccp.serving.clients import FleetProfile, GatewayClient, NullGateway, VllmClient
from mccp.settings import Settings
from mccp.storage.objectstore import ObjectStore, open_store
from mccp.training.backends import TrainerBackend, make_backend


@dataclass
class Platform:
    settings: Settings
    engine: AsyncEngine
    store: ObjectStore
    signer: Signer
    fleet: FleetProfile
    vllm: VllmClient
    gateway: GatewayClient | NullGateway
    mirror: RegistryMirror
    opa: OpaClient
    backend: TrainerBackend

    @classmethod
    def from_settings(cls, s: Settings) -> "Platform":
        vllm_key = s.vllm_api_key.get_secret_value() if s.vllm_api_key else None
        gateway: GatewayClient | NullGateway
        if s.gateway_kind == "litellm":
            gateway = GatewayClient(
                s.litellm_base_url,
                s.litellm_master_key.get_secret_value() if s.litellm_master_key else None,
                s.litellm_vllm_api_base or s.vllm_base_url,
            )
        else:
            gateway = NullGateway()
        return cls(
            settings=s,
            engine=make_engine(s),
            store=open_store(s.object_store_url, s.s3_endpoint_url),
            signer=Signer(s.signing_mode, s.signing_private_key_path, s.signing_public_key_path),
            fleet=FleetProfile.load(s.fleet_profile_path),
            vllm=VllmClient(s.vllm_base_url, vllm_key, timeout=s.eval_request_timeout_s),
            gateway=gateway,
            mirror=MlflowMirror(s.mlflow_tracking_uri) if s.mlflow_tracking_uri else NullMirror(),
            opa=OpaClient(s.opa_url),
            backend=make_backend(
                s.trainer_backend,
                python=s.trainer_python,
                entry=s.trainer_entry,
                prefix=s.trainer_command_prefix,
                timeout_s=s.trainer_timeout_s,
            ),
        )

    async def close(self) -> None:
        await self.engine.dispose()
