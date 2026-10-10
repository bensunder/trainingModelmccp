"""Runtime configuration. Every value comes from the environment (prefix
`MCCP_`) so the same image runs in dev, CI and production. Secrets are read
from env/secret mounts only and never logged."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCCP_", env_file=".env", extra="ignore")

    # --- database -------------------------------------------------------
    database_url: str = "postgresql+asyncpg://mccp_app:mccp_app@127.0.0.1:5432/mccp"
    """Connection as the non-owner `mccp_app` role, so row-level security applies."""
    owner_database_url: str = "postgresql+asyncpg://mccp_owner:mccp_owner@127.0.0.1:5432/mccp"
    """Connection as the table owner; used only for migrations."""
    db_pool_size: int = 10
    db_statement_timeout_ms: int = 15_000

    # --- auth -----------------------------------------------------------
    auth_mode: Literal["oidc", "dev"] = "dev"
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_roles_claim: str = "roles"
    oidc_tenant_claim: str = "tid"
    dev_jwt_secret: SecretStr = SecretStr("dev-only-secret-change-me-0123456789abcdef")

    # --- storage --------------------------------------------------------
    object_store_url: str = "file:///var/lib/mccp/objects"
    """`file:///abs/path` or `s3://bucket/prefix` (S3-compatible, e.g. MinIO)."""
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    work_dir: Path = Path("/var/lib/mccp/work")
    max_dataset_upload_bytes: int = 256 * 1024 * 1024

    # --- signing --------------------------------------------------------
    signing_mode: Literal["key", "sigstore"] = "key"
    signing_private_key_path: Path | None = None
    signing_public_key_path: Path | None = None

    # --- training engine ------------------------------------------------
    trainer_backend: Literal["soup", "rehearsal"] = "soup"
    trainer_command_prefix: list[str] = Field(default_factory=list)
    """Prepended to the trainer command, e.g. a `docker run --gpus all ...` wrapper."""
    trainer_python: str = "python"
    trainer_entry: Path = Path("trainer/mccp_trainer_entry.py")
    trainer_timeout_s: int = 12 * 3600

    # --- serving / gateway / registry -----------------------------------
    vllm_base_url: str = "http://127.0.0.1:8000"
    """Private vLLM endpoint (admin + inference). Never exposed publicly (NFR-4)."""
    vllm_api_key: SecretStr | None = None
    vllm_lora_root: Path = Path("/var/lib/mccp/lora")
    """Directory vLLM reads adapters from (shared volume with the vLLM pod)."""
    vllm_lora_root_in_server: str | None = None
    """Same directory as seen from inside the vLLM container, if it differs."""
    fleet_profile_path: Path = Path("deploy/fleet.yaml")

    gateway_kind: Literal["litellm", "none"] = "litellm"
    litellm_base_url: str = "http://127.0.0.1:4000"
    litellm_master_key: SecretStr | None = None
    litellm_vllm_api_base: str | None = None
    """vLLM base URL as reachable from the LiteLLM container."""

    mlflow_tracking_uri: str | None = None
    opa_url: str = "http://127.0.0.1:8181"

    # --- evaluation -----------------------------------------------------
    eval_concurrency: int = 4
    eval_request_timeout_s: float = 120.0
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None

    # --- pipeline runner ------------------------------------------------
    worker_id: str = "worker-local"
    worker_lanes: list[Literal["control", "gpu", "serving"]] = ["control", "gpu", "serving"]
    """Lanes this worker serves. GPU host: gpu,serving. Control-plane host: control."""
    worker_slots_gpu: int = 1
    """Concurrent training jobs (one per GPU dedicated to training)."""
    worker_slots_serving: int = 2
    worker_slots_control: int = 4
    lease_seconds: int = 120
    heartbeat_seconds: int = 30
    max_stage_attempts: int = 5
    canary_window_s: int = 900
    canary_max_negative_feedback_rate: float = 0.2
    retire_after_s: int = 7 * 24 * 3600

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
