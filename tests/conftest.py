from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import asyncpg
import httpx
import pytest
import pytest_asyncio

from mccp.api.app import create_app
from mccp.api.auth import Authenticator, mint_dev_token
from mccp.db.migrate import migrate
from mccp.examples.invoices import OUTPUT_SCHEMA, generate
from mccp.platform import Platform
from mccp.registry.artifacts import generate_signing_keypair
from mccp.services.core import create_tenant
from mccp.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DSN = os.environ.get("MCCP_TEST_ADMIN_DSN", "postgresql://postgres@127.0.0.1:5432/postgres")
OPA_BIN = os.environ.get("MCCP_TEST_OPA_BIN") or shutil.which("opa") or "/home/claude/bin/opa"
SUBMIT, APPROVE, ADMIN = "Customizer.Submit", "Customizer.Approve", "Customizer.Admin"


def _pg_available() -> bool:
    async def probe() -> bool:
        try:
            conn = await asyncpg.connect(ADMIN_DSN, timeout=3)
            await conn.close()
            return True
        except Exception:
            return False

    return asyncio.run(probe())


PG = _pg_available()
needs_db = pytest.mark.skipif(not PG, reason=f"Postgres not reachable at {ADMIN_DSN}")
needs_opa = pytest.mark.skipif(not Path(OPA_BIN).exists(), reason="opa binary not found")


@pytest.fixture(scope="session")
def database() -> dict[str, str]:
    if not PG:
        pytest.skip("Postgres not reachable")
    name = f"mccp_test_{uuid.uuid4().hex[:8]}"

    async def setup() -> None:
        conn = await asyncpg.connect(ADMIN_DSN)
        try:
            await conn.execute((ROOT / "deploy/sql/bootstrap_roles.sql").read_text())
            await conn.execute(f'CREATE DATABASE "{name}" OWNER mccp_owner')
            await conn.execute(f'GRANT CONNECT ON DATABASE "{name}" TO mccp_app')
        finally:
            await conn.close()

    asyncio.run(setup())
    host = ADMIN_DSN.split("@", 1)[1].split("/", 1)[0]
    urls = {
        "name": name,
        "owner": f"postgresql+asyncpg://mccp_owner:mccp_owner@{host}/{name}",
        "app": f"postgresql+asyncpg://mccp_app:mccp_app@{host}/{name}",
        "admin": ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}",
    }
    asyncio.run(migrate(urls["owner"]))
    yield urls

    async def teardown() -> None:
        conn = await asyncpg.connect(ADMIN_DSN)
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(teardown())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def opa_url() -> str:
    if not Path(OPA_BIN).exists():
        pytest.skip("opa binary not found")
    port = _free_port()
    proc = subprocess.Popen(
        [OPA_BIN, "run", "--server", "--addr", f"127.0.0.1:{port}", "--log-level", "error", str(ROOT / "policy")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("OPA did not start")
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="session")
def fake_serving():
    from tests.fake_serving import FakeServing

    fake = FakeServing()
    fake.start()
    yield fake
    fake.stop()


@pytest.fixture(scope="session")
def keys(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("keys")
    generate_signing_keypair(d / "k.pem", d / "pub.pem")
    return d / "k.pem", d / "pub.pem"


@pytest.fixture
def settings(database, opa_url, fake_serving, keys, tmp_path) -> Settings:
    return Settings(
        database_url=database["app"],
        owner_database_url=database["owner"],
        auth_mode="dev",
        object_store_url=f"file://{tmp_path}/objects",
        work_dir=tmp_path / "work",
        signing_mode="key",
        signing_private_key_path=keys[0],
        signing_public_key_path=keys[1],
        trainer_backend="rehearsal",
        vllm_base_url=fake_serving.url,
        vllm_lora_root=tmp_path / "lora",
        fleet_profile_path=ROOT / "deploy/fleet.yaml",
        profiles_dir=ROOT / "profiles",
        gateway_kind="litellm",
        litellm_base_url=fake_serving.url,
        litellm_vllm_api_base="http://vllm:8000",
        mlflow_tracking_uri=fake_serving.url,
        opa_url=opa_url,
        canary_window_s=0,
        heartbeat_seconds=5,
        max_stage_attempts=2,
        db_pool_size=5,
    )


@pytest_asyncio.fixture
async def platform(settings, fake_serving):
    fake_serving.adapter_mode = "good"
    pf = Platform.from_settings(settings)
    yield pf
    await pf.close()


class Api:
    """Thin authenticated client against the ASGI app (no network)."""

    def __init__(self, app, secret: str, tenant: str) -> None:
        self.app, self.secret, self.tenant = app, secret, tenant

    def client(self, sub: str, *roles: str) -> httpx.AsyncClient:
        token = mint_dev_token(self.secret, subject=sub, tenant=self.tenant, roles=list(roles))
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://mccp",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )


@pytest_asyncio.fixture
async def api(platform):
    app = create_app(platform.settings, platform)
    app.state.pf = platform
    app.state.auth = Authenticator(platform)
    idp = f"tid-{uuid.uuid4().hex[:8]}"
    tenant_id = await create_tenant(platform, f"acme-{idp}", idp)
    a = Api(app, platform.settings.dev_jwt_secret.get_secret_value(), idp)
    a.tenant_id = tenant_id
    a.tenant_name = f"acme-{idp}"
    a.logical = f"acme-{idp}.finance-ops.invoice-triage"
    return a


RECIPE = {
    "ref": "qwen3-8b-qlora-sft@v1.0.0+L40S",
    "trainer_image": "ghcr.io/example/mccp-trainer@sha256:0123456789abcdef",
    "base_model": {
        "id": "Qwen/Qwen3-8B",
        "revision": "9c925d64d72725edaf899c6cb9c377fd0709d9c5",
        "license": "Apache-2.0",
        "architecture": "qwen3",
    },
    # Tiny dims keep the rehearsal adapter small in tests; real recipe in examples/.
    "dims": {
        "parameters": 8_190_000_000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
    },
    "finetuning_types": ["qlora"],
    "target_modules": "all-linear",
    "enable_thinking": False,
    "chat_template_sha256": "*",
    "engine": {"trl": "0.24.0", "peft": "0.17.1", "transformers": "4.57.1"},
    "soup": {"version": "0.72.4"},
    "defaults": {
        "n_epochs": 2,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 4096,
        "warmup_ratio": 0.03,
    },
    "allowed_overrides": {
        "n_epochs": {"min": 1, "max": 5},
        "learning_rate": {"min": 1e-5, "max": 5e-4},
        "lora_r": {"min": 4, "max": 64},
    },
    "max_lora_rank": 64,
    "gpu_hourly_usd": 0.86,
    "serving": {"min_vllm_version": "0.29.0"},
}


def suite_body(version: int = 1, dataset_version: int = 1) -> dict:
    return {
        "name": "invoice-triage-suite",
        "version": version,
        "golden": f"finance-ops/invoice-exceptions@v{dataset_version}:test",
        "output_schema": OUTPUT_SCHEMA,
        "compare_fields": ["decision", "reason_code"],
        "max_items": 60,
        "effect_probes": 3,
        "performance": {"requests": 4, "concurrency": 2},
    }


def job_body(**over) -> dict:
    body = {
        "project": "finance-ops",
        "recipe": "qwen3-8b-qlora-sft@v1.0.0+L40S",
        "method": {"type": "sft", "sft": {"hyperparameters": {"n_epochs": 2}}},
        "finetuning_type": "qlora",
        "training_dataset": "finance-ops/invoice-exceptions@v1:train",
        "validation_dataset": "finance-ops/invoice-exceptions@v1:val",
        "eval_suite": "invoice-triage-suite@v1",
        "output_adapter": "invoice-triage",
        "risk_tier": "medium",
        "intended_use": "Classify AP invoice exceptions and propose next action; human-in-loop",
        "seed": 42,
        "metadata": {"ticket": "AIGOV-112"},
    }
    body.update(over)
    return body


def invoices_ndjson(n: int = 240, seed: int = 7) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in generate(n, seed))


@pytest_asyncio.fixture
async def seeded(api):
    """Recipe, dataset v1 and suite v1 in a fresh tenant."""
    async with api.client("admin@acme", ADMIN) as c:
        r = await c.post("/v1/recipes", json=RECIPE)
        assert r.status_code in (201, 409), r.text
    async with api.client("alice@acme", SUBMIT) as c:
        r = await c.post("/v1/datasets/finance-ops/invoice-exceptions/versions", content=invoices_ndjson())
        assert r.status_code == 201, r.text
        r = await c.post("/v1/eval-suites/finance-ops", json=suite_body())
        assert r.status_code == 201, r.text
    return api
