"""`mccp` command line: operations (db, tenants, keys, api, worker) and an API
client for the end-to-end flow (datasets, suites, recipes, jobs, approvals)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import yaml

app = typer.Typer(help="Enterprise Model Customization Control Plane", no_args_is_help=True)
db_app = typer.Typer(help="Database operations", no_args_is_help=True)
tenant_app = typer.Typer(help="Tenant onboarding", no_args_is_help=True)
keys_app = typer.Typer(help="Adapter signing keys", no_args_is_help=True)
examples_app = typer.Typer(help="Demo data", no_args_is_help=True)
job_app = typer.Typer(help="Customization jobs (API client)", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(tenant_app, name="tenant")
app.add_typer(keys_app, name="keys")
app.add_typer(examples_app, name="examples")
app.add_typer(job_app, name="job")


def _settings():
    from mccp.settings import get_settings

    return get_settings()


# ------------------------------------------------------------------ ops --
@db_app.command("bootstrap")
def db_bootstrap(
    admin_dsn: Annotated[str, typer.Option(envvar="MCCP_ADMIN_DSN", help="Superuser DSN (postgresql://...)")],
    database: str = "mccp",
) -> None:
    """Create the owner/app roles and the database (idempotent)."""
    import asyncpg

    async def run() -> None:
        sql = Path(__file__).resolve().parents[2] / "deploy/sql/bootstrap_roles.sql"
        script = sql.read_text() if sql.exists() else _BOOTSTRAP_SQL
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(script)
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
            if not exists:
                await conn.execute(f'CREATE DATABASE "{database}" OWNER mccp_owner')
            await conn.execute(f'GRANT CONNECT ON DATABASE "{database}" TO mccp_app')
        finally:
            await conn.close()

    asyncio.run(run())
    typer.echo(f"roles and database `{database}` ready")


_BOOTSTRAP_SQL = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mccp_owner') THEN CREATE ROLE mccp_owner LOGIN PASSWORD 'mccp_owner'; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mccp_app') THEN CREATE ROLE mccp_app LOGIN PASSWORD 'mccp_app' NOBYPASSRLS; END IF;
END $$;
"""


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending schema migrations as the owner role."""
    from mccp.db.migrate import migrate

    applied = asyncio.run(migrate(_settings().owner_database_url))
    typer.echo("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))


@tenant_app.command("create")
def tenant_create(
    name: str,
    idp_tenant_id: Annotated[str, typer.Option(help="IdP tenant claim value (Entra tid)")],
    region: str = "us",
) -> None:
    from mccp.platform import Platform
    from mccp.services.core import create_tenant

    async def run() -> uuid.UUID:
        pf = Platform.from_settings(_settings())
        try:
            return await create_tenant(pf, name, idp_tenant_id, region)
        finally:
            await pf.close()

    typer.echo(str(asyncio.run(run())))


@app.command("storage-init")
def storage_init(wait: int = typer.Option(0, help="Seconds to keep retrying while the store starts")) -> None:
    """Create the object-store bucket/root if it does not exist (idempotent)."""
    from mccp.storage.objectstore import open_store

    s = _settings()
    store = open_store(s.object_store_url, s.s3_endpoint_url, s.s3_region)
    deadline = time.monotonic() + wait
    while True:
        try:
            asyncio.run(store.ensure_bucket())
            break
        except Exception as e:  # noqa: BLE001 - retried until the deadline, then surfaced
            if time.monotonic() >= deadline:
                raise
            typer.echo(f"object store not ready ({type(e).__name__}); retrying", err=True)
            time.sleep(3)
    typer.echo(f"object store ready: {store.url}")


@keys_app.command("generate")
def keys_generate(out_dir: Path = Path("secrets")) -> None:
    """Generate an ECDSA P-256 keypair for OpenSSF model signing (key mode)."""
    from mccp.registry.artifacts import generate_signing_keypair

    generate_signing_keypair(out_dir / "signing-key.pem", out_dir / "signing-pub.pem")
    typer.echo(f"wrote {out_dir}/signing-key.pem (0600) and {out_dir}/signing-pub.pem")


@app.command("dev-token")
def dev_token(
    sub: str,
    tenant: Annotated[str, typer.Option(help="IdP tenant id (tid claim)")],
    role: Annotated[list[str], typer.Option(help="Customizer.Submit | Customizer.Approve | Customizer.Admin")],
    ttl: int = 3600,
) -> None:
    """Mint a token for MCCP_AUTH_MODE=dev. Never enable dev auth in production."""
    from mccp.api.auth import mint_dev_token

    s = _settings()
    if s.auth_mode != "dev":
        raise typer.BadParameter("dev tokens are only valid when MCCP_AUTH_MODE=dev")
    typer.echo(mint_dev_token(s.dev_jwt_secret.get_secret_value(), subject=sub, tenant=tenant, roles=role, ttl_s=ttl))


@app.command("api")
def api(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    from mccp.api.app import configure_logging, create_app

    s = _settings()
    configure_logging(s.log_level)
    uvicorn.run(create_app(s), host=host, port=port, proxy_headers=True, log_config=None)


@app.command("worker")
def worker(once: bool = typer.Option(False, help="Process ready work then exit")) -> None:
    """Run the durable pipeline runner (one per GPU host)."""
    from mccp.api.app import configure_logging
    from mccp.pipeline.runner import PipelineRunner
    from mccp.platform import Platform

    s = _settings()
    configure_logging(s.log_level)

    async def run() -> None:
        pf = Platform.from_settings(s)
        runner = PipelineRunner(pf)
        try:
            if once:
                n = await runner.run_until_idle()
                typer.echo(f"processed {n} work item(s)")
                return
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await runner.run_forever(stop)
        finally:
            await pf.close()

    asyncio.run(run())


@app.command("gateway-key")
def gateway_key(
    tenant: Annotated[str, typer.Argument(help="Tenant slug (tenants.name)")],
    alias: Annotated[str, typer.Option(help="Key alias, e.g. the agent's name")],
    duration: Annotated[str | None, typer.Option(help="LiteLLM duration, e.g. 30d")] = None,
) -> None:
    """Create a LiteLLM virtual key that can call only this tenant's adapters."""
    from mccp.domain.refs import require_slug
    from mccp.serving.clients import GatewayClient

    s = _settings()
    require_slug("tenant", tenant)
    if not s.litellm_master_key:
        raise typer.BadParameter("MCCP_LITELLM_MASTER_KEY is required")
    gw = GatewayClient(
        s.litellm_base_url, s.litellm_master_key.get_secret_value(), s.litellm_vllm_api_base or s.vllm_base_url
    )
    typer.echo(asyncio.run(gw.tenant_key(tenant, alias=alias, duration=duration)))


@examples_app.command("invoices")
def examples_invoices(out: Path = Path("examples/out"), n: int = 3000, seed: int = 7) -> None:
    """Write the AP invoice-triage demo dataset and the agent's tool fixtures."""
    from mccp.examples.invoices import fixtures, generate

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "invoices.jsonl", "w", encoding="utf-8") as f:
        for r in generate(n, seed):
            f.write(json.dumps(r) + "\n")
    (out / "fixtures.json").write_text(json.dumps(fixtures(n, seed), indent=1))
    typer.echo(f"wrote {n} records to {out}/invoices.jsonl and agent fixtures to {out}/fixtures.json")


# ---------------------------------------------------------------- client --
def _client() -> httpx.Client:
    url = os.environ.get("MCCP_API_URL", "http://127.0.0.1:8080")
    token = os.environ.get("MCCP_TOKEN")
    if not token:
        raise typer.BadParameter("set MCCP_TOKEN (see `mccp dev-token`)")
    return httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=600)


def _show(r: httpx.Response) -> Any:
    try:
        body = r.json()
    except ValueError:
        body = r.text
    if r.status_code >= 400:
        typer.echo(json.dumps(body, indent=2), err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(body, indent=2))
    return body


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@app.command("dataset-upload")
def dataset_upload(project: str, name: str, file: Path, redactor: str = "pattern") -> None:
    with _client() as c:
        _show(
            c.post(
                f"/v1/datasets/{project}/{name}/versions",
                params={"redactor": redactor},
                content=file.read_bytes(),
                headers={"Content-Type": "application/x-ndjson"},
            )
        )


@app.command("suite-create")
def suite_create(project: str, file: Path) -> None:
    with _client() as c:
        _show(c.post(f"/v1/eval-suites/{project}", json=_load(file)))


@app.command("recipe-put")
def recipe_put(file: Path) -> None:
    with _client() as c:
        _show(c.post("/v1/recipes", json=_load(file)))


@job_app.command("submit")
def job_submit(file: Path, idempotency_key: str = typer.Option(None, help="Defaults to a hash of the file")) -> None:
    body = _load(file)
    key = idempotency_key or f"cli-{uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(body, sort_keys=True))}"
    with _client() as c:
        _show(c.post("/v1/customization/jobs", json=body, headers={"Idempotency-Key": key}))


@job_app.command("get")
def job_get(job_id: str) -> None:
    with _client() as c:
        _show(c.get(f"/v1/customization/jobs/{job_id}"))


@job_app.command("watch")
def job_watch(job_id: str, interval: float = 5.0) -> None:
    parked = {"pending_approval", "deployed", "rejected", "failed", "cancelled", "paused", "retired", "superseded"}
    last = 0
    with _client() as c:
        while True:
            evs = c.get(f"/v1/customization/jobs/{job_id}/events", params={"after": last}).json()["data"]
            for e in evs:
                typer.echo(f"{e['ts']}  {e['action']:<24} {json.dumps(e['data'])[:160]}")
                last = e["seq"]
            job = c.get(f"/v1/customization/jobs/{job_id}").json()
            if job["lifecycle_status"] in parked:
                typer.echo(f"-> {job['lifecycle_status']}")
                return
            time.sleep(interval)


@job_app.command("cancel")
def job_cancel(job_id: str) -> None:
    with _client() as c:
        _show(c.post(f"/v1/customization/jobs/{job_id}:cancel"))


@app.command("approvals")
def approvals_list() -> None:
    with _client() as c:
        _show(c.get("/v1/approvals"))


@app.command("approve")
def approve(job_id: str, justification: str, reject: bool = False) -> None:
    with _client() as c:
        _show(
            c.post(f"/v1/approvals/{job_id}:{'reject' if reject else 'approve'}", json={"justification": justification})
        )


@app.command("rollback")
def rollback(project: str, name: str, reason: str) -> None:
    with _client() as c:
        _show(c.post(f"/v1/deployments/{project}/{name}:rollback", json={"reason": reason}))


@app.command("audit-verify")
def audit_verify() -> None:
    with _client() as c:
        body = _show(c.get("/v1/audit:verify"))
    if not body.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    app()
