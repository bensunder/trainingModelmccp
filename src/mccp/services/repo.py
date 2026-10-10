"""Data access for jobs, queue, adapters, deployments and approvals.

Every function takes an open tenant-scoped connection (see db.engine.tenant_tx)
so RLS applies, and every state change appends its audit event in the same
transaction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from mccp.audit.chain import append_event
from mccp.db.engine import as_obj, jsonb
from mccp.domain.errors import Conflict, NotFound
from mccp.domain.status import AdapterStatus, LifecycleStatus, ensure_transition


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: uuid.UUID
    roles: frozenset[str]

    SUBMIT = "Customizer.Submit"
    APPROVE = "Customizer.Approve"
    ADMIN = "Customizer.Admin"

    def has(self, role: str) -> bool:
        return role in self.roles or self.ADMIN in self.roles


SYSTEM_ACTOR = "system:pipeline"

_JOB_COLS = """id, tenant_id, project, idempotency_key, request_hash, request, resolved, config_hash,
lifecycle_status, risk_tier, output_adapter, adapter_version_id, submitted_by, trained_tokens,
gpu_seconds, cost_usd, flop_estimate, error, stage_state, cancel_requested, pause_requested,
created_at, updated_at, finished_at"""


def _row(r: Any) -> dict[str, Any]:
    d = dict(r._mapping)
    for k in ("request", "resolved", "error", "stage_state", "card", "body", "report", "decision", "payload", "data"):
        if k in d and d[k] is not None:
            d[k] = as_obj(d[k])
    return d


# ------------------------------------------------------------------ jobs --
async def get_job(conn: AsyncConnection, job_id: str, *, for_update: bool = False) -> dict[str, Any]:
    q = f"SELECT {_JOB_COLS} FROM jobs WHERE id = :id" + (" FOR UPDATE" if for_update else "")
    r = (await conn.execute(text(q), {"id": job_id})).first()
    if r is None:
        raise NotFound(f"job {job_id} not found")
    return _row(r)


async def find_job_by_key(conn: AsyncConnection, key: str) -> dict[str, Any] | None:
    r = (await conn.execute(text(f"SELECT {_JOB_COLS} FROM jobs WHERE idempotency_key = :k"), {"k": key})).first()
    return _row(r) if r else None


async def list_jobs(
    conn: AsyncConnection, *, lifecycle: list[str] | None, after: str | None, limit: int, project: str | None
) -> list[dict[str, Any]]:
    clauses, params = [], {"limit": limit}
    if lifecycle:
        clauses.append("lifecycle_status = ANY(:lc)")
        params["lc"] = lifecycle
    if after:
        clauses.append("id < :after")
        params["after"] = after
    if project:
        clauses.append("project = :project")
        params["project"] = project
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = (
        await conn.execute(text(f"SELECT {_JOB_COLS} FROM jobs {where} ORDER BY id DESC LIMIT :limit"), params)
    ).all()
    return [_row(r) for r in rows]


async def insert_job(conn: AsyncConnection, job: dict[str, Any]) -> bool:
    """Insert unless the tenant already used this idempotency key."""
    r = await conn.execute(
        text(
            """INSERT INTO jobs (id, tenant_id, project, idempotency_key, request_hash, request, resolved,
                                 config_hash, lifecycle_status, risk_tier, output_adapter, submitted_by)
               VALUES (:id, CAST(:tenant AS uuid), :project, :key, :rhash, CAST(:request AS jsonb),
                       CAST(:resolved AS jsonb), :chash, 'created', :risk, :out, :sub)
               ON CONFLICT (tenant_id, idempotency_key) DO NOTHING
               RETURNING id"""
        ),
        {
            "id": job["id"],
            "tenant": str(job["tenant_id"]),
            "project": job["project"],
            "key": job["idempotency_key"],
            "rhash": job["request_hash"],
            "request": jsonb(job["request"]),
            "resolved": jsonb(job["resolved"]),
            "chash": job["config_hash"],
            "risk": job["risk_tier"],
            "out": job["output_adapter"],
            "sub": job["submitted_by"],
        },
    )
    return r.first() is not None


_UPDATABLE = {
    "adapter_version_id",
    "trained_tokens",
    "gpu_seconds",
    "cost_usd",
    "flop_estimate",
    "error",
    "stage_state",
    "cancel_requested",
    "pause_requested",
}
_JSON_FIELDS = {"error", "stage_state"}


async def update_job(conn: AsyncConnection, job_id: str, **fields: Any) -> None:
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"not updatable: {bad}")
    if not fields:
        return
    sets, params = [], {"id": job_id}
    for k, v in fields.items():
        if k in _JSON_FIELDS:
            sets.append(f"{k} = CAST(:{k} AS jsonb)")
            params[k] = jsonb(v) if v is not None else None
        else:
            sets.append(f"{k} = :{k}")
            params[k] = v
    await conn.execute(text(f"UPDATE jobs SET {', '.join(sets)}, updated_at = now() WHERE id = :id"), params)


async def transition(
    conn: AsyncConnection,
    job: dict[str, Any],
    to: LifecycleStatus,
    *,
    actor: str,
    data: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Compare-and-set transition guarded by the domain table, with audit."""
    src = LifecycleStatus(job["lifecycle_status"])
    ensure_transition(src, to)
    await update_job(conn, job["id"], **fields)
    terminal = to in (
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.REJECTED,
        LifecycleStatus.RETIRED,
    )
    r = await conn.execute(
        text(
            """UPDATE jobs SET lifecycle_status = :to, updated_at = now(),
                      finished_at = CASE WHEN :terminal THEN now() ELSE finished_at END
                WHERE id = :id AND lifecycle_status = :from"""
        ),
        {"to": to.value, "id": job["id"], "from": src.value, "terminal": terminal},
    )
    if r.rowcount != 1:
        raise Conflict(f"job {job['id']} changed concurrently (expected {src})")
    await append_event(
        conn,
        job["tenant_id"],
        actor=actor,
        action="job.transition",
        resource_type="job",
        resource_id=job["id"],
        data={"from": src.value, "to": to.value, **(data or {})},
    )
    job = {**job, **fields, "lifecycle_status": to.value}
    return job


# ------------------------------------------------------------------ queue --
async def enqueue(conn: AsyncConnection, job: dict[str, Any], *, delay_s: float = 0) -> None:
    await conn.execute(
        text(
            """INSERT INTO job_queue (job_id, tenant_id, run_after)
               VALUES (:id, CAST(:t AS uuid), now() + make_interval(secs => :d))
               ON CONFLICT (job_id) DO UPDATE SET run_after = EXCLUDED.run_after,
                     lease_owner = NULL, lease_expires_at = NULL"""
        ),
        {"id": job["id"], "t": str(job["tenant_id"]), "d": float(delay_s)},
    )


async def dequeue(conn: AsyncConnection, job_id: str) -> None:
    await conn.execute(text("DELETE FROM job_queue WHERE job_id = :id"), {"id": job_id})


async def queue_row(conn: AsyncConnection, job_id: str) -> dict[str, Any] | None:
    r = (await conn.execute(text("SELECT * FROM job_queue WHERE job_id = :id"), {"id": job_id})).first()
    return dict(r._mapping) if r else None


async def release(
    conn: AsyncConnection, job_id: str, owner: str, *, delay_s: float, error: str | None, reset_attempts: bool
) -> None:
    await conn.execute(
        text(
            """UPDATE job_queue SET lease_owner = NULL, lease_expires_at = NULL,
                      run_after = now() + make_interval(secs => :d), last_error = :err,
                      attempts = CASE WHEN :reset THEN 0 ELSE attempts END
                WHERE job_id = :id AND lease_owner = :owner"""
        ),
        {"id": job_id, "owner": owner, "d": float(delay_s), "err": error, "reset": reset_attempts},
    )


async def heartbeat(conn: AsyncConnection, job_id: str, owner: str, lease_seconds: int) -> bool:
    r = await conn.execute(
        text(
            """UPDATE job_queue SET lease_expires_at = now() + make_interval(secs => :s)
                WHERE job_id = :id AND lease_owner = :owner"""
        ),
        {"id": job_id, "owner": owner, "s": lease_seconds},
    )
    return r.rowcount == 1


# ------------------------------------------------------------ metrics etc --
async def record_metrics(conn: AsyncConnection, job: dict[str, Any], step: int, metrics: dict[str, Any]) -> None:
    await conn.execute(
        text(
            """INSERT INTO job_metrics (tenant_id, job_id, step, metrics)
               VALUES (CAST(:t AS uuid), :id, :step, CAST(:m AS jsonb))
               ON CONFLICT (job_id, step) DO NOTHING"""
        ),
        {"t": str(job["tenant_id"]), "id": job["id"], "step": step, "m": jsonb(metrics)},
    )


async def record_checkpoint(conn: AsyncConnection, job: dict[str, Any], step: int, path: str) -> None:
    await conn.execute(
        text(
            """INSERT INTO job_checkpoints (tenant_id, job_id, step, path)
               VALUES (CAST(:t AS uuid), :id, :step, :p) ON CONFLICT (job_id, step) DO NOTHING"""
        ),
        {"t": str(job["tenant_id"]), "id": job["id"], "step": step, "p": path},
    )


async def job_events(conn: AsyncConnection, job_id: str, after_seq: int, limit: int) -> list[dict[str, Any]]:
    rows = (
        await conn.execute(
            text(
                """SELECT seq, ts, actor, action, data FROM events
                    WHERE resource_type = 'job' AND resource_id = :id AND seq > :after
                    ORDER BY seq LIMIT :limit"""
            ),
            {"id": job_id, "after": after_seq, "limit": limit},
        )
    ).all()
    return [_row(r) for r in rows]


async def job_metrics(conn: AsyncConnection, job_id: str) -> list[dict[str, Any]]:
    rows = (
        await conn.execute(
            text("SELECT step, metrics, ts FROM job_metrics WHERE job_id = :id ORDER BY step"), {"id": job_id}
        )
    ).all()
    return [{**dict(r._mapping), "metrics": as_obj(r.metrics)} for r in rows]


async def job_checkpoints(conn: AsyncConnection, job_id: str) -> list[dict[str, Any]]:
    rows = (
        await conn.execute(
            text("SELECT step, path, ts FROM job_checkpoints WHERE job_id = :id ORDER BY step"), {"id": job_id}
        )
    ).all()
    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------------- recipes --
async def get_recipe(conn: AsyncConnection, ref: str) -> dict[str, Any]:
    r = (
        await conn.execute(
            text("SELECT ref, body, body_sha256, created_by, created_at FROM recipes WHERE ref = :r"), {"r": ref}
        )
    ).first()
    if r is None:
        raise NotFound(f"recipe {ref} not found")
    return _row(r)


# --------------------------------------------------------------- datasets --
async def ensure_project(conn: AsyncConnection, tenant_id: uuid.UUID, project: str) -> None:
    await conn.execute(
        text("INSERT INTO projects (tenant_id, name) VALUES (CAST(:t AS uuid), :p) ON CONFLICT DO NOTHING"),
        {"t": str(tenant_id), "p": project},
    )


async def get_dataset_version(conn: AsyncConnection, project: str, name: str, version: int) -> dict[str, Any]:
    r = (
        await conn.execute(
            text(
                """SELECT tenant_id, project, name, version, ref, manifest_sha256, artifact_prefix, card, created_by, created_at
                     FROM dataset_versions WHERE project = :p AND name = :n AND version = :v"""
            ),
            {"p": project, "n": name, "v": version},
        )
    ).first()
    if r is None:
        raise NotFound(f"dataset {project}/{name}@v{version} not found")
    return _row(r)


async def next_dataset_version(conn: AsyncConnection, project: str, name: str) -> int:
    await conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": f"dataset:{project}/{name}"}
    )
    v = (
        await conn.execute(
            text("SELECT COALESCE(MAX(version), 0) FROM dataset_versions WHERE project = :p AND name = :n"),
            {"p": project, "n": name},
        )
    ).scalar_one()
    return int(v) + 1


async def get_suite(conn: AsyncConnection, project: str, name: str, version: int) -> dict[str, Any]:
    r = (
        await conn.execute(
            text(
                "SELECT project, name, version, ref, body, body_sha256 FROM eval_suites WHERE project = :p AND name = :n AND version = :v"
            ),
            {"p": project, "n": name, "v": version},
        )
    ).first()
    if r is None:
        raise NotFound(f"eval suite {name}@v{version} not found in project {project}")
    return _row(r)


# --------------------------------------------------------------- adapters --
_ADAPTER_COLS = """id, tenant_id, project, name, version, serving_name, artifact_uri, artifact_sha256,
signature_uri, card, status, job_id, created_at, updated_at"""


async def get_adapter_version(conn: AsyncConnection, version_id: str) -> dict[str, Any]:
    r = (
        await conn.execute(text(f"SELECT {_ADAPTER_COLS} FROM adapter_versions WHERE id = :id"), {"id": version_id})
    ).first()
    if r is None:
        raise NotFound(f"adapter version {version_id} not found")
    return _row(r)


async def find_adapter_version(conn: AsyncConnection, project: str, name: str, version: int) -> dict[str, Any]:
    r = (
        await conn.execute(
            text(f"SELECT {_ADAPTER_COLS} FROM adapter_versions WHERE project = :p AND name = :n AND version = :v"),
            {"p": project, "n": name, "v": version},
        )
    ).first()
    if r is None:
        raise NotFound(f"adapter {project}/{name} v{version} not found")
    return _row(r)


async def list_adapter_versions(conn: AsyncConnection, project: str | None) -> list[dict[str, Any]]:
    q = (
        f"SELECT {_ADAPTER_COLS} FROM adapter_versions"
        + (" WHERE project = :p" if project else "")
        + " ORDER BY project, name, version DESC"
    )
    return [_row(r) for r in (await conn.execute(text(q), {"p": project} if project else {})).all()]


async def set_adapter_status(
    conn: AsyncConnection, version: dict[str, Any], status: AdapterStatus, *, actor: str
) -> None:
    await conn.execute(
        text("UPDATE adapter_versions SET status = :s, updated_at = now() WHERE id = :id"),
        {"s": status.value, "id": version["id"]},
    )
    await append_event(
        conn,
        version["tenant_id"],
        actor=actor,
        action="adapter.status",
        resource_type="adapter_version",
        resource_id=version["id"],
        data={"from": version["status"], "to": status.value},
    )


async def set_alias(conn: AsyncConnection, version: dict[str, Any], alias: str, *, actor: str) -> None:
    await conn.execute(
        text(
            """INSERT INTO adapter_aliases (tenant_id, project, name, alias, version_id)
               VALUES (CAST(:t AS uuid), :p, :n, :a, :v)
               ON CONFLICT (tenant_id, project, name, alias) DO UPDATE SET version_id = EXCLUDED.version_id, updated_at = now()"""
        ),
        {"t": str(version["tenant_id"]), "p": version["project"], "n": version["name"], "a": alias, "v": version["id"]},
    )
    await append_event(
        conn,
        version["tenant_id"],
        actor=actor,
        action="adapter.alias",
        resource_type="adapter_version",
        resource_id=version["id"],
        data={"alias": alias, "name": f"{version['project']}/{version['name']}"},
    )


async def aliases(conn: AsyncConnection, project: str, name: str) -> dict[str, str]:
    rows = (
        await conn.execute(
            text("SELECT alias, version_id FROM adapter_aliases WHERE project = :p AND name = :n"),
            {"p": project, "n": name},
        )
    ).all()
    return {r.alias: r.version_id for r in rows}


# ------------------------------------------------------------ deployments --
async def get_deployment(
    conn: AsyncConnection, project: str, logical_name: str, *, for_update: bool = False
) -> dict[str, Any] | None:
    q = "SELECT * FROM deployments WHERE project = :p AND logical_name = :n" + (" FOR UPDATE" if for_update else "")
    r = (await conn.execute(text(q), {"p": project, "n": logical_name})).first()
    return dict(r._mapping) if r else None


async def upsert_deployment(
    conn: AsyncConnection, tenant_id: uuid.UUID | str, project: str, logical_name: str, **fields: Any
) -> None:
    allowed = {"champion_version_id", "challenger_version_id", "challenger_weight", "previous_champion_id"}
    if set(fields) - allowed:
        raise ValueError(set(fields) - allowed)
    cols = ["tenant_id", "project", "logical_name", *fields]
    vals = ["CAST(:tenant_id AS uuid)", ":project", ":logical_name", *[f":{k}" for k in fields]]
    updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in fields) or "logical_name = EXCLUDED.logical_name"
    await conn.execute(
        text(
            f"""INSERT INTO deployments ({", ".join(cols)}) VALUES ({", ".join(vals)})
                ON CONFLICT (tenant_id, project, logical_name) DO UPDATE SET {updates}, updated_at = now()"""
        ),
        {"tenant_id": str(tenant_id), "project": project, "logical_name": logical_name, **fields},
    )


async def list_deployments(conn: AsyncConnection) -> list[dict[str, Any]]:
    return [
        dict(r._mapping)
        for r in (await conn.execute(text("SELECT * FROM deployments ORDER BY project, logical_name"))).all()
    ]


# --------------------------------------------------------------- approvals --
async def approvals_for(conn: AsyncConnection, job_id: str) -> list[dict[str, Any]]:
    rows = (
        await conn.execute(
            text(
                "SELECT id, approver, decision, justification, evidence_sha256, created_at FROM approvals WHERE job_id = :j ORDER BY created_at"
            ),
            {"j": job_id},
        )
    ).all()
    return [dict(r._mapping) for r in rows]


async def latest_evidence(conn: AsyncConnection, job_id: str) -> dict[str, Any] | None:
    r = (
        await conn.execute(
            text(
                """SELECT bundle_sha256 AS sha256, sha256 AS decision_sha256, target_env, body, decision, created_at
                     FROM evidence_bundles WHERE job_id = :j ORDER BY created_at DESC LIMIT 1"""
            ),
            {"j": job_id},
        )
    ).first()
    return _row(r) if r else None


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None
