"""Use cases invoked by the API (transport-agnostic, no HTTP types here)."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from sqlalchemy import text

from mccp.audit.chain import append_event, verify_chain
from mccp.datasets.pii import make_redactor
from mccp.datasets.pipeline import BuildParams, build_dataset
from mccp.db.engine import jsonb, system_tx, tenant_tx
from mccp.domain.errors import Conflict, Forbidden, NotFound, ValidationFailed
from mccp.domain.hashing import digest_of, sha256_hex
from mccp.domain.recipe import Recipe
from mccp.domain.refs import new_id, new_job_id, require_slug
from mccp.domain.spec import JobRequest
from mccp.domain.status import LifecycleStatus, is_cancellable, job_status
from mccp.evals.suite import EvalSuite
from mccp.platform import Platform
from mccp.storage.objectstore import tenant_prefix

from . import repo
from .repo import Principal


def _require(p: Principal, role: str) -> None:
    if not p.has(role):
        raise Forbidden(f"requires role {role}")


# ------------------------------------------------------------------ views --
def job_view(job: dict[str, Any]) -> dict[str, Any]:
    lc = LifecycleStatus(job["lifecycle_status"])
    req = job["request"]
    resolved = job["resolved"]
    return {
        "id": job["id"],
        "object": "customization.job",
        "project": job["project"],
        "status": job_status(lc).value,
        "lifecycle_status": lc.value,
        "recipe": req["recipe"],
        "method": {"type": req["method"]["type"]},
        "finetuning_type": req["finetuning_type"],
        "base_model": f"{resolved['base_model']['id']}@{resolved['base_model']['revision']}",
        "training_dataset": req["training_dataset"],
        "validation_dataset": req["validation_dataset"],
        "eval_suite": req["eval_suite"],
        "output_adapter": (job.get("stage_state") or {}).get("serving_name"),
        "adapter_version_id": job["adapter_version_id"],
        "risk_tier": job["risk_tier"],
        "trained_tokens": job["trained_tokens"],
        "gpu_seconds": job["gpu_seconds"],
        "cost_usd": job["cost_usd"],
        "flop_estimate": job["flop_estimate"],
        "config_hash": job["config_hash"],
        "seed": req.get("seed"),
        "metadata": req.get("metadata", {}),
        "submitted_by": job["submitted_by"],
        "created_at": int(job["created_at"].timestamp()),
        "finished_at": int(job["finished_at"].timestamp()) if job["finished_at"] else None,
        "estimated_finish": None,
        "error": job["error"],
    }


# ---------------------------------------------------------------- recipes --
async def put_recipe(pf: Platform, p: Principal, body: dict[str, Any]) -> dict[str, Any]:
    _require(p, Principal.ADMIN)
    recipe = Recipe.model_validate(body)
    canonical = recipe.model_dump(mode="json")
    sha = digest_of(canonical)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        existing = (
            await conn.execute(text("SELECT body_sha256 FROM recipes WHERE ref = :r"), {"r": str(recipe.ref)})
        ).first()
        if existing and existing.body_sha256 != sha:
            raise Conflict(f"recipe {recipe.ref} already exists with different content; bump its version")
        if not existing:
            await conn.execute(
                text("INSERT INTO recipes (ref, body, body_sha256, created_by) VALUES (:r, CAST(:b AS jsonb), :s, :c)"),
                {"r": str(recipe.ref), "b": jsonb(canonical), "s": sha, "c": p.subject},
            )
            await append_event(
                conn,
                p.tenant_id,
                actor=p.subject,
                action="recipe.create",
                resource_type="recipe",
                resource_id=str(recipe.ref),
                data={"sha256": sha},
            )
    return {"ref": str(recipe.ref), "sha256": sha, "recipe": canonical}


async def list_recipes(pf: Platform, p: Principal) -> list[dict[str, Any]]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        rows = (await conn.execute(text("SELECT ref, body, body_sha256 FROM recipes ORDER BY ref"))).all()
    return [{"ref": r.ref, "sha256": r.body_sha256, "recipe": repo.as_obj(r.body)} for r in rows]


# --------------------------------------------------------------- datasets --
async def create_dataset_version(
    pf: Platform, p: Principal, project: str, name: str, raw: bytes, *, redactor: str = "pattern", seed: int = 1234
) -> dict[str, Any]:
    _require(p, Principal.SUBMIT)
    require_slug("project", project)
    require_slug("dataset", name)
    if len(raw) > pf.settings.max_dataset_upload_bytes:
        raise ValidationFailed("body", f"dataset exceeds {pf.settings.max_dataset_upload_bytes} bytes")
    result = await asyncio.to_thread(build_dataset, raw, make_redactor(redactor), BuildParams(seed=seed))
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await repo.ensure_project(conn, p.tenant_id, project)
        version = await repo.next_dataset_version(conn, project, name)
        prefix = f"{tenant_prefix(p.tenant_id)}/datasets/{project}/{name}/v{version}"
        for fname, data in result.files.items():
            await pf.store.put_bytes(f"{prefix}/{fname}", data)
        card = {**result.card, "ref": f"{project}/{name}@v{version}"}
        await pf.store.put_bytes(f"{prefix}/card.json", json.dumps(card, indent=2).encode())
        await conn.execute(
            text(
                """INSERT INTO dataset_versions (tenant_id, project, name, version, ref, manifest_sha256, artifact_prefix, card, created_by)
                   VALUES (CAST(:t AS uuid), :p, :n, :v, :ref, :m, :prefix, CAST(:card AS jsonb), :by)"""
            ),
            {
                "t": str(p.tenant_id),
                "p": project,
                "n": name,
                "v": version,
                "ref": card["ref"],
                "m": result.manifest_sha256,
                "prefix": prefix,
                "card": jsonb(card),
                "by": p.subject,
            },
        )
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action="dataset.version.create",
            resource_type="dataset_version",
            resource_id=card["ref"],
            data={"manifest_sha256": result.manifest_sha256, "raw_sha256": sha256_hex(raw)},
        )
    return card


async def get_dataset_card(pf: Platform, p: Principal, project: str, name: str, version: int) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        return (await repo.get_dataset_version(conn, project, name, version))["card"]


# ------------------------------------------------------------ eval suites --
async def create_suite(pf: Platform, p: Principal, project: str, body: dict[str, Any]) -> dict[str, Any]:
    _require(p, Principal.SUBMIT)
    suite = EvalSuite.model_validate(body)
    if suite.golden.project != project:
        raise ValidationFailed("golden", "golden set must belong to the suite's project")
    canonical = suite.model_dump(mode="json")
    sha = digest_of(canonical)
    ref = f"{suite.name}@v{suite.version}"
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await repo.ensure_project(conn, p.tenant_id, project)
        g = suite.golden
        await repo.get_dataset_version(conn, g.project, g.name, g.version)
        existing = (
            await conn.execute(
                text("SELECT body_sha256 FROM eval_suites WHERE project = :p AND name = :n AND version = :v"),
                {"p": project, "n": suite.name, "v": suite.version},
            )
        ).first()
        if existing:
            if existing.body_sha256 != sha:
                raise Conflict(f"suite {ref} exists with different content; versions are immutable")
            return {"ref": ref, "sha256": sha}
        await conn.execute(
            text(
                """INSERT INTO eval_suites (tenant_id, project, name, version, ref, body, body_sha256, created_by)
                   VALUES (CAST(:t AS uuid), :p, :n, :v, :ref, CAST(:b AS jsonb), :s, :by)"""
            ),
            {
                "t": str(p.tenant_id),
                "p": project,
                "n": suite.name,
                "v": suite.version,
                "ref": ref,
                "b": jsonb(canonical),
                "s": sha,
                "by": p.subject,
            },
        )
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action="eval_suite.create",
            resource_type="eval_suite",
            resource_id=f"{project}/{ref}",
            data={"sha256": sha},
        )
    return {"ref": ref, "sha256": sha}


# -------------------------------------------------------------------- jobs --
async def submit_job(pf: Platform, p: Principal, req: JobRequest, idempotency_key: str) -> tuple[dict[str, Any], bool]:
    _require(p, Principal.SUBMIT)
    if not (8 <= len(idempotency_key) <= 255):
        raise ValidationFailed("Idempotency-Key", "must be 8..255 characters")
    req.check_method_enabled()
    request_json = req.model_dump(mode="json")
    request_hash = digest_of(request_json)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        existing = await repo.find_job_by_key(conn, idempotency_key)
        if existing:
            return _replay(existing, request_hash)
        recipe = Recipe.model_validate((await repo.get_recipe(conn, str(req.recipe)))["body"])
        resolved = req.resolve(recipe)
        td = req.training_dataset
        await repo.get_dataset_version(conn, td.project, td.name, td.version)
        await repo.get_suite(conn, req.project, req.eval_suite.name, req.eval_suite.version)
        job = {
            "id": new_job_id(),
            "tenant_id": p.tenant_id,
            "project": req.project,
            "idempotency_key": idempotency_key,
            "request_hash": request_hash,
            "request": request_json,
            "resolved": resolved.model_dump(mode="json"),
            "config_hash": resolved.config_hash(),
            "risk_tier": req.risk_tier,
            "output_adapter": req.output_adapter,
            "submitted_by": p.subject,
        }
        inserted = await repo.insert_job(conn, job)
        if not inserted:
            # Lost a race with a concurrent identical submit; it has committed.
            existing = await repo.find_job_by_key(conn, idempotency_key)
            assert existing is not None
            return _replay(existing, request_hash)
        stored = await repo.get_job(conn, job["id"])
        await repo.enqueue(conn, stored)
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action="job.create",
            resource_type="job",
            resource_id=job["id"],
            data={"config_hash": job["config_hash"], "request_hash": request_hash, "recipe": str(req.recipe)},
        )
    return job_view(stored), True


def _replay(existing: dict[str, Any], request_hash: str) -> tuple[dict[str, Any], bool]:
    if existing["request_hash"] != request_hash:
        raise Conflict("Idempotency-Key was already used with a different request body")
    return job_view(existing), False


async def get_job(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        return job_view(await repo.get_job(conn, job_id))


async def list_jobs(
    pf: Platform, p: Principal, *, lifecycle: list[str] | None, after: str | None, limit: int, project: str | None
) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        rows = await repo.list_jobs(conn, lifecycle=lifecycle, after=after, limit=limit + 1, project=project)
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {
        "object": "list",
        "data": [job_view(r) for r in rows],
        "has_more": has_more,
        "last_id": rows[-1]["id"] if rows else None,
    }


async def cancel_job(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    _require(p, Principal.SUBMIT)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        job = await repo.get_job(conn, job_id, for_update=True)
        lc = LifecycleStatus(job["lifecycle_status"])
        if not is_cancellable(lc):
            raise Conflict(f"job in state {lc} cannot be cancelled")
        q = await repo.queue_row(conn, job_id)
        leased = bool(
            q and q["lease_owner"] and q["lease_expires_at"] and q["lease_expires_at"].timestamp() > _now_ts()
        )
        if (
            lc in (LifecycleStatus.CREATED, LifecycleStatus.VALIDATING, LifecycleStatus.QUEUED, LifecycleStatus.PAUSED)
            and not leased
        ):
            job = await repo.transition(
                conn, job, LifecycleStatus.CANCELLED, actor=p.subject, data={"reason": "user cancel"}
            )
            await repo.dequeue(conn, job_id)
        else:
            # A worker owns it: it honours the flag at the next control point.
            await repo.update_job(conn, job_id, cancel_requested=True)
            await append_event(
                conn,
                p.tenant_id,
                actor=p.subject,
                action="job.cancel_requested",
                resource_type="job",
                resource_id=job_id,
                data={},
            )
            job = await repo.get_job(conn, job_id)
        return job_view(job)


async def pause_job(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    _require(p, Principal.SUBMIT)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        job = await repo.get_job(conn, job_id, for_update=True)
        if job["lifecycle_status"] != LifecycleStatus.RUNNING:
            raise Conflict("only running jobs can be paused")
        await repo.update_job(conn, job_id, pause_requested=True)
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action="job.pause_requested",
            resource_type="job",
            resource_id=job_id,
            data={},
        )
        return job_view(await repo.get_job(conn, job_id))


async def resume_job(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    _require(p, Principal.SUBMIT)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        job = await repo.get_job(conn, job_id, for_update=True)
        job = await repo.transition(
            conn, job, LifecycleStatus.RUNNING, actor=p.subject, data={"reason": "resume"}, pause_requested=False
        )
        await repo.enqueue(conn, job)
        return job_view(job)


async def job_events(pf: Platform, p: Principal, job_id: str, after: int, limit: int) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await repo.get_job(conn, job_id)
        rows = await repo.job_events(conn, job_id, after, limit)
    return {"object": "list", "data": [{**r, "ts": repo.iso(r["ts"])} for r in rows]}


async def job_metrics(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await repo.get_job(conn, job_id)
        rows = await repo.job_metrics(conn, job_id)
    return {"object": "list", "data": [{**r, "ts": repo.iso(r["ts"])} for r in rows]}


async def job_checkpoints(pf: Platform, p: Principal, job_id: str) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await repo.get_job(conn, job_id)
        rows = await repo.job_checkpoints(conn, job_id)
    return {"object": "list", "data": [{**r, "ts": repo.iso(r["ts"])} for r in rows]}


# --------------------------------------------------------------- approvals --
async def pending_approvals(pf: Platform, p: Principal) -> list[dict[str, Any]]:
    _require(p, Principal.APPROVE)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        jobs = await repo.list_jobs(
            conn, lifecycle=[LifecycleStatus.PENDING_APPROVAL.value], after=None, limit=200, project=None
        )
        out = []
        for j in jobs:
            ev = await repo.latest_evidence(conn, j["id"])
            out.append(
                {
                    "job": job_view(j),
                    "evidence_sha256": ev["sha256"] if ev else None,
                    "staging_decision": ev["decision"] if ev else None,
                    "model_card": ev["body"].get("model_card") if ev else None,
                    "approvals": [
                        {**a, "created_at": repo.iso(a["created_at"])} for a in await repo.approvals_for(conn, j["id"])
                    ],
                }
            )
        return out


async def decide_approval(pf: Platform, p: Principal, job_id: str, decision: str, justification: str) -> dict[str, Any]:
    _require(p, Principal.APPROVE)
    if decision not in ("approve", "reject"):
        raise ValidationFailed("decision", "approve or reject")
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        job = await repo.get_job(conn, job_id, for_update=True)
        if job["lifecycle_status"] != LifecycleStatus.PENDING_APPROVAL:
            raise Conflict(f"job is {job['lifecycle_status']}, not pending_approval")
        if decision == "approve" and job["submitted_by"] == p.subject:
            # Also enforced by policy; refusing early gives a clear error.
            raise Forbidden("separation of duties: the submitter cannot approve their own adapter")
        ev = await repo.latest_evidence(conn, job_id)
        if ev is None:
            raise Conflict("no evidence bundle recorded for this job")
        r = await conn.execute(
            text(
                """INSERT INTO approvals (id, tenant_id, job_id, adapter_version_id, approver, decision, justification, evidence_sha256)
                   VALUES (:id, CAST(:t AS uuid), :j, :av, :who, :d, :why, :ev)
                   ON CONFLICT (job_id, approver) DO NOTHING RETURNING id"""
            ),
            {
                "id": new_id("apr"),
                "t": str(p.tenant_id),
                "j": job_id,
                "av": job["adapter_version_id"],
                "who": p.subject,
                "d": decision,
                "why": justification,
                "ev": ev["sha256"],
            },
        )
        if r.first() is None:
            raise Conflict("you already recorded a decision for this job")
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action=f"approval.{decision}",
            resource_type="job",
            resource_id=job_id,
            data={"evidence_sha256": ev["sha256"], "justification": justification},
        )
        await repo.enqueue(conn, job)  # the pipeline re-evaluates the production gate
        return {"job_id": job_id, "decision": decision, "evidence_sha256": ev["sha256"]}


# ---------------------------------------------------------------- adapters --
def adapter_view(v: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": v["id"],
        "project": v["project"],
        "name": v["name"],
        "version": v["version"],
        "serving_name": v["serving_name"],
        "status": v["status"],
        "artifact_uri": v["artifact_uri"],
        "artifact_sha256": v["artifact_sha256"],
        "signature_uri": v["signature_uri"],
        "job_id": v["job_id"],
        "card": v["card"],
        "created_at": repo.iso(v["created_at"]),
    }


async def list_adapters(pf: Platform, p: Principal, project: str | None) -> list[dict[str, Any]]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        return [adapter_view(v) for v in await repo.list_adapter_versions(conn, project)]


async def get_adapter(pf: Platform, p: Principal, project: str, name: str, version: int) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        v = await repo.find_adapter_version(conn, project, name, version)
        return {**adapter_view(v), "aliases": await repo.aliases(conn, project, name)}


async def adapter_evidence(pf: Platform, p: Principal, project: str, name: str, version: int) -> dict[str, Any]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        v = await repo.find_adapter_version(conn, project, name, version)
        ev = await repo.latest_evidence(conn, v["job_id"])
        if ev is None:
            raise NotFound("no evidence recorded yet")
        rep = (
            await conn.execute(
                text("SELECT report FROM eval_reports WHERE adapter_version_id = :id ORDER BY created_at DESC LIMIT 1"),
                {"id": v["id"]},
            )
        ).first()
        return {
            "evidence_sha256": ev["sha256"],
            "target": ev["target_env"],
            "decision": ev["decision"],
            "bundle": ev["body"],
            "eval_report": repo.as_obj(rep.report) if rep else None,
        }


async def dry_run_policy(pf: Platform, p: Principal, job_id: str, target: str) -> dict[str, Any]:
    if target not in ("staging", "production"):
        raise ValidationFailed("target", "staging or production")
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        ev = await repo.latest_evidence(conn, job_id)
        if ev is None:
            raise NotFound("no evidence recorded for this job yet")
        approvals = [
            {"approver": a["approver"], "decision": a["decision"]} for a in await repo.approvals_for(conn, job_id)
        ]
    return await pf.opa.decide(ev["body"], target=target, approvals=approvals)


# ------------------------------------------------------------- deployments --
async def list_deployments(pf: Platform, p: Principal) -> list[dict[str, Any]]:
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        out = []
        for d in await repo.list_deployments(conn):
            d = {k: (repo.iso(v) if k == "updated_at" else v) for k, v in d.items() if k != "tenant_id"}
            out.append(d)
        return out


async def request_rollback(pf: Platform, p: Principal, project: str, logical_name: str, reason: str) -> dict[str, Any]:
    _require(p, Principal.APPROVE)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        dep = await repo.get_deployment(conn, project, logical_name, for_update=True)
        if dep is None:
            raise NotFound(f"no deployment {project}/{logical_name}")
        target_version = dep["challenger_version_id"] or (
            dep["champion_version_id"] if dep["previous_champion_id"] else None
        )
        if target_version is None:
            raise Conflict("nothing to roll back: no canary in progress and no previous champion")
        v = await repo.get_adapter_version(conn, target_version)
        job = await repo.get_job(conn, v["job_id"], for_update=True)
        state = {**(job["stage_state"] or {}), "rollback_requested": {"by": p.subject, "reason": reason}}
        await repo.update_job(conn, job["id"], stage_state=state)
        await repo.enqueue(conn, job)
        await append_event(
            conn,
            p.tenant_id,
            actor=p.subject,
            action="deployment.rollback_requested",
            resource_type="deployment",
            resource_id=f"{project}/{logical_name}",
            data={"job_id": job["id"], "reason": reason},
        )
        return {"job_id": job["id"], "rolling_back": v["serving_name"]}


# ---------------------------------------------------------------- feedback --
async def record_feedback(pf: Platform, p: Principal, body: dict[str, Any]) -> dict[str, Any]:
    serving_name = body.get("adapter_serving_name")
    rating = body.get("rating")
    if not isinstance(serving_name, str) or rating not in (-1, 1):
        raise ValidationFailed("feedback", "adapter_serving_name (str) and rating (-1|1) are required")
    comment = body.get("comment")
    if comment is not None and (not isinstance(comment, str) or len(comment) > 4000):
        raise ValidationFailed("comment", "string <= 4000 chars")
    payload = body.get("payload") or {}
    if len(json.dumps(payload)) > 64_000:
        raise ValidationFailed("payload", "too large")
    fid = new_id("fb")
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        await conn.execute(
            text(
                """INSERT INTO feedback (id, tenant_id, adapter_serving_name, logical_name, request_id, rating, comment, payload, created_by)
                   VALUES (:id, CAST(:t AS uuid), :s, :l, :r, :rating, :c, CAST(:pl AS jsonb), :by)"""
            ),
            {
                "id": fid,
                "t": str(p.tenant_id),
                "s": serving_name,
                "l": body.get("logical_name"),
                "r": body.get("request_id"),
                "rating": rating,
                "c": comment,
                "pl": jsonb(payload),
                "by": p.subject,
            },
        )
    return {"id": fid}


# ------------------------------------------------------------------- audit --
async def verify_audit(pf: Platform, p: Principal) -> dict[str, Any]:
    _require(p, Principal.ADMIN)
    async with tenant_tx(pf.engine, p.tenant_id) as conn:
        v = await verify_chain(conn, p.tenant_id)
    return {
        "ok": v.ok,
        "events_checked": v.events_checked,
        "head_sha": v.head_sha,
        "first_bad_seq": v.first_bad_seq,
        "reason": v.reason,
    }


# ----------------------------------------------------------------- tenants --
async def resolve_tenant(pf: Platform, idp_tenant: str) -> uuid.UUID | None:
    async with system_tx(pf.engine) as conn:
        v = (await conn.execute(text("SELECT mccp_resolve_tenant(:t)"), {"t": idp_tenant})).scalar_one_or_none()
    return v


async def create_tenant(pf: Platform, name: str, idp_tenant_id: str, region: str = "us") -> uuid.UUID:
    # The name becomes part of serving and gateway names, so it must be a slug.
    require_slug("tenant", name)
    tid = uuid.uuid4()
    async with tenant_tx(pf.engine, tid) as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, idp_tenant_id, region) VALUES (CAST(:id AS uuid), :n, :i, :r)"),
            {"id": str(tid), "n": name, "i": idp_tenant_id, "r": region},
        )
        await append_event(
            conn,
            tid,
            actor="system:admin",
            action="tenant.create",
            resource_type="tenant",
            resource_id=str(tid),
            data={"name": name},
        )
    return tid


def _now_ts() -> float:
    import time

    return time.time()
