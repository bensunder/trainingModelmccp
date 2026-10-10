"""FastAPI application: transport only. Business rules live in services/."""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text

from mccp.db.engine import system_tx
from mccp.domain.errors import DomainError, ValidationFailed
from mccp.domain.refs import parse_job_id
from mccp.domain.spec import JobRequest
from mccp.platform import Platform
from mccp.services import core
from mccp.services.repo import Principal
from mccp.settings import Settings, get_settings

from .auth import Authenticator

log = logging.getLogger("mccp.api")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
PROBLEM = "application/problem+json"


def _problem(status: int, title: str, detail: str, type_suffix: str, **extra: Any) -> JSONResponse:
    body = {
        "type": f"https://mccp.dev/problems/{type_suffix}",
        "title": title,
        "status": status,
        "detail": detail,
        "request_id": request_id_var.get(),
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM)


def _pf(request: Request) -> Platform:
    return request.app.state.pf


async def _principal(request: Request) -> Principal:
    return await request.app.state.auth.principal(request)


PF = Annotated[Platform, Depends(_pf)]
P = Annotated[Principal, Depends(_principal)]


class ApprovalBody(BaseModel):
    justification: str = Field(min_length=3, max_length=4000)


class RollbackBody(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


def create_app(settings: Settings | None = None, platform: Platform | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pf = platform or Platform.from_settings(settings)
        app.state.pf = pf
        app.state.auth = Authenticator(pf)
        try:
            yield
        finally:
            if platform is None:
                await pf.close()

    app = FastAPI(
        title="MCCP - Enterprise Model Customization Control Plane",
        version="0.1.0",
        lifespan=lifespan,
        description="Governed Qwen QLoRA SFT adapter pipeline: dataset -> Soup training -> eval -> OPA gate -> registry -> vLLM -> agent.",
    )

    # ---------------------------------------------------------- middleware --
    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        if len(rid) > 128:
            rid = rid[:128]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(DomainError)
    async def domain_error(_: Request, e: DomainError):
        return _problem(
            e.status,
            e.title,
            e.detail,
            e.type_suffix,
            **{k: v for k, v in e.extra.items() if isinstance(v, (str, int, float))},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError):
        errors = [{"loc": list(map(str, err["loc"])), "msg": err["msg"]} for err in e.errors()][:20]
        return _problem(
            422, "Validation failed", "request body or parameters are invalid", "validation-failed", errors=errors
        )

    @app.exception_handler(ValidationError)
    async def pydantic_error(_: Request, e: ValidationError):
        errors = [{"loc": list(map(str, err["loc"])), "msg": err["msg"]} for err in e.errors()][:20]
        return _problem(422, "Validation failed", "payload is invalid", "validation-failed", errors=errors)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, e: Exception):
        log.exception("unhandled error (request %s)", request_id_var.get())
        return _problem(
            500, "Internal error", "an unexpected error occurred; quote the request_id to support", "internal"
        )

    # ---------------------------------------------------------------- health --
    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def readyz(platform: PF) -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            async with system_tx(platform.engine) as conn:
                await conn.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["postgres"] = f"error: {type(e).__name__}"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)

    # --------------------------------------------------------------- recipes --
    @app.get("/v1/recipes", tags=["recipes"])
    async def list_recipes(platform: PF, p: P):
        return {"object": "list", "data": await core.list_recipes(platform, p)}

    @app.post("/v1/recipes", status_code=201, tags=["recipes"])
    async def put_recipe(platform: PF, p: P, body: Annotated[dict[str, Any], Body()]):
        return await core.put_recipe(platform, p, body)

    # -------------------------------------------------------------- datasets --
    @app.post("/v1/datasets/{project}/{name}/versions", status_code=201, tags=["datasets"])
    async def create_dataset_version(
        project: str,
        name: str,
        request: Request,
        platform: PF,
        p: P,
        redactor: Annotated[str, Query(pattern="^(pattern|presidio)$")] = "pattern",
        seed: Annotated[int, Query(ge=0, le=2**31 - 1)] = 1234,
    ):
        limit = platform.settings.max_dataset_upload_bytes
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise ValidationFailed("body", f"dataset exceeds {limit} bytes")
            chunks.append(chunk)
        return await core.create_dataset_version(
            platform, p, project, name, b"".join(chunks), redactor=redactor, seed=seed
        )

    @app.get("/v1/datasets/{project}/{name}/versions/{version}", tags=["datasets"])
    async def get_dataset(project: str, name: str, version: int, platform: PF, p: P):
        return await core.get_dataset_card(platform, p, project, name, version)

    # ------------------------------------------------------------ eval suites --
    @app.post("/v1/eval-suites/{project}", status_code=201, tags=["evaluations"])
    async def create_suite(project: str, platform: PF, p: P, body: Annotated[dict[str, Any], Body()]):
        return await core.create_suite(platform, p, project, body)

    # ------------------------------------------------------------------ jobs --
    @app.post("/v1/customization/jobs", tags=["jobs"])
    async def submit_job(
        platform: PF,
        p: P,
        body: Annotated[dict[str, Any], Body()],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    ):
        req = JobRequest.model_validate(body)
        view, created = await core.submit_job(platform, p, req, idempotency_key)
        return JSONResponse(
            view, status_code=201 if created else 200, headers={"Idempotent-Replayed": "false" if created else "true"}
        )

    @app.get("/v1/customization/jobs", tags=["jobs"])
    async def list_jobs(
        platform: PF,
        p: P,
        lifecycle_status: Annotated[list[str] | None, Query()] = None,
        project: str | None = None,
        after: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        return await core.list_jobs(platform, p, lifecycle=lifecycle_status, after=after, limit=limit, project=project)

    @app.get("/v1/customization/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str, platform: PF, p: P):
        return await core.get_job(platform, p, parse_job_id(job_id))

    @app.post("/v1/customization/jobs/{job_id}:cancel", tags=["jobs"])
    async def cancel_job(job_id: str, platform: PF, p: P):
        return await core.cancel_job(platform, p, parse_job_id(job_id))

    @app.post("/v1/customization/jobs/{job_id}:pause", tags=["jobs"])
    async def pause_job(job_id: str, platform: PF, p: P):
        return await core.pause_job(platform, p, parse_job_id(job_id))

    @app.post("/v1/customization/jobs/{job_id}:resume", tags=["jobs"])
    async def resume_job(job_id: str, platform: PF, p: P):
        return await core.resume_job(platform, p, parse_job_id(job_id))

    @app.get("/v1/customization/jobs/{job_id}/events", tags=["jobs"])
    async def job_events(
        job_id: str, platform: PF, p: P, after: int = 0, limit: Annotated[int, Query(ge=1, le=500)] = 100
    ):
        return await core.job_events(platform, p, parse_job_id(job_id), after, limit)

    @app.get("/v1/customization/jobs/{job_id}/metrics", tags=["jobs"])
    async def job_metrics(job_id: str, platform: PF, p: P):
        return await core.job_metrics(platform, p, parse_job_id(job_id))

    @app.get("/v1/customization/jobs/{job_id}/checkpoints", tags=["jobs"])
    async def job_checkpoints(job_id: str, platform: PF, p: P):
        return await core.job_checkpoints(platform, p, parse_job_id(job_id))

    # -------------------------------------------------------------- approvals --
    @app.get("/v1/approvals", tags=["governance"])
    async def approvals(platform: PF, p: P):
        return {"object": "list", "data": await core.pending_approvals(platform, p)}

    @app.post("/v1/approvals/{job_id}:approve", tags=["governance"])
    async def approve(job_id: str, platform: PF, p: P, body: ApprovalBody):
        return await core.decide_approval(platform, p, parse_job_id(job_id), "approve", body.justification)

    @app.post("/v1/approvals/{job_id}:reject", tags=["governance"])
    async def reject(job_id: str, platform: PF, p: P, body: ApprovalBody):
        return await core.decide_approval(platform, p, parse_job_id(job_id), "reject", body.justification)

    @app.post("/v1/policies:evaluate", tags=["governance"])
    async def policy_dry_run(
        platform: PF,
        p: P,
        job_id: Annotated[str, Body(embed=True)],
        target: Annotated[str, Body(embed=True)] = "production",
    ):
        return await core.dry_run_policy(platform, p, parse_job_id(job_id), target)

    # --------------------------------------------------------------- adapters --
    @app.get("/v1/adapters", tags=["registry"])
    async def adapters(platform: PF, p: P, project: str | None = None):
        return {"object": "list", "data": await core.list_adapters(platform, p, project)}

    @app.get("/v1/adapters/{project}/{name}/versions/{version}", tags=["registry"])
    async def adapter(project: str, name: str, version: int, platform: PF, p: P):
        return await core.get_adapter(platform, p, project, name, version)

    @app.get("/v1/adapters/{project}/{name}/versions/{version}/evidence", tags=["registry"])
    async def adapter_evidence(project: str, name: str, version: int, platform: PF, p: P):
        return await core.adapter_evidence(platform, p, project, name, version)

    # ------------------------------------------------------------ deployments --
    @app.get("/v1/deployments", tags=["serving"])
    async def deployments(platform: PF, p: P):
        return {"object": "list", "data": await core.list_deployments(platform, p)}

    @app.post("/v1/deployments/{project}/{name}:rollback", tags=["serving"])
    async def rollback(project: str, name: str, platform: PF, p: P, body: RollbackBody):
        return await core.request_rollback(platform, p, project, name, body.reason)

    # --------------------------------------------------------------- feedback --
    @app.post("/v1/feedback", status_code=201, tags=["feedback"])
    async def feedback(platform: PF, p: P, body: Annotated[dict[str, Any], Body()]):
        return await core.record_feedback(platform, p, body)

    # ------------------------------------------------------------------ audit --
    @app.get("/v1/audit:verify", tags=["audit"])
    async def audit_verify(platform: PF, p: P):
        return await core.verify_audit(platform, p)

    return app


def configure_logging(level: str) -> None:
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "request_id": request_id_var.get(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
