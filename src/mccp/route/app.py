"""HTTP layer of the routing gateway. Transport only; the request path is service.py.

Client API (what coding tools call):
  POST /v1/messages                Anthropic Messages (Claude Code)
  POST /v1/messages/count_tokens   estimate (Claude Code context bookkeeping)
  POST /v1/chat/completions        OpenAI Chat Completions (OpenCode, SDKs)
  POST /v1/responses               OpenAI Responses (Codex), translated to Chat Completions
  GET  /v1/models                  route profiles and the models this key may use
  GET  /v1/usage                   this key's own usage this month
Admin API (/admin/v1), two kinds of principal:
  operator   ROUTE_ADMIN_TOKEN (only its SHA-256 is configured): every org, creates orgs
  org admin  `ra_...` token bound to one org; role owner (manage the org) or viewer (read)
  Another org's objects answer 404, not 403: an org admin cannot learn what exists.
  Every change is written to the org's audit log (GET /admin/v1/orgs/{id}/audit).
Health: /healthz (process), /readyz (store, cache, providers)

Errors are RFC 9457 problem+json. Client API keys are accepted as
`Authorization: Bearer rk_...` or `x-api-key: rk_...` (the Anthropic SDK header).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .cache import Cache, MemoryCache, RedisCache
from .config import Catalog, Wire, load_catalog
from .guardrails import GuardrailPolicy
from .policy import Evidence, explain
from .service import Gateway, GatewayError, KeyContext
from .state import (
    ADMIN_PREFIX,
    AuditEvent,
    MemoryStore,
    NotFound,
    Store,
    event_dict,
    hash_key,
    mint_key,
    new_id,
    now,
    period_start,
    summarize,
)  # fmt: skip
from .upstream import Upstream

log = logging.getLogger("mccp.route.api")
PROBLEM = "application/problem+json"
MAX_BODY_BYTES = 32 * 1024 * 1024


def problem(
    status: int, kind: str, title: str, detail: str, headers: dict[str, str] | None = None, **extra: Any
) -> JSONResponse:
    body = {
        "type": f"https://route.whyaidata.com/problems/{kind}",
        "title": title,
        "status": status,
        "detail": detail,
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM, headers=headers)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog_path: str
    evidence_path: str | None = None
    database_url: str | None = None
    redis_url: str | None = None
    key_pepper: bytes
    admin_token_sha256: str
    production: bool = False

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ) if env is None else env
        pepper_raw = e.get("ROUTE_KEY_PEPPER", "")
        try:
            pepper = base64.b64decode(pepper_raw, validate=True) if pepper_raw else b""
        except ValueError as err:
            raise ValueError("ROUTE_KEY_PEPPER must be base64") from err
        prod = e.get("ROUTE_ENV", "development") == "production"
        s = cls(
            catalog_path=e.get("ROUTE_CATALOG", "deploy/route/route.yaml"),
            evidence_path=e.get("ROUTE_EVIDENCE") or None,
            database_url=e.get("ROUTE_DATABASE_URL") or None,
            redis_url=e.get("ROUTE_REDIS_URL") or None,
            key_pepper=pepper,
            admin_token_sha256=e.get("ROUTE_ADMIN_TOKEN_SHA256", "").lower(),
            production=prod,
        )
        if len(s.key_pepper) < 32:
            raise ValueError("ROUTE_KEY_PEPPER must decode to at least 32 bytes")
        if len(s.admin_token_sha256) != 64:
            raise ValueError("ROUTE_ADMIN_TOKEN_SHA256 must be the hex SHA-256 of the admin token")
        if prod and not (s.database_url and s.redis_url):
            raise ValueError("production needs ROUTE_DATABASE_URL and ROUTE_REDIS_URL")
        return s


# -- admin request bodies ---------------------------------------------------------
class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrgIn(_Body):
    name: str = Field(min_length=1, max_length=200)
    default_profile: str | None = None


class OrgPatch(_Body):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    default_profile: str | None = None
    rpm_per_key: int | None = Field(default=None, ge=1, le=100_000)


class TeamIn(_Body):
    name: str = Field(min_length=1, max_length=200)
    approved_models: list[str] | None = None
    default_profile: str | None = None


class TeamPatch(_Body):
    approved_models: list[str] | None = None
    default_profile: str | None = None


class KeyIn(_Body):
    label: str = Field(min_length=1, max_length=200)
    default_profile: str | None = None


class AdminIn(_Body):
    label: str = Field(min_length=1, max_length=200)
    role: Literal["owner", "viewer"]


class BudgetIn(_Body):
    scope: Literal["org", "team", "key"]
    scope_id: str
    period: Literal["day", "month"]
    limit_usd: float = Field(ge=0, le=10_000_000)
    hard: bool = True


# -- dependencies (module level: FastAPI resolves postponed annotations from module globals)
def _gw(request: Request) -> Gateway:
    return request.app.state.gw


async def _client(request: Request) -> KeyContext:
    token = request.headers.get("x-api-key")
    auth = request.headers.get("authorization", "")
    if not token and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    return await request.app.state.gw.authenticate(token)


@dataclass(frozen=True)
class Principal:
    kind: Literal["operator", "org"]
    role: str  # operator | owner | viewer
    actor: str  # recorded in the audit log
    org_id: str | None = None
    label: str = "operator"
    admin_id: str | None = None


OPERATOR = Principal("operator", "operator", "operator")


async def _admin(request: Request) -> Principal:
    auth = request.headers.get("authorization", "")
    tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    unauthorized = GatewayError(401, "unauthorized", "Unauthorized", "admin token required")
    if not tok:
        raise unauthorized
    want = request.app.state.settings.admin_token_sha256
    if hmac.compare_digest(hashlib.sha256(tok.encode()).hexdigest(), want):
        return OPERATOR
    if tok.startswith(ADMIN_PREFIX):
        g: Gateway = request.app.state.gw
        a = await g.store.admin_by_hash(hash_key(g.pepper, tok))
        if a is not None and a.revoked_at is None:
            return Principal("org", a.role, f"admin:{a.id}", a.org_id, a.label, a.id)
    raise unauthorized


G = Annotated[Gateway, Depends(_gw)]
C = Annotated[KeyContext, Depends(_client)]
P = Annotated[Principal, Depends(_admin)]
FORBIDDEN_RO = ("forbidden", "Forbidden", "this admin token is read-only (viewer)")


def allow(p: Principal, org_id: str, write: bool = False) -> None:
    """Operator: any org. Org admin: its own org only (others look absent: 404);
    viewers cannot change anything (403)."""
    if p.kind == "operator":
        return
    if p.org_id != org_id:
        raise NotFound("org")
    if write and p.role != "owner":
        raise GatewayError(403, *FORBIDDEN_RO)


def operator_only(p: Principal, what: str) -> None:
    if p.kind != "operator":
        raise GatewayError(403, "forbidden", "Forbidden", f"{what} is reserved for the platform operator")


def create_app(
    settings: Settings | None = None,
    *,
    catalog: Catalog | None = None,
    store: Store | None = None,
    cache: Cache | None = None,
    upstream: Upstream | None = None,
    evidence: Evidence | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or Settings.from_env()
        cat = catalog or load_catalog(s.catalog_path, production=s.production)
        st = store
        if st is None:
            if s.database_url:
                from .pgstore import PgStore

                st = await PgStore.connect(s.database_url)
            else:
                log.warning("ROUTE_DATABASE_URL not set: using in-memory state (development only)")
                st = MemoryStore()
        ca = cache or (RedisCache.from_url(s.redis_url) if s.redis_url else MemoryCache())
        up = upstream or Upstream(cat)
        gw = Gateway(cat, evidence or Evidence.load(s.evidence_path), st, ca, up, s.key_pepper)
        gw.writer.start()
        app.state.gw, app.state.settings = gw, s
        try:
            yield
        finally:
            await gw.writer.stop()
            await up.aclose()
            await ca.close()
            await st.close()

    app = FastAPI(title="Route gateway", version="1", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(GatewayError)
    async def _gw_err(_: Request, e: GatewayError) -> JSONResponse:
        return problem(e.status, e.kind, e.title, e.detail, e.headers, **e.extra)

    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound) -> JSONResponse:
        return problem(404, "not-found", "Not found", f"{e} not found")

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, e: RequestValidationError) -> JSONResponse:
        return problem(422, "validation", "Invalid request", "request failed validation", errors=e.errors())

    async def json_body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise GatewayError(413, "too-large", "Request too large", f"body over {MAX_BODY_BYTES} bytes")
        import json

        try:
            data = json.loads(raw or b"null")
        except json.JSONDecodeError as e:
            raise GatewayError(400, "invalid-json", "Invalid JSON", str(e)) from e
        if not isinstance(data, dict):
            raise GatewayError(400, "invalid-json", "Invalid JSON", "body must be a JSON object")
        return data

    def respond(res) -> Response:
        if res.stream is not None:
            return StreamingResponse(res.stream, status_code=res.status, headers=res.headers)
        return Response(
            res.body or b"",
            status_code=res.status,
            headers=res.headers,
            media_type=res.headers.get("content-type", "application/json"),
        )

    # -- client API -------------------------------------------------------------------
    @app.post("/v1/messages")
    async def messages(request: Request, g: G, ctx: C) -> Response:
        return respond(await g.handle(Wire.ANTHROPIC, await json_body(request), dict(request.headers), ctx))

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request, g: G, ctx: C) -> dict[str, int]:
        body = await json_body(request)
        try:
            return {"input_tokens": g.count_tokens(Wire.ANTHROPIC, body)}
        except ValueError as e:
            raise GatewayError(400, "invalid-request", "Invalid request", str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat(request: Request, g: G, ctx: C) -> Response:
        return respond(await g.handle(Wire.OPENAI_CHAT, await json_body(request), dict(request.headers), ctx))

    @app.post("/v1/responses")
    async def responses_api(request: Request, g: G, ctx: C) -> Response:
        return respond(await g.handle_responses(await json_body(request), dict(request.headers), ctx))

    @app.get("/v1/models")
    async def models(g: G, ctx: C) -> dict[str, Any]:
        approved = set(ctx.team.approved_models) if ctx.team.approved_models is not None else None
        created = int(ctx.org.created_at.timestamp())
        data = [
            {
                "id": p.id,
                "object": "model",
                "type": "model",
                "display_name": f"Route: {p.name}",
                "owned_by": "route",
                "created": created,
                "created_at": ctx.org.created_at.isoformat(),
            }
            for p in g.catalog.profiles
        ]
        for m in g.catalog.models:
            if approved is None or m.id in approved:
                data.append(
                    {
                        "id": m.id,
                        "object": "model",
                        "type": "model",
                        "display_name": m.id,
                        "owned_by": m.provider,
                        "created": created,
                        "created_at": ctx.org.created_at.isoformat(),
                    }
                )
        return {"object": "list", "data": data, "has_more": False}

    @app.get("/v1/usage")
    async def my_usage(g: G, ctx: C) -> dict[str, Any]:
        since = period_start("month")
        rows = await g.store.usage_rows(ctx.org.id, since, now() + timedelta(seconds=1), key_id=ctx.key.id)
        return {"since": since.isoformat(), **summarize(rows)}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(g: G) -> JSONResponse:
        cache_ok = await g.cache.ping()
        providers = g.upstream.status()
        ok = cache_ok and any(p["key_configured"] for p in providers.values())
        return JSONResponse(
            {"ready": ok, "cache": cache_ok, "providers": providers, "usage_dropped": g.writer.dropped},
            status_code=200 if ok else 503,
        )

    # -- admin API ----------------------------------------------------------------------
    def _check_profile(g: Gateway, pid: str | None) -> None:
        if pid is not None and g.catalog.profile(pid) is None:
            raise GatewayError(422, "unknown-profile", "Unknown profile", f"no route profile {pid}")

    def _check_models(g: Gateway, ids: list[str] | None) -> None:
        bad = [m for m in ids or [] if not g.catalog.has_model(m)]
        if bad:
            raise GatewayError(422, "unknown-model", "Unknown model", f"not in the catalog: {bad}")

    def _ser(obj: Any) -> dict[str, Any]:
        d = asdict(obj)
        d.pop("key_hash", None)
        d.pop("token_hash", None)
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()}

    _ser_admin = _ser

    async def audit(g: Gateway, p: Principal, org_id: str, action: str, target: str, **detail: Any) -> None:
        """Written after the change succeeds, before the response: a client that got a
        2xx can rely on the change being in the log. If the audit write fails the
        request fails (500) even though the change itself was applied; the error is logged."""
        await g.store.record_audit(AuditEvent(new_id(), now(), org_id, p.actor, action, target, detail))

    @app.get("/admin/v1/me")
    async def me(p: P) -> dict[str, Any]:
        return {"kind": p.kind, "role": p.role, "org_id": p.org_id, "label": p.label, "admin_id": p.admin_id}

    @app.post("/admin/v1/orgs", status_code=201)
    async def create_org(body: OrgIn, g: G, p: P) -> dict[str, Any]:
        operator_only(p, "creating organisations")
        _check_profile(g, body.default_profile)
        o = await g.store.create_org(body.name, body.default_profile)
        await audit(g, p, o.id, "org.create", o.id, name=o.name)
        return _ser(o)

    @app.get("/admin/v1/orgs")
    async def list_orgs(g: G, p: P) -> list[dict[str, Any]]:
        if p.kind == "org":
            return [_ser(await g.store.get_org(p.org_id or ""))]
        return [_ser(o) for o in await g.store.list_orgs()]

    @app.get("/admin/v1/orgs/{org_id}")
    async def get_org(org_id: str, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id)
        return _ser(await g.store.get_org(org_id))

    @app.patch("/admin/v1/orgs/{org_id}")
    async def patch_org(org_id: str, body: OrgPatch, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        fields = body.model_dump(exclude_unset=True)
        if "rpm_per_key" in fields:
            operator_only(p, "changing the per-key rate limit")
        _check_profile(g, body.default_profile)
        o = await g.store.update_org(org_id, **fields)
        await audit(g, p, org_id, "org.update", org_id, **fields)
        return _ser(o)

    @app.put("/admin/v1/orgs/{org_id}/guardrails")
    async def put_guardrails(org_id: str, body: GuardrailPolicy, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        policy = body.model_dump(mode="json")
        o = await g.store.update_org(org_id, guardrails=policy)
        await audit(g, p, org_id, "guardrails.update", org_id, **policy)
        return _ser(o)

    @app.post("/admin/v1/orgs/{org_id}/teams", status_code=201)
    async def create_team(org_id: str, body: TeamIn, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        _check_profile(g, body.default_profile)
        _check_models(g, body.approved_models)
        try:
            t = await g.store.create_team(org_id, body.name, body.approved_models, body.default_profile)
        except ValueError as e:
            raise GatewayError(409, "conflict", "Conflict", str(e)) from e
        await audit(g, p, org_id, "team.create", t.id, name=t.name, approved_models=body.approved_models)
        return _ser(t)

    @app.get("/admin/v1/orgs/{org_id}/teams")
    async def list_teams(org_id: str, g: G, p: P) -> list[dict[str, Any]]:
        allow(p, org_id)
        return [_ser(t) for t in await g.store.list_teams(org_id)]

    @app.patch("/admin/v1/orgs/{org_id}/teams/{team_id}")
    async def patch_team(org_id: str, team_id: str, body: TeamPatch, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        _check_profile(g, body.default_profile)
        _check_models(g, body.approved_models)
        fields = body.model_dump(exclude_unset=True)
        t = await g.store.update_team(org_id, team_id, **fields)
        await audit(g, p, org_id, "team.update", team_id, **fields)
        return _ser(t)

    @app.post("/admin/v1/orgs/{org_id}/teams/{team_id}/keys", status_code=201)
    async def create_key(org_id: str, team_id: str, body: KeyIn, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        _check_profile(g, body.default_profile)
        plain, prefix = mint_key()
        k = await g.store.create_key(
            org_id, team_id, body.label, hash_key(g.pepper, plain), prefix, body.default_profile
        )
        await audit(g, p, org_id, "key.create", k.id, team_id=team_id, label=k.label, prefix=k.prefix)
        return {**_ser(k), "key": plain, "note": "shown once; store it now"}

    @app.get("/admin/v1/orgs/{org_id}/keys")
    async def list_keys(org_id: str, g: G, p: P, team_id: str | None = None) -> list[dict[str, Any]]:
        allow(p, org_id)
        return [_ser(k) for k in await g.store.list_keys(org_id, team_id)]

    @app.delete("/admin/v1/orgs/{org_id}/keys/{key_id}")
    async def revoke_key(org_id: str, key_id: str, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        k = await g.store.revoke_key(org_id, key_id)
        await g.on_revoke(k)
        await audit(g, p, org_id, "key.revoke", key_id, prefix=k.prefix, label=k.label)
        return _ser(k)

    @app.put("/admin/v1/orgs/{org_id}/budgets")
    async def put_budget(org_id: str, body: BudgetIn, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        b = await g.store.set_budget(org_id, body.scope, body.scope_id, body.period, body.limit_usd, body.hard)
        await audit(g, p, org_id, "budget.put", b.id, **body.model_dump())
        return _ser(b)

    @app.get("/admin/v1/orgs/{org_id}/budgets")
    async def list_budgets(org_id: str, g: G, p: P) -> list[dict[str, Any]]:
        allow(p, org_id)
        return [_ser(b) for b in await g.store.list_budgets(org_id)]

    @app.delete("/admin/v1/orgs/{org_id}/budgets/{budget_id}", status_code=204)
    async def delete_budget(org_id: str, budget_id: str, g: G, p: P) -> Response:
        allow(p, org_id, write=True)
        await g.store.delete_budget(org_id, budget_id)
        await audit(g, p, org_id, "budget.delete", budget_id)
        return Response(status_code=204)

    @app.get("/admin/v1/orgs/{org_id}/usage")
    async def org_usage(
        org_id: str, g: G, p: P,
        since: datetime | None = None, until: datetime | None = None,
        events: bool = Query(default=False, description="include raw events (max 1000)"),
    ) -> dict[str, Any]:  # fmt: skip
        allow(p, org_id)
        await g.store.get_org(org_id)
        s = since or period_start("month")
        u = until or (now() + timedelta(seconds=1))
        rows = await g.store.usage_rows(org_id, s, u)
        out = {"since": s.isoformat(), "until": u.isoformat(), **summarize(rows)}
        if events:
            out["events"] = [event_dict(e) for e in rows[-1000:]]
        return out

    # -- org admins and the audit log ----------------------------------------------------
    @app.post("/admin/v1/orgs/{org_id}/admins", status_code=201)
    async def create_admin(org_id: str, body: AdminIn, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        plain, prefix = mint_key(ADMIN_PREFIX)
        a = await g.store.create_admin(org_id, body.label, body.role, hash_key(g.pepper, plain), prefix)
        await audit(g, p, org_id, "admin.create", a.id, label=a.label, role=a.role, prefix=a.prefix)
        return {**_ser_admin(a), "token": plain, "note": "shown once; store it now"}

    @app.get("/admin/v1/orgs/{org_id}/admins")
    async def list_admins(org_id: str, g: G, p: P) -> list[dict[str, Any]]:
        allow(p, org_id)
        return [_ser_admin(a) for a in await g.store.list_admins(org_id)]

    @app.delete("/admin/v1/orgs/{org_id}/admins/{admin_id}")
    async def revoke_admin(org_id: str, admin_id: str, g: G, p: P) -> dict[str, Any]:
        allow(p, org_id, write=True)
        a = await g.store.revoke_admin(org_id, admin_id)
        await audit(g, p, org_id, "admin.revoke", admin_id, label=a.label, prefix=a.prefix)
        return _ser_admin(a)

    @app.get("/admin/v1/orgs/{org_id}/audit")
    async def list_audit(
        org_id: str, g: G, p: P,
        limit: int = Query(default=100, ge=1, le=500), before: datetime | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        allow(p, org_id)
        await g.store.get_org(org_id)
        rows = await g.store.list_audit(org_id, limit, before)
        return {
            "events": [{**asdict(e), "created_at": e.created_at.isoformat()} for e in rows],
            "next_before": rows[-1].created_at.isoformat() if len(rows) == limit else None,
        }

    @app.get("/admin/v1/catalog")
    async def catalog_view(g: G, p: P) -> dict[str, Any]:
        c = g.catalog
        return {
            "version": c.version,
            "baseline_model": c.baseline_model,
            "default_profile": c.default_profile,
            "models": [
                {"id": m.id, "provider": m.provider, "tier": m.tier, "wires": [w.value for w in m.wires],
                 "price": m.price.model_dump(), "price_source": m.price_source, "context_window": m.context_window,
                 "provider_ready": g.upstream.configured(m.provider)}
                for m in c.models
            ],
            "profiles": [{"id": p.id, "name": p.name, "strategy": p.strategy.value, "candidates": p.candidates,
                          "fallback": p.fallback, "background_model": p.background_model} for p in c.profiles],
        }  # fmt: skip

    @app.get("/admin/v1/routing")
    async def routing(g: G, p: P) -> dict[str, Any]:
        return {
            "catalog_version": g.catalog.version,
            "evidence": {"suite": g.evidence.suite, "generated_at": g.evidence.generated_at},
            "profiles": {
                p.id: {"strategy": p.strategy.value, "routes": explain(g.catalog, p, g.evidence)}
                for p in g.catalog.profiles
            },
        }

    _mount_dashboard(app)
    return app


DASHBOARD_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


def _mount_dashboard(app: FastAPI) -> None:
    """Serve the built dashboard (web/route-dashboard/dist) at /dashboard when present.
    Same origin as the admin API, strict CSP, no third-party code."""
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    root = Path(
        os.environ.get("ROUTE_DASHBOARD_DIR", Path(__file__).resolve().parents[3] / "web" / "route-dashboard" / "dist")
    )
    if not (root / "index.html").exists():
        return
    app.mount("/dashboard", StaticFiles(directory=root, html=True), name="dashboard")

    @app.middleware("http")
    async def _dashboard_headers(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/dashboard"):
            resp.headers["content-security-policy"] = DASHBOARD_CSP
            resp.headers["x-content-type-options"] = "nosniff"
            resp.headers["referrer-policy"] = "no-referrer"
            resp.headers["x-frame-options"] = "DENY"
            if request.url.path.endswith((".js", ".css")) and "/assets/" in request.url.path:
                resp.headers["cache-control"] = "public, max-age=31536000, immutable"  # hashed file names
            else:
                resp.headers["cache-control"] = "no-cache"
        return resp


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("ROUTE_HOST", "127.0.0.1"),
        port=int(os.environ.get("ROUTE_PORT", "8090")),
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
