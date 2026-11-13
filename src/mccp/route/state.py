"""Durable state: organisations, teams, API keys, budgets, usage events, org
admins and the admin audit log.

Two implementations of one interface: `MemoryStore` (tests, local development)
and `PgStore` (production; schema `route`, its own append-only migrations). Every
query that reads tenant data takes the org id and filters on it; tenant isolation
is tested, not assumed.

API keys (`rk_`) and org-admin tokens (`ra_`) are stored as HMAC-SHA256 under a
server-side pepper, never in clear. The plaintext is returned once, at creation.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol

KEY_PREFIX = "rk_"
ADMIN_PREFIX = "ra_"
ROLES = ("owner", "viewer")


def now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Org:
    id: str
    name: str
    created_at: datetime
    guardrails: dict[str, Any] = field(default_factory=dict)  # GuardrailPolicy fields
    default_profile: str | None = None
    rpm_per_key: int = 600


@dataclass(frozen=True)
class Team:
    id: str
    org_id: str
    name: str
    created_at: datetime
    approved_models: tuple[str, ...] | None = None  # None: every catalog model
    default_profile: str | None = None


@dataclass(frozen=True)
class ApiKey:
    id: str
    org_id: str
    team_id: str
    label: str  # who/what uses it (developer, CI job)
    prefix: str  # first characters, shown in UIs and logs to identify a key
    key_hash: str
    created_at: datetime
    revoked_at: datetime | None = None
    default_profile: str | None = None


@dataclass(frozen=True)
class Budget:
    id: str
    org_id: str
    scope: str  # org | team | key
    scope_id: str
    period: str  # day | month
    limit_usd: float
    hard: bool  # hard: requests are refused once spent; soft: reported only


@dataclass(frozen=True)
class UsageEvent:
    id: str
    request_id: str
    created_at: datetime
    org_id: str
    team_id: str
    key_id: str
    wire: str
    requested_model: str
    profile: str | None
    task: str | None
    model: str | None  # model that served it (None: nothing served)
    provider: str | None
    status: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usage_reported: bool = False
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    latency_ms: int = 0
    ttft_ms: int | None = None
    attempts: int = 1
    route_reason: str = ""
    guardrail: dict[str, Any] = field(default_factory=dict)
    stream: bool = False


@dataclass(frozen=True)
class OrgAdmin:
    """A customer-side administrator of one organisation. owner: manage everything in
    the org (teams, keys, budgets, guardrails, other admins); viewer: read only."""

    id: str
    org_id: str
    label: str  # who holds it (person, team)
    role: str  # owner | viewer
    prefix: str
    token_hash: str
    created_at: datetime
    revoked_at: datetime | None = None


@dataclass(frozen=True)
class AuditEvent:
    """One admin change: who (operator or admin prefix), what, on which object. Never
    carries secrets: created keys/tokens are recorded by id and prefix only."""

    id: str
    created_at: datetime
    org_id: str
    actor: str  # "operator" | "admin:<id>"
    action: str  # e.g. key.create, budget.delete
    target: str  # object id
    detail: dict[str, Any] = field(default_factory=dict)


def hash_key(pepper: bytes, plaintext: str) -> str:
    return hmac.new(pepper, plaintext.encode(), hashlib.sha256).hexdigest()


def mint_key(prefix: str = KEY_PREFIX) -> tuple[str, str]:
    """(plaintext, prefix). 32 random bytes; the prefix identifies, never authenticates."""
    body = secrets.token_urlsafe(32)
    plain = f"{prefix}{body}"
    return plain, plain[: len(prefix) + 8]


def period_start(period: str, at: datetime | None = None) -> datetime:
    at = at or now()
    if period == "day":
        return at.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        return at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unknown period {period}")


def period_end(period: str, at: datetime | None = None) -> datetime:
    start = period_start(period, at)
    if period == "day":
        from datetime import timedelta

        return start + timedelta(days=1)
    return start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)


class NotFound(LookupError):
    pass


class Store(Protocol):
    async def create_org(self, name: str, default_profile: str | None = None) -> Org: ...
    async def get_org(self, org_id: str) -> Org: ...
    async def list_orgs(self) -> list[Org]: ...
    async def update_org(self, org_id: str, **fields: Any) -> Org: ...
    async def create_team(
        self, org_id: str, name: str, approved_models: list[str] | None, default_profile: str | None
    ) -> Team: ...
    async def get_team(self, org_id: str, team_id: str) -> Team: ...
    async def list_teams(self, org_id: str) -> list[Team]: ...
    async def update_team(self, org_id: str, team_id: str, **fields: Any) -> Team: ...
    async def create_key(
        self, org_id: str, team_id: str, label: str, key_hash: str, prefix: str, default_profile: str | None
    ) -> ApiKey: ...
    async def key_by_hash(self, key_hash: str) -> ApiKey | None: ...
    async def list_keys(self, org_id: str, team_id: str | None = None) -> list[ApiKey]: ...
    async def revoke_key(self, org_id: str, key_id: str) -> ApiKey: ...
    async def set_budget(
        self, org_id: str, scope: str, scope_id: str, period: str, limit_usd: float, hard: bool
    ) -> Budget: ...
    async def list_budgets(self, org_id: str) -> list[Budget]: ...
    async def delete_budget(self, org_id: str, budget_id: str) -> None: ...
    async def create_admin(self, org_id: str, label: str, role: str, token_hash: str, prefix: str) -> OrgAdmin: ...
    async def admin_by_hash(self, token_hash: str) -> OrgAdmin | None: ...
    async def list_admins(self, org_id: str) -> list[OrgAdmin]: ...
    async def revoke_admin(self, org_id: str, admin_id: str) -> OrgAdmin: ...
    async def record_audit(self, event: AuditEvent) -> None: ...
    async def list_audit(self, org_id: str, limit: int = 100, before: datetime | None = None) -> list[AuditEvent]: ...
    async def record_usage(self, events: list[UsageEvent]) -> None: ...
    async def spend(self, org_id: str, scope: str, scope_id: str, since: datetime) -> float: ...
    async def usage_rows(
        self, org_id: str, since: datetime, until: datetime, key_id: str | None = None
    ) -> list[UsageEvent]: ...
    async def close(self) -> None: ...


class MemoryStore:
    """Thread-unsafe, single-process. Same semantics as PgStore (tested alike)."""

    def __init__(self) -> None:
        self.orgs: dict[str, Org] = {}
        self.teams: dict[str, Team] = {}
        self.keys: dict[str, ApiKey] = {}
        self.budgets: dict[str, Budget] = {}
        self.usage: list[UsageEvent] = []
        self.admins: dict[str, OrgAdmin] = {}
        self.audit: list[AuditEvent] = []

    async def create_org(self, name: str, default_profile: str | None = None) -> Org:
        o = Org(new_id(), name, now(), default_profile=default_profile)
        self.orgs[o.id] = o
        return o

    async def list_orgs(self) -> list[Org]:
        return sorted(self.orgs.values(), key=lambda o: o.name)

    async def get_org(self, org_id: str) -> Org:
        if org_id not in self.orgs:
            raise NotFound("org")
        return self.orgs[org_id]

    async def update_org(self, org_id: str, **fields: Any) -> Org:
        o = replace(await self.get_org(org_id), **fields)
        self.orgs[org_id] = o
        return o

    async def create_team(self, org_id, name, approved_models, default_profile) -> Team:
        await self.get_org(org_id)
        if any(t.org_id == org_id and t.name == name for t in self.teams.values()):
            raise ValueError(f"team {name} exists")
        t = Team(
            new_id(),
            org_id,
            name,
            now(),
            tuple(approved_models) if approved_models is not None else None,
            default_profile,
        )
        self.teams[t.id] = t
        return t

    async def get_team(self, org_id, team_id) -> Team:
        t = self.teams.get(team_id)
        if t is None or t.org_id != org_id:
            raise NotFound("team")
        return t

    async def list_teams(self, org_id) -> list[Team]:
        return sorted((t for t in self.teams.values() if t.org_id == org_id), key=lambda t: t.name)

    async def update_team(self, org_id, team_id, **fields) -> Team:
        t = await self.get_team(org_id, team_id)
        if "approved_models" in fields and fields["approved_models"] is not None:
            fields["approved_models"] = tuple(fields["approved_models"])
        t = replace(t, **fields)
        self.teams[t.id] = t
        return t

    async def create_key(self, org_id, team_id, label, key_hash, prefix, default_profile) -> ApiKey:
        await self.get_team(org_id, team_id)
        k = ApiKey(new_id(), org_id, team_id, label, prefix, key_hash, now(), None, default_profile)
        self.keys[k.id] = k
        return k

    async def key_by_hash(self, key_hash) -> ApiKey | None:
        return next((k for k in self.keys.values() if hmac.compare_digest(k.key_hash, key_hash)), None)

    async def list_keys(self, org_id, team_id=None) -> list[ApiKey]:
        return [k for k in self.keys.values() if k.org_id == org_id and (team_id is None or k.team_id == team_id)]

    async def revoke_key(self, org_id, key_id) -> ApiKey:
        k = self.keys.get(key_id)
        if k is None or k.org_id != org_id:
            raise NotFound("key")
        k = replace(k, revoked_at=k.revoked_at or now())
        self.keys[k.id] = k
        return k

    async def set_budget(self, org_id, scope, scope_id, period, limit_usd, hard) -> Budget:
        await self.get_org(org_id)
        for b in self.budgets.values():
            if (b.org_id, b.scope, b.scope_id, b.period) == (org_id, scope, scope_id, period):
                nb = replace(b, limit_usd=limit_usd, hard=hard)
                self.budgets[b.id] = nb
                return nb
        b = Budget(new_id(), org_id, scope, scope_id, period, limit_usd, hard)
        self.budgets[b.id] = b
        return b

    async def list_budgets(self, org_id) -> list[Budget]:
        return [b for b in self.budgets.values() if b.org_id == org_id]

    async def delete_budget(self, org_id, budget_id) -> None:
        b = self.budgets.get(budget_id)
        if b is None or b.org_id != org_id:
            raise NotFound("budget")
        del self.budgets[budget_id]

    async def create_admin(self, org_id, label, role, token_hash, prefix) -> OrgAdmin:
        await self.get_org(org_id)
        if role not in ROLES:
            raise ValueError(f"unknown role {role}")
        a = OrgAdmin(new_id(), org_id, label, role, prefix, token_hash, now())
        self.admins[a.id] = a
        return a

    async def admin_by_hash(self, token_hash) -> OrgAdmin | None:
        return next((a for a in self.admins.values() if hmac.compare_digest(a.token_hash, token_hash)), None)

    async def list_admins(self, org_id) -> list[OrgAdmin]:
        return sorted((a for a in self.admins.values() if a.org_id == org_id), key=lambda a: a.created_at)

    async def revoke_admin(self, org_id, admin_id) -> OrgAdmin:
        a = self.admins.get(admin_id)
        if a is None or a.org_id != org_id:
            raise NotFound("admin")
        a = replace(a, revoked_at=a.revoked_at or now())
        self.admins[a.id] = a
        return a

    async def record_audit(self, event) -> None:
        self.audit.append(event)

    async def list_audit(self, org_id, limit=100, before=None) -> list[AuditEvent]:
        rows = [e for e in self.audit if e.org_id == org_id and (before is None or e.created_at < before)]
        return sorted(rows, key=lambda e: e.created_at, reverse=True)[:limit]

    async def record_usage(self, events) -> None:
        self.usage.extend(events)

    async def spend(self, org_id, scope, scope_id, since) -> float:
        col = {"org": "org_id", "team": "team_id", "key": "key_id"}[scope]
        return sum(
            e.cost_usd
            for e in self.usage
            if e.org_id == org_id and getattr(e, col) == scope_id and e.created_at >= since
        )

    async def usage_rows(self, org_id, since, until, key_id=None) -> list[UsageEvent]:
        return [
            e for e in self.usage
            if e.org_id == org_id and since <= e.created_at < until and (key_id is None or e.key_id == key_id)
        ]  # fmt: skip

    async def close(self) -> None:
        return None


def summarize(rows: list[UsageEvent]) -> dict[str, Any]:
    """Spend, tokens, cache hit rate, savings, and breakdowns for dashboards."""
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_day: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_team: dict[str, float] = defaultdict(float)
    by_task: dict[str, int] = defaultdict(int)
    by_day_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    spend = baseline = 0.0
    inp = out = cread = cwrite = 0
    errors = unreported = 0
    for e in rows:
        spend += e.cost_usd
        baseline += e.baseline_cost_usd
        inp += e.input_tokens
        out += e.output_tokens
        cread += e.cache_read_tokens
        cwrite += e.cache_write_tokens
        errors += e.status >= 400
        unreported += e.status < 400 and not e.usage_reported
        if e.model:
            m = by_model[e.model]
            m["requests"] += 1
            m["cost_usd"] += e.cost_usd
            m["tokens"] += e.input_tokens + e.output_tokens + e.cache_read_tokens + e.cache_write_tokens
        day = e.created_at.date().isoformat()
        d = by_day[day]
        if e.model:
            by_day_model[day][e.model] += e.cost_usd
        d["cost_usd"] += e.cost_usd
        d["requests"] += 1
        by_team[e.team_id] += e.cost_usd
        if e.task:
            by_task[e.task] += 1
    prompt = inp + cread + cwrite
    return {
        "requests": len(rows),
        "errors": errors,
        "responses_without_usage": unreported,
        "spend_usd": round(spend, 6),
        "baseline_usd": round(baseline, 6),
        "saved_usd": round(max(0.0, baseline - spend), 6),
        "saved_pct": round((baseline - spend) / baseline, 4) if baseline > 0 else None,
        "tokens": {"input": inp, "output": out, "cache_read": cread, "cache_write": cwrite},
        "cache_hit_rate": round(cread / prompt, 4) if prompt else None,
        "by_model": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sorted(by_model.items())},
        "by_day": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sorted(by_day.items())},
        "by_team": {k: round(v, 6) for k, v in sorted(by_team.items())},
        "by_task": dict(sorted(by_task.items())),
        "by_day_model": {d: {m: round(v, 6) for m, v in sorted(ms.items())} for d, ms in sorted(by_day_model.items())},
    }


def event_dict(e: UsageEvent) -> dict[str, Any]:
    d = asdict(e)
    d["created_at"] = e.created_at.isoformat()
    return d
