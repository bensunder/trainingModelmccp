"""Postgres implementation of `Store` (schema `route`).

Migrations live in this module, are append-only, and are recorded with their
SHA-256 in `route.schema_migrations`; an edited applied migration is refused.
Every tenant query filters on org_id in SQL; usage rows are written in batches by
the gateway's background writer (`UsageWriter`), never on the request path.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import asyncpg

from .state import ApiKey, Budget, NotFound, Org, Team, UsageEvent, new_id, now

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_init",
        """
        CREATE SCHEMA IF NOT EXISTS route;
        CREATE TABLE route.orgs (
          id uuid PRIMARY KEY, name text NOT NULL, created_at timestamptz NOT NULL,
          guardrails jsonb NOT NULL DEFAULT '{}', default_profile text,
          rpm_per_key integer NOT NULL DEFAULT 600 CHECK (rpm_per_key > 0)
        );
        CREATE TABLE route.teams (
          id uuid PRIMARY KEY, org_id uuid NOT NULL REFERENCES route.orgs(id) ON DELETE CASCADE,
          name text NOT NULL, created_at timestamptz NOT NULL,
          approved_models text[], default_profile text, UNIQUE (org_id, name)
        );
        CREATE TABLE route.api_keys (
          id uuid PRIMARY KEY, org_id uuid NOT NULL REFERENCES route.orgs(id) ON DELETE CASCADE,
          team_id uuid NOT NULL REFERENCES route.teams(id) ON DELETE CASCADE,
          label text NOT NULL, prefix text NOT NULL, key_hash text NOT NULL UNIQUE,
          created_at timestamptz NOT NULL, revoked_at timestamptz, default_profile text
        );
        CREATE INDEX ON route.api_keys (org_id, team_id);
        CREATE TABLE route.budgets (
          id uuid PRIMARY KEY, org_id uuid NOT NULL REFERENCES route.orgs(id) ON DELETE CASCADE,
          scope text NOT NULL CHECK (scope IN ('org','team','key')), scope_id uuid NOT NULL,
          period text NOT NULL CHECK (period IN ('day','month')),
          limit_usd numeric(14,4) NOT NULL CHECK (limit_usd >= 0), hard boolean NOT NULL,
          UNIQUE (org_id, scope, scope_id, period)
        );
        CREATE TABLE route.usage (
          id uuid PRIMARY KEY, request_id text NOT NULL, created_at timestamptz NOT NULL,
          org_id uuid NOT NULL, team_id uuid NOT NULL, key_id uuid NOT NULL,
          wire text NOT NULL, requested_model text NOT NULL, profile text, task text,
          model text, provider text, status integer NOT NULL,
          input_tokens bigint NOT NULL, output_tokens bigint NOT NULL,
          cache_read_tokens bigint NOT NULL, cache_write_tokens bigint NOT NULL,
          usage_reported boolean NOT NULL, cost_usd numeric(18,8) NOT NULL,
          baseline_cost_usd numeric(18,8) NOT NULL, latency_ms integer NOT NULL, ttft_ms integer,
          attempts integer NOT NULL, route_reason text NOT NULL, guardrail jsonb NOT NULL, stream boolean NOT NULL
        );
        CREATE INDEX ON route.usage (org_id, created_at);
        CREATE INDEX ON route.usage (org_id, team_id, created_at);
        CREATE INDEX ON route.usage (org_id, key_id, created_at);
        """,
    ),
]


async def migrate(conn: asyncpg.Connection) -> list[str]:
    await conn.execute("CREATE SCHEMA IF NOT EXISTS route")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS route.schema_migrations (name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    applied = {r["name"]: r["sha256"] for r in await conn.fetch("SELECT name, sha256 FROM route.schema_migrations")}
    done = []
    for name, sql in MIGRATIONS:
        digest = hashlib.sha256(sql.encode()).hexdigest()
        if name in applied:
            if applied[name] != digest:
                raise RuntimeError(f"route migration {name} was edited after it was applied")
            continue
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute("INSERT INTO route.schema_migrations (name, sha256) VALUES ($1, $2)", name, digest)
        done.append(name)
    return done


def _org(r: asyncpg.Record) -> Org:
    return Org(
        str(r["id"]),
        r["name"],
        r["created_at"],
        json.loads(r["guardrails"]) if isinstance(r["guardrails"], str) else dict(r["guardrails"]),
        r["default_profile"],
        r["rpm_per_key"],
    )


def _team(r: asyncpg.Record) -> Team:
    am = r["approved_models"]
    return Team(
        str(r["id"]),
        str(r["org_id"]),
        r["name"],
        r["created_at"],
        tuple(am) if am is not None else None,
        r["default_profile"],
    )


def _key(r: asyncpg.Record) -> ApiKey:
    return ApiKey(
        str(r["id"]),
        str(r["org_id"]),
        str(r["team_id"]),
        r["label"],
        r["prefix"],
        r["key_hash"],
        r["created_at"],
        r["revoked_at"],
        r["default_profile"],
    )


def _budget(r: asyncpg.Record) -> Budget:
    return Budget(
        str(r["id"]), str(r["org_id"]), r["scope"], str(r["scope_id"]), r["period"], float(r["limit_usd"]), r["hard"]
    )


_USAGE_COLS = [
    "id",
    "request_id",
    "created_at",
    "org_id",
    "team_id",
    "key_id",
    "wire",
    "requested_model",
    "profile",
    "task",
    "model",
    "provider",
    "status",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "usage_reported",
    "cost_usd",
    "baseline_cost_usd",
    "latency_ms",
    "ttft_ms",
    "attempts",
    "route_reason",
    "guardrail",
    "stream",
]


class PgStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str, min_size: int = 1, max_size: int = 10) -> PgStore:
        async def init(conn: asyncpg.Connection) -> None:
            await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")

        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=init, command_timeout=30)
        async with pool.acquire() as conn:
            await migrate(conn)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def _one(self, sql: str, *args: Any) -> asyncpg.Record:
        r = await self.pool.fetchrow(sql, *args)
        if r is None:
            raise NotFound(sql.split()[3] if len(sql.split()) > 3 else "row")
        return r

    async def create_org(self, name: str, default_profile: str | None = None) -> Org:
        r = await self._one(
            "INSERT INTO route.orgs (id, name, created_at, default_profile) VALUES ($1,$2,$3,$4) RETURNING *",
            new_id(), name, now(), default_profile,
        )  # fmt: skip
        return _org(r)

    async def get_org(self, org_id: str) -> Org:
        return _org(await self._one("SELECT * FROM route.orgs WHERE id = $1", org_id))

    async def update_org(self, org_id: str, **fields: Any) -> Org:
        allowed = {"name", "guardrails", "default_profile", "rpm_per_key"}
        sets = [k for k in fields if k in allowed]
        if not sets:
            return await self.get_org(org_id)
        sql = (
            "UPDATE route.orgs SET "
            + ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(sets))
            + " WHERE id = $1 RETURNING *"
        )
        return _org(await self._one(sql, org_id, *[fields[k] for k in sets]))

    async def create_team(self, org_id, name, approved_models, default_profile) -> Team:
        await self.get_org(org_id)
        try:
            r = await self._one(
                "INSERT INTO route.teams (id, org_id, name, created_at, approved_models, default_profile) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *",
                new_id(), org_id, name, now(), approved_models, default_profile,
            )  # fmt: skip
        except asyncpg.UniqueViolationError as e:
            raise ValueError(f"team {name} exists") from e
        return _team(r)

    async def get_team(self, org_id, team_id) -> Team:
        return _team(await self._one("SELECT * FROM route.teams WHERE org_id = $1 AND id = $2", org_id, team_id))

    async def list_teams(self, org_id) -> list[Team]:
        return [
            _team(r) for r in await self.pool.fetch("SELECT * FROM route.teams WHERE org_id = $1 ORDER BY name", org_id)
        ]

    async def update_team(self, org_id, team_id, **fields) -> Team:
        allowed = {"name", "approved_models", "default_profile"}
        sets = [k for k in fields if k in allowed]
        if not sets:
            return await self.get_team(org_id, team_id)
        vals = [list(fields[k]) if k == "approved_models" and fields[k] is not None else fields[k] for k in sets]
        sql = (
            "UPDATE route.teams SET "
            + ", ".join(f"{k} = ${i + 3}" for i, k in enumerate(sets))
            + " WHERE org_id = $1 AND id = $2 RETURNING *"
        )
        return _team(await self._one(sql, org_id, team_id, *vals))

    async def create_key(self, org_id, team_id, label, key_hash, prefix, default_profile) -> ApiKey:
        await self.get_team(org_id, team_id)
        r = await self._one(
            "INSERT INTO route.api_keys (id, org_id, team_id, label, prefix, key_hash, created_at, default_profile) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *",
            new_id(), org_id, team_id, label, prefix, key_hash, now(), default_profile,
        )  # fmt: skip
        return _key(r)

    async def key_by_hash(self, key_hash) -> ApiKey | None:
        r = await self.pool.fetchrow("SELECT * FROM route.api_keys WHERE key_hash = $1", key_hash)
        return _key(r) if r else None

    async def list_keys(self, org_id, team_id=None) -> list[ApiKey]:
        if team_id is None:
            rows = await self.pool.fetch("SELECT * FROM route.api_keys WHERE org_id = $1 ORDER BY created_at", org_id)
        else:
            rows = await self.pool.fetch(
                "SELECT * FROM route.api_keys WHERE org_id = $1 AND team_id = $2 ORDER BY created_at", org_id, team_id
            )
        return [_key(r) for r in rows]

    async def revoke_key(self, org_id, key_id) -> ApiKey:
        r = await self._one(
            "UPDATE route.api_keys SET revoked_at = COALESCE(revoked_at, $3) WHERE org_id = $1 AND id = $2 RETURNING *",
            org_id,
            key_id,
            now(),
        )
        return _key(r)

    async def set_budget(self, org_id, scope, scope_id, period, limit_usd, hard) -> Budget:
        await self.get_org(org_id)
        r = await self._one(
            "INSERT INTO route.budgets (id, org_id, scope, scope_id, period, limit_usd, hard) VALUES ($1,$2,$3,$4,$5,$6,$7) "
            "ON CONFLICT (org_id, scope, scope_id, period) DO UPDATE SET limit_usd = EXCLUDED.limit_usd, hard = EXCLUDED.hard RETURNING *",
            new_id(), org_id, scope, scope_id, period, limit_usd, hard,
        )  # fmt: skip
        return _budget(r)

    async def list_budgets(self, org_id) -> list[Budget]:
        return [_budget(r) for r in await self.pool.fetch("SELECT * FROM route.budgets WHERE org_id = $1", org_id)]

    async def delete_budget(self, org_id, budget_id) -> None:
        r = await self.pool.execute("DELETE FROM route.budgets WHERE org_id = $1 AND id = $2", org_id, budget_id)
        if r.endswith(" 0"):
            raise NotFound("budget")

    async def record_usage(self, events: list[UsageEvent]) -> None:
        if not events:
            return
        rows = [tuple(getattr(e, c) for c in _USAGE_COLS) for e in events]
        cols = ", ".join(_USAGE_COLS)
        params = ", ".join(f"${i + 1}" for i in range(len(_USAGE_COLS)))
        # one statement, one round trip per batch (batches are at most a few hundred rows)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                f"INSERT INTO route.usage ({cols}) VALUES ({params}) ON CONFLICT (id) DO NOTHING", rows
            )

    async def spend(self, org_id, scope, scope_id, since: datetime) -> float:
        col = {"org": "org_id", "team": "team_id", "key": "key_id"}[scope]
        v = await self.pool.fetchval(
            f"SELECT COALESCE(SUM(cost_usd), 0) FROM route.usage WHERE org_id = $1 AND {col} = $2 AND created_at >= $3",
            org_id, scope_id, since,
        )  # fmt: skip
        return float(v)

    async def usage_rows(self, org_id, since, until, key_id=None) -> list[UsageEvent]:
        sql = "SELECT * FROM route.usage WHERE org_id = $1 AND created_at >= $2 AND created_at < $3"
        args: list[Any] = [org_id, since, until]
        if key_id:
            sql += " AND key_id = $4"
            args.append(key_id)
        out = []
        for r in await self.pool.fetch(sql + " ORDER BY created_at", *args):
            d = {c: r[c] for c in _USAGE_COLS}
            for c in ("id", "org_id", "team_id", "key_id"):
                d[c] = str(d[c])
            d["cost_usd"], d["baseline_cost_usd"] = float(d["cost_usd"]), float(d["baseline_cost_usd"])
            out.append(UsageEvent(**d))
        return out
