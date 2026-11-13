"""One contract, two stores: MemoryStore and PgStore must behave identically,
including tenant isolation and the usage summary the dashboard reads."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from mccp.route.pgstore import MIGRATIONS, PgStore, migrate
from mccp.route.state import (
    ADMIN_PREFIX,
    AuditEvent,
    MemoryStore,
    NotFound,
    UsageEvent,
    hash_key,
    mint_key,
    new_id,
    now,
    summarize,
)  # fmt: skip
from tests.conftest import ADMIN_DSN, PG

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    if request.param == "memory":
        yield MemoryStore()
        return
    if not PG:
        pytest.skip(f"Postgres not reachable at {ADMIN_DSN}")
    import asyncpg

    name = f"route_test_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    s = await PgStore.connect(dsn)
    try:
        yield s
    finally:
        await s.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


def _event(org, team, key, cost=0.01, baseline=0.05, model="glm", at=None, **kw) -> UsageEvent:
    return UsageEvent(
        id=new_id(), request_id="req", created_at=at or now(), org_id=org, team_id=team, key_id=key,
        wire="anthropic", requested_model="auto-cost", profile="auto-cost", task="refactor", model=model,
        provider="zai", status=200, input_tokens=1000, output_tokens=100, cache_read_tokens=500,
        usage_reported=True, cost_usd=cost, baseline_cost_usd=baseline, latency_ms=900, ttft_ms=120,
        route_reason="r", guardrail={"findings": {"secrets:jwt": 1}}, **kw,
    )  # fmt: skip


async def test_orgs_teams_keys_and_isolation(store):
    a = await store.create_org("A", "auto-cost")
    b = await store.create_org("B")
    ta = await store.create_team(a.id, "eng", ["glm", "kimi"], None)
    tb = await store.create_team(b.id, "eng", None, "auto-balanced")
    assert ta.approved_models == ("glm", "kimi") and tb.approved_models is None
    with pytest.raises(ValueError):
        await store.create_team(a.id, "eng", None, None)
    with pytest.raises(NotFound):
        await store.get_team(a.id, tb.id)  # B's team through A's org
    plain, prefix = mint_key()
    k = await store.create_key(a.id, ta.id, "dev", hash_key(b"p" * 32, plain), prefix, None)
    assert (await store.key_by_hash(hash_key(b"p" * 32, plain))).id == k.id
    assert await store.key_by_hash(hash_key(b"q" * 32, plain)) is None  # pepper matters
    with pytest.raises(NotFound):
        await store.revoke_key(b.id, k.id)
    r = await store.revoke_key(a.id, k.id)
    assert r.revoked_at is not None and (await store.revoke_key(a.id, k.id)).revoked_at == r.revoked_at
    t2 = await store.update_team(a.id, ta.id, approved_models=["glm"], default_profile="auto-cost")
    assert t2.approved_models == ("glm",) and t2.default_profile == "auto-cost"
    o = await store.update_org(a.id, guardrails={"secrets": "block"}, rpm_per_key=60)
    assert o.guardrails == {"secrets": "block"} and o.rpm_per_key == 60
    assert [t.name for t in await store.list_teams(a.id)] == ["eng"] and len(await store.list_keys(b.id)) == 0


async def test_budgets_upsert_and_spend_by_scope(store):
    o = await store.create_org("A")
    t = await store.create_team(o.id, "eng", None, None)
    other = await store.create_org("B")
    ot = await store.create_team(other.id, "eng", None, None)
    k1, k2 = new_id(), new_id()
    b = await store.set_budget(o.id, "team", t.id, "month", 100, True)
    b2 = await store.set_budget(o.id, "team", t.id, "month", 50, False)
    assert b2.id == b.id and b2.limit_usd == 50 and not b2.hard and len(await store.list_budgets(o.id)) == 1
    await store.record_usage([
        _event(o.id, t.id, k1, cost=0.5), _event(o.id, t.id, k2, cost=0.25),
        _event(o.id, t.id, k1, cost=9.0, at=now() - timedelta(days=40)),
        _event(other.id, ot.id, new_id(), cost=100.0),
    ])  # fmt: skip
    since = now() - timedelta(days=1)
    assert await store.spend(o.id, "team", t.id, since) == pytest.approx(0.75)
    assert await store.spend(o.id, "key", k1, since) == pytest.approx(0.5)
    assert await store.spend(o.id, "org", o.id, since) == pytest.approx(0.75)  # B's spend never counted
    rows = await store.usage_rows(o.id, since, now() + timedelta(seconds=5))
    s = summarize(rows)
    assert s["requests"] == 2 and s["saved_usd"] == 0  # spent more than baseline: savings never negative
    assert s["spend_usd"] == pytest.approx(0.75) and s["baseline_usd"] == pytest.approx(0.10)
    assert s["cache_hit_rate"] == pytest.approx(500 / 1500, abs=1e-4) and s["by_model"]["glm"]["requests"] == 2
    assert rows[0].guardrail == {"findings": {"secrets:jwt": 1}}
    await store.delete_budget(o.id, b.id)
    with pytest.raises(NotFound):
        await store.delete_budget(o.id, b.id)


async def test_org_admins_and_audit_are_per_org(store):
    a = await store.create_org("A")
    b = await store.create_org("B")
    plain, prefix = mint_key(ADMIN_PREFIX)
    assert plain.startswith("ra_") and prefix.startswith("ra_")
    adm = await store.create_admin(a.id, "alice", "owner", hash_key(b"p" * 32, plain), prefix)
    assert (await store.admin_by_hash(hash_key(b"p" * 32, plain))).id == adm.id
    with pytest.raises(ValueError):
        await store.create_admin(a.id, "x", "superuser", "h", "ra_x")
    with pytest.raises(NotFound):
        await store.revoke_admin(b.id, adm.id)  # A's admin through B's org
    r = await store.revoke_admin(a.id, adm.id)
    assert r.revoked_at is not None and (await store.revoke_admin(a.id, adm.id)).revoked_at == r.revoked_at
    assert [x.id for x in await store.list_admins(a.id)] == [adm.id] and await store.list_admins(b.id) == []
    t0 = now()
    for i in range(3):
        await store.record_audit(AuditEvent(new_id(), t0 + timedelta(seconds=i), a.id, "operator", f"act.{i}", "t",
                                            {"n": i}))  # fmt: skip
    await store.record_audit(AuditEvent(new_id(), t0, b.id, "operator", "other", "t", {}))
    rows = await store.list_audit(a.id, limit=2)
    assert [e.action for e in rows] == ["act.2", "act.1"] and rows[0].detail == {"n": 2}  # newest first
    older = await store.list_audit(a.id, before=rows[-1].created_at)
    assert [e.action for e in older] == ["act.0"]
    assert [e.action for e in await store.list_audit(b.id)] == ["other"]


async def test_pg_migrations_are_append_only():
    if not PG:
        pytest.skip("no Postgres")
    import asyncpg

    name = f"route_mig_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    conn = await asyncpg.connect(ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}")
    try:
        assert await migrate(conn) == [m for m, _ in MIGRATIONS]
        assert await migrate(conn) == []  # idempotent
        await conn.execute("UPDATE route.schema_migrations SET sha256 = 'x' WHERE name = '0001_init'")
        with pytest.raises(RuntimeError, match="edited"):
            await migrate(conn)
    finally:
        await conn.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()
