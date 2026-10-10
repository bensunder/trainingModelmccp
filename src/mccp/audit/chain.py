"""Tamper-evident audit chain (NFR-3).

Each tenant has an independent chain. Appends are serialized per tenant by a
transaction-scoped advisory lock, so `seq` is gap-free and every event links
to its predecessor:

    event_sha = sha256(prev_event_sha || 0x00 || canonical(body))

The append happens in the same transaction as the state change it records,
so a committed state change always has its audit event and vice versa.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from mccp.db.engine import as_obj, jsonb
from mccp.domain.hashing import GENESIS, canonical_json, chain_link


def _body(tenant_id: str, seq: int, ts: datetime, actor: str, action: str, rtype: str, rid: str, data: Any) -> dict:
    return {
        "tenant_id": tenant_id,
        "seq": seq,
        "ts": ts.astimezone(UTC).isoformat(timespec="microseconds"),
        "actor": actor,
        "action": action,
        "resource_type": rtype,
        "resource_id": rid,
        "data": data,
    }


async def append_event(
    conn: AsyncConnection,
    tenant_id: uuid.UUID | str,
    *,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tid = str(tenant_id)
    data = data or {}
    await conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": "mccp-events:" + tid})
    row = (
        await conn.execute(
            text("SELECT seq, event_sha FROM events WHERE tenant_id = CAST(:t AS uuid) ORDER BY seq DESC LIMIT 1"),
            {"t": tid},
        )
    ).first()
    seq, prev = (row.seq + 1, row.event_sha) if row else (1, GENESIS)
    # Python datetimes carry microseconds, which is exactly what timestamptz
    # stores, so the hashed ISO string roundtrips on verification.
    ts = datetime.now(UTC)
    body = _body(tid, seq, ts, actor, action, resource_type, resource_id, data)
    event_sha = chain_link(prev, canonical_json(body))
    await conn.execute(
        text(
            """INSERT INTO events (tenant_id, seq, ts, actor, action, resource_type, resource_id,
                                   data, prev_event_sha, event_sha)
               VALUES (CAST(:t AS uuid), :seq, :ts, :actor, :action, :rtype, :rid,
                       CAST(:data AS jsonb), :prev, :sha)"""
        ),
        {
            "t": tid,
            "seq": seq,
            "ts": ts,
            "actor": actor,
            "action": action,
            "rtype": resource_type,
            "rid": resource_id,
            "data": jsonb(data),
            "prev": prev,
            "sha": event_sha,
        },
    )
    return {"seq": seq, "event_sha": event_sha, **body}


@dataclass
class ChainVerification:
    ok: bool
    events_checked: int
    head_sha: str
    first_bad_seq: int | None = None
    reason: str | None = None


async def verify_chain(conn: AsyncConnection, tenant_id: uuid.UUID | str, batch: int = 1000) -> ChainVerification:
    tid = str(tenant_id)
    prev = GENESIS
    expected_seq = 1
    checked = 0
    while True:
        rows = (
            await conn.execute(
                text(
                    """SELECT seq, ts, actor, action, resource_type, resource_id, data, prev_event_sha, event_sha
                         FROM events WHERE tenant_id = CAST(:t AS uuid) AND seq >= :from
                        ORDER BY seq LIMIT :n"""
                ),
                {"t": tid, "from": expected_seq, "n": batch},
            )
        ).all()
        if not rows:
            return ChainVerification(True, checked, prev)
        for r in rows:
            if r.seq != expected_seq:
                return ChainVerification(False, checked, prev, expected_seq, f"missing event seq {expected_seq}")
            if r.prev_event_sha != prev:
                return ChainVerification(False, checked, prev, r.seq, "prev_event_sha does not link to predecessor")
            body = _body(tid, r.seq, r.ts, r.actor, r.action, r.resource_type, r.resource_id, as_obj(r.data))
            if chain_link(prev, canonical_json(body)) != r.event_sha:
                return ChainVerification(False, checked, prev, r.seq, "event content does not match its hash")
            prev = r.event_sha
            expected_seq += 1
            checked += 1
