"""Versioned SQL migrations.

Files `sql/NNNN_name.sql` are applied in order, each in its own transaction,
as the owner role. Applied files are recorded with their SHA-256; editing an
applied migration is detected and refused (migrations are append-only).
"""

from __future__ import annotations

import hashlib
import logging
import re
from importlib import resources

import asyncpg

log = logging.getLogger(__name__)

_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    pass


def _migrations() -> list[tuple[int, str, str]]:
    out = []
    for entry in resources.files("mccp.db").joinpath("sql").iterdir():
        m = _NAME.match(entry.name)
        if m:
            sql = entry.read_text(encoding="utf-8")
            out.append((int(m.group(1)), entry.name, sql))
    out.sort()
    versions = [v for v, _, _ in out]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration numbering must be contiguous from 0001, got {versions}")
    return out


def _asyncpg_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def migrate(owner_dsn: str) -> list[str]:
    """Apply pending migrations. Returns the names applied."""
    conn = await asyncpg.connect(_asyncpg_dsn(owner_dsn))
    applied_now: list[str] = []
    try:
        # Serialize concurrent migrators (e.g. several API replicas starting).
        await conn.execute("SELECT pg_advisory_lock(727274)")
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS mccp_schema_migrations (
                   version integer PRIMARY KEY,
                   name text NOT NULL,
                   sha256 text NOT NULL,
                   applied_at timestamptz NOT NULL DEFAULT now())"""
        )
        rows = await conn.fetch("SELECT version, name, sha256 FROM mccp_schema_migrations")
        applied = {r["version"]: (r["name"], r["sha256"]) for r in rows}
        for version, name, sql in _migrations():
            digest = hashlib.sha256(sql.encode()).hexdigest()
            if version in applied:
                if applied[version][1] != digest:
                    raise MigrationError(f"applied migration {name} was modified (sha mismatch)")
                continue
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO mccp_schema_migrations(version, name, sha256) VALUES ($1, $2, $3)",
                    version,
                    name,
                    digest,
                )
            log.info("applied migration %s", name)
            applied_now.append(name)
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock(727274)")
        finally:
            await conn.close()
    return applied_now
