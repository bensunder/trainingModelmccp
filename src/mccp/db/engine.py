"""Engine and tenant-scoped transactions.

All tenant data access goes through `tenant_tx`, which sets
`app.tenant_id` for the duration of the transaction (SET LOCAL semantics via
set_config(..., true)). Postgres RLS then filters every statement, so a
missing WHERE clause cannot leak another tenant's rows.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from mccp.settings import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        json_serializer=lambda v: json.dumps(v, separators=(",", ":"), allow_nan=False),
        connect_args={"server_settings": {"statement_timeout": str(settings.db_statement_timeout_ms)}},
    )


@asynccontextmanager
async def tenant_tx(engine: AsyncEngine, tenant_id: uuid.UUID | str) -> AsyncIterator[AsyncConnection]:
    tid = str(uuid.UUID(str(tenant_id)))  # validates
    async with engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tid})
        yield conn


@asynccontextmanager
async def system_tx(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Transaction with no tenant context: only SECURITY DEFINER entry points
    and the global `recipes` table are visible."""
    async with engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.tenant_id', '', true)"))
        yield conn


def jsonb(value: Any) -> str:
    """Serialize a value for a `CAST(:param AS jsonb)` bind."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False, default=_default)


def _default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(type(obj).__name__)


def as_obj(value: Any) -> Any:
    """jsonb columns decode to Python objects with the asyncpg dialect; this
    also accepts raw strings so code is robust to codec configuration."""
    if isinstance(value, (str, bytes)):
        return json.loads(value)
    return value
