"""Fast shared state: sticky sessions, API-key lookups, spend counters, rate limits.

`RedisCache` in production (shared by every gateway process), `MemoryCache` for
tests and single-process development. Everything here is a cache or a counter
that can be rebuilt from Postgres; losing Redis costs speed, not data.
"""

from __future__ import annotations

import time
from typing import Protocol

# Token bucket in one round trip, atomic across gateway processes.
_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local b = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(b[1]) or burst
local ts = tonumber(b[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
local wait = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  wait = (1 - tokens) / rate
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / rate) + 60)
return {allowed, tostring(wait)}
"""


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr_float(self, key: str, amount: float, ttl_s: int) -> float: ...
    async def get_float(self, key: str) -> float | None: ...
    async def take(self, key: str, per_minute: int) -> tuple[bool, float]: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class MemoryCache:
    def __init__(self, clock=time.monotonic) -> None:
        self._d: dict[str, tuple[str, float]] = {}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._clock = clock

    def _live(self, key: str) -> str | None:
        v = self._d.get(key)
        if v is None:
            return None
        if v[1] < self._clock():
            del self._d[key]
            return None
        return v[0]

    async def get(self, key):
        return self._live(key)

    async def set(self, key, value, ttl_s):
        self._d[key] = (value, self._clock() + ttl_s)

    async def delete(self, key):
        self._d.pop(key, None)

    async def incr_float(self, key, amount, ttl_s):
        cur = float(self._live(key) or 0.0) + amount
        self._d[key] = (repr(cur), self._clock() + ttl_s)
        return cur

    async def get_float(self, key):
        v = self._live(key)
        return float(v) if v is not None else None

    async def take(self, key, per_minute):
        rate, burst, t = per_minute / 60.0, float(per_minute), self._clock()
        tokens, ts = self._buckets.get(key, (burst, t))
        tokens = min(burst, tokens + (t - ts) * rate)
        if tokens >= 1:
            self._buckets[key] = (tokens - 1, t)
            return True, 0.0
        self._buckets[key] = (tokens, t)
        return False, (1 - tokens) / rate

    async def ping(self):
        return True

    async def close(self):
        return None


class RedisCache:
    def __init__(self, client) -> None:  # redis.asyncio.Redis
        self.r = client
        self._bucket = client.register_script(_BUCKET_LUA)

    @classmethod
    def from_url(cls, url: str) -> RedisCache:
        import redis.asyncio as redis

        return cls(redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2))

    async def get(self, key):
        return await self.r.get(key)

    async def set(self, key, value, ttl_s):
        await self.r.set(key, value, ex=ttl_s)

    async def delete(self, key):
        await self.r.delete(key)

    async def incr_float(self, key, amount, ttl_s):
        async with self.r.pipeline(transaction=True) as p:
            p.incrbyfloat(key, amount)
            p.expire(key, ttl_s)
            v, _ = await p.execute()
        return float(v)

    async def get_float(self, key):
        v = await self.r.get(key)
        return float(v) if v is not None else None

    async def take(self, key, per_minute):
        allowed, wait = await self._bucket(keys=[key], args=[per_minute / 60.0, per_minute, time.time()])
        return bool(int(allowed)), float(wait)

    async def ping(self):
        try:
            return bool(await self.r.ping())
        except Exception:  # noqa: BLE001 - health check
            return False

    async def close(self):
        await self.r.aclose()
