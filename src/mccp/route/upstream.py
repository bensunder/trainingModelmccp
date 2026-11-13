"""Calls to model providers: auth, timeouts, circuit breaking, streaming passthrough.

Failover is only safe before the client has received a byte. So the response
status is checked before streaming starts; a connection failure, timeout or a
retryable status (429/5xx/529) at that point raises `Retryable` and the caller
tries the next model. Once bytes flow, the stream is passed through as-is and
never retried (a retry would duplicate tool calls and charges).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Catalog, Provider, Wire

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
PATHS = {Wire.ANTHROPIC: "/v1/messages", Wire.OPENAI_CHAT: "/chat/completions"}
# Client headers that change provider behaviour and are safe to forward.
FORWARD_HEADERS = {
    Wire.ANTHROPIC: ("anthropic-version", "anthropic-beta"),
    Wire.OPENAI_CHAT: (),
}
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class Retryable(Exception):
    def __init__(self, provider: str, reason: str, status: int | None = None) -> None:
        super().__init__(f"{provider}: {reason}")
        self.provider, self.reason, self.status = provider, reason, status


class ProviderUnavailable(Exception):
    pass


@dataclass
class Breaker:
    """Opens after `threshold` consecutive failures; after `cooldown_s` one trial
    request is let through (half-open); a success closes it."""

    threshold: int = 5
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None
    trial_in_flight: bool = False

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.cooldown_s and not self.trial_in_flight:
            self.trial_in_flight = True
            return True
        return False

    def success(self) -> None:
        self.failures, self.opened_at, self.trial_in_flight = 0, None, False

    def failure(self, now: float | None = None) -> None:
        self.failures += 1
        self.trial_in_flight = False
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic() if now is None else now

    @property
    def state(self) -> str:
        return "closed" if self.opened_at is None else "open"


@dataclass
class UpstreamResponse:
    status: int
    headers: dict[str, str]
    body: bytes | None = None  # non-streaming
    stream: AsyncIterator[bytes] | None = None  # streaming: raw bytes, already started
    _close: Any = None

    async def aclose(self) -> None:
        if self._close is not None:
            await self._close()


class Upstream:
    def __init__(self, catalog: Catalog, client: httpx.AsyncClient | None = None, env: dict[str, str] | None = None):
        self.catalog = catalog
        self._client = client or httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=200))
        env = dict(os.environ) if env is None else env
        self._keys = {p.id: env.get(p.api_key_env, "") for p in catalog.providers}
        self.breakers = {p.id: Breaker() for p in catalog.providers}

    def configured(self, provider_id: str) -> bool:
        return bool(self._keys.get(provider_id))

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            p.id: {"key_configured": self.configured(p.id), "breaker": self.breakers[p.id].state}
            for p in self.catalog.providers
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, p: Provider, wire: Wire, client_headers: dict[str, str]) -> dict[str, str]:
        key = self._keys[p.id]
        h = {"content-type": "application/json", "accept": "application/json", **p.extra_headers}
        if p.auth == "x-api-key":
            h["x-api-key"] = key
        else:
            h["authorization"] = f"Bearer {key}"
        lower = {k.lower(): v for k, v in client_headers.items()}
        for name in FORWARD_HEADERS[wire]:
            if name in lower:
                h[name] = lower[name]
        if wire is Wire.ANTHROPIC:
            h.setdefault("anthropic-version", DEFAULT_ANTHROPIC_VERSION)
        return h

    async def send(
        self, model_id: str, wire: Wire, body: dict[str, Any], client_headers: dict[str, str], stream: bool
    ) -> UpstreamResponse:
        m = self.catalog.model(model_id)
        p = self.catalog.provider(m.provider)
        if not self.configured(p.id):
            raise Retryable(p.id, "no API key configured")
        br = self.breakers[p.id]
        if not br.allow():
            raise Retryable(p.id, "circuit open")
        url = p.base_urls[wire] + PATHS[wire]
        timeout = httpx.Timeout(p.timeout_s, connect=p.connect_timeout_s)
        req = self._client.build_request(
            "POST", url, json=body, headers=self._headers(p, wire, client_headers), timeout=timeout
        )
        try:
            resp = await self._client.send(req, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            br.failure()
            raise Retryable(p.id, f"{type(e).__name__}") from e
        if resp.status_code in RETRYABLE_STATUS:
            detail = (await resp.aread())[:300].decode("utf-8", "replace")
            await resp.aclose()
            br.failure()
            raise Retryable(p.id, f"HTTP {resp.status_code}: {detail}", resp.status_code)
        headers = {
            k.lower(): v
            for k, v in resp.headers.items()
            if k.lower() in ("content-type", "retry-after", "request-id", "x-request-id")
        }
        if resp.status_code >= 400 or not stream:
            data = await resp.aread()
            await resp.aclose()
            if resp.status_code < 400:
                br.success()
            return UpstreamResponse(resp.status_code, headers, body=data)
        br.success()

        async def gen() -> AsyncIterator[bytes]:
            try:
                # decoded bytes: content-encoding is not forwarded to the client
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return UpstreamResponse(resp.status_code, headers, stream=gen(), _close=resp.aclose)
