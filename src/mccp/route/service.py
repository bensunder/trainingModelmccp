"""The request path: authenticate -> rate limit -> budget -> guardrails ->
classify -> choose a model -> call it (failover before the first byte) -> meter.

Transport-free: the FastAPI layer (app.py) only maps HTTP to these calls, so the
whole path is testable in-process with fake providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .cache import Cache
from .classify import BACKGROUND_SUFFIX, Classification, classify
from .config import Catalog, Profile, TaskClass, Wire
from .guardrails import GuardrailPolicy
from .guardrails import apply as apply_guardrails
from .metering import StreamMeter, Usage, cost, usage_from_body
from .policy import Choice, Evidence, NoEligibleModel, choose
from .state import ApiKey, Budget, Org, Store, Team, UsageEvent, hash_key, now, period_end, period_start
from .upstream import Retryable, Upstream
from .wire import BadRequest, Features, features, rewrite_for_upstream

log = logging.getLogger("mccp.route")
KEY_CACHE_TTL_S = 60


class GatewayError(Exception):
    def __init__(
        self, status: int, kind: str, title: str, detail: str, headers: dict[str, str] | None = None, **extra: Any
    ):
        super().__init__(detail)
        self.status, self.kind, self.title, self.detail = status, kind, title, detail
        self.headers = headers or {}
        self.extra = extra


@dataclass(frozen=True)
class KeyContext:
    key: ApiKey
    org: Org
    team: Team


@dataclass
class Result:
    status: int
    headers: dict[str, str]
    body: bytes | None = None
    stream: AsyncIterator[bytes] | None = None


@dataclass
class Plan:
    profile: Profile | None
    pinned: str | None
    task: Classification
    choice: Choice
    sticky_key: str | None


@dataclass
class _Attempt:
    model: str
    outcome: str


class UsageWriter:
    """Batches usage events to the store off the request path. Bounded: if the
    store is down for long, the oldest events are dropped and counted."""

    def __init__(self, store: Store, max_batch: int = 500, interval_s: float = 1.0, max_buffer: int = 50_000):
        self.store, self.max_batch, self.interval_s, self.max_buffer = store, max_batch, interval_s, max_buffer
        self._buf: list[UsageEvent] = []
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self.dropped = 0
        self.written = 0

    def submit(self, ev: UsageEvent) -> None:
        self._buf.append(ev)
        if len(self._buf) > self.max_buffer:
            over = len(self._buf) - self.max_buffer
            del self._buf[:over]
            self.dropped += over
        if len(self._buf) >= self.max_batch:
            self._wake.set()

    async def flush(self) -> None:
        while self._buf:
            batch = self._buf[: self.max_batch]
            await self.store.record_usage(batch)
            del self._buf[: len(batch)]
            self.written += len(batch)

    async def _run(self) -> None:
        backoff = self.interval_s
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=backoff)
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.flush()
                backoff = self.interval_s
            except Exception:  # noqa: BLE001 - keep buffering, retry with backoff
                log.exception("usage write failed; retrying")
                backoff = min(backoff * 2, 30.0)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="route-usage-writer")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.flush()


def _empty_or_refused(wire: Wire, body: Any) -> bool:
    """A non-streamed answer not worth returning when a better model is available."""
    if not isinstance(body, dict):
        return True
    if wire is Wire.ANTHROPIC:
        if body.get("stop_reason") == "refusal":
            return True
        return not any(
            (b.get("type") == "text" and (b.get("text") or "").strip()) or b.get("type") == "tool_use"
            for b in body.get("content") or []
            if isinstance(b, dict)
        )
    ch = (body.get("choices") or [{}])[0] or {}
    msg = ch.get("message") or {}
    if ch.get("finish_reason") == "content_filter":
        return True
    return not ((msg.get("content") or "").strip() or msg.get("tool_calls"))


class Gateway:
    def __init__(
        self,
        catalog: Catalog,
        evidence: Evidence,
        store: Store,
        cache: Cache,
        upstream: Upstream,
        pepper: bytes,
        writer: UsageWriter | None = None,
    ) -> None:
        if len(pepper) < 32:
            raise ValueError("key pepper must be at least 32 bytes")
        self.catalog, self.evidence, self.store, self.cache, self.upstream = catalog, evidence, store, cache, upstream
        self.pepper = pepper
        self.writer = writer or UsageWriter(store)

    # -- auth -------------------------------------------------------------------
    async def authenticate(self, token: str | None) -> KeyContext:
        if not token or not token.startswith("rk_"):
            raise GatewayError(401, "unauthorized", "Unauthorized", "missing or malformed API key")
        h = hash_key(self.pepper, token)
        cached = await self.cache.get(f"key:{h}")
        if cached == "revoked":
            raise GatewayError(401, "unauthorized", "Unauthorized", "API key revoked")
        key = await self.store.key_by_hash(h)
        if key is None:
            raise GatewayError(401, "unauthorized", "Unauthorized", "unknown API key")
        if key.revoked_at is not None:
            await self.cache.set(f"key:{h}", "revoked", KEY_CACHE_TTL_S)
            raise GatewayError(401, "unauthorized", "Unauthorized", "API key revoked")
        org = await self.store.get_org(key.org_id)
        team = await self.store.get_team(key.org_id, key.team_id)
        return KeyContext(key, org, team)

    async def on_revoke(self, key: ApiKey) -> None:
        await self.cache.set(f"key:{key.key_hash}", "revoked", 24 * 3600)

    # -- planning ---------------------------------------------------------------
    def _approved(self, team: Team) -> set[str] | None:
        return set(team.approved_models) if team.approved_models is not None else None

    def _resolve(self, f: Features, ctx: KeyContext) -> tuple[Profile | None, str | None]:
        req = f.requested_model
        base = req[: -len(BACKGROUND_SUFFIX)] if req.endswith(BACKGROUND_SUFFIX) else req
        p = self.catalog.profile(base)
        if p is not None:
            return p, None
        if self.catalog.has_model(req):
            return None, req
        pid = (
            ctx.key.default_profile
            or ctx.team.default_profile
            or ctx.org.default_profile
            or self.catalog.default_profile
        )
        p = self.catalog.profile(pid) or self.catalog.profile(self.catalog.default_profile)
        return p, None

    def _sticky_key(self, ctx: KeyContext, profile: Profile, f: Features, task: TaskClass) -> str | None:
        if task is TaskClass.BACKGROUND:
            return None
        sess = f.session_hint or (f.first_user_digest if f.n_messages > 1 else None)
        return f"sticky:{ctx.key.id}:{profile.id}:{f.wire.value}:{sess}" if sess else None

    async def plan(self, f: Features, ctx: KeyContext) -> Plan:
        approved = self._approved(ctx.team)
        profile, pinned = self._resolve(f, ctx)
        task = classify(f)
        if pinned is not None:
            m = self.catalog.model(pinned)
            if approved is not None and pinned not in approved:
                raise GatewayError(
                    403,
                    "model-not-approved",
                    "Model not approved",
                    f"{pinned} is not approved for team {ctx.team.name}",
                )
            if f.wire not in m.wires:
                raise GatewayError(
                    400, "wire-unsupported", "Unsupported API", f"{pinned} is not served over the {f.wire.value} API"
                )
            return Plan(None, pinned, task, Choice(pinned, (), "pinned by request"), None)
        assert profile is not None
        sk = self._sticky_key(ctx, profile, f, task.task)
        sticky = await self.cache.get(sk) if sk else None
        try:
            choice = choose(self.catalog, profile, task.task, f.wire, self.evidence, approved, f.has_tools, sticky)
        except NoEligibleModel as e:
            raise GatewayError(400, "no-eligible-model", "No eligible model", str(e)) from e
        return Plan(profile, None, task, choice, sk)

    # -- budgets and limits -------------------------------------------------------
    def _scope_ids(self, ctx: KeyContext) -> dict[str, str]:
        return {"org": ctx.org.id, "team": ctx.team.id, "key": ctx.key.id}

    def _spend_key(self, b: Budget | tuple[str, str, str], at=None) -> str:
        scope, sid, period = (b.scope, b.scope_id, b.period) if isinstance(b, Budget) else b
        return f"spend:{scope}:{sid}:{period}:{period_start(period, at).date().isoformat()}"

    async def _spent(self, ctx: KeyContext, b: Budget) -> float:
        k = self._spend_key(b)
        v = await self.cache.get_float(k)
        if v is None:
            v = await self.store.spend(ctx.org.id, b.scope, b.scope_id, period_start(b.period))
            ttl = int((period_end(b.period) - now() + timedelta(days=1)).total_seconds())
            await self.cache.incr_float(k, v, ttl)
        return v

    async def check_budgets(self, ctx: KeyContext) -> list[str]:
        ids = self._scope_ids(ctx)
        warnings = []
        for b in await self.store.list_budgets(ctx.org.id):
            if ids.get(b.scope) != b.scope_id:
                continue
            spent = await self._spent(ctx, b)
            if spent >= b.limit_usd:
                if b.hard:
                    retry = int((period_end(b.period) - now()).total_seconds())
                    raise GatewayError(
                        429, "budget-exceeded", "Budget exceeded",
                        f"{b.scope} {b.period} budget ${b.limit_usd:.2f} reached (spent ${spent:.2f})",
                        {"retry-after": str(max(1, retry))},
                    )  # fmt: skip
                warnings.append(f"{b.scope}-{b.period}-budget-exceeded")
        return warnings

    async def add_spend(self, ctx: KeyContext, usd: float) -> None:
        if usd <= 0:
            return
        for b in await self.store.list_budgets(ctx.org.id):
            if self._scope_ids(ctx).get(b.scope) == b.scope_id:
                ttl = int((period_end(b.period) - now() + timedelta(days=1)).total_seconds())
                await self.cache.incr_float(self._spend_key(b), usd, ttl)

    async def rate_limit(self, ctx: KeyContext) -> None:
        ok, wait = await self.cache.take(f"rl:{ctx.key.id}", ctx.org.rpm_per_key)
        if not ok:
            raise GatewayError(429, "rate-limited", "Rate limited", f"limit {ctx.org.rpm_per_key} requests/minute per key",
                               {"retry-after": str(max(1, int(wait + 0.999)))})  # fmt: skip

    # -- the request ----------------------------------------------------------------
    def _event(
        self, rid: str, ctx: KeyContext, f: Features | None, wire: Wire, body: dict[str, Any], **kw: Any
    ) -> UsageEvent:
        return UsageEvent(
            id=str(uuid.uuid4()), request_id=rid, created_at=now(), org_id=ctx.org.id, team_id=ctx.team.id,
            key_id=ctx.key.id, wire=wire.value, requested_model=str(body.get("model", ""))[:200],
            stream=bool(f.stream) if f else False, **kw,
        )  # fmt: skip

    async def handle(self, wire: Wire, body: dict[str, Any], headers: dict[str, str], ctx: KeyContext) -> Result:
        rid = f"req_{uuid.uuid4().hex[:24]}"
        t0 = time.monotonic()
        await self.rate_limit(ctx)
        try:
            f = features(wire, body, headers)
        except BadRequest as e:
            raise GatewayError(400, "invalid-request", "Invalid request", str(e)) from e
        warnings = await self.check_budgets(ctx)
        policy = GuardrailPolicy(**ctx.org.guardrails) if ctx.org.guardrails else GuardrailPolicy()
        fwd, verdict = apply_guardrails(body, policy)
        if verdict.blocked:
            self.writer.submit(self._event(rid, ctx, f, wire, body, profile=None, task=None, model=None, provider=None,
                                           status=400, route_reason="guardrail block", guardrail=verdict.summary()))  # fmt: skip
            raise GatewayError(400, "guardrail-blocked", "Blocked by guardrail policy",
                               "the request contains content your organisation does not allow to leave: " + ", ".join(sorted(verdict.findings)),
                               findings=dict(verdict.findings))  # fmt: skip
        plan = await self.plan(f, ctx)
        order = [plan.choice.model, *plan.choice.fallbacks]
        attempts: list[_Attempt] = []
        spent_failed = Usage()
        out_headers = {
            "x-route-request-id": rid,
            "x-route-profile": plan.profile.id if plan.profile else "pinned",
            "x-route-task": plan.task.task.value,
            **({"x-route-warning": ",".join(warnings)} if warnings else {}),
            **({"x-route-guardrail": ",".join(sorted(verdict.findings))} if verdict.findings else {}),
        }
        for i, mid in enumerate(order):
            m = self.catalog.model(mid)
            try:
                resp = await self.upstream.send(
                    mid, wire, rewrite_for_upstream(wire, fwd, m.upstream), headers, f.stream
                )
            except Retryable as e:
                attempts.append(_Attempt(mid, e.reason[:120]))
                continue
            if f.stream and resp.stream is not None:
                return self._stream_result(rid, ctx, f, wire, body, plan, mid, resp, attempts, t0, out_headers, verdict)
            data: Any = None
            try:
                data = json.loads(resp.body or b"null")
            except json.JSONDecodeError:
                pass
            if resp.status < 400 and _empty_or_refused(wire, data) and i < len(order) - 1:
                u = usage_from_body(wire, data or {})
                spent_failed.input_tokens += u.input_tokens
                spent_failed.output_tokens += u.output_tokens
                attempts.append(_Attempt(mid, "empty or refused answer"))
                continue
            u = usage_from_body(wire, data) if isinstance(data, dict) else Usage()
            c = cost(self.catalog, mid, u)
            extra = cost(self.catalog, attempts[-1].model, spent_failed).usd if attempts and spent_failed.total else 0.0
            ev = self._event(
                rid, ctx, f, wire, body, profile=plan.profile.id if plan.profile else None, task=plan.task.task.value,
                model=mid, provider=m.provider, status=resp.status, input_tokens=u.input_tokens,
                output_tokens=u.output_tokens, cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
                usage_reported=u.reported, cost_usd=c.usd + extra, baseline_cost_usd=c.baseline_usd,
                latency_ms=int((time.monotonic() - t0) * 1000), attempts=len(attempts) + 1,
                route_reason=self._reason(plan, attempts), guardrail=verdict.summary() if verdict.findings else {},
            )  # fmt: skip
            self.writer.submit(ev)
            await self.add_spend(ctx, ev.cost_usd)
            if resp.status < 400 and plan.sticky_key and plan.profile:
                await self.cache.set(plan.sticky_key, mid, plan.profile.sticky_ttl_s)
            return Result(resp.status, {**out_headers, **resp.headers, "x-route-model": mid}, body=resp.body)
        self.writer.submit(self._event(rid, ctx, f, wire, body, profile=plan.profile.id if plan.profile else None,
                                       task=plan.task.task.value, model=None, provider=None, status=503,
                                       attempts=len(attempts), route_reason=self._reason(plan, attempts)))  # fmt: skip
        raise GatewayError(503, "upstream-unavailable", "No model available",
                           "every model on this route failed before answering",
                           headers={"retry-after": "5", **out_headers},
                           attempts=[{"model": a.model, "outcome": a.outcome} for a in attempts])  # fmt: skip

    def _reason(self, plan: Plan, attempts: list[_Attempt]) -> str:
        r = f"{plan.task.task.value}: {'; '.join(plan.task.reasons)} | {plan.choice.reason}"
        if attempts:
            r += " | failover: " + ", ".join(f"{a.model} ({a.outcome})" for a in attempts)
        return r[:1000]

    def _stream_result(self, rid, ctx, f, wire, body, plan, mid, resp, attempts, t0, out_headers, verdict) -> Result:
        meter = StreamMeter(wire)
        m = self.catalog.model(mid)

        async def gen() -> AsyncIterator[bytes]:
            ttft: int | None = None
            completed = False
            try:
                async for chunk in resp.stream:
                    if ttft is None:
                        ttft = int((time.monotonic() - t0) * 1000)
                    meter.feed(chunk)
                    yield chunk
                completed = True
            finally:
                meter.close()
                c = cost(self.catalog, mid, meter.usage)
                status = resp.status if completed and not meter.error else 499 if not completed else 502
                ev = self._event(
                    rid, ctx, f, wire, body, profile=plan.profile.id if plan.profile else None, task=plan.task.task.value,
                    model=mid, provider=m.provider, status=status, input_tokens=meter.usage.input_tokens,
                    output_tokens=meter.usage.output_tokens, cache_read_tokens=meter.usage.cache_read_tokens,
                    cache_write_tokens=meter.usage.cache_write_tokens, usage_reported=meter.usage.reported,
                    cost_usd=c.usd, baseline_cost_usd=c.baseline_usd, latency_ms=int((time.monotonic() - t0) * 1000),
                    ttft_ms=ttft, attempts=len(attempts) + 1, route_reason=self._reason(plan, attempts),
                    guardrail=verdict.summary() if verdict.findings else {},
                )  # fmt: skip
                self.writer.submit(ev)
                try:
                    await self.add_spend(ctx, ev.cost_usd)
                    if completed and not meter.error and plan.sticky_key and plan.profile:
                        await self.cache.set(plan.sticky_key, mid, plan.profile.sticky_ttl_s)
                except Exception:  # noqa: BLE001 - metering must not break a finished stream
                    log.exception("post-stream bookkeeping failed for %s", rid)

        hdrs = {**out_headers, "x-route-model": mid, "content-type": resp.headers.get("content-type", "text/event-stream"),
                "cache-control": "no-cache"}  # fmt: skip
        return Result(resp.status, hdrs, stream=gen())

    def count_tokens(self, wire: Wire, body: dict[str, Any]) -> int:
        """Estimate for /v1/messages/count_tokens (no provider call): ~4 chars/token.
        Harnesses use it for context-window bookkeeping, not billing."""
        f = features(wire, {**body, "model": body.get("model") or "x"})
        return max(1, (f.total_chars + 3) // 4)
