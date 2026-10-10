"""Authentication: OIDC (Entra ID or any OIDC IdP) or a local dev issuer.

Tokens carry the subject, the IdP tenant id (mapped to an internal tenant)
and app roles: Customizer.Submit / Customizer.Approve / Customizer.Admin.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import jwt
from fastapi import Request

from mccp.domain.errors import DomainError
from mccp.services.core import resolve_tenant
from mccp.services.repo import Principal

DEV_ISSUER = "mccp-dev"
DEV_AUDIENCE = "mccp"


class Unauthorized(DomainError):
    status = 401
    title = "Unauthorized"
    type_suffix = "unauthorized"


def mint_dev_token(secret: str, *, subject: str, tenant: str, roles: list[str], ttl_s: int = 3600) -> str:
    now = int(time.time())
    claims = {
        "iss": DEV_ISSUER,
        "aud": DEV_AUDIENCE,
        "sub": subject,
        "tid": tenant,
        "roles": roles,
        "iat": now,
        "exp": now + ttl_s,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


class Authenticator:
    def __init__(self, platform) -> None:  # Platform; untyped to avoid import cycle
        self.pf = platform
        s = platform.settings
        self.mode = s.auth_mode
        self._jwks = (
            jwt.PyJWKClient(s.oidc_jwks_url, cache_keys=True, lifespan=3600)
            if s.auth_mode == "oidc" and s.oidc_jwks_url
            else None
        )
        self._tenant_cache: dict[str, tuple[float, uuid.UUID]] = {}

    def _decode(self, token: str) -> dict[str, Any]:
        s = self.pf.settings
        try:
            if self.mode == "dev":
                return jwt.decode(
                    token,
                    s.dev_jwt_secret.get_secret_value(),
                    algorithms=["HS256"],
                    audience=DEV_AUDIENCE,
                    issuer=DEV_ISSUER,
                )
            if self._jwks is None or not s.oidc_issuer or not s.oidc_audience:
                raise Unauthorized("OIDC is not configured (issuer, audience, jwks url)")
            key = self._jwks.get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=["RS256", "ES256"], audience=s.oidc_audience, issuer=s.oidc_issuer)
        except jwt.PyJWTError as e:
            raise Unauthorized(f"invalid token: {e}") from e

    async def _tenant(self, idp_tenant: str) -> uuid.UUID:
        hit = self._tenant_cache.get(idp_tenant)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        tid = await resolve_tenant(self.pf, idp_tenant)
        if tid is None:
            raise Unauthorized("tenant is not onboarded")
        self._tenant_cache[idp_tenant] = (time.monotonic() + 300, tid)
        return tid

    async def principal(self, request: Request) -> Principal:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Unauthorized("missing bearer token")
        claims = await asyncio.to_thread(self._decode, token)
        s = self.pf.settings
        sub = claims.get("sub")
        tenant_claim = claims.get(s.oidc_tenant_claim if self.mode == "oidc" else "tid")
        if not sub or not tenant_claim:
            raise Unauthorized("token lacks subject or tenant claim")
        roles = claims.get(s.oidc_roles_claim if self.mode == "oidc" else "roles") or []
        if not isinstance(roles, list):
            raise Unauthorized("roles claim must be a list")
        return Principal(
            subject=str(sub), tenant_id=await self._tenant(str(tenant_claim)), roles=frozenset(map(str, roles))
        )
