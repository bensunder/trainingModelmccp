"""Domain errors. Pure: transport layers map these to RFC 9457 problem documents."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain-rule violations."""

    status: int = 422
    title: str = "Domain rule violated"
    type_suffix: str = "domain-error"

    def __init__(self, detail: str, **extra: object) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra


class ValidationFailed(DomainError):
    title = "Validation failed"
    type_suffix = "validation-failed"

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}", field=field)
        self.field = field


class InvalidReference(DomainError):
    title = "Invalid reference"
    type_suffix = "invalid-reference"


class InvalidTransition(DomainError):
    status = 409
    title = "Invalid state transition"
    type_suffix = "invalid-transition"

    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"cannot transition from {current} to {target}", current=current, target=target)


class MethodNotEnabled(DomainError):
    title = "Method not enabled"
    type_suffix = "method-not-enabled"

    def __init__(self, method: str) -> None:
        super().__init__(f"method `{method}` is not enabled on this control plane (Phase 2)", method=method)


class NotFound(DomainError):
    status = 404
    title = "Not found"
    type_suffix = "not-found"


class Conflict(DomainError):
    status = 409
    title = "Conflict"
    type_suffix = "conflict"


class Forbidden(DomainError):
    status = 403
    title = "Forbidden"
    type_suffix = "forbidden"
