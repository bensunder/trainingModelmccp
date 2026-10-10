"""Job state machine (spec D5/D6).

`LifecycleStatus` is authoritative; the OpenAI-compatible `JobStatus` is always
derived from it so the two can never disagree. `TRANSITIONS` is the only place
transitions are permitted. The database mirrors the enum domain with CHECK
constraints, and the store refuses writes that are not in this table.
"""

from __future__ import annotations

from enum import StrEnum

from .errors import InvalidTransition


class JobStatus(StrEnum):
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LifecycleStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    PAUSED = "paused"
    REGISTERING = "registering"
    PUBLISH_GUARD = "publish_guard"
    EVALUATING = "evaluating"
    POLICY_CHECK = "policy_check"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    CANARY = "canary"
    DEPLOYED = "deployed"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


L = LifecycleStatus

TRANSITIONS: frozenset[tuple[LifecycleStatus, LifecycleStatus]] = frozenset(
    {
        (L.CREATED, L.VALIDATING),
        (L.CREATED, L.CANCELLED),
        (L.VALIDATING, L.QUEUED),
        (L.VALIDATING, L.FAILED),
        (L.VALIDATING, L.CANCELLED),
        (L.QUEUED, L.PROVISIONING),
        (L.QUEUED, L.CANCELLED),
        (L.PROVISIONING, L.RUNNING),
        (L.PROVISIONING, L.FAILED),
        (L.PROVISIONING, L.CANCELLED),
        (L.RUNNING, L.PAUSED),
        (L.RUNNING, L.CANCELLED),
        (L.RUNNING, L.FAILED),
        (L.RUNNING, L.REGISTERING),
        (L.PAUSED, L.RUNNING),
        (L.PAUSED, L.CANCELLED),
        (L.REGISTERING, L.PUBLISH_GUARD),
        (L.REGISTERING, L.FAILED),
        (L.PUBLISH_GUARD, L.EVALUATING),
        (L.PUBLISH_GUARD, L.FAILED),
        (L.EVALUATING, L.POLICY_CHECK),
        (L.EVALUATING, L.FAILED),
        (L.POLICY_CHECK, L.PENDING_APPROVAL),
        (L.POLICY_CHECK, L.REJECTED),
        (L.POLICY_CHECK, L.FAILED),
        (L.PENDING_APPROVAL, L.APPROVED),
        (L.PENDING_APPROVAL, L.REJECTED),
        (L.APPROVED, L.CANARY),
        (L.APPROVED, L.FAILED),
        (L.CANARY, L.DEPLOYED),
        (L.CANARY, L.REJECTED),
        (L.CANARY, L.FAILED),
        (L.DEPLOYED, L.SUPERSEDED),
        # Rollback: a superseded champion is reinstated.
        (L.SUPERSEDED, L.DEPLOYED),
        (L.SUPERSEDED, L.RETIRED),
    }
)

TERMINAL: frozenset[LifecycleStatus] = frozenset({L.REJECTED, L.FAILED, L.CANCELLED, L.RETIRED})

# States the pipeline runner actively drives (leases). Parked states wait for
# an external signal (approval, resume, rollback/supersede).
WORKER_DRIVEN: frozenset[LifecycleStatus] = frozenset(
    {
        L.CREATED,
        L.VALIDATING,
        L.QUEUED,
        L.PROVISIONING,
        L.RUNNING,
        L.REGISTERING,
        L.PUBLISH_GUARD,
        L.EVALUATING,
        L.POLICY_CHECK,
        L.PENDING_APPROVAL,  # re-driven only when an approval decision arrives
        L.APPROVED,
        L.CANARY,
    }
)


def can_transition(src: LifecycleStatus, dst: LifecycleStatus) -> bool:
    return (src, dst) in TRANSITIONS


def ensure_transition(src: LifecycleStatus, dst: LifecycleStatus) -> None:
    if not can_transition(src, dst):
        raise InvalidTransition(src.value, dst.value)


def is_cancellable(s: LifecycleStatus) -> bool:
    return can_transition(s, L.CANCELLED)


def job_status(s: LifecycleStatus) -> JobStatus:
    match s:
        case L.CREATED | L.VALIDATING:
            return JobStatus.VALIDATING_FILES
        case L.QUEUED | L.PROVISIONING:
            return JobStatus.QUEUED
        case L.RUNNING:
            return JobStatus.RUNNING
        case L.PAUSED:
            return JobStatus.PAUSED
        case L.FAILED:
            return JobStatus.FAILED
        case L.CANCELLED:
            return JobStatus.CANCELLED
        case _:
            # Training produced an adapter; the governance outcome is in lifecycle.
            return JobStatus.SUCCEEDED


class AdapterStatus(StrEnum):
    PENDING = "pending"
    CANDIDATE = "candidate"
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    REJECTED = "rejected"


class Lane(StrEnum):
    """Which kind of worker may run a stage (must match SQL mccp_lane())."""

    CONTROL = "control"
    GPU = "gpu"
    SERVING = "serving"


LANE: dict[LifecycleStatus, Lane] = {
    **{s: Lane.CONTROL for s in LifecycleStatus},
    **{s: Lane.GPU for s in (L.QUEUED, L.PROVISIONING, L.RUNNING, L.PAUSED, L.REGISTERING)},
    **{
        s: Lane.SERVING
        for s in (L.PUBLISH_GUARD, L.EVALUATING, L.APPROVED, L.CANARY, L.DEPLOYED, L.SUPERSEDED)
    },
}


def lane_of(s: LifecycleStatus) -> Lane:
    return LANE[s]
