"""S1: work lanes. A long training run must not stall another job's approval
and canary, and the SQL lane mapping must equal the Python one."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from sqlalchemy import text

from mccp.db.engine import system_tx
from mccp.domain.status import LANE, LifecycleStatus
from mccp.pipeline.runner import PipelineRunner
from mccp.training.backends import RehearsalBackend
from tests.conftest import APPROVE, SUBMIT, job_body, needs_db, needs_opa

pytestmark = [needs_db, needs_opa, pytest.mark.e2e]


async def test_sql_lane_function_matches_domain(platform):
    async with system_tx(platform.engine) as conn:
        for s in LifecycleStatus:
            got = (await conn.execute(text("SELECT mccp_lane(:s)"), {"s": s.value})).scalar_one()
            assert got == LANE[s].value, s


async def _status(api, job_id):
    async with api.client("alice@acme", SUBMIT) as c:
        return (await c.get(f"/v1/customization/jobs/{job_id}")).json()


async def test_long_training_does_not_block_other_jobs(seeded, platform):
    api = seeded
    # Job B reaches pending_approval with the fast backend.
    async with api.client("alice@acme", SUBMIT) as c:
        b = (
            await c.post("/v1/customization/jobs", json=job_body(), headers={"Idempotency-Key": f"k-{uuid.uuid4()}"})
        ).json()
    await PipelineRunner(platform).run_until_idle()
    assert (await _status(api, b["id"]))["lifecycle_status"] == "pending_approval"

    # Job A trains slowly (~6 s). While it trains, B is approved and must deploy.
    platform.backend = RehearsalBackend(step_delay_s=1.0, steps=6)
    async with api.client("alice@acme", SUBMIT) as c:
        a = (
            await c.post(
                "/v1/customization/jobs", json=job_body(seed=9), headers={"Idempotency-Key": f"k-{uuid.uuid4()}"}
            )
        ).json()
    async with api.client("bob@acme", APPROVE) as c:
        assert (await c.post(f"/v1/approvals/{b['id']}:approve", json={"justification": "reviewed"})).status_code == 200

    stop = asyncio.Event()
    runner = PipelineRunner(platform)
    task = asyncio.create_task(runner.run_forever(stop, idle_sleep_s=0.1))
    try:
        deadline = time.monotonic() + 30
        b_deployed_while_a_running = False
        while time.monotonic() < deadline:
            sb = (await _status(api, b["id"]))["lifecycle_status"]
            sa = (await _status(api, a["id"]))["lifecycle_status"]
            if sb == "deployed":
                b_deployed_while_a_running = sa in ("provisioning", "running")
                break
            await asyncio.sleep(0.1)
        assert b_deployed_while_a_running, f"B={sb} A={sa}: B waited for A's training to finish"
        # A still completes on its own lane afterwards.
        while time.monotonic() < deadline:
            if (await _status(api, a["id"]))["lifecycle_status"] == "pending_approval":
                break
            await asyncio.sleep(0.2)
        assert (await _status(api, a["id"]))["lifecycle_status"] == "pending_approval"
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)


async def test_control_only_worker_never_runs_gpu_stages(seeded, platform):
    platform.settings.worker_lanes = ["control"]
    try:
        async with seeded.client("alice@acme", SUBMIT) as c:
            j = (
                await c.post(
                    "/v1/customization/jobs", json=job_body(), headers={"Idempotency-Key": f"k-{uuid.uuid4()}"}
                )
            ).json()
        await PipelineRunner(platform).run_until_idle()
        st = await _status(seeded, j["id"])
        # Validation (control lane) ran; training (gpu lane) was left for a GPU worker.
        assert st["lifecycle_status"] == "queued"
        assert st["trained_tokens"] is None
    finally:
        platform.settings.worker_lanes = ["control", "gpu", "serving"]
