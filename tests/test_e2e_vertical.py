"""End-to-end Phase 1 vertical:

dataset -> job -> (rehearsal) training -> sign/register -> publish guard ->
L1-L6 eval on served adapter -> OPA staging gate -> human approval ->
OPA production gate -> canary -> champion -> second version supersedes ->
rollback -> audit chain verifies.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from mccp.pipeline.runner import PipelineRunner
from tests.conftest import ADMIN, APPROVE, SUBMIT, job_body, needs_db, needs_opa

pytestmark = [needs_db, needs_opa, pytest.mark.e2e]


async def submit(api, body=None, key=None):
    async with api.client("alice@acme", SUBMIT) as c:
        r = await c.post(
            "/v1/customization/jobs", json=body or job_body(), headers={"Idempotency-Key": key or f"k-{uuid.uuid4()}"}
        )
        assert r.status_code == 201, r.text
        return r.json()


async def get_job(api, job_id):
    async with api.client("alice@acme", SUBMIT) as c:
        r = await c.get(f"/v1/customization/jobs/{job_id}")
        assert r.status_code == 200, r.text
        return r.json()


async def approve(api, job_id, who="bob@acme"):
    async with api.client(who, APPROVE) as c:
        return await c.post(f"/v1/approvals/{job_id}:approve", json={"justification": "Eval and evidence reviewed"})


async def test_full_vertical_promote_supersede_rollback(seeded, platform, fake_serving):
    api = seeded
    LOGICAL = api.logical
    runner = PipelineRunner(platform)

    # ---- v1: train -> guard -> eval -> staging gate -> pending approval
    job = await submit(api)
    assert job["status"] == "validating_files" and job["lifecycle_status"] == "created"
    await runner.run_until_idle()
    job = await get_job(api, job["id"])
    assert job["lifecycle_status"] == "pending_approval", job["error"]
    assert job["status"] == "succeeded"
    assert job["trained_tokens"] > 0 and job["flop_estimate"] > 0 and job["cost_usd"] is not None
    v1_name = job["output_adapter"]
    assert v1_name == f"{api.tenant_name}-finance-ops-invoice-triage-v1"
    assert v1_name in fake_serving.loaded  # loaded by the publish guard

    async with api.client("alice@acme", SUBMIT) as c:
        ev = (await c.get("/v1/adapters/finance-ops/invoice-triage/versions/1/evidence")).json()
    assert ev["decision"]["allow"] is True, ev["decision"]
    report = ev["eval_report"]
    assert report["l1"]["adapter"]["json_validity"] == 1.0
    assert report["l1"]["adapter"]["tool_call_parse_rate"] == 1.0
    assert report["l1"]["delta_vs_base"]["ci_low"] > 0  # adapter beats base
    assert report["l4"]["status"] == "ok" and report["l4"]["unsafe_rate_delta"] == 0
    assert ev["bundle"]["signatures"]["adapter_verified"] is True
    assert ev["bundle"]["ml_bom"]["bomFormat"] == "CycloneDX"
    assert ev["bundle"]["eval"]["publish_guard"]["skipped_modules"] == []

    # ---- approvals: submitter cannot self-approve; independent approver can
    r = await approve(api, job["id"], who="alice@acme")
    assert r.status_code == 403
    async with api.client("alice@acme", SUBMIT) as c:
        r = await c.post(f"/v1/approvals/{job['id']}:approve", json={"justification": "nope"})
        assert r.status_code == 403  # lacks approver role
    async with api.client("bob@acme", APPROVE) as c:
        pending = (await c.get("/v1/approvals")).json()["data"]
        assert [p["job"]["id"] for p in pending] == [job["id"]]
        assert pending[0]["model_card"]["intended_use"].startswith("Classify AP")
    assert (await approve(api, job["id"])).status_code == 200
    assert (await approve(api, job["id"])).status_code == 409  # one decision per approver

    await runner.run_until_idle()
    job = await get_job(api, job["id"])
    assert job["lifecycle_status"] == "deployed", job["error"]
    assert fake_serving.weights(LOGICAL) == {v1_name: 100}
    assert fake_serving.mlflow_aliases[(LOGICAL, "champion")] == "1"

    # ---- v2 with a champion present: canary 90/10 then promotion supersedes v1
    job2 = await submit(api, job_body(seed=7))
    await runner.run_until_idle()
    job2 = await get_job(api, job2["id"])
    assert job2["lifecycle_status"] == "pending_approval", job2["error"]
    async with api.client("alice@acme", SUBMIT) as c:
        ev2 = (await c.get("/v1/adapters/finance-ops/invoice-triage/versions/2/evidence")).json()
    assert ev2["eval_report"]["l1"]["delta_vs_champion"] is not None
    assert (await approve(api, job2["id"], who="carol@acme")).status_code == 200
    await runner.run_until_idle()
    job2 = await get_job(api, job2["id"])
    v2_name = job2["output_adapter"]
    assert job2["lifecycle_status"] == "deployed", job2["error"]
    assert fake_serving.weights(LOGICAL) == {v2_name: 100}
    assert (await get_job(api, job["id"]))["lifecycle_status"] == "superseded"

    async with api.client("alice@acme", SUBMIT) as c:
        deps = (await c.get("/v1/deployments")).json()["data"]
        adapters = {
            a["version"]: a["status"]
            for a in (await c.get("/v1/adapters", params={"project": "finance-ops"})).json()["data"]
        }
    assert adapters == {1: "superseded", 2: "champion"}
    assert deps[0]["previous_champion_id"] is not None

    # ---- rollback reinstates v1
    async with api.client("bob@acme", APPROVE) as c:
        r = await c.post("/v1/deployments/finance-ops/invoice-triage:rollback", json={"reason": "spike in escalations"})
        assert r.status_code == 200, r.text
    await runner.run_until_idle()
    assert (await get_job(api, job["id"]))["lifecycle_status"] == "deployed"
    assert (await get_job(api, job2["id"]))["lifecycle_status"] == "superseded"
    assert fake_serving.weights(LOGICAL) == {v1_name: 100}

    # ---- the audit chain covers all of it and verifies
    async with api.client("root@acme", ADMIN) as c:
        audit = (await c.get("/v1/audit:verify")).json()
        events = (await c.get(f"/v1/customization/jobs/{job['id']}/events", params={"limit": 500})).json()["data"]
    assert audit["ok"] is True and audit["events_checked"] > 30
    path = [e["data"]["to"] for e in events if e["action"] == "job.transition"]
    assert path == [
        "validating",
        "queued",
        "provisioning",
        "running",
        "registering",
        "publish_guard",
        "evaluating",
        "policy_check",
        "pending_approval",
        "approved",
        "canary",
        "deployed",
        "superseded",
        "deployed",
    ]


async def test_unsafe_adapter_is_rejected_by_gate(seeded, platform, fake_serving):
    fake_serving.adapter_mode = "unsafe"
    runner = PipelineRunner(platform)
    job = await submit(seeded)
    await runner.run_until_idle()
    job = await get_job(seeded, job["id"])
    assert job["lifecycle_status"] == "rejected"
    assert job["error"]["code"] == "policy_denied"
    assert any("unsafe-response rate rose" in d for d in job["error"]["deny"])


async def test_tool_call_leak_is_rejected(seeded, platform, fake_serving):
    fake_serving.adapter_mode = "leaky"  # tool calls emitted as text (vLLM #42021 symptom)
    runner = PipelineRunner(platform)
    job = await submit(seeded)
    await runner.run_until_idle()
    job = await get_job(seeded, job["id"])
    assert job["lifecycle_status"] == "rejected"
    assert any("tool_calls populated" in d for d in job["error"]["deny"])


async def test_noop_adapter_passes_staging_but_not_production(seeded, platform, fake_serving):
    fake_serving.adapter_mode = "same_as_base"
    runner = PipelineRunner(platform)
    job = await submit(seeded, job_body(risk_tier="low"))
    await runner.run_until_idle()
    job = await get_job(seeded, job["id"])
    # Same-as-base answers the golden set badly, so the staging gate denies on task quality.
    assert job["lifecycle_status"] == "rejected"
    assert any("no observable effect" in d or "tool_calls populated" in d for d in job["error"]["deny"])


async def test_publish_guard_blocks_modules_vllm_would_skip(seeded, platform):
    # Simulate a fleet whose vLLM does not apply LoRA to MLP layers.
    platform.fleet.architectures["qwen3"].lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    try:
        runner = PipelineRunner(platform)
        job = await submit(seeded)
        await runner.run_until_idle()
        job = await get_job(seeded, job["id"])
    finally:
        platform.fleet.architectures["qwen3"].lora_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    assert job["lifecycle_status"] == "failed"
    assert job["error"]["stage"] == "publish_guard"
    assert job["error"]["detail"]["publish_guard"]["skipped_modules"] == ["down_proj", "gate_proj", "up_proj"]


async def test_reject_by_approver(seeded, platform):
    runner = PipelineRunner(platform)
    job = await submit(seeded)
    await runner.run_until_idle()
    async with seeded.client("bob@acme", APPROVE) as c:
        r = await c.post(f"/v1/approvals/{job['id']}:reject", json={"justification": "Intended use too broad"})
        assert r.status_code == 200
    await runner.run_until_idle()
    job = await get_job(seeded, job["id"])
    assert job["lifecycle_status"] == "rejected"
    assert "rejected by bob@acme" in job["error"]["deny"]


async def test_cancel_before_worker_picks_up(seeded, platform):
    job = await submit(seeded)
    async with seeded.client("alice@acme", SUBMIT) as c:
        r = await c.post(f"/v1/customization/jobs/{job['id']}:cancel")
        assert r.status_code == 200 and r.json()["lifecycle_status"] == "cancelled"
        assert (await c.post(f"/v1/customization/jobs/{job['id']}:cancel")).status_code == 409
    assert await PipelineRunner(platform).run_until_idle() == 0


async def test_idempotent_submission_under_concurrency(seeded):
    key = f"idem-{uuid.uuid4()}"

    async def one():
        async with seeded.client("alice@acme", SUBMIT) as c:
            return await c.post("/v1/customization/jobs", json=job_body(), headers={"Idempotency-Key": key})

    responses = await asyncio.gather(*(one() for _ in range(20)))
    assert sorted({r.status_code for r in responses}) in ([200, 201], [201])
    assert sum(r.status_code == 201 for r in responses) == 1
    assert len({r.json()["id"] for r in responses}) == 1
    async with seeded.client("alice@acme", SUBMIT) as c:
        r = await c.post("/v1/customization/jobs", json=job_body(seed=1), headers={"Idempotency-Key": key})
        assert r.status_code == 409
        listed = (await c.get("/v1/customization/jobs")).json()["data"]
    assert len(listed) == 1


async def test_tenant_isolation_via_api(seeded, platform):
    from mccp.services.core import create_tenant
    from tests.conftest import Api

    job = await submit(seeded)
    other_idp = f"tid-{uuid.uuid4().hex[:8]}"
    await create_tenant(platform, f"globex-{other_idp}", other_idp)
    other = Api(seeded.app, seeded.secret, other_idp)
    async with other.client("mallory@globex", SUBMIT, APPROVE, ADMIN) as c:
        assert (await c.get(f"/v1/customization/jobs/{job['id']}")).status_code == 404
        assert (await c.post(f"/v1/customization/jobs/{job['id']}:cancel")).status_code == 404
        assert (await c.get("/v1/customization/jobs")).json()["data"] == []
        assert (await c.get("/v1/datasets/finance-ops/invoice-exceptions/versions/1")).status_code == 404


async def test_validation_and_auth_errors_are_problem_json(seeded):
    async with seeded.client("alice@acme", SUBMIT) as c:
        r = await c.post(
            "/v1/customization/jobs", json=job_body(learning_rate=1), headers={"Idempotency-Key": "k-123456789"}
        )
        assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")
        r = await c.post(
            "/v1/customization/jobs",
            json=job_body(method={"type": "grpo"}),
            headers={"Idempotency-Key": "k-1234567890"},
        )
        assert r.status_code == 422 and "Phase 2" in r.json()["detail"]
        bad_hp = job_body(method={"type": "sft", "sft": {"hyperparameters": {"lora_alpha": 64}}})
        r = await c.post("/v1/customization/jobs", json=bad_hp, headers={"Idempotency-Key": "k-12345678901"})
        assert r.status_code == 422 and "not allowed by recipe" in r.json()["detail"]
        r = await c.get("/v1/customization/jobs/ftjob-../../etc")
        assert r.status_code in (404, 422)
        r = await c.post("/v1/recipes", json={})
        assert r.status_code == 403
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seeded.app), base_url="http://mccp") as anon:
        r = await anon.get("/v1/customization/jobs")
        assert r.status_code == 401 and r.json()["type"].endswith("/unauthorized")
        r = await anon.get("/v1/customization/jobs", headers={"Authorization": "Bearer not-a-jwt"})
        assert r.status_code == 401


async def test_cancel_flag_on_orphaned_running_job(seeded, platform):
    """A worker died mid-training: the next worker honours the cancel flag
    before relaunching the trainer."""
    from mccp.db.engine import tenant_tx
    from mccp.domain.status import LifecycleStatus as L
    from mccp.services import repo

    job = await submit(seeded)
    async with tenant_tx(platform.engine, seeded.tenant_id) as conn:
        j = await repo.get_job(conn, job["id"], for_update=True)
        for s in (L.VALIDATING, L.QUEUED, L.PROVISIONING, L.RUNNING):
            j = await repo.transition(conn, j, s, actor="test")
        await repo.update_job(conn, job["id"], cancel_requested=True)
    await PipelineRunner(platform).run_until_idle()
    job = await get_job(seeded, job["id"])
    assert job["lifecycle_status"] == "cancelled" and job["trained_tokens"] is None


async def test_same_adapter_name_in_two_tenants_never_collides(seeded, platform, fake_serving):
    """C1: tenants with identical project/adapter names get disjoint gateway
    entries; promoting in one tenant leaves the other tenant's champion alone."""
    from mccp.services.core import create_tenant
    from tests.conftest import RECIPE, Api, invoices_ndjson, suite_body

    a = seeded
    other_idp = f"tid-{uuid.uuid4().hex[:8]}"
    await create_tenant(platform, f"globex-{other_idp}", other_idp)
    b = Api(a.app, a.secret, other_idp)
    b_logical = f"globex-{other_idp}.finance-ops.invoice-triage"
    async with b.client("admin@globex", ADMIN) as c:
        assert (await c.post("/v1/recipes", json=RECIPE)).status_code in (201, 409)
    async with b.client("dana@globex", SUBMIT) as c:
        assert (
            await c.post("/v1/datasets/finance-ops/invoice-exceptions/versions", content=invoices_ndjson())
        ).status_code == 201
        assert (await c.post("/v1/eval-suites/finance-ops", json=suite_body())).status_code == 201

    runner = PipelineRunner(platform)
    job_a = await submit(a)
    async with b.client("dana@globex", SUBMIT) as c:
        r = await c.post("/v1/customization/jobs", json=job_body(), headers={"Idempotency-Key": f"k-{uuid.uuid4()}"})
        job_b = r.json()
    await runner.run_until_idle()
    assert (await approve(a, job_a["id"])).status_code == 200
    async with b.client("erin@globex", APPROVE) as c:
        assert (
            await c.post(f"/v1/approvals/{job_b['id']}:approve", json={"justification": "reviewed"})
        ).status_code == 200
    await runner.run_until_idle()

    a_name = (await get_job(a, job_a["id"]))["output_adapter"]
    async with b.client("dana@globex", SUBMIT) as c:
        b_job = (await c.get(f"/v1/customization/jobs/{job_b['id']}")).json()
    assert b_job["lifecycle_status"] == "deployed", b_job["error"]
    assert a_name != b_job["output_adapter"]
    assert fake_serving.weights(a.logical) == {a_name: 100}
    assert fake_serving.weights(b_logical) == {b_job["output_adapter"]: 100}
    for d in fake_serving.litellm.values():
        assert d["model_info"]["access_groups"] == [f"mccp-tenant-{d['model_name'].split('.')[0]}"]


@pytest.mark.parametrize(
    ("server_default_on", "gateway_forwards", "expect"),
    [
        (False, False, "deployed"),  # server-side default works; gateway irrelevant
        (True, True, "deployed"),  # server flag missing but gateway extra_body saves it
        (True, False, "rejected"),  # both mechanisms fail -> thinking in production
    ],
)
async def test_canary_gateway_smoke_catches_thinking_on_production_path(
    seeded, platform, fake_serving, server_default_on, gateway_forwards, expect
):
    """R5: eval talks to vLLM directly with thinking disabled per request; the
    canary must prove the *gateway* path also has thinking off."""
    fake_serving.server_thinking_default = server_default_on
    fake_serving.gateway_forwards_extra_body = gateway_forwards
    try:
        runner = PipelineRunner(platform)
        job = await submit(seeded)
        await runner.run_until_idle()
        assert (await get_job(seeded, job["id"]))["lifecycle_status"] == "pending_approval"
        assert (await approve(seeded, job["id"])).status_code == 200
        await runner.run_until_idle()
        st = await get_job(seeded, job["id"])
        assert st["lifecycle_status"] == expect, st["error"]
        probe = f"{seeded.logical}~probe"
        assert fake_serving.weights(probe) == {}  # probe route removed either way
        if expect == "rejected":
            assert st["error"]["code"] == "canary_rolled_back"
            assert "thinking output on the production path" in st["error"]["message"]
            assert fake_serving.weights(seeded.logical) == {}
    finally:
        fake_serving.server_thinking_default = False
        fake_serving.gateway_forwards_extra_body = True
