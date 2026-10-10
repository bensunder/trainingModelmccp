"""S3: the split topology stores everything in an S3-compatible bucket shared
by the VPS control plane and the GPU host. Exercised against a moto server."""

from __future__ import annotations

import uuid

import pytest

from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest
from mccp.pipeline.runner import PipelineRunner
from mccp.platform import Platform
from mccp.registry.artifacts import ArtifactError, Signer, materialize_verified, publish_adapter
from mccp.storage.objectstore import S3Store, StorageError, open_store
from mccp.training.backends import write_peft_adapter
from tests.conftest import RECIPE, SUBMIT, Api, invoices_ndjson, job_body, needs_db, needs_opa, suite_body


@pytest.fixture(scope="module")
def s3_endpoint(monkeypatch_module):
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "test")
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
async def store(s3_endpoint) -> S3Store:
    st = open_store(f"s3://mccp-{uuid.uuid4().hex[:8]}/objects", s3_endpoint)
    await st.ensure_bucket()
    await st.ensure_bucket()  # idempotent
    return st


async def test_roundtrip_list_and_not_found(store, tmp_path):
    await store.put_bytes("t/a/1.json", b"{}")
    await store.put_bytes("t/ab/2.json", b"[]")
    assert await store.list("t/a") == ["t/a/1.json"]  # directory semantics
    assert await store.get_bytes("t/a/1.json") == b"{}"
    assert await store.exists("t/a/1.json") and not await store.exists("t/a/missing")
    with pytest.raises(StorageError):
        await store.get_bytes("t/a/missing")
    with pytest.raises(StorageError):
        await store.download_file("t/a/missing", tmp_path / "x")
    assert not list(tmp_path.glob(".tmp-*"))


async def test_signed_adapter_roundtrip_with_verify_only_reader(store, tmp_path, keys):
    cfg = JobRequest.model_validate(job_body()).resolve(Recipe.model_validate(RECIPE))
    adapter = write_peft_adapter(tmp_path / "a", cfg, Recipe.model_validate(RECIPE).dims)
    writer = Signer("key", keys[0], keys[1])  # GPU worker
    reader = Signer("key", None, keys[1])  # control plane: public key only
    art = await publish_adapter(store, "tenants/t1", adapter, tmp_path / "stage", writer)
    dest = await materialize_verified(store, art.key_prefix, art.digest, tmp_path / "lora" / "v1", reader)
    assert (dest / "adapter_model.safetensors").exists()
    with pytest.raises(ArtifactError, match="GPU worker only"):
        reader.sign(dest, tmp_path / "x.sig")
    with pytest.raises(ArtifactError, match="PUBLIC_KEY_PATH"):
        Signer("key", None, None).verify(dest, tmp_path / "x.sig")


@needs_db
@needs_opa
async def test_vertical_runs_on_s3_store(settings, s3_endpoint, fake_serving):
    settings.object_store_url = f"s3://mccp-{uuid.uuid4().hex[:8]}/objects"
    settings.s3_endpoint_url = s3_endpoint
    pf = Platform.from_settings(settings)
    await pf.store.ensure_bucket()
    try:
        from mccp.api.app import create_app
        from mccp.api.auth import Authenticator
        from mccp.services.core import create_tenant

        app = create_app(settings, pf)
        app.state.pf, app.state.auth = pf, Authenticator(pf)
        idp = f"tid-{uuid.uuid4().hex[:8]}"
        await create_tenant(pf, f"s3t-{idp}", idp)
        api = Api(app, settings.dev_jwt_secret.get_secret_value(), idp)
        async with api.client("admin@s3", "Customizer.Admin") as c:
            assert (await c.post("/v1/recipes", json=RECIPE)).status_code in (201, 409)
        async with api.client("alice@s3", SUBMIT) as c:
            assert (
                await c.post("/v1/datasets/finance-ops/invoice-exceptions/versions", content=invoices_ndjson())
            ).status_code == 201
            assert (await c.post("/v1/eval-suites/finance-ops", json=suite_body())).status_code == 201
            job = (
                await c.post(
                    "/v1/customization/jobs", json=job_body(), headers={"Idempotency-Key": f"k-{uuid.uuid4()}"}
                )
            ).json()
        await PipelineRunner(pf).run_until_idle()
        async with api.client("alice@s3", SUBMIT) as c:
            st = (await c.get(f"/v1/customization/jobs/{job['id']}")).json()
        assert st["lifecycle_status"] == "pending_approval", st["error"]
        keys = await pf.store.list("tenants")
        assert any("/datasets/finance-ops/invoice-exceptions/v1/train.jsonl" in k for k in keys)
        assert any("/adapters/sha256/" in k and k.endswith("model.sig") for k in keys)
        assert any("/evidence/" in k for k in keys)
    finally:
        await pf.close()
