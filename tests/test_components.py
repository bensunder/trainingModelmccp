from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest
from mccp.evals.guard import adapter_modules, static_guard
from mccp.evals.stats import bootstrap_mean_ci, cohen_kappa, paired_bootstrap_ci
from mccp.examples.invoices import generate
from mccp.registry.artifacts import (
    ArtifactError,
    Signer,
    generate_signing_keypair,
    materialize_verified,
    publish_adapter,
)
from mccp.serving.clients import FleetProfile
from mccp.storage.objectstore import LocalStore, StorageError, check_key
from mccp.training.backends import RehearsalBackend, SoupBackend, TrainContext, write_peft_adapter
from tests.conftest import RECIPE, ROOT, job_body


@pytest.fixture
def recipe() -> Recipe:
    return Recipe.model_validate(RECIPE)


@pytest.fixture
def cfg(recipe):
    return JobRequest.model_validate(job_body()).resolve(recipe)


@pytest.fixture
def signer(tmp_path) -> Signer:
    generate_signing_keypair(tmp_path / "k.pem", tmp_path / "p.pem")
    return Signer("key", tmp_path / "k.pem", tmp_path / "p.pem")


class Cb:
    def __init__(self, control="continue"):
        self.metrics, self.ckpts, self._control = [], [], control

    async def on_metrics(self, step, m):
        self.metrics.append((step, m))

    async def on_checkpoint(self, step, path):
        self.ckpts.append((step, path))

    async def control(self):
        return self._control


def ctx(tmp_path, cfg, recipe) -> TrainContext:
    recs = generate(10)
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(r) for r in recs[:2]))
    return TrainContext(
        "ftjob-" + "0" * 32,
        cfg,
        recipe.dims,
        tmp_path / "wd",
        [{"messages": r["messages"]} for r in recs[:8]],
        [{"messages": recs[8]["messages"]}],
        golden,
    )


class TestArtifacts:
    async def test_sign_publish_verify_roundtrip(self, tmp_path, cfg, recipe, signer):
        adapter = write_peft_adapter(tmp_path / "a", cfg, recipe.dims)
        store = LocalStore(tmp_path / "store")
        art = await publish_adapter(store, "tenants/t1", adapter, tmp_path / "stage", signer)
        assert set(art.attested) == {"adapter_config.json", "adapter_model.safetensors"}
        assert art.key_prefix == f"tenants/t1/adapters/sha256/{art.digest}"
        dest = await materialize_verified(store, art.key_prefix, art.digest, tmp_path / "lora" / "x-v1", signer)
        assert (dest / "adapter_model.safetensors").exists()

    async def test_tampered_artifact_never_materializes(self, tmp_path, cfg, recipe, signer):
        adapter = write_peft_adapter(tmp_path / "a", cfg, recipe.dims)
        store = LocalStore(tmp_path / "store")
        art = await publish_adapter(store, "tenants/t1", adapter, tmp_path / "stage", signer)
        target = tmp_path / "store" / art.key_prefix / "adapter_config.json"
        target.write_text(target.read_text().replace('"r": 16', '"r": 32'))
        dest = tmp_path / "lora" / "x-v1"
        with pytest.raises(ArtifactError):
            await materialize_verified(store, art.key_prefix, art.digest, dest, signer)
        assert not dest.exists()

    async def test_wrong_key_fails_verification(self, tmp_path, cfg, recipe, signer):
        adapter = write_peft_adapter(tmp_path / "a", cfg, recipe.dims)
        store = LocalStore(tmp_path / "store")
        art = await publish_adapter(store, "tenants/t1", adapter, tmp_path / "stage", signer)
        generate_signing_keypair(tmp_path / "k2.pem", tmp_path / "p2.pem")
        other = Signer("key", tmp_path / "k.pem", tmp_path / "p2.pem")
        with pytest.raises(ArtifactError):
            await materialize_verified(store, art.key_prefix, art.digest, tmp_path / "lora" / "y", other)

    async def test_wrong_digest_rejected(self, tmp_path, cfg, recipe, signer):
        adapter = write_peft_adapter(tmp_path / "a", cfg, recipe.dims)
        store = LocalStore(tmp_path / "store")
        art = await publish_adapter(store, "tenants/t1", adapter, tmp_path / "stage", signer)
        with pytest.raises(ArtifactError, match="digest"):
            await materialize_verified(store, art.key_prefix, "0" * 64, tmp_path / "lora" / "z", signer)


class TestStorage:
    @pytest.mark.parametrize("bad", ["/abs", "../x", "a/../b", "a//b", "a/./b", "", "a b"])
    def test_key_validation(self, bad):
        with pytest.raises(StorageError):
            check_key(bad)

    async def test_roundtrip_and_list(self, tmp_path):
        s = LocalStore(tmp_path)
        await s.put_bytes("t/a/1.json", b"{}")
        await s.put_bytes("t/a/2.json", b"[]")
        assert await s.list("t/a") == ["t/a/1.json", "t/a/2.json"]
        assert await s.get_bytes("t/a/2.json") == b"[]"
        with pytest.raises(StorageError):
            await s.get_bytes("t/a/missing")


class TestGuard:
    def test_static_guard_on_rehearsal_adapter(self, tmp_path, cfg, recipe):
        adapter = write_peft_adapter(tmp_path / "a", cfg, recipe.dims)
        mods, conf = adapter_modules(adapter)
        assert mods == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} and conf["r"] == 16
        fleet = FleetProfile.load(ROOT / "deploy/fleet.yaml")
        assert static_guard(adapter, fleet=fleet, base_model_id="Qwen/Qwen3-8B")["problems"] == []

    def test_static_guard_flags_skipped_modules_and_rank(self, tmp_path, cfg, recipe):
        adapter = write_peft_adapter(
            tmp_path / "a", cfg, recipe.dims, modules={"linear_attn.in_proj_a": (64, 8), "self_attn.q_proj": (64, 64)}
        )
        fleet = FleetProfile.load(ROOT / "deploy/fleet.yaml")
        fleet.max_lora_rank = 8
        g = static_guard(adapter, fleet=fleet, base_model_id="Qwen/Qwen3-8B")
        assert g["skipped_modules"] == ["in_proj_a"]
        assert any("rank 16 exceeds" in p for p in g["problems"])


class TestTrainingBackends:
    async def test_rehearsal_writes_noop_peft_adapter(self, tmp_path, cfg, recipe):
        import numpy as np
        from safetensors.numpy import load_file

        cb = Cb()
        out = await RehearsalBackend().train(ctx(tmp_path, cfg, recipe), cb)
        assert out.status == "succeeded" and out.trained_tokens > 0 and len(cb.metrics) == 3
        tensors = load_file(str(out.adapter_dir / "adapter_model.safetensors"))
        assert all(not np.any(v) for k, v in tensors.items() if ".lora_B." in k)
        run = json.loads((tmp_path / "wd" / "run.json").read_text())
        assert run["soup_yaml"]["data"]["format"] == "chatml" and run["soup_yaml"]["training"]["quantization"] == "4bit"
        lines = (tmp_path / "wd" / "data" / "train_and_val.jsonl").read_text().splitlines()
        assert len(lines) == 9  # 8 train + 1 val appended for Soup's val_split

    def _fake_entry(self, tmp_path, body: str) -> Path:
        p = tmp_path / "fake_entry.py"
        p.write_text(textwrap.dedent(body))
        return p

    async def test_soup_backend_consumes_event_protocol(self, tmp_path, cfg, recipe):
        entry = self._fake_entry(
            tmp_path,
            f"""
            import json, os, sys
            sys.path.insert(0, {str(ROOT / "src")!r})
            run = json.load(open("run.json"))
            assert run["base_model"]["revision"] and run["soup_yaml"]["data"]["val_split"] > 0
            def ev(**k):
                with open("events.jsonl", "a") as f: f.write(json.dumps(k) + "\\n")
            ev(type="preflight", trained_tokens=1234, chat_template_sha256="abc")
            ev(type="metrics", step=10, metrics={{"loss": 1.5}})
            ev(type="checkpoint", step=10, path="output/checkpoint-10")
            os.makedirs("output", exist_ok=True)
            open("output/adapter_config.json", "w").write("{{}}")
            open("output/adapter_model.safetensors", "wb").write(b"x")
            ev(type="done", adapter_dir="output", trained_tokens=1234)
            """,
        )
        backend = SoupBackend(sys.executable, entry, [], timeout_s=60, poll_s=0.05)
        cb = Cb()
        out = await backend.train(ctx(tmp_path, cfg, recipe), cb)
        assert out.status == "succeeded", out.error
        assert out.trained_tokens == 1234 and out.adapter_dir.name == "output"
        assert cb.metrics == [(10, {"loss": 1.5})] and cb.ckpts == [(10, "output/checkpoint-10")]

    async def test_soup_backend_cancel_terminates_process_group(self, tmp_path, cfg, recipe):
        entry = self._fake_entry(tmp_path, "import time\nwhile True: time.sleep(0.1)\n")
        backend = SoupBackend(sys.executable, entry, [], timeout_s=60, poll_s=0.05, kill_grace_s=5)
        out = await asyncio.wait_for(backend.train(ctx(tmp_path, cfg, recipe), Cb("cancel")), 30)
        assert out.status == "cancelled"

    async def test_soup_backend_reports_trainer_error(self, tmp_path, cfg, recipe):
        entry = self._fake_entry(
            tmp_path,
            """
            import json, sys
            open("events.jsonl", "a").write(json.dumps({"type": "error", "message": "render-check failed"}) + "\\n")
            sys.exit(2)
            """,
        )
        out = await SoupBackend(sys.executable, entry, [], timeout_s=60, poll_s=0.05).train(
            ctx(tmp_path, cfg, recipe), Cb()
        )
        assert out.status == "failed" and out.error == "render-check failed"


class TestStats:
    def test_paired_bootstrap(self):
        r = paired_bootstrap_ci([1.0] * 50, [0.0] * 50)
        assert r["delta"] == 1.0 and r["ci_low"] == 1.0
        r = paired_bootstrap_ci([1, 0] * 25, [0, 1] * 25)
        assert r["ci_low"] < 0 < r["ci_high"]

    def test_mean_ci_and_kappa(self):
        assert bootstrap_mean_ci([])["n"] == 0
        assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
        assert cohen_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]) < 0


def test_golden_items_use_openai_wire_format():
    """vLLM rejects tool calls without `id`, non-string `arguments` and tool
    results without `tool_call_id` (found in the first real GPU pilot)."""
    from mccp.evals.layers import golden_items
    from mccp.examples.invoices import generate

    items = golden_items(generate(2, seed=3), limit=10)
    assert items
    for it in items:
        ids: list[str] = []
        for m in it.messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    assert tc["id"] and isinstance(tc["function"]["arguments"], str)
                    ids.append(tc["id"])
            if m["role"] == "tool":
                assert m["tool_call_id"] == ids.pop(0)
    # Deterministic: identical requests on every evaluation run.
    again = golden_items(generate(2, seed=3), limit=10)
    assert [i.messages for i in again] == [i.messages for i in items]
