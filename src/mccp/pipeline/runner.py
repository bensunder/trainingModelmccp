"""Durable pipeline runner (the customization workflow).

Each job is a row in `jobs` plus, while a worker should act on it, a row in
`job_queue`. Workers lease rows with `FOR UPDATE SKIP LOCKED` (via the
`mccp_lease_job` SECURITY DEFINER function), heartbeat while working and
release with backoff on failure. Every stage is idempotent: it records what
it produced in `stage_state` before advancing, so a crash at any point
resumes cleanly on another worker. Every transition commits together with its
audit event.

Stage map (lifecycle -> handler):
    created/validating  -> validate inputs              -> queued
    queued/provisioning -> stage data on the GPU worker  -> running
    running             -> train (Soup)                  -> registering | paused | cancelled
    registering         -> sign + content-address + MLflow -> publish_guard
    publish_guard       -> static + live vLLM guard      -> evaluating
    evaluating          -> L1-L6 on the served adapter   -> policy_check
    policy_check        -> OPA (staging)                 -> pending_approval | rejected
    pending_approval    -> OPA (production + approvals)  -> approved | rejected | wait
    approved            -> load + gateway canary split   -> canary
    canary              -> health window + smoke eval    -> deployed | rejected
    deployed            -> rollback request              -> superseded (previous reinstated)
    superseded          -> retirement sweep              -> retired
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text

from mccp.audit.chain import append_event
from mccp.db.engine import jsonb, system_tx, tenant_tx
from mccp.domain.errors import Conflict, NotFound
from mccp.domain.hashing import digest_of
from mccp.domain.recipe import Recipe
from mccp.domain.refs import DatasetRef, SuiteRef, new_id
from mccp.domain.spec import ResolvedConfig
from mccp.domain.status import AdapterStatus
from mccp.domain.status import LifecycleStatus as L
from mccp.evals import guard as publish_guard
from mccp.evals import layers
from mccp.evals.runner import EvalTargets, run_suite
from mccp.evals.suite import EvalSuite
from mccp.governance.evidence import evidence_bundle
from mccp.platform import Platform
from mccp.registry.artifacts import materialize_verified, publish_adapter
from mccp.services import repo
from mccp.services.repo import SYSTEM_ACTOR
from mccp.storage.objectstore import tenant_prefix
from mccp.training.backends import TrainContext

log = logging.getLogger(__name__)


class StageFailed(Exception):
    """Deterministic failure: the job transitions to `failed` without retries."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class Park(Exception):
    """Stop driving this job for now (parked state or scheduled later)."""

    def __init__(self, delay_s: float | None = None) -> None:
        super().__init__()
        self.delay_s = delay_s  # None => remove from queue


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


class PipelineRunner:
    def __init__(self, pf: Platform) -> None:
        self.pf = pf
        self.s = pf.settings
        self.owner = f"{self.s.worker_id}:{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------ main loop --
    async def run_forever(self, stop: asyncio.Event, idle_sleep_s: float = 2.0) -> None:
        last_sweep = 0.0
        while not stop.is_set():
            try:
                worked = await self.run_once()
                if time.monotonic() - last_sweep > 60:
                    await self.retire_sweep()
                    last_sweep = time.monotonic()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("pipeline loop error")
                worked = False
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), idle_sleep_s)

    async def run_until_idle(self, max_iterations: int = 200) -> int:
        """Process ready work until none is left (tests, CLI `--once`)."""
        n = 0
        while n < max_iterations and await self.run_once():
            n += 1
        return n

    async def run_once(self) -> bool:
        async with system_tx(self.pf.engine) as conn:
            row = (
                await conn.execute(
                    text("SELECT * FROM mccp_lease_job(:o, :s)"), {"o": self.owner, "s": self.s.lease_seconds}
                )
            ).first()
        if row is None:
            return False
        await self._drive(row.job_id, row.tenant_id, row.attempts)
        return True

    async def _drive(self, job_id: str, tenant_id: uuid.UUID, attempts: int) -> None:
        hb = asyncio.create_task(self._heartbeat(job_id, tenant_id))
        try:
            progressed = False
            while True:
                async with tenant_tx(self.pf.engine, tenant_id) as conn:
                    job = await repo.get_job(conn, job_id)
                lc = L(job["lifecycle_status"])
                if lc.value in ("failed", "cancelled", "rejected", "retired"):
                    await self._dequeue(job)
                    return
                if job["cancel_requested"] and lc in (L.CREATED, L.VALIDATING, L.QUEUED, L.PROVISIONING, L.RUNNING, L.PAUSED):
                    await self._move(job, L.CANCELLED, data={"reason": "user cancel"}, cancel_requested=False)
                    continue
                handler = self._handlers().get(lc)
                if handler is None:
                    await self._dequeue(job)
                    return
                try:
                    await handler(job)
                    progressed = True
                except Park as p:
                    if p.delay_s is None:
                        await self._dequeue(job)
                    else:
                        await self._release(job, p.delay_s, None, reset=True)
                    return
                except StageFailed as e:
                    await self._fail(job, str(e), e.detail)
                    return
        except Conflict as e:
            # Another actor changed the job (e.g. user cancel); re-drive soon.
            log.info("job %s changed concurrently: %s", job_id, e)
            await self._release_raw(job_id, tenant_id, 1.0, str(e), reset=False)
        except Exception as e:  # noqa: BLE001 - transient: retry with backoff
            log.exception("stage error on job %s (attempt %s)", job_id, attempts)
            if attempts >= self.s.max_stage_attempts and not progressed:
                async with tenant_tx(self.pf.engine, tenant_id) as conn:
                    job = await repo.get_job(conn, job_id)
                with contextlib.suppress(Exception):
                    await self._fail(job, f"stage failed after {attempts} attempts: {type(e).__name__}: {e}", {})
                    return
            delay = min(300.0, 5.0 * 2 ** max(0, attempts - 1))
            await self._release_raw(job_id, tenant_id, delay, f"{type(e).__name__}: {e}"[:1000], reset=progressed)
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    def _handlers(self):
        return {
            L.CREATED: self.stage_created,
            L.VALIDATING: self.stage_validate,
            L.QUEUED: self.stage_admit,
            L.PROVISIONING: self.stage_provision,
            L.RUNNING: self.stage_train,
            L.REGISTERING: self.stage_register,
            L.PUBLISH_GUARD: self.stage_publish_guard,
            L.EVALUATING: self.stage_evaluate,
            L.POLICY_CHECK: self.stage_policy_staging,
            L.PENDING_APPROVAL: self.stage_policy_production,
            L.APPROVED: self.stage_deploy_canary,
            L.CANARY: self.stage_canary,
            L.DEPLOYED: self.stage_deployed,
            L.PAUSED: self._park,
            L.SUPERSEDED: self._park,
        }

    # ---------------------------------------------------------- primitives --
    async def _move(
        self, job: dict[str, Any], to: L, *, data: dict[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            current = await repo.get_job(conn, job["id"], for_update=True)
            return await repo.transition(conn, current, to, actor=SYSTEM_ACTOR, data=data, **fields)

    async def _save_state(self, job: dict[str, Any], **updates: Any) -> dict[str, Any]:
        state = {**(job.get("stage_state") or {}), **updates}
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await repo.update_job(conn, job["id"], stage_state=state)
        job["stage_state"] = state
        return state

    async def _fail(self, job: dict[str, Any], message: str, detail: dict[str, Any]) -> None:
        log.warning("job %s failed: %s", job["id"], message)
        err = {"code": "stage_failed", "stage": job["lifecycle_status"], "message": message[:2000], "detail": detail}
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            current = await repo.get_job(conn, job["id"], for_update=True)
            await repo.transition(
                conn, current, L.FAILED, actor=SYSTEM_ACTOR, data={"message": message[:500]}, error=err
            )
            await repo.dequeue(conn, job["id"])
            if current["adapter_version_id"]:
                v = await repo.get_adapter_version(conn, current["adapter_version_id"])
                if v["status"] in (AdapterStatus.PENDING, AdapterStatus.CANDIDATE, AdapterStatus.CHALLENGER):
                    await repo.set_adapter_status(conn, v, AdapterStatus.REJECTED, actor=SYSTEM_ACTOR)

    async def _dequeue(self, job: dict[str, Any]) -> None:
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await repo.dequeue(conn, job["id"])

    async def _release(self, job: dict[str, Any], delay: float, err: str | None, *, reset: bool) -> None:
        await self._release_raw(job["id"], job["tenant_id"], delay, err, reset=reset)

    async def _release_raw(self, job_id: str, tenant_id: Any, delay: float, err: str | None, *, reset: bool) -> None:
        async with tenant_tx(self.pf.engine, tenant_id) as conn:
            await repo.release(conn, job_id, self.owner, delay_s=delay, error=err, reset_attempts=reset)

    async def _heartbeat(self, job_id: str, tenant_id: Any) -> None:
        while True:
            await asyncio.sleep(self.s.heartbeat_seconds)
            try:
                async with tenant_tx(self.pf.engine, tenant_id) as conn:
                    await repo.heartbeat(conn, job_id, self.owner, self.s.lease_seconds)
            except Exception:  # noqa: BLE001
                log.warning("heartbeat failed for %s", job_id, exc_info=True)

    async def _park(self, job: dict[str, Any]) -> None:
        raise Park(None)

    def _resolved(self, job: dict[str, Any]) -> ResolvedConfig:
        return ResolvedConfig.model_validate(job["resolved"])

    def _workdir(self, job: dict[str, Any]) -> Path:
        return self.s.work_dir / "jobs" / job["id"]

    def _extra_body(self, cfg: ResolvedConfig) -> dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}}

    async def _dataset(self, conn, ref: DatasetRef) -> dict[str, Any]:
        return await repo.get_dataset_version(conn, ref.project, ref.name, ref.version)

    # --------------------------------------------------------------- stages --
    async def stage_created(self, job: dict[str, Any]) -> None:
        await self._move(job, L.VALIDATING)

    async def stage_validate(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        problems = []
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            try:
                Recipe.model_validate((await repo.get_recipe(conn, str(cfg.recipe)))["body"])
                ds = await self._dataset(conn, cfg.training_dataset)
                suite_ref = SuiteRef.parse(job["request"]["eval_suite"])
                await repo.get_suite(conn, job["project"], suite_ref.name, suite_ref.version)
            except NotFound as e:
                raise StageFailed(str(e)) from e
        card = ds["card"]
        if not card["pii"]["passed"]:
            problems.append("training dataset failed its PII scan")
        if card["split"]["samples"]["train"] < 1:
            problems.append("training split is empty")
        try:
            self.pf.fleet.base_for(cfg.base_model.id)
        except KeyError as e:
            problems.append(str(e))
        if problems:
            raise StageFailed("; ".join(problems))
        await self._move(job, L.QUEUED)

    async def stage_admit(self, job: dict[str, Any]) -> None:
        # The leasing worker is the GPU host; admission = this worker took it.
        await self._move(job, L.PROVISIONING, data={"worker": self.owner})

    async def stage_provision(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        wd = self._workdir(job)
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            ds = await self._dataset(conn, cfg.training_dataset)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            await self.pf.store.download_file(f"{ds['artifact_prefix']}/{name}", wd / "dataset" / name)
        await self._save_state(job, workdir=str(wd), dataset_manifest_sha256=ds["manifest_sha256"])
        await self._move(job, L.RUNNING, data={"workdir": str(wd)})

    async def stage_train(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            recipe = Recipe.model_validate((await repo.get_recipe(conn, str(cfg.recipe)))["body"])
        wd = self._workdir(job)
        ds_dir = wd / "dataset"
        if not (ds_dir / "train.jsonl").exists():
            # Resumed on a different worker: re-stage the data.
            async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
                ds = await self._dataset(conn, cfg.training_dataset)
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                await self.pf.store.download_file(f"{ds['artifact_prefix']}/{name}", ds_dir / name)
        ctx = TrainContext(
            job_id=job["id"],
            config=cfg,
            dims=recipe.dims,
            workdir=wd,
            train_rows=_jsonl((ds_dir / "train.jsonl").read_bytes()),
            val_rows=_jsonl((ds_dir / "val.jsonl").read_bytes()),
            golden_path=ds_dir / "test.jsonl",
        )
        runner = self

        class Cb:
            async def on_metrics(self, step: int, metrics: dict[str, Any]) -> None:
                async with tenant_tx(runner.pf.engine, job["tenant_id"]) as conn:
                    await repo.record_metrics(conn, job, step, metrics)

            async def on_checkpoint(self, step: int, path: str) -> None:
                async with tenant_tx(runner.pf.engine, job["tenant_id"]) as conn:
                    await repo.record_checkpoint(conn, job, step, path)

            async def control(self):
                async with tenant_tx(runner.pf.engine, job["tenant_id"]) as conn:
                    j = await repo.get_job(conn, job["id"])
                if j["cancel_requested"]:
                    return "cancel"
                if j["pause_requested"]:
                    return "pause"
                return "continue"

        outcome = await self.pf.backend.train(ctx, Cb())
        prior = job.get("gpu_seconds") or 0.0
        gpu_seconds = prior + outcome.wall_seconds
        cost = round(gpu_seconds / 3600 * recipe.gpu_hourly_usd, 4)
        if outcome.status == "cancelled":
            await self._move(
                job,
                L.CANCELLED,
                data={"reason": "user cancel"},
                cancel_requested=False,
                gpu_seconds=gpu_seconds,
                cost_usd=cost,
            )
            raise Park(None)
        if outcome.status == "paused":
            await self._move(
                job,
                L.PAUSED,
                data={"reason": "user pause"},
                pause_requested=False,
                gpu_seconds=gpu_seconds,
                cost_usd=cost,
            )
            raise Park(None)
        if outcome.status != "succeeded" or outcome.adapter_dir is None:
            async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
                await repo.update_job(conn, job["id"], gpu_seconds=gpu_seconds, cost_usd=cost)
            raise StageFailed(outcome.error or "training failed", {"preflight": outcome.preflight})
        tokens = int(outcome.trained_tokens or 0)
        await self._save_state(
            job, adapter_dir=str(outcome.adapter_dir), preflight=outcome.preflight, trainer=self.pf.backend.name
        )
        await self._move(
            job,
            L.REGISTERING,
            data={"trained_tokens": tokens, "gpu_seconds": round(gpu_seconds, 1)},
            trained_tokens=tokens,
            gpu_seconds=gpu_seconds,
            cost_usd=cost,
            flop_estimate=recipe.flop_estimate(tokens),
        )

    async def stage_register(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        state = job.get("stage_state") or {}
        if job["adapter_version_id"]:
            await self._move(job, L.PUBLISH_GUARD)
            return
        adapter_dir = Path(state["adapter_dir"])
        if not adapter_dir.exists():
            raise StageFailed(f"adapter directory {adapter_dir} is gone; retrain required")
        art = await publish_adapter(
            self.pf.store, tenant_prefix(job["tenant_id"]), adapter_dir, self._workdir(job) / "staging", self.pf.signer
        )
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        name = job["output_adapter"]
        tshort = str(job["tenant_id"]).split("-")[0]
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                {"k": f"adapter:{job['tenant_id']}:{job['project']}/{name}"},
            )
            version = int(
                (
                    await conn.execute(
                        text(
                            "SELECT COALESCE(MAX(version),0)+1 FROM adapter_versions WHERE project = :p AND name = :n"
                        ),
                        {"p": job["project"], "n": name},
                    )
                ).scalar_one()
            )
            vid = new_id("adv")
            serving_name = f"{tshort}-{job['project']}-{name}-v{version}"
            card = {
                "base_model": cfg.base_model.model_dump(),
                "peft": {
                    k: adapter_cfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora")
                },
                "quantization_at_train": "nf4" if cfg.finetuning_type == "qlora" else "bf16",
                "serving_dtype": "bfloat16",
                "chat_template_sha256": state.get("preflight", {}).get(
                    "chat_template_sha256", cfg.chat_template_sha256
                ),
                "enable_thinking": cfg.enable_thinking,
                "train": {"job_id": job["id"], "config_hash": job["config_hash"], "trainer": state.get("trainer")},
                "manifest": art.manifest,
                "artifact_prefix": art.key_prefix,
                "attested_files": art.attested,
                "signature_sha256": art.signature_sha256,
                "risk_tier": job["risk_tier"],
            }
            await conn.execute(
                text(
                    """INSERT INTO adapter_versions (id, tenant_id, project, name, version, serving_name, artifact_uri,
                                                     artifact_sha256, signature_uri, card, status, job_id)
                       VALUES (:id, CAST(:t AS uuid), :p, :n, :v, :sn, :uri, :sha, :sig, CAST(:card AS jsonb), 'pending', :job)"""
                ),
                {
                    "id": vid,
                    "t": str(job["tenant_id"]),
                    "p": job["project"],
                    "n": name,
                    "v": version,
                    "sn": serving_name,
                    "uri": self.pf.store.uri(art.key_prefix),
                    "sha": art.digest,
                    "sig": self.pf.store.uri(f"{art.key_prefix}/model.sig"),
                    "card": jsonb(card),
                    "job": job["id"],
                },
            )
            await append_event(
                conn,
                job["tenant_id"],
                actor=SYSTEM_ACTOR,
                action="adapter.register",
                resource_type="adapter_version",
                resource_id=vid,
                data={"serving_name": serving_name, "artifact_sha256": art.digest, "job_id": job["id"]},
            )
            await repo.update_job(conn, job["id"], adapter_version_id=vid)
            job["adapter_version_id"] = vid
        mlflow_version = await self.pf.mirror.register_version(
            f"{job['project']}.{name}",
            self.pf.store.uri(art.key_prefix),
            {
                "mccp.serving_name": serving_name,
                "mccp.artifact_sha256": art.digest,
                "mccp.job_id": job["id"],
                "mccp.config_hash": job["config_hash"],
                "mccp.base_model": f"{cfg.base_model.id}@{cfg.base_model.revision}",
                "mccp.approval_status": "pending",
            },
            f"QLoRA SFT adapter from {job['id']}",
        )
        await self._save_state(
            job,
            serving_name=serving_name,
            artifact_digest=art.digest,
            artifact_prefix=art.key_prefix,
            mlflow_version=mlflow_version,
        )
        await self._move(job, L.PUBLISH_GUARD, data={"adapter_version_id": vid, "serving_name": serving_name})

    async def _materialize(self, serving_name: str, artifact_prefix: str, artifact_digest: str) -> tuple[str, str]:
        """Verified copy into the vLLM adapter root; returns (local, server) paths."""
        dest = self.s.vllm_lora_root / serving_name
        if not (dest / "adapter_model.safetensors").exists():
            await materialize_verified(self.pf.store, artifact_prefix, artifact_digest, dest, self.pf.signer)
        server_root = self.s.vllm_lora_root_in_server or str(self.s.vllm_lora_root)
        return str(dest), f"{server_root.rstrip('/')}/{serving_name}"

    async def _materialize_job(self, job: dict[str, Any]) -> tuple[str, str]:
        st = job["stage_state"]
        return await self._materialize(st["serving_name"], st["artifact_prefix"], st["artifact_digest"])

    async def _golden(self, job: dict[str, Any]) -> tuple[EvalSuite, list[dict[str, Any]]]:
        suite_ref = SuiteRef.parse(job["request"]["eval_suite"])
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            suite = EvalSuite.model_validate(
                (await repo.get_suite(conn, job["project"], suite_ref.name, suite_ref.version))["body"]
            )
            g = suite.golden
            ds = await repo.get_dataset_version(conn, g.project, g.name, g.version)
        golden = _jsonl(await self.pf.store.get_bytes(f"{ds['artifact_prefix']}/test.jsonl"))
        return suite, golden

    async def stage_publish_guard(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        local, server_path = await self._materialize_job(job)
        static = publish_guard.static_guard(Path(local), fleet=self.pf.fleet, base_model_id=cfg.base_model.id)
        if static["problems"]:
            await self._save_state(job, publish_guard=static)
            raise StageFailed(
                "publish guard rejected the adapter: " + "; ".join(static["problems"]), {"publish_guard": static}
            )
        suite, golden = await self._golden(job)
        probes = []
        for it in layers.golden_items(golden, suite.effect_probes * 4)[: suite.effect_probes]:
            p: dict[str, Any] = {"messages": it.messages}
            if it.tools:
                p["tools"] = it.tools
            probes.append(p)
        base_served = self.pf.fleet.base_for(cfg.base_model.id).served_name
        live = await publish_guard.live_guard(
            self.pf.vllm,
            serving_name=job["stage_state"]["serving_name"],
            lora_path=server_path,
            base_served_name=base_served,
            probes=probes,
            extra_body=self._extra_body(cfg),
        )
        await self._save_state(job, publish_guard={**static, **live})
        await self._move(job, L.EVALUATING, data={"no_observable_effect": live["no_observable_effect"]})

    async def _champion_serving_name(self, job: dict[str, Any]) -> str | None:
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            dep = await repo.get_deployment(conn, job["project"], job["output_adapter"])
            if dep and dep["champion_version_id"]:
                return (await repo.get_adapter_version(conn, dep["champion_version_id"]))["serving_name"]
        return None

    async def stage_evaluate(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        suite, golden = await self._golden(job)
        champion = await self._champion_serving_name(job)
        base_served = self.pf.fleet.base_for(cfg.base_model.id).served_name
        report = await run_suite(
            suite=suite,
            golden=golden,
            targets=EvalTargets(adapter=job["stage_state"]["serving_name"], base=base_served, champion=champion),
            vllm=self.pf.vllm,
            vllm_base_url=self.s.vllm_base_url,
            extra_body=self._extra_body(cfg),
            concurrency=self.s.eval_concurrency,
            judge_base_url=self.s.judge_base_url,
            judge_api_key=self.s.judge_api_key.get_secret_value() if self.s.judge_api_key else None,
        )
        rid = new_id("evr")
        sha = digest_of(report).removeprefix("sha256:")
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await conn.execute(
                text(
                    """INSERT INTO eval_reports (id, tenant_id, adapter_version_id, suite_ref, report, report_sha256)
                       VALUES (:id, CAST(:t AS uuid), :av, :s, CAST(:r AS jsonb), :sha)"""
                ),
                {
                    "id": rid,
                    "t": str(job["tenant_id"]),
                    "av": job["adapter_version_id"],
                    "s": report["suite"],
                    "r": jsonb(report),
                    "sha": sha,
                },
            )
        await self.pf.store.put_bytes(
            f"{tenant_prefix(job['tenant_id'])}/evals/{rid}.json", json.dumps(report).encode()
        )
        await self._save_state(job, eval_report_id=rid)
        await self._move(job, L.POLICY_CHECK, data={"eval_report_id": rid, "summary": report["summary"]})

    async def _bundle(self, job: dict[str, Any]) -> dict[str, Any]:
        cfg = self._resolved(job)
        st = job["stage_state"]
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            v = await repo.get_adapter_version(conn, job["adapter_version_id"])
            rep = (
                await conn.execute(text("SELECT report FROM eval_reports WHERE id = :id"), {"id": st["eval_report_id"]})
            ).scalar_one()
            ds = await self._dataset(conn, cfg.training_dataset)
        card = v["card"]
        # Re-verify the stored artifact right before the decision.
        verified = True
        try:
            await materialize_verified(
                self.pf.store,
                st["artifact_prefix"],
                st["artifact_digest"],
                self._workdir(job) / "verify",
                self.pf.signer,
            )
        except Exception as e:  # noqa: BLE001 - recorded as unverified evidence
            log.warning("re-verification failed for %s: %s", job["id"], e)
            verified = False
        adapter = {
            "serving_name": v["serving_name"],
            "version": v["version"],
            "artifact_sha256": v["artifact_sha256"],
            "artifact_uri": v["artifact_uri"],
            "peft": card["peft"],
            "signatures": {
                "adapter_verified": verified,
                "attested_files": card["attested_files"],
                "manifest_files": {f["path"]: f["sha256"] for f in card["manifest"]["files"]},
                "signature_sha256": card["signature_sha256"],
            },
        }
        datasets = [
            {
                "ref": ds["ref"],
                "manifest_sha256": ds["manifest_sha256"],
                "licenses": ds["card"]["licenses"],
                "pii_scan": ds["card"]["pii"],
            }
        ]
        compute = {
            "trained_tokens": job["trained_tokens"],
            "flop_estimate": job["flop_estimate"],
            "method": "6ND full-model estimate",
            "gpu_seconds": job["gpu_seconds"],
            "cost_usd": job["cost_usd"],
        }
        return evidence_bundle(
            job=job,
            resolved=job["resolved"],
            adapter=adapter,
            datasets=datasets,
            report=repo.as_obj(rep),
            guard=st.get("publish_guard", {}),
            compute=compute,
        )

    async def _record_decision(
        self, job: dict[str, Any], bundle: dict[str, Any], target: str, decision: dict[str, Any]
    ) -> str:
        bundle_sha = digest_of(bundle).removeprefix("sha256:")
        decision_sha = decision["policy_input_sha256"].removeprefix("sha256:")
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await conn.execute(
                text(
                    """INSERT INTO evidence_bundles (sha256, bundle_sha256, tenant_id, job_id, target_env, body, decision)
                       VALUES (:sha, :bsha, CAST(:t AS uuid), :j, :target, CAST(:b AS jsonb), CAST(:d AS jsonb))
                       ON CONFLICT (tenant_id, sha256) DO NOTHING"""
                ),
                {
                    "sha": decision_sha,
                    "bsha": bundle_sha,
                    "t": str(job["tenant_id"]),
                    "j": job["id"],
                    "target": target,
                    "b": jsonb(bundle),
                    "d": jsonb(decision),
                },
            )
            await append_event(
                conn,
                job["tenant_id"],
                actor="system:policy",
                action=f"policy.{target}",
                resource_type="job",
                resource_id=job["id"],
                data={
                    "allow": decision["allow"],
                    "deny": decision["deny"],
                    "evidence_sha256": bundle_sha,
                    "decision_sha256": decision_sha,
                },
            )
        await self.pf.store.put_bytes(
            f"{tenant_prefix(job['tenant_id'])}/evidence/{bundle_sha}.json", json.dumps(bundle).encode()
        )
        return bundle_sha

    async def _adapter_status(self, job: dict[str, Any], status: AdapterStatus, alias: str | None) -> None:
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            v = await repo.get_adapter_version(conn, job["adapter_version_id"])
            await repo.set_adapter_status(conn, v, status, actor=SYSTEM_ACTOR)
            if alias:
                await repo.set_alias(conn, v, alias, actor=SYSTEM_ACTOR)
        mv = (job.get("stage_state") or {}).get("mlflow_version")
        if alias and mv:
            await self.pf.mirror.set_alias(f"{job['project']}.{job['output_adapter']}", alias, mv)

    async def stage_policy_staging(self, job: dict[str, Any]) -> None:
        bundle = await self._bundle(job)
        decision = await self.pf.opa.decide(bundle, target="staging", approvals=[])
        sha = await self._record_decision(job, bundle, "staging", decision)
        if decision["allow"]:
            await self._adapter_status(job, AdapterStatus.CANDIDATE, "candidate")
            await self._move(job, L.PENDING_APPROVAL, data={"evidence_sha256": sha})
            raise Park(None)  # waits for an approval decision
        await self._adapter_status(job, AdapterStatus.REJECTED, None)
        await self._move(
            job,
            L.REJECTED,
            data={"deny": decision["deny"], "evidence_sha256": sha},
            error={"code": "policy_denied", "target": "staging", "deny": decision["deny"]},
        )
        raise Park(None)

    async def stage_policy_production(self, job: dict[str, Any]) -> None:
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            approvals = [
                {"approver": a["approver"], "decision": a["decision"]}
                for a in await repo.approvals_for(conn, job["id"])
            ]
        if not approvals:
            raise Park(None)
        bundle = await self._bundle(job)
        decision = await self.pf.opa.decide(bundle, target="production", approvals=approvals)
        sha = await self._record_decision(job, bundle, "production", decision)
        if decision["allow"]:
            await self._adapter_status(job, AdapterStatus.CHALLENGER, "challenger")
            await self._move(
                job, L.APPROVED, data={"evidence_sha256": sha, "approvers": [a["approver"] for a in approvals]}
            )
            return
        if decision.get("awaiting_approvals"):
            raise Park(None)
        await self._adapter_status(job, AdapterStatus.REJECTED, None)
        await self._move(
            job,
            L.REJECTED,
            data={"deny": decision["deny"], "evidence_sha256": sha},
            error={"code": "policy_denied", "target": "production", "deny": decision["deny"]},
        )
        raise Park(None)

    async def stage_deploy_canary(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        _, server_path = await self._materialize_job(job)
        name = job["stage_state"]["serving_name"]
        await self.pf.vllm.load_lora(name, server_path)
        champion = await self._champion_serving_name(job)
        weights = {champion: 90, name: 10} if champion else {name: 100}
        await self.pf.gateway.set_weights(self._logical(job), weights, extra_body=self._extra_body(cfg))
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await repo.upsert_deployment(
                conn,
                job["tenant_id"],
                job["project"],
                job["output_adapter"],
                challenger_version_id=job["adapter_version_id"],
                challenger_weight=weights[name],
            )
            await append_event(
                conn,
                job["tenant_id"],
                actor=SYSTEM_ACTOR,
                action="deployment.canary",
                resource_type="deployment",
                resource_id=f"{job['project']}/{job['output_adapter']}",
                data={"weights": weights},
            )
        until = time.time() + self.s.canary_window_s
        await self._save_state(job, canary_started=time.time(), canary_until=until, canary_weights=weights)
        await self._move(job, L.CANARY, data={"weights": weights, "window_s": self.s.canary_window_s})
        raise Park(self.s.canary_window_s)

    def _logical(self, job: dict[str, Any]) -> str:
        return f"{job['project']}.{job['output_adapter']}"

    async def stage_canary(self, job: dict[str, Any]) -> None:
        st = job["stage_state"]
        if st.get("rollback_requested"):
            await self._abort_canary(
                job, f"rollback requested by {st['rollback_requested']['by']}: {st['rollback_requested']['reason']}"
            )
            raise Park(None)
        remaining = st["canary_until"] - time.time()
        if remaining > 0:
            raise Park(remaining)
        name = st["serving_name"]
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            fb = (
                await conn.execute(
                    text("""SELECT count(*) AS n, count(*) FILTER (WHERE rating < 0) AS neg FROM feedback
                             WHERE adapter_serving_name = :s AND created_at >= to_timestamp(:since)"""),
                    {"s": name, "since": st["canary_started"]},
                )
            ).one()
        neg_rate = (fb.neg / fb.n) if fb.n else 0.0
        cfg = self._resolved(job)
        suite, golden = await self._golden(job)
        smoke = layers.summarize_l1(
            await layers.run_items(
                self.pf.vllm.chat, name, layers.golden_items(golden, 10), suite, self._extra_body(cfg), 2
            )
        )
        problems = []
        if fb.n >= 5 and neg_rate > self.s.canary_max_negative_feedback_rate:
            problems.append(f"negative feedback rate {neg_rate:.2f} > {self.s.canary_max_negative_feedback_rate}")
        if smoke["errors"]:
            problems.append(f"{smoke['errors']} smoke requests errored")
        if smoke["json_validity"] is not None and smoke["json_validity"] < 1.0:
            problems.append(f"smoke JSON validity {smoke['json_validity']:.2f} < 1.0")
        await self._save_state(
            job, canary_result={"feedback": {"n": fb.n, "negative": fb.neg}, "smoke": smoke, "problems": problems}
        )
        if problems:
            await self._abort_canary(job, "; ".join(problems))
            raise Park(None)
        await self._promote(job)
        raise Park(None)

    async def _abort_canary(self, job: dict[str, Any], reason: str) -> None:
        cfg = self._resolved(job)
        champion = await self._champion_serving_name(job)
        await self.pf.gateway.set_weights(
            self._logical(job), {champion: 100} if champion else {}, extra_body=self._extra_body(cfg)
        )
        await self.pf.vllm.unload_lora(job["stage_state"]["serving_name"])
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await repo.upsert_deployment(
                conn,
                job["tenant_id"],
                job["project"],
                job["output_adapter"],
                challenger_version_id=None,
                challenger_weight=0,
            )
        await self._adapter_status(job, AdapterStatus.REJECTED, None)
        await self._move(
            job, L.REJECTED, data={"reason": reason}, error={"code": "canary_rolled_back", "message": reason}
        )

    async def _promote(self, job: dict[str, Any]) -> None:
        cfg = self._resolved(job)
        name = job["stage_state"]["serving_name"]
        await self.pf.gateway.set_weights(self._logical(job), {name: 100}, extra_body=self._extra_body(cfg))
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            dep = await repo.get_deployment(conn, job["project"], job["output_adapter"], for_update=True)
            old_id = dep["champion_version_id"] if dep else None
            await repo.upsert_deployment(
                conn,
                job["tenant_id"],
                job["project"],
                job["output_adapter"],
                champion_version_id=job["adapter_version_id"],
                challenger_version_id=None,
                challenger_weight=0,
                previous_champion_id=old_id,
            )
            v = await repo.get_adapter_version(conn, job["adapter_version_id"])
            await repo.set_adapter_status(conn, v, AdapterStatus.CHAMPION, actor=SYSTEM_ACTOR)
            await repo.set_alias(conn, v, "champion", actor=SYSTEM_ACTOR)
            if old_id:
                old = await repo.get_adapter_version(conn, old_id)
                await repo.set_adapter_status(conn, old, AdapterStatus.SUPERSEDED, actor=SYSTEM_ACTOR)
                old_job = await repo.get_job(conn, old["job_id"], for_update=True)
                if old_job["lifecycle_status"] == L.DEPLOYED:
                    await repo.transition(
                        conn,
                        old_job,
                        L.SUPERSEDED,
                        actor=SYSTEM_ACTOR,
                        data={"superseded_by": job["adapter_version_id"]},
                    )
            current = await repo.get_job(conn, job["id"], for_update=True)
            await repo.transition(
                conn,
                current,
                L.DEPLOYED,
                actor=SYSTEM_ACTOR,
                data={"logical_name": self._logical(job), "previous_champion": old_id},
            )
        mv = job["stage_state"].get("mlflow_version")
        if mv:
            await self.pf.mirror.set_alias(self._logical(job), "champion", mv)

    async def stage_deployed(self, job: dict[str, Any]) -> None:
        """Only reached when a rollback was requested on the current champion."""
        st = job.get("stage_state") or {}
        req = st.get("rollback_requested")
        if not req:
            raise Park(None)
        cfg = self._resolved(job)
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            dep = await repo.get_deployment(conn, job["project"], job["output_adapter"], for_update=True)
            prev_id = dep["previous_champion_id"] if dep else None
            if not prev_id or dep["champion_version_id"] != job["adapter_version_id"]:
                state = {k: v for k, v in st.items() if k != "rollback_requested"}
                await repo.update_job(conn, job["id"], stage_state=state)
                raise Park(None)
            prev = await repo.get_adapter_version(conn, prev_id)
        _, server_path = await self._materialize(
            prev["serving_name"], prev["card"]["artifact_prefix"], prev["artifact_sha256"]
        )
        await self.pf.vllm.load_lora(prev["serving_name"], server_path)
        await self.pf.gateway.set_weights(
            self._logical(job), {prev["serving_name"]: 100}, extra_body=self._extra_body(cfg)
        )
        async with tenant_tx(self.pf.engine, job["tenant_id"]) as conn:
            await repo.upsert_deployment(
                conn,
                job["tenant_id"],
                job["project"],
                job["output_adapter"],
                champion_version_id=prev_id,
                previous_champion_id=None,
                challenger_version_id=None,
                challenger_weight=0,
            )
            cur_v = await repo.get_adapter_version(conn, job["adapter_version_id"])
            await repo.set_adapter_status(conn, cur_v, AdapterStatus.SUPERSEDED, actor=req["by"])
            await repo.set_adapter_status(conn, prev, AdapterStatus.CHAMPION, actor=req["by"])
            await repo.set_alias(conn, prev, "champion", actor=req["by"])
            prev_job = await repo.get_job(conn, prev["job_id"], for_update=True)
            if prev_job["lifecycle_status"] == L.SUPERSEDED:
                await repo.transition(conn, prev_job, L.DEPLOYED, actor=req["by"], data={"rollback": True})
            current = await repo.get_job(conn, job["id"], for_update=True)
            state = {k: v for k, v in (current["stage_state"] or {}).items() if k != "rollback_requested"}
            await repo.transition(
                conn,
                current,
                L.SUPERSEDED,
                actor=req["by"],
                data={"rollback": True, "reason": req["reason"]},
                stage_state=state,
            )
        raise Park(None)

    # ---------------------------------------------------------- retirement --
    async def retire_sweep(self) -> int:
        async with system_tx(self.pf.engine) as conn:
            due = (
                await conn.execute(text("SELECT * FROM mccp_due_retirements(:s)"), {"s": self.s.retire_after_s})
            ).all()
        n = 0
        for row in due:
            async with tenant_tx(self.pf.engine, row.tenant_id) as conn:
                job = await repo.get_job(conn, row.job_id, for_update=True)
                if job["lifecycle_status"] != L.SUPERSEDED:
                    continue
                v = await repo.get_adapter_version(conn, job["adapter_version_id"])
            await self.pf.vllm.unload_lora(v["serving_name"])
            async with tenant_tx(self.pf.engine, row.tenant_id) as conn:
                job = await repo.get_job(conn, row.job_id, for_update=True)
                await repo.transition(conn, job, L.RETIRED, actor=SYSTEM_ACTOR, data={"unloaded": v["serving_name"]})
                await repo.set_adapter_status(conn, v, AdapterStatus.RETIRED, actor=SYSTEM_ACTOR)
            n += 1
        return n
