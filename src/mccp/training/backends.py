"""Training backends behind one interface.

* `SoupBackend` – production: launches the trainer entrypoint (which runs
  `soup train`) as a supervised subprocess, optionally inside a container via
  `MCCP_TRAINER_COMMAND_PREFIX`, and turns its JSON event stream into job
  metrics, checkpoints and a final outcome. Cancel/pause send SIGTERM to the
  whole process group; resume restarts from the latest checkpoint.
* `RehearsalBackend` – exercises the full pipeline without a GPU. It writes a
  valid PEFT LoRA adapter whose B matrices are zero, i.e. an exact no-op
  adapter. The publish guard flags it as having no observable effect, so the
  governance gate will not let it reach production. Use it to rehearse the
  platform, never to ship a model.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from safetensors.numpy import save_file

from mccp.domain.recipe import ModelDims
from mccp.domain.spec import ResolvedConfig

from .soup_config import build_soup_yaml

Control = Literal["continue", "cancel", "pause"]


@dataclass
class TrainContext:
    job_id: str
    config: ResolvedConfig
    dims: ModelDims
    workdir: Path
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    golden_path: Path


@dataclass
class TrainOutcome:
    status: Literal["succeeded", "cancelled", "paused", "failed"]
    adapter_dir: Path | None = None
    trained_tokens: int | None = None
    wall_seconds: float = 0.0
    preflight: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Callbacks(Protocol):
    async def on_metrics(self, step: int, metrics: dict[str, Any]) -> None: ...
    async def on_checkpoint(self, step: int, path: str) -> None: ...
    async def control(self) -> Control: ...


class TrainerBackend(Protocol):
    name: str

    async def train(self, ctx: TrainContext, cb: Callbacks) -> TrainOutcome: ...


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_workdir(ctx: TrainContext) -> dict[str, Any]:
    """Write data and run.json. Train rows are shuffled with the job seed
    (Soup 0.72 exposes no trainer seed, so data order is the seed we control)."""
    rows = list(ctx.train_rows)
    random.Random(ctx.config.seed).shuffle(rows)
    _write_jsonl(ctx.workdir / "data" / "train_and_val.jsonl", rows + list(ctx.val_rows))
    _write_jsonl(ctx.workdir / "data" / "train.jsonl", rows)
    soup_yaml = build_soup_yaml(ctx.config, n_train=len(rows), n_val=len(ctx.val_rows))
    run = {
        "job_id": ctx.job_id,
        "base_model": {"id": ctx.config.base_model.id, "revision": ctx.config.base_model.revision},
        "chat_template_sha256": ctx.config.chat_template_sha256,
        "enable_thinking": ctx.config.enable_thinking,
        "chat": ctx.config.trainer_chat_spec(),
        "chat_template_kwargs": ctx.config.chat_template_kwargs(),
        "train_path": "data/train.jsonl",
        "golden_path": os.path.relpath(ctx.golden_path, ctx.workdir),
        "soup_yaml": soup_yaml,
    }
    (ctx.workdir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


class SoupBackend:
    name = "soup"

    def __init__(
        self,
        python: str,
        entry: Path,
        command_prefix: list[str],
        timeout_s: int,
        poll_s: float = 2.0,
        kill_grace_s: float = 120.0,
    ) -> None:
        self.kill_grace_s = kill_grace_s
        self.python = python
        self.entry = entry.resolve()
        self.prefix = command_prefix
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    async def train(self, ctx: TrainContext, cb: Callbacks) -> TrainOutcome:
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        prepare_workdir(ctx)
        events_path = ctx.workdir / "events.jsonl"
        offset = events_path.stat().st_size if events_path.exists() else 0
        cmd = [*self.prefix, self.python, str(self.entry), "--run", "run.json"]
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=ctx.workdir,
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        outcome = TrainOutcome(status="failed")
        requested: Control = "continue"
        terminated_at: float | None = None
        try:
            while True:
                offset = await self._drain(events_path, offset, cb, outcome)
                if proc.returncode is not None:
                    break
                if requested == "continue":
                    requested = await cb.control()
                    if requested != "continue":
                        self._signal(proc, signal.SIGTERM)
                        terminated_at = time.monotonic()
                if terminated_at is None and time.monotonic() - started > self.timeout_s:
                    outcome.error = f"training exceeded timeout of {self.timeout_s}s"
                    self._signal(proc, signal.SIGTERM)
                    terminated_at = time.monotonic()
                if terminated_at is not None and time.monotonic() - terminated_at > self.kill_grace_s:
                    self._signal(proc, signal.SIGKILL)
                try:
                    await asyncio.wait_for(proc.wait(), self.poll_s)
                except TimeoutError:
                    pass
            await self._drain(events_path, offset, cb, outcome)
        except BaseException:
            self._signal(proc, signal.SIGTERM)
            raise
        outcome.wall_seconds = time.monotonic() - started
        rc = proc.returncode
        if requested == "cancel":
            outcome.status = "cancelled"
        elif requested == "pause":
            outcome.status = "paused"
        elif rc == 0 and outcome.adapter_dir is not None:
            outcome.status = "succeeded"
        else:
            outcome.status = "failed"
            outcome.error = outcome.error or f"trainer exited with code {rc}"
        return outcome

    @staticmethod
    def _signal(proc: asyncio.subprocess.Process, sig: int) -> None:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass

    @staticmethod
    async def _drain(path: Path, offset: int, cb: Callbacks, outcome: TrainOutcome) -> int:
        def read() -> bytes:
            if not path.exists():
                return b""
            with open(path, "rb") as f:
                f.seek(offset)
                return f.read()

        data = await asyncio.to_thread(read)
        # Only consume complete lines.
        end = data.rfind(b"\n") + 1
        for line in data[:end].splitlines():
            ev = json.loads(line)
            kind = ev.get("type")
            if kind == "metrics":
                await cb.on_metrics(int(ev["step"]), ev.get("metrics", {}))
            elif kind == "checkpoint":
                await cb.on_checkpoint(int(ev["step"]), str(ev["path"]))
            elif kind == "preflight":
                outcome.preflight = {k: v for k, v in ev.items() if k not in ("type", "ts")}
                outcome.trained_tokens = ev.get("trained_tokens")
            elif kind == "done":
                outcome.adapter_dir = path.parent / ev["adapter_dir"]
                outcome.trained_tokens = ev.get("trained_tokens", outcome.trained_tokens)
            elif kind == "error":
                outcome.error = ev.get("message")
        return offset + end


def lora_module_shapes(dims: ModelDims) -> dict[str, tuple[int, int]]:
    """(in_features, out_features) for Qwen3-style dense decoder linears."""
    q = dims.num_attention_heads * dims.head_dim
    kv = dims.num_kv_heads * dims.head_dim
    h, i = dims.hidden_size, dims.intermediate_size
    return {
        "self_attn.q_proj": (h, q),
        "self_attn.k_proj": (h, kv),
        "self_attn.v_proj": (h, kv),
        "self_attn.o_proj": (q, h),
        "mlp.gate_proj": (h, i),
        "mlp.up_proj": (h, i),
        "mlp.down_proj": (i, h),
    }


def write_peft_adapter(
    out: Path, cfg: ResolvedConfig, dims: ModelDims, *, modules: dict[str, tuple[int, int]] | None = None, seed: int = 0
) -> Path:
    """Write a PEFT-format LoRA adapter with zero-initialised B (a no-op)."""
    out.mkdir(parents=True, exist_ok=True)
    hp = cfg.hyperparameters
    modules = modules or lora_module_shapes(dims)
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for layer in range(dims.num_layers):
        for mod, (fin, fout) in modules.items():
            stem = f"base_model.model.model.layers.{layer}.{mod}"
            tensors[f"{stem}.lora_A.weight"] = (rng.standard_normal((hp.lora_r, fin)) * 0.01).astype(np.float16)
            tensors[f"{stem}.lora_B.weight"] = np.zeros((fout, hp.lora_r), dtype=np.float16)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": cfg.base_model.id,
        "revision": cfg.base_model.revision,
        "r": hp.lora_r,
        "lora_alpha": hp.lora_alpha,
        "lora_dropout": hp.lora_dropout,
        "target_modules": sorted({m.split(".")[-1] for m in modules}),
        "bias": "none",
        "use_rslora": False,
        "inference_mode": True,
    }
    (out / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    return out


class RehearsalBackend:
    name = "rehearsal"

    def __init__(self, step_delay_s: float = 0.0, steps: int = 3) -> None:
        self.step_delay_s = step_delay_s
        self.steps = steps

    async def train(self, ctx: TrainContext, cb: Callbacks) -> TrainOutcome:
        started = time.monotonic()
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        prepare_workdir(ctx)
        words = sum(len(json.dumps(r["messages"]).split()) for r in ctx.train_rows)
        trained_tokens = int(words * 1.3) * ctx.config.hyperparameters.n_epochs
        for step in range(1, self.steps + 1):
            if (c := await cb.control()) != "continue":
                return TrainOutcome(
                    status="cancelled" if c == "cancel" else "paused", wall_seconds=time.monotonic() - started
                )
            await cb.on_metrics(
                step, {"loss": round(2.0 / step, 4), "learning_rate": ctx.config.hyperparameters.learning_rate}
            )
            if self.step_delay_s:
                await asyncio.sleep(self.step_delay_s)
        adapter = await asyncio.to_thread(
            write_peft_adapter, ctx.workdir / "output", ctx.config, ctx.dims, seed=ctx.config.seed
        )
        await cb.on_checkpoint(self.steps, "output")
        return TrainOutcome(
            status="succeeded",
            adapter_dir=adapter,
            trained_tokens=trained_tokens,
            wall_seconds=time.monotonic() - started,
            preflight={
                "token_count_method": "approximate (words x 1.3); rehearsal backend",
                "trained_tokens": trained_tokens,
            },
        )


def make_backend(kind: str, *, python: str, entry: Path, prefix: list[str], timeout_s: int) -> TrainerBackend:
    if kind == "soup":
        return SoupBackend(python, entry, prefix, timeout_s)
    if kind == "rehearsal":
        return RehearsalBackend()
    raise ValueError(f"unknown trainer backend {kind!r}")
