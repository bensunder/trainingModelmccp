"""Single-GPU pilot for MCCP (no Docker, no control-plane database).

Validates the three things the sandbox build could not:
  1. a real `soup train` QLoRA run on Qwen3-8B (via trainer/mccp_trainer_entry.py,
     including its pre-flight render-check against the real Qwen3 tokenizer);
  2. vLLM loading the adapter at runtime, with the hermes tool parser and
     thinking disabled;
  3. the L0 publish guard and L1 task evaluation on the *served* adapter vs base.

It reuses the production modules (dataset pipeline, recipe/job resolution,
trainer workdir contract, guard, L1 scoring), so a pass here is evidence about
the real code path, not a parallel reimplementation.

Steps (run from the repo root with the control-plane venv: `uv run python deploy/gpu/pilot.py ...`):
  prepare   build dataset + run.json in WORKDIR (pins the Qwen3-8B revision)
  (train)   run the trainer entrypoint yourself with the trainer venv (see README-pilot)
  evaluate  static guard + live guard + L1 on a running vLLM; writes report.json

Pilot thresholds in `evaluate` are engineering judgment for a smoke test, not
the production Rego thresholds in policy/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from mccp.datasets.pii import PatternRedactor
from mccp.datasets.pipeline import BuildParams, DatasetBuildError, build_dataset
from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest
from mccp.evals import layers
from mccp.evals.guard import live_guard, static_guard
from mccp.evals.suite import EvalSuite
from mccp.examples.invoices import generate
from mccp.serving.clients import FleetProfile, ServingError, VllmClient
from mccp.training.backends import TrainContext, prepare_workdir

REPO = Path(__file__).resolve().parents[2]
BASE_ID = "Qwen/Qwen3-8B"
BASE_SERVED = "qwen3-8b-base"
ADAPTER_SERVED = "invoice-triage-pilot"
PILOT_RECIPE_REF = "qwen3-8b-qlora-sft@v1.0.0+A5000"
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _jsonl(b: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in b.decode("utf-8").splitlines() if line.strip()]


def resolve_revision(model_id: str, timeout: float = 30.0) -> str:
    r = httpx.get(f"https://huggingface.co/api/models/{model_id}", timeout=timeout)
    r.raise_for_status()
    sha = r.json().get("sha")
    if not sha or len(sha) != 40:
        raise SystemExit(f"could not resolve a commit SHA for {model_id}: {r.text[:200]}")
    return sha


def prepare(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    if (workdir / "run.json").exists() and not args.force:
        print(f"{workdir}/run.json already exists; pass --force to rebuild", file=sys.stderr)
        return 2
    raw_recipe = yaml.safe_load((REPO / "examples/recipe-qwen3-8b-qlora-sft.yaml").read_text(encoding="utf-8"))
    pinned = str(raw_recipe["base_model"].get("revision", ""))
    # Prefer an explicit --revision, then the recipe's validated pin; only fall back
    # to "whatever Hugging Face serves now" when the recipe has no real pin.
    revision = args.revision or (pinned if len(pinned) == 40 else resolve_revision(BASE_ID))

    # 1. Dataset through the production pipeline (schema, PII, dedup, group split, decontam, export).
    raw = b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in generate(args.n, seed=args.seed))
    try:
        built = build_dataset(raw, PatternRedactor(), BuildParams(seed=args.seed))
    except DatasetBuildError as e:
        print(f"dataset build failed: {e}\n{json.dumps(e.report, indent=2)[:2000]}", file=sys.stderr)
        return 3
    if not built.card["pii"]["passed"]:
        print("dataset build: residual PII scan failed; refusing to train", file=sys.stderr)
        return 3
    train_rows = _jsonl(built.files["train.jsonl"])
    val_rows = _jsonl(built.files["val.jsonl"])
    test_rows = _jsonl(built.files["test.jsonl"])

    # 2. Recipe (pilot variant of the shipped one) and job resolution through the domain models.
    raw_recipe["ref"] = PILOT_RECIPE_REF
    raw_recipe["trainer_image"] = "local:gridshare-pilot"  # no image: trainer runs from a hash-locked venv
    raw_recipe["base_model"]["revision"] = revision
    # 24 GB card: halve the micro-batch, keep the effective batch at 16.
    raw_recipe["defaults"]["per_device_batch_size"] = 2
    raw_recipe["defaults"]["gradient_accumulation_steps"] = 8
    raw_recipe["soup"]["save_steps"] = args.save_steps
    raw_recipe["gpu_hourly_usd"] = args.gpu_hourly_usd
    recipe = Recipe.model_validate(raw_recipe)

    raw_job = yaml.safe_load((REPO / "examples/job-invoice-triage.yaml").read_text(encoding="utf-8"))
    raw_job["recipe"] = PILOT_RECIPE_REF
    raw_job["method"]["sft"]["hyperparameters"]["n_epochs"] = args.epochs
    raw_job["seed"] = args.seed
    cfg = JobRequest.model_validate(raw_job).resolve(recipe)

    # 3. Trainer workdir: data, golden set, run.json (the trainer entrypoint's contract).
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    golden = workdir / "data" / "golden.jsonl"
    golden.write_bytes(built.files["test.jsonl"])
    run = prepare_workdir(
        TrainContext(
            job_id="pilot",
            config=cfg,
            dims=recipe.dims,
            workdir=workdir,
            train_rows=train_rows,
            val_rows=val_rows,
            golden_path=golden,
        )
    )
    manifest = {
        "base_model": {"id": BASE_ID, "revision": revision},
        "recipe": PILOT_RECIPE_REF,
        "config_hash": cfg.config_hash(),
        "dataset": {"manifest_sha256": built.manifest_sha256, "card": built.card},
        "samples": {"train": len(train_rows), "val": len(val_rows), "golden_records": len(test_rows)},
        "soup_yaml": run["soup_yaml"],
    }
    (workdir / "pilot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    steps = -(-len(train_rows) // 16) * args.epochs
    print(
        json.dumps(
            {
                "workdir": str(workdir),
                "revision": revision,
                "config_hash": manifest["config_hash"],
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "golden_records": len(test_rows),
                "approx_optimizer_steps": steps,
                "pii_placeholders": built.card["pii"]["placeholders_by_type"],
            },
            indent=2,
        )
    )
    return 0


def _verdict(static: dict[str, Any], live: dict[str, Any], l1: dict[str, Any], think_leaks: int) -> dict[str, Any]:
    a, b = l1["adapter"], l1["base"]
    checks = {
        "static_guard_clean": not static["problems"],
        "adapter_has_effect": not live["no_observable_effect"],
        "tool_calls_parsed_ge_95pct": (a["tool_call_parse_rate"] or 0) >= 0.95,
        "no_tool_calls_leaked_into_content": a["tool_calls_leaked_into_content"] == 0,
        "json_validity_ge_90pct": (a["json_validity"] or 0) >= 0.90,
        "no_thinking_in_answers": think_leaks == 0,
        "beats_base_on_task_score": a["task_score"]["mean"] > b["task_score"]["mean"],
        "no_endpoint_errors": a["errors"] == 0 and b["errors"] == 0,
    }
    return {"pass": all(checks.values()), "checks": checks}


async def _evaluate(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    adapter_dir = workdir / "output"
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / f).exists():
            print(f"missing {adapter_dir / f}: training has not produced an adapter", file=sys.stderr)
            return 2

    fleet = FleetProfile.load(REPO / "deploy/fleet.yaml")
    static = static_guard(adapter_dir, fleet=fleet, base_model_id=BASE_ID)

    suite = EvalSuite.model_validate(
        yaml.safe_load((REPO / "examples/suite-invoice-triage.yaml").read_text(encoding="utf-8"))
    )
    golden = _jsonl((workdir / "data" / "golden.jsonl").read_bytes())
    items = layers.golden_items(golden, args.max_items)
    if not items:
        print("golden set produced no evaluation items", file=sys.stderr)
        return 2

    vllm = VllmClient(args.vllm_url, args.vllm_api_key, timeout=args.timeout)
    try:
        if not await vllm.has_model(BASE_SERVED):
            print(f"vLLM at {args.vllm_url} does not serve `{BASE_SERVED}`", file=sys.stderr)
            return 2
        probes = [
            {"messages": it.messages, **({"tools": it.tools} if it.tools else {})}
            for it in items[: suite.effect_probes]
        ]
        live = await live_guard(
            vllm,
            serving_name=ADAPTER_SERVED,
            lora_path=str(adapter_dir),
            base_served_name=BASE_SERVED,
            probes=probes,
            extra_body=EXTRA_BODY,
        )
        adapter_scores = await layers.run_items(vllm.chat, ADAPTER_SERVED, items, suite, EXTRA_BODY, args.concurrency)
        base_scores = await layers.run_items(vllm.chat, BASE_SERVED, items, suite, EXTRA_BODY, args.concurrency)
    except (ServingError, httpx.HTTPError) as e:
        print(f"vLLM error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    l1 = {
        "adapter": layers.summarize_l1(adapter_scores),
        "base": layers.summarize_l1(base_scores),
        "delta_vs_base": layers.delta_vs(adapter_scores, base_scores),
    }
    think_leaks = sum("<think>" in (s.get("content") or "") for s in adapter_scores if s["kind"] == "final")
    report = {
        "pilot": json.loads((workdir / "pilot.json").read_text(encoding="utf-8")).get("base_model"),
        "static_guard": static,
        "live_guard": live,
        "l1": l1,
        "thinking_leaks_in_final_answers": think_leaks,
        "verdict": _verdict(static, live, l1, think_leaks),
        "samples": {
            "adapter_finals": [s.get("content") for s in adapter_scores if s["kind"] == "final"][:3],
            "base_finals": [s.get("content") for s in base_scores if s["kind"] == "final"][:3],
        },
    }
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    a, b = l1["adapter"], l1["base"]
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "trained_modules": static["trained_modules"],
                "skipped_modules": static["skipped_modules"],
                "identical_to_base": f"{live['identical_to_base']}/{live['effect_probes']}",
                "task_score": {"adapter": a["task_score"]["mean"], "base": b["task_score"]["mean"]},
                "tool_call_parse_rate": {"adapter": a["tool_call_parse_rate"], "base": b["tool_call_parse_rate"]},
                "json_validity": {"adapter": a["json_validity"], "base": b["json_validity"]},
                "field_accuracy": {"adapter": a["field_accuracy"], "base": b["field_accuracy"]},
                "report": str(workdir / "report.json"),
            },
            indent=2,
        )
    )
    return 0 if report["verdict"]["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--workdir", required=True)
    p.add_argument("--n", type=int, default=400, help="synthetic invoice records (pilot default 400)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--revision", help="Qwen3-8B commit; default is the recipe pin")
    p.add_argument("--gpu-hourly-usd", type=float, default=0.33)
    p.add_argument("--force", action="store_true")

    e = sub.add_parser("evaluate")
    e.add_argument("--workdir", required=True)
    e.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    e.add_argument("--vllm-api-key", required=True)
    e.add_argument("--max-items", type=int, default=150)
    e.add_argument("--concurrency", type=int, default=4)
    e.add_argument("--timeout", type=float, default=180.0)

    sub.add_parser("new-key", help="print a random vLLM API key")

    args = ap.parse_args()
    if args.cmd == "prepare":
        return prepare(args)
    if args.cmd == "evaluate":
        return asyncio.run(_evaluate(args))
    print(secrets.token_urlsafe(32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
