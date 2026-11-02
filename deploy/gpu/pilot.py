"""Single-GPU pilot and benchmark for MCCP (no Docker, no control-plane database).

Validates on real hardware what the sandbox build cannot:
  1. a real `soup train` QLoRA run (trainer/mccp_trainer_entry.py, including its
     pre-flight render-checks against the real tokenizer);
  2. vLLM loading the adapter at runtime with the profile's tool/reasoning parsers;
  3. the L0 publish guard, then a benchmark of the *served* adapter against the
     base model and, optionally, a prompted frontier model, on the held-out
     in-distribution split and an out-of-distribution (OOD) set.

It reuses the production modules (dataset pipeline, recipe/job resolution,
trainer workdir contract, guard, L1 scoring, mccp.evals.benchmark), so a pass
here is evidence about the real code path, not a parallel reimplementation.

Commands (repo root, control-plane venv: `uv run python deploy/gpu/pilot.py ...`):
  prepare   build dataset, OOD set and run.json in WORKDIR (pins the base revision)
  evaluate  guard + benchmark on a running vLLM; writes WORKDIR/report.json
  compare   aggregate several report.json files (e.g. one per seed) into a table
  new-key   print a random vLLM API key

Pilot thresholds in the verdict are engineering judgment for a smoke test, not
the production Rego thresholds in policy/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from mccp.datasets.pii import PatternRedactor, RedactionScope, redact_value
from mccp.datasets.pipeline import BuildParams, DatasetBuildError, build_dataset
from mccp.datasets.quality import contaminated, input_text
from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest
from mccp.evals import layers
from mccp.evals.benchmark import (
    OpenAICompatClient,
    Target,
    compare_reports,
    comparison_markdown,
    run_benchmark,
)
from mccp.evals.guard import live_guard, static_guard
from mccp.evals.suite import EvalSuite
from mccp.examples.invoices import generate, generate_ood
from mccp.serving.clients import FleetProfile, ServingError, VllmClient
from mccp.training.backends import TrainContext, prepare_workdir

REPO = Path(__file__).resolve().parents[2]
BASE_ID = "Qwen/Qwen3-8B"
BASE_SERVED = "qwen3-8b-base"
PILOT_RECIPE_REF = "qwen3-8b-qlora-sft@v1.0.0+A5000"
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
SET_FILES = {"in_dist": "golden.jsonl", "ood": "golden_ood.jsonl"}


def _jsonl(b: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in b.decode("utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def resolve_revision(model_id: str, timeout: float = 30.0) -> str:
    r = httpx.get(f"https://huggingface.co/api/models/{model_id}", timeout=timeout)
    r.raise_for_status()
    sha = r.json().get("sha")
    if not sha or len(sha) != 40:
        raise SystemExit(f"could not resolve a commit SHA for {model_id}: {r.text[:200]}")
    return sha


def redact_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Apply the same PII redaction the dataset pipeline applies to golden data."""
    redactor = PatternRedactor()
    out, total = [], 0
    for rec in records:
        scope = RedactionScope()
        d = json.loads(json.dumps(rec))
        for m in d["messages"]:
            m["content"] = redactor.redact(m.get("content") or "", scope)
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = redact_value(tc["function"]["arguments"], redactor, scope)
        total += sum(scope.counters.values())
        out.append(d)
    return out, total


def adapter_serving_name(workdir: Path) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", workdir.name.lower()).strip("-") or "run"
    return f"invoice-triage-{slug}"[:64]


# ------------------------------------------------------------------- prepare --
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
    #    `--data-seed` fixes the data and the held-out split, so runs that differ only in
    #    `--seed` (training data order / LoRA init) are scored on identical test items.
    raw = b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in generate(args.n, seed=args.data_seed))
    try:
        built = build_dataset(raw, PatternRedactor(), BuildParams(seed=args.data_seed))
    except DatasetBuildError as e:
        print(f"dataset build failed: {e}\n{json.dumps(e.report, indent=2)[:2000]}", file=sys.stderr)
        return 3
    if not built.card["pii"]["passed"]:
        print("dataset build: residual PII scan failed; refusing to train", file=sys.stderr)
        return 3
    train_rows = _jsonl(built.files["train.jsonl"])
    val_rows = _jsonl(built.files["val.jsonl"])
    test_rows = _jsonl(built.files["test.jsonl"])

    # 2. OOD set: fixed seed (independent of the training seed) so every run is
    #    scored on identical OOD records; redacted like the golden split.
    ood_rows, ood_pii = redact_records(generate_ood(args.ood_n, seed=args.ood_seed))
    overlap = sum(
        contaminated(
            [input_text(r["messages"]) for r in ood_rows],
            [input_text(r["messages"]) for r in train_rows],
            n=13,
        )
    )
    if overlap:
        print(f"OOD set shares {overlap} non-boilerplate 13-grams with training inputs; refusing", file=sys.stderr)
        return 3

    # 3. Recipe (pilot variant of the shipped one) and job resolution through the domain models.
    raw_recipe["ref"] = PILOT_RECIPE_REF
    raw_recipe["trainer_image"] = "local:gridshare-pilot"  # no image: trainer runs from a hash-locked venv
    raw_recipe["base_model"]["revision"] = revision
    # 20-24 GB card: halve the micro-batch, keep the effective batch at 16.
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

    # 4. Trainer workdir: data, golden sets, run.json (the trainer entrypoint's contract).
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    golden = workdir / "data" / SET_FILES["in_dist"]
    golden.write_bytes(built.files["test.jsonl"])
    _write_jsonl(workdir / "data" / SET_FILES["ood"], ood_rows)
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
        "seed": args.seed,
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "dataset": {"manifest_sha256": built.manifest_sha256, "card": built.card},
        "ood": {"records": len(ood_rows), "seed": args.ood_seed, "pii_placeholders": ood_pii, "train_overlap": 0},
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
                "seed": args.seed,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "golden_records": len(test_rows),
                "ood_records": len(ood_rows),
                "approx_optimizer_steps": steps,
                "approx_train_hours_rtx4000ada": round(steps * 18 / 3600, 1),
                "pii_placeholders": built.card["pii"]["placeholders_by_type"],
            },
            indent=2,
        )
    )
    return 0


# ------------------------------------------------------------------ evaluate --
def _verdict(static: dict[str, Any] | None, live: dict[str, Any] | None, in_dist: dict[str, Any]) -> dict[str, Any]:
    t = in_dist["targets"]
    a = t["adapter"]["l1"]
    b = t["base"]["l1"] if "base" in t else None
    checks: dict[str, bool | None] = {
        "static_guard_clean": not static["problems"] if static else None,
        "adapter_has_effect": not live["no_observable_effect"] if live else None,
        "tool_calls_parsed_ge_95pct": (a["tool_call_parse_rate"] or 0) >= 0.95,
        "no_tool_calls_leaked_into_content": a["tool_calls_leaked_into_content"] == 0,
        "json_validity_ge_90pct": (a["json_validity"] or 0) >= 0.90,
        "no_thinking_in_answers": a["thinking_leaks"] == 0,
        "beats_base_on_task_score": a["task_score"]["mean"] > b["task_score"]["mean"] if b else None,
        "no_endpoint_errors": t["adapter"]["calls"]["errors"] == 0,
    }
    return {"pass": all(v for v in checks.values() if v is not None), "checks": checks}


def _frontier_target(args: argparse.Namespace) -> tuple[Target, OpenAICompatClient] | None:
    if not args.baseline_url:
        return None
    if not args.baseline_model:
        raise SystemExit("--baseline-model is required with --baseline-url")
    key = os.environ.get(args.baseline_key_env)
    if not key:
        raise SystemExit(f"set the frontier API key in ${args.baseline_key_env} (never on the command line)")
    client = OpenAICompatClient(args.baseline_url, key, auth=args.baseline_auth, timeout=args.timeout)

    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        return await client.chat({**payload, "parallel_tool_calls": False})

    target = Target(
        name="frontier",
        kind="frontier",
        model=args.baseline_model,
        chat=chat,
        extra_body={},  # no vLLM-only fields (chat_template_kwargs) to a hosted API
        concurrency=args.baseline_concurrency,
        usd_per_m_input=args.baseline_price_in,
        usd_per_m_output=args.baseline_price_out,
    )
    return target, client


async def _evaluate(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    wanted = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = set(wanted) - {"adapter", "base", "frontier"}
    if unknown or not wanted:
        print(f"--targets must be a subset of adapter,base,frontier (got {args.targets})", file=sys.stderr)
        return 2
    if "frontier" in wanted and not args.baseline_url:
        print("--targets includes frontier but no --baseline-url was given", file=sys.stderr)
        return 2
    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    sets: dict[str, list[dict[str, Any]]] = {}
    for s in set_names:
        path = workdir / "data" / SET_FILES.get(s, f"{s}.jsonl")
        if not path.exists():
            print(f"test set `{s}` not found at {path} (re-run prepare)", file=sys.stderr)
            return 2
        sets[s] = _jsonl(path.read_bytes())
    if "in_dist" not in sets:
        print("--sets must include in_dist (the verdict is computed on it)", file=sys.stderr)
        return 2

    adapter_dir = workdir / "output"
    adapter_name = args.adapter_name or adapter_serving_name(workdir)
    suite = EvalSuite.model_validate(
        yaml.safe_load((REPO / "examples/suite-invoice-triage.yaml").read_text(encoding="utf-8"))
    )
    vllm = VllmClient(args.vllm_url, args.vllm_api_key, timeout=args.timeout)
    static = live = None
    frontier = None
    loaded = False
    try:
        if {"adapter", "base"} & set(wanted) and not await vllm.has_model(BASE_SERVED):
            print(f"vLLM at {args.vllm_url} does not serve `{BASE_SERVED}`", file=sys.stderr)
            return 2
        targets: list[Target] = []
        if "adapter" in wanted:
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapter_dir / f).exists():
                    print(f"missing {adapter_dir / f}: training has not produced an adapter", file=sys.stderr)
                    return 2
            static = static_guard(
                adapter_dir, fleet=FleetProfile.load(REPO / "deploy/fleet.yaml"), base_model_id=BASE_ID
            )
            probe_items = layers.golden_items(sets["in_dist"], suite.effect_probes)
            probes = [{"messages": it.messages, **({"tools": it.tools} if it.tools else {})} for it in probe_items]
            live = await live_guard(
                vllm,
                serving_name=adapter_name,
                lora_path=str(adapter_dir),
                base_served_name=BASE_SERVED,
                probes=probes,
                extra_body=EXTRA_BODY,
            )
            loaded = True
            targets.append(
                Target(
                    "adapter",
                    "adapter",
                    adapter_name,
                    vllm.chat,
                    EXTRA_BODY,
                    args.concurrency,
                    gpu_usd_per_hour=args.gpu_usd_per_hour,
                )
            )
        if "base" in wanted:
            targets.append(
                Target(
                    "base",
                    "base",
                    BASE_SERVED,
                    vllm.chat,
                    EXTRA_BODY,
                    args.concurrency,
                    gpu_usd_per_hour=args.gpu_usd_per_hour,
                )
            )
        if "frontier" in wanted:
            ft = _frontier_target(args)
            if ft:
                targets.append(ft[0])
                frontier = ft[1]
        bench = await run_benchmark(sets=sets, targets=targets, suite=suite, max_items=args.max_items)
    except (ServingError, httpx.HTTPError) as e:
        print(f"vLLM error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4
    finally:
        if loaded:
            try:
                await vllm.unload_lora(adapter_name)
            except (ServingError, httpx.HTTPError) as e:
                print(f"warning: could not unload {adapter_name}: {e}", file=sys.stderr)
        if frontier:
            await frontier.aclose()

    in_dist = bench["sets"]["in_dist"]
    report: dict[str, Any] = {
        "pilot": json.loads((workdir / "pilot.json").read_text(encoding="utf-8")),
        "adapter_serving_name": adapter_name if "adapter" in wanted else None,
        "static_guard": static,
        "live_guard": live,
        "benchmark": bench,
        "frontier_adaptations": frontier.adaptations if frontier else None,
    }
    if "adapter" in in_dist["targets"]:
        report["verdict"] = _verdict(static, live, in_dist)
        # Legacy shape (in-distribution adapter vs base), kept for older tooling.
        report["l1"] = {k: v["l1"] for k, v in in_dist["targets"].items() if k in ("adapter", "base")}
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"verdict": report.get("verdict"), "report": str(workdir / "report.json")}
    for s, body in bench["sets"].items():
        summary[s] = {
            name: {
                "valid": t["valid"],
                "task_score": round(t["l1"]["task_score"]["mean"], 4),
                "decision_accuracy": None
                if t["l1"]["decision_accuracy"] is None
                else round(t["l1"]["decision_accuracy"]["mean"], 4),
                "tool_call_accuracy": t["l1"]["tool_call_accuracy"],
                "json_validity": t["l1"]["json_validity"],
                "latency_p50_s": t["calls"]["latency_p50_s"],
                "usd_per_1k_decisions": t["cost"]["usd_per_1k_decisions"],
            }
            for name, t in body["targets"].items()
        }
    print(json.dumps(summary, indent=2, default=str))
    invalid = [n for b in bench["sets"].values() for n, t in b["targets"].items() if not t["valid"]]
    if invalid:
        print(
            f"warning: >5% endpoint errors for {sorted(set(invalid))}; those rows are not valid evidence",
            file=sys.stderr,
        )
    if "verdict" not in report:
        return 0 if not invalid else 1
    return 0 if report["verdict"]["pass"] and not invalid else 1


# ------------------------------------------------------------------- compare --
def compare(args: argparse.Namespace) -> int:
    reports: dict[str, dict[str, Any]] = {}
    for p in args.reports:
        path = Path(p)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "benchmark" not in data:
            print(f"{path} has no benchmark section (older report); skipping", file=sys.stderr)
            continue
        seed = data.get("pilot", {}).get("seed")
        reports[f"{path.parent.name}" + (f" (seed {seed})" if seed is not None else "")] = data["benchmark"]
    if not reports:
        print("no comparable reports", file=sys.stderr)
        return 2
    cmp = compare_reports(reports)
    md = comparison_markdown(cmp)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--workdir", required=True)
    p.add_argument("--n", type=int, default=400, help="synthetic invoice records (pilot default 400)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42, help="training seed (data order); vary it for reproducibility runs")
    p.add_argument("--data-seed", type=int, default=42, help="dataset generation + split seed; keep fixed across seeds")
    p.add_argument("--ood-n", type=int, default=200, help="OOD evaluation records")
    p.add_argument("--ood-seed", type=int, default=1009, help="fixed across runs so OOD sets are identical")
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--revision", help="base model commit; default is the recipe pin")
    p.add_argument("--gpu-hourly-usd", type=float, default=0.35)
    p.add_argument("--force", action="store_true")

    e = sub.add_parser("evaluate")
    e.add_argument("--workdir", required=True)
    e.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    e.add_argument("--vllm-api-key", required=True)
    e.add_argument("--targets", default="adapter,base", help="comma list: adapter,base,frontier")
    e.add_argument("--sets", default="in_dist,ood", help="comma list of test sets: in_dist,ood")
    e.add_argument("--adapter-name", help="vLLM serving name; default derived from the workdir")
    e.add_argument("--max-items", type=int, default=None, help="cap items per set (default: all)")
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--timeout", type=float, default=180.0)
    e.add_argument("--gpu-usd-per-hour", type=float, default=0.35, help="for self-hosted cost per 1k decisions")
    e.add_argument("--baseline-url", help="OpenAI-compatible base URL (before /v1) of the frontier model")
    e.add_argument("--baseline-model", help="model / deployment name at the baseline endpoint")
    e.add_argument("--baseline-key-env", default="MCCP_BASELINE_API_KEY", help="env var holding the API key")
    e.add_argument("--baseline-auth", choices=["bearer", "api-key"], default="bearer")
    e.add_argument("--baseline-concurrency", type=int, default=8)
    e.add_argument("--baseline-price-in", type=float, help="USD per 1M input tokens")
    e.add_argument("--baseline-price-out", type=float, help="USD per 1M output tokens")

    c = sub.add_parser("compare")
    c.add_argument("reports", nargs="+", help="report.json files")
    c.add_argument("--out", help="directory for comparison.json and comparison.md")

    sub.add_parser("new-key", help="print a random vLLM API key")

    args = ap.parse_args()
    if args.cmd == "prepare":
        return prepare(args)
    if args.cmd == "evaluate":
        return asyncio.run(_evaluate(args))
    if args.cmd == "compare":
        return compare(args)
    print(secrets.token_urlsafe(32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
