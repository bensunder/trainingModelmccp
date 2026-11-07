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
import shlex
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from mccp.datasets.pii import PatternRedactor, RedactionScope, redact_value
from mccp.datasets.pipeline import BuildParams, DatasetBuildError, build_dataset
from mccp.datasets.quality import contaminated, input_text
from mccp.domain.profile import ModelProfile, ProfileRegistry
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
from mccp.examples.invoices import TASKS, TRAIN_MIXES, generate, generate_ood
from mccp.examples.messy import generate_messy
from mccp.rgt import Attempt, build_rgt_rows, final_turn_prompt, hinge_rows
from mccp.rgt.invoices import InvoiceRuleGrader, InvoiceTemplateDistiller
from mccp.rgt.pairs import prompt_key
from mccp.serving.clients import FleetProfile, ServingError, VllmClient
from mccp.training.backends import TrainContext, prepare_workdir

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "qwen3-8b@v1"
SET_FILES = {"in_dist": "golden.jsonl", "ood": "golden_ood.jsonl", "messy": "golden_messy.jsonl"}


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


# task -> (eval suite file, decision policy or None when the model decides)
SUITE_FILES = {
    "agent": ("examples/suite-invoice-triage.yaml", None),
    "extract": ("examples/suite-invoice-extract.yaml", "invoice-triage-rules@v1"),
}


def load_suite(task: str) -> EvalSuite:
    if task not in SUITE_FILES:
        raise SystemExit(f"unknown task {task!r} in pilot.json")
    suite = EvalSuite.model_validate(yaml.safe_load((REPO / SUITE_FILES[task][0]).read_text(encoding="utf-8")))
    if suite.decision_rules != SUITE_FILES[task][1]:
        raise SystemExit(f"{SUITE_FILES[task][0]} decision_rules does not match task {task}")
    return suite


def load_profile(ref: str) -> ModelProfile:
    return ProfileRegistry.load(REPO / "profiles").get(ref)


def served_base_name(profile: ModelProfile) -> str:
    return FleetProfile.load(REPO / "deploy/fleet.yaml").base_for(profile.base.hf_id).served_name


def extra_body(profile: ModelProfile) -> dict[str, Any]:
    kwargs = profile.chat_template_kwargs()
    return {"chat_template_kwargs": kwargs} if kwargs else {}


def pilot_recipe_ref(profile: ModelProfile) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", profile.id) + "-qlora-sft@v1.0.0+pilot"


def adapter_serving_name(workdir: Path) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", workdir.name.lower()).strip("-") or "run"
    return f"invoice-triage-{slug}"[:64]


TEST_ID_OFFSET = 1_000_000


async def collect_attempts(
    samples: list[dict[str, Any]],
    *,
    url: str,
    key: str | None,
    model: str,
    extra_body: dict[str, Any],
    cache: Path | None,
    concurrency: int = 8,
) -> dict[str, Attempt]:
    """The served model's own answer to each decision prompt (on-policy, greedy).

    Cached by prompt hash in `cache` (JSONL), so nested training sizes built
    from the same data seed never ask twice. A failed request is recorded as an
    error attempt (the sample still gets its rule strategy, without a mistake)."""
    have: dict[str, Attempt] = {}
    if cache and cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = Attempt.model_validate_json(line)
                if a.model == model and not a.error:
                    have[a.key] = a
    todo = {prompt_key(s): s for s in samples if prompt_key(s) not in have}
    client = VllmClient(url, key, timeout=120.0)
    sem = asyncio.Semaphore(concurrency)

    async def one(k: str, sample: dict[str, Any]) -> Attempt:
        p = final_turn_prompt(sample)
        payload: dict[str, Any] = {
            "model": model,
            "messages": layers.to_openai_messages(p["messages"]),
            "temperature": 0.0,
            "max_tokens": 256,
            **extra_body,
        }
        if p.get("tools"):
            payload["tools"] = p["tools"]
        err = None
        async with sem:
            for _ in range(3):
                try:
                    resp = await client.chat(payload)
                    msg = resp["choices"][0]["message"]
                    return Attempt(key=k, content=msg.get("content") or "", model=model)
                except (ServingError, httpx.HTTPError, KeyError) as e:
                    err = f"{type(e).__name__}: {e}"[:300]
                    await asyncio.sleep(1)
        return Attempt(key=k, content=None, model=model, error=err)

    fresh = await asyncio.gather(*(one(k, s) for k, s in todo.items()))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as f:
            for a in fresh:
                f.write(a.model_dump_json() + "\n")
    have.update({a.key: a for a in fresh})
    errors = sum(1 for a in fresh if a.error)
    print(f"attempts: {len(samples)} prompts, {len(fresh)} new, {errors} errors", file=sys.stderr)
    return have


# ------------------------------------------------------------------- prepare --
def prepare(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    if (workdir / "run.json").exists() and not args.force:
        print(f"{workdir}/run.json already exists; pass --force to rebuild", file=sys.stderr)
        return 2
    if args.task == "extract" and args.method == "rgt":
        print("--method rgt supervises the decision turn; the extract task has none (use sft)", file=sys.stderr)
        return 2
    if args.attempts_url and (args.method != "rgt" or not args.attempts_model):
        print("--attempts-url needs --method rgt and --attempts-model", file=sys.stderr)
        return 2
    profile = load_profile(args.profile)
    if profile.status != "validated" and not args.allow_draft:
        print(
            f"profile {profile.ref} is {profile.status}; run `uv run mccp profile validate {profile.ref} "
            "--trainer-python /root/venvs/trainer/bin/python` first (CPU only)",
            file=sys.stderr,
        )
        return 2
    # Prefer an explicit --revision, then the profile's validated pin; only a draft
    # profile (with --allow-draft) falls back to whatever Hugging Face serves now.
    revision = args.revision or profile.base.revision or resolve_revision(profile.base.hf_id)

    # 1. Dataset through the production pipeline (schema, PII, dedup, group split, decontam, export).
    #    `--data-seed` fixes the data and the held-out split, so runs that differ only in
    #    `--seed` (training data order / LoRA init) are scored on identical test items.
    #    With `--test-n` the test set is generated separately (own seed, disjoint ids)
    #    and is identical for every training size, so learning curves compare like
    #    with like; all `--n` records go to train/val.
    #    `--train-mix` changes only the training scenarios; the fixed test, OOD and
    #    messy sets are always generated from their own seeds and stay identical.
    raw = b"".join(
        json.dumps(r, ensure_ascii=False).encode() + b"\n"
        for r in generate(args.n, seed=args.data_seed, mix=args.train_mix, task=args.task)
    )
    params = BuildParams(seed=args.data_seed)
    test_dropped = 0
    if args.test_n:
        params = BuildParams(seed=args.data_seed, ratios=(0.9, 0.1, 0.0), min_records={"train": 1, "val": 1, "test": 0})
    try:
        built = build_dataset(raw, PatternRedactor(), params)
    except DatasetBuildError as e:
        print(f"dataset build failed: {e}\n{json.dumps(e.report, indent=2)[:2000]}", file=sys.stderr)
        return 3
    if not built.card["pii"]["passed"]:
        print("dataset build: residual PII scan failed; refusing to train", file=sys.stderr)
        return 3
    train_rows = _jsonl(built.files["train.jsonl"])
    val_rows = _jsonl(built.files["val.jsonl"])
    if args.test_n:
        test_rows, _ = redact_records(
            generate(args.test_n, seed=args.test_seed, id_offset=TEST_ID_OFFSET, task=args.task)
        )
        # Same rule as the pipeline's own split: drop train/val samples that share a
        # non-boilerplate 13-gram with the test inputs. With templated text a few
        # chance matches (e.g. a repeated random PO number) are expected; more than
        # 2% would mean real overlap between the generators, so that refuses.
        test_texts = [input_text(r["messages"]) for r in test_rows]
        before = len(train_rows) + len(val_rows)
        flags_train = contaminated([input_text(r["messages"]) for r in train_rows], test_texts, n=13)
        flags_val = contaminated([input_text(r["messages"]) for r in val_rows], test_texts, n=13)
        train_rows = [r for r, bad in zip(train_rows, flags_train, strict=True) if not bad]
        val_rows = [r for r, bad in zip(val_rows, flags_val, strict=True) if not bad]
        test_dropped = before - len(train_rows) - len(val_rows)
        if test_dropped > 0.02 * before:
            print(
                f"{test_dropped}/{before} train/val samples share 13-grams with the fixed test set; refusing",
                file=sys.stderr,
            )
            return 3
        if not train_rows or not val_rows:
            print("no train/val samples left after decontamination", file=sys.stderr)
            return 3
        golden_bytes = b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in test_rows)
    else:
        test_rows = _jsonl(built.files["test.jsonl"])
        golden_bytes = built.files["test.jsonl"]

    # 1b. Rationale-Guided Training: add a strategy-explicit copy of every decision
    #     sample. Optional on-policy attempts (the base model answering the same
    #     prompts) let the strategy name the mistake the model actually makes.
    rgt_provenance: dict[str, Any] | None = None
    strategies: list[dict[str, Any]] = []
    if args.method == "rgt":
        attempts = None
        if args.attempts_url:
            attempts = asyncio.run(
                collect_attempts(
                    [train_rows[i] for i in hinge_rows(train_rows)],
                    url=args.attempts_url,
                    key=args.attempts_key,
                    model=args.attempts_model,
                    extra_body=extra_body(profile),
                    cache=Path(args.attempts_cache) if args.attempts_cache else None,
                    concurrency=args.attempts_concurrency,
                )
            )
        try:
            rgt = build_rgt_rows(
                train_rows,
                grader=InvoiceRuleGrader(),
                distiller=InvoiceTemplateDistiller(),
                reasoning=profile.reasoning.model_dump() if profile.reasoning else None,
                attempts=attempts,
            )
        except ValueError as e:
            print(f"RGT: {e}", file=sys.stderr)
            return 2
        train_rows = rgt.rows
        rgt_provenance = {**rgt.provenance, "attempts_model": args.attempts_model if attempts is not None else None}
        strategies = [r["rgt"] for r in train_rows if r.get("rgt", {}).get("pair") == "B"]

    # 2. OOD set: fixed seed (independent of the training seed) so every run is
    #    scored on identical OOD records; redacted like the golden split.
    ood_rows, ood_pii = redact_records(generate_ood(args.ood_n, seed=args.ood_seed, task=args.task))
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

    # 3. Recipe (the shipped one, re-based on the chosen profile) and job resolution.
    raw_recipe = yaml.safe_load((REPO / "examples/recipe-qwen3-8b-qlora-sft.yaml").read_text(encoding="utf-8"))
    recipe_ref = pilot_recipe_ref(profile)
    raw_recipe["ref"] = recipe_ref
    raw_recipe["trainer_image"] = "local:gridshare-pilot"  # no image: trainer runs from a hash-locked venv
    raw_recipe["profile"] = profile.ref
    raw_recipe["base_model"] = {
        "id": profile.base.hf_id,
        "revision": revision,
        "license": profile.base.license,
        "architecture": profile.base.architecture,
    }
    raw_recipe["dims"] = profile.dims.model_dump()
    raw_recipe["chat_template_sha256"] = profile.base.chat_template_sha256 or "*"
    raw_recipe["enable_thinking"] = False
    raw_recipe["serving"] = {
        **raw_recipe["serving"],
        "tool_call_parser": profile.serving.tool_call_parser,
        "reasoning_parser": profile.serving.reasoning_parser,
    }
    # 20-24 GB card: halve the micro-batch, keep the effective batch at 16.
    raw_recipe["defaults"]["per_device_batch_size"] = 2
    raw_recipe["defaults"]["gradient_accumulation_steps"] = 8
    raw_recipe["soup"]["save_steps"] = args.save_steps
    raw_recipe["gpu_hourly_usd"] = args.gpu_hourly_usd
    recipe = Recipe.model_validate(raw_recipe)

    raw_job = yaml.safe_load((REPO / "examples/job-invoice-triage.yaml").read_text(encoding="utf-8"))
    raw_job["recipe"] = recipe_ref
    raw_job["method"]["sft"]["hyperparameters"]["n_epochs"] = args.epochs
    raw_job["seed"] = args.seed
    if args.task == "extract":
        raw_job["training_dataset"] = "finance-ops/invoice-exceptions-extract@v1:train"
        raw_job["validation_dataset"] = "finance-ops/invoice-exceptions-extract@v1:val"
        raw_job["eval_suite"] = "invoice-extract-suite@v1"
    cfg = JobRequest.model_validate(raw_job).resolve(recipe, profile, allow_draft_profile=args.allow_draft)

    # 4. Trainer workdir: data, golden sets, run.json (the trainer entrypoint's contract).
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    golden = workdir / "data" / SET_FILES["in_dist"]
    golden.write_bytes(golden_bytes)
    if strategies:
        _write_jsonl(workdir / "data" / "rgt_strategies.jsonl", strategies)
    _write_jsonl(workdir / "data" / SET_FILES["ood"], ood_rows)
    messy_meta = None
    if args.messy_n:
        messy_rows, messy_pii = redact_records(generate_messy(args.messy_n, seed=args.messy_seed, task=args.task))
        _write_jsonl(workdir / "data" / SET_FILES["messy"], messy_rows)
        messy_meta = {"records": len(messy_rows), "seed": args.messy_seed, "pii_placeholders": messy_pii}
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
        "base_model": {"id": profile.base.hf_id, "revision": revision},
        "profile": profile.ref,
        "profile_status": profile.status,
        "recipe": recipe_ref,
        "config_hash": cfg.config_hash(),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "method": args.method,
        "task": args.task,
        "decision_rules": SUITE_FILES[args.task][1],
        "train_mix": args.train_mix,
        "records": args.n,
        "rgt": rgt_provenance,
        "test_set": {
            "fixed": bool(args.test_n),
            "records": len(test_rows),
            "seed": args.test_seed if args.test_n else args.data_seed,
            "dropped_train_val_samples": test_dropped,
        },
        "dataset": {"manifest_sha256": built.manifest_sha256, "card": built.card},
        "ood": {"records": len(ood_rows), "seed": args.ood_seed, "pii_placeholders": ood_pii, "train_overlap": 0},
        "messy": messy_meta,
        "samples": {"train": len(train_rows), "val": len(val_rows), "golden_records": len(test_rows)},
        "soup_yaml": run["soup_yaml"],
    }
    (workdir / "pilot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    steps = -(-len(train_rows) // 16) * args.epochs
    print(
        json.dumps(
            {
                "workdir": str(workdir),
                "profile": f"{profile.ref} ({profile.status}, origin {profile.base.origin_country})",
                "revision": revision,
                "config_hash": manifest["config_hash"],
                "seed": args.seed,
                "method": args.method,
                "task": args.task,
                "train_mix": args.train_mix,
                "rgt_b_rows": rgt_provenance["b_rows"] if rgt_provenance else 0,
                "rgt_attempt_pass_rate": rgt_provenance["attempts"]["pass_rate"] if rgt_provenance else None,
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
    tools = a["tool_call_parse_rate"]
    checks: dict[str, bool | None] = {
        "static_guard_clean": not static["problems"] if static else None,
        "adapter_has_effect": not live["no_observable_effect"] if live else None,
        # A task without tool calls (extract) has no tool-call checks, not failed ones.
        "tool_calls_parsed_ge_95pct": None if tools is None else tools >= 0.95,
        "no_tool_calls_leaked_into_content": None if tools is None else a["tool_calls_leaked_into_content"] == 0,
        "json_validity_ge_90pct": (a["json_validity"] or 0) >= 0.90,
        "no_thinking_in_answers": a["thinking_leaks"] == 0,
        "beats_base_on_task_score": a["task_score"]["mean"] > b["task_score"]["mean"] if b else None,
        "no_endpoint_errors": t["adapter"]["calls"]["errors"] == 0,
    }
    return {"pass": all(v for v in checks.values() if v is not None), "checks": checks}


class _OutputSaver:
    """Writes every scored answer (raw model output, the canonicalized form, field
    hits) to `<report-stem>.<set>.<target>.outputs.jsonl` in the workdir, so a
    failure can be read instead of guessed. Contains only test-set text (already
    PII-redacted) and model answers; no keys."""

    def __init__(self, workdir: Path, report_name: str) -> None:
        self.workdir = workdir
        self.stem = Path(report_name).stem
        self.written: list[Path] = []

    def __call__(self, set_name: str, target: str, items: list, records: list, scores: list) -> None:
        path = self.workdir / f"{self.stem}.{set_name}.{target}.outputs.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for it, sc in zip(items, scores, strict=True):
                meta = records[it.record].get("meta") or {}
                row = {
                    "record": it.record,
                    "source_id": meta.get("source_id"),
                    "scenario": meta.get("scenario"),
                    "ood_kind": meta.get("ood_kind"),
                    "messy_kinds": meta.get("messy_kinds"),
                    "kind": sc.get("kind"),
                    "score": sc.get("score"),
                    "json_valid": sc.get("json_valid"),
                    "json_valid_raw": sc.get("json_valid_raw"),
                    "normalized": sc.get("normalized"),
                    "fields": sc.get("fields"),
                    "error": sc.get("error"),
                    "expected": it.expected.get("content") or it.expected.get("tool_calls"),
                    "content": sc.get("content"),
                }
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.written.append(path)


def _frontier_target(args: argparse.Namespace) -> tuple[Target, OpenAICompatClient] | None:
    if not args.baseline_url:
        return None
    if not args.baseline_model:
        raise SystemExit("--baseline-model is required with --baseline-url")
    key = os.environ.get(args.baseline_key_env)
    if not key:
        raise SystemExit(f"set the frontier API key in ${args.baseline_key_env} (never on the command line)")
    drop = tuple(x.strip() for x in (args.baseline_drop or "").split(",") if x.strip())
    try:
        extra = json.loads(args.baseline_extra_body) if args.baseline_extra_body else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"--baseline-extra-body is not valid JSON: {e}") from e
    if not isinstance(extra, dict):
        raise SystemExit("--baseline-extra-body must be a JSON object")
    client = OpenAICompatClient(args.baseline_url, key, auth=args.baseline_auth, timeout=args.timeout, drop_params=drop)

    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        # One call per turn, like the adapter; only meaningful (and only allowed) with tools.
        return await client.chat({**payload, "parallel_tool_calls": False} if payload.get("tools") else payload)

    target = Target(
        name="frontier",
        kind="frontier",
        model=args.baseline_model,
        chat=chat,
        extra_body=extra,  # provider fields only (e.g. Kimi `thinking`); never vLLM's chat_template_kwargs
        concurrency=args.baseline_concurrency,
        usd_per_m_input=args.baseline_price_in,
        usd_per_m_output=args.baseline_price_out,
    )
    return target, client


async def _frontier_preflight(target: Target) -> str | None:
    """One tiny call before hundreds of real ones. Returns why the frontier must be
    skipped (bad key, no credit, unknown model), or None when it answered."""
    try:
        await target.chat(
            {
                "model": target.model,
                "messages": [{"role": "user", "content": "Reply OK."}],
                "max_tokens": 16,
                **target.extra_body,
            }
        )
    except Exception as e:  # noqa: BLE001 - reported, the run continues without the frontier row
        return str(e)[:400]
    return None


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
    if "in_dist" not in sets and args.report_name == "report.json":
        print(
            "--sets must include in_dist for the main report (the verdict is computed on it); "
            "use --report-name for an additional set",
            file=sys.stderr,
        )
        return 2
    # Adapter-effect probes: in-dist when available; a workdir restored from /workspace
    # (machine lost, only saved results left) has only the set being scored.
    in_dist_path = workdir / "data" / SET_FILES["in_dist"]
    probe_records = (
        sets.get("in_dist")
        or (_jsonl(in_dist_path.read_bytes()) if in_dist_path.exists() else None)
        or next(iter(sets.values()))
    )

    adapter_dir = workdir / "output"
    adapter_name = args.adapter_name or adapter_serving_name(workdir)
    pilot_meta = json.loads((workdir / "pilot.json").read_text(encoding="utf-8"))
    profile = load_profile(pilot_meta.get("profile", DEFAULT_PROFILE))
    base_served = served_base_name(profile)
    xbody = extra_body(profile)
    suite = load_suite(pilot_meta.get("task", "agent"))
    vllm = VllmClient(args.vllm_url, args.vllm_api_key, timeout=args.timeout)
    static = live = None
    frontier = None
    frontier_skipped: str | None = None
    loaded = False
    try:
        if {"adapter", "base"} & set(wanted) and not await vllm.has_model(base_served):
            print(f"vLLM at {args.vllm_url} does not serve `{base_served}`", file=sys.stderr)
            return 2
        targets: list[Target] = []
        if "adapter" in wanted:
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapter_dir / f).exists():
                    print(f"missing {adapter_dir / f}: training has not produced an adapter", file=sys.stderr)
                    return 2
            static = static_guard(
                adapter_dir, fleet=FleetProfile.load(REPO / "deploy/fleet.yaml"), base_model_id=profile.base.hf_id
            )
            probe_items = layers.golden_items(probe_records, suite.effect_probes)
            probes = [{"messages": it.messages, **({"tools": it.tools} if it.tools else {})} for it in probe_items]
            live = await live_guard(
                vllm,
                serving_name=adapter_name,
                lora_path=str(adapter_dir),
                base_served_name=base_served,
                probes=probes,
                extra_body=xbody,
            )
            loaded = True
            targets.append(
                Target(
                    "adapter",
                    "adapter",
                    adapter_name,
                    vllm.chat,
                    xbody,
                    args.concurrency,
                    gpu_usd_per_hour=args.gpu_usd_per_hour,
                )
            )
        if "base" in wanted:
            targets.append(
                Target(
                    "base",
                    "base",
                    base_served,
                    vllm.chat,
                    xbody,
                    args.concurrency,
                    gpu_usd_per_hour=args.gpu_usd_per_hour,
                )
            )
        if "frontier" in wanted:
            ft = _frontier_target(args)
            if ft:
                frontier = ft[1]
                frontier_skipped = await _frontier_preflight(ft[0])
                if frontier_skipped:
                    print(f"frontier baseline skipped: {frontier_skipped}", file=sys.stderr)
                else:
                    targets.append(ft[0])
        saver = _OutputSaver(workdir, args.report_name) if args.save_outputs else None
        bench = await run_benchmark(sets=sets, targets=targets, suite=suite, max_items=args.max_items, on_scores=saver)
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

    in_dist = bench["sets"].get("in_dist")
    report: dict[str, Any] = {
        "pilot": pilot_meta,
        "adapter_serving_name": adapter_name if "adapter" in wanted else None,
        "static_guard": static,
        "live_guard": live,
        "benchmark": bench,
        "frontier_adaptations": frontier.adaptations if frontier else None,
        "frontier_skipped": frontier_skipped,
    }
    if in_dist and "adapter" in in_dist["targets"]:
        report["verdict"] = _verdict(static, live, in_dist)
        # Legacy shape (in-distribution adapter vs base), kept for older tooling.
        report["l1"] = {k: v["l1"] for k, v in in_dist["targets"].items() if k in ("adapter", "base")}
    report_path = workdir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"verdict": report.get("verdict"), "report": str(report_path)}
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
                "json_validity_raw": t["l1"].get("json_validity_raw"),
                "normalized_answers": t["l1"].get("normalized_answers"),
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


# --------------------------------------------------------------- profile-env --
def profile_env(args: argparse.Namespace) -> int:
    """Shell exports for the batch scripts (eval "$(pilot.py profile-env ...)")."""
    p = load_profile(args.profile)
    exports = {
        "PROFILE_REF": p.ref,
        "PROFILE_ID": p.id,
        "PROFILE_STATUS": p.status,
        "PROFILE_ORIGIN": p.base.origin_country,
        "HF_ID": p.base.hf_id,
        "HF_REVISION": p.base.revision or "",
        "BASE_SERVED": served_base_name(p),
        "TOOL_PARSER": p.serving.tool_call_parser,
        "REASONING_PARSER": p.serving.reasoning_parser or "",
        "THINKING_KWARG": p.reasoning.template_kwarg if p.reasoning else "",
    }
    for k, v in exports.items():
        print(f"export {k}={shlex.quote(str(v))}")
    return 0


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


# ------------------------------------------------------------------- curve --
def _stage_seconds(workdir: Path) -> dict[str, float]:
    """Stage wall times recorded by run_batch.sh (timing.jsonl) and the trainer (events)."""
    out: dict[str, float] = {}
    t = workdir / "timing.jsonl"
    if t.exists():
        for line in t.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["stage"]] = out.get(e["stage"], 0.0) + float(e["seconds"])
    ev = workdir / "events.jsonl"
    if ev.exists() and "train" not in out:
        started = done = None
        for line in ev.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") == "started" and started is None:
                started = e["ts"]
            if e.get("type") == "done":
                done = e["ts"]
        if started is not None and done is not None:
            out["train"] = done - started
    return out


def _l1(report: dict[str, Any], set_name: str, target: str) -> dict[str, Any] | None:
    t = report.get("benchmark", {}).get("sets", {}).get(set_name, {}).get("targets", {}).get(target)
    return t["l1"] if t else None


def _mean(v: Any) -> float | None:
    if v is None:
        return None
    return float(v["mean"]) if isinstance(v, dict) else float(v)


def curve_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("pilot", {})
        ind, ood = _l1(data, "in_dist", "adapter"), _l1(data, "ood", "adapter")
        if ind is None:
            continue
        stages = _stage_seconds(path.parent)
        messy_path = path.parent / "report_messy.json"
        messy = json.loads(messy_path.read_text(encoding="utf-8")) if messy_path.exists() else {}
        m_ad = _l1(messy, "messy", "adapter") or _l1(data, "messy", "adapter")
        m_base = _l1(messy, "messy", "base") or _l1(data, "messy", "base")
        rgt = meta.get("rgt") or {}
        rows.append(
            {
                "run": path.parent.name,
                "method": meta.get("method", "sft")
                + ("" if meta.get("train_mix", "standard") == "standard" else f"+{meta['train_mix']}")
                + ("" if meta.get("task", "agent") == "agent" else f"+{meta['task']}"),
                "records": meta.get("records"),
                "seed": meta.get("seed"),
                "train_samples": meta.get("samples", {}).get("train"),
                "rgt_attempt_pass_rate": (rgt.get("attempts") or {}).get("pass_rate"),
                "fixed_test": meta.get("test_set", {}).get("fixed", False),
                "test_records": meta.get("test_set", {}).get("records"),
                "decision_in_dist": _mean(ind.get("decision_accuracy")),
                "task_score_in_dist": _mean(ind.get("task_score")),
                "tool_call_in_dist": ind.get("tool_call_accuracy"),
                "json_validity_in_dist": ind.get("json_validity"),
                "decision_ood": _mean(ood.get("decision_accuracy")) if ood else None,
                "decision_messy": _mean(m_ad.get("decision_accuracy")) if m_ad else None,
                "tool_call_messy": m_ad.get("tool_call_accuracy") if m_ad else None,
                "base_decision_messy": _mean(m_base.get("decision_accuracy")) if m_base else None,
                "base_decision_in_dist": _mean((_l1(data, "in_dist", "base") or {}).get("decision_accuracy")),
                "prepare_s": stages.get("prepare"),
                "train_s": stages.get("train"),
                "eval_s": stages.get("eval"),
            }
        )
    rows.sort(key=lambda r: (r["method"], r["records"] or 0, r["seed"] or 0))
    return rows


def curve_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Fewest records at which each method reaches `threshold` in-dist decision accuracy."""
    out: dict[str, Any] = {"threshold": threshold, "methods": {}}
    for m in sorted({r["method"] for r in rows}):
        hit = next(
            (r for r in rows if r["method"] == m and (r["decision_in_dist"] or 0) >= threshold),
            None,
        )
        out["methods"][m] = (
            None
            if hit is None
            else {
                "records": hit["records"],
                "minutes_to_adapter": round(((hit["prepare_s"] or 0) + (hit["train_s"] or 0)) / 60, 1),
            }
        )
    a, b = out["methods"].get("sft"), out["methods"].get("rgt")
    if a and b and b["records"]:
        out["sft_over_rgt_records"] = round(a["records"] / b["records"], 2)
    return out


def curve_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.1f}%"

    def mins(*vals: float | None) -> str:
        known = [v for v in vals if v is not None]
        return "n/a" if not known else f"{sum(known) / 60:.1f}"

    lines = [
        "# Learning curve: standard SFT vs Rationale-Guided Training (RGT)",
        "",
        "| Method | Records | Train samples | Prepare min | Train min | Minutes to adapter | Decision acc (in-dist) "
        "| Task score | Decision acc (OOD) | Decision acc (messy) | Tool-call acc (messy) | Tool-call acc "
        "| Base attempts correct |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method'].upper()} | {r['records']} | {r['train_samples']} | {mins(r['prepare_s'])} | "
            f"{mins(r['train_s'])} | {mins(r['prepare_s'], r['train_s'])} | {pct(r['decision_in_dist'])} | "
            f"{pct(r['task_score_in_dist'])} | {pct(r['decision_ood'])} | {pct(r['decision_messy'])} | "
            f"{pct(r['tool_call_messy'])} | {pct(r['tool_call_in_dist'])} | "
            f"{pct(r['rgt_attempt_pass_rate'])} |"
        )
    base = next((r["base_decision_in_dist"] for r in rows if r["base_decision_in_dist"] is not None), None)
    base_m = next((r["base_decision_messy"] for r in rows if r["base_decision_messy"] is not None), None)
    lines += [
        "",
        f"Base model, no fine-tuning: {pct(base)} decision accuracy (in-dist)"
        + (f", {pct(base_m)} (messy)." if base_m is not None else "."),
    ]
    th = summary["threshold"]
    for m, hit in summary["methods"].items():
        lines.append(
            f"{m.upper()}: "
            + (
                f"reaches {th:.0%} decision accuracy at {hit['records']} records "
                f"({hit['minutes_to_adapter']} min data prep + training)."
                if hit
                else f"does not reach {th:.0%} at the sizes tested."
            )
        )
    if "sft_over_rgt_records" in summary:
        lines.append(f"SFT needs {summary['sft_over_rgt_records']}x the records RGT needs to reach that threshold.")
    fixed = all(r["fixed_test"] for r in rows)
    lines += [
        "",
        "Caveats: "
        + ("all runs scored on the same fixed test set; " if fixed else "WARNING: test sets differ between runs; ")
        + "synthetic invoice data from a rules generator; one training seed per point; RGT trains one extra "
        "(strategy) sample per decision, about 4 samples per record vs 3 for SFT, so it uses ~33% more "
        "training compute per record (compare the train-minutes column, not only records).",
    ]
    return "\n".join(lines) + "\n"


def curve(args: argparse.Namespace) -> int:
    rows = curve_rows([Path(p) for p in args.reports])
    if not rows:
        print("no evaluated reports", file=sys.stderr)
        return 2
    summary = curve_summary(rows, args.threshold)
    md = curve_markdown(rows, summary)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "curve.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
        (out / "curve.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def add_messy(args: argparse.Namespace) -> int:
    """Write the messy-invoice set into an existing workdir (no retraining needed)."""
    workdir = Path(args.workdir).resolve()
    if not (workdir / "pilot.json").exists():
        print(f"{workdir} is not a prepared workdir", file=sys.stderr)
        return 2
    task = json.loads((workdir / "pilot.json").read_text(encoding="utf-8")).get("task", "agent")
    rows, pii = redact_records(generate_messy(args.n, seed=args.seed, task=task))
    _write_jsonl(workdir / "data" / SET_FILES["messy"], rows)
    print(json.dumps({"workdir": str(workdir), "messy_records": len(rows), "seed": args.seed, "pii_placeholders": pii}))
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
    p.add_argument("--profile", default=DEFAULT_PROFILE, help="model profile ref (profiles/<ref>.yaml)")
    p.add_argument("--allow-draft", action="store_true", help="allow an unvalidated profile (not for evidence)")
    p.add_argument("--method", choices=["sft", "rgt"], default="sft", help="rgt = Rationale-Guided Training data")
    p.add_argument(
        "--train-mix",
        choices=list(TRAIN_MIXES),
        default="standard",
        help="training scenario mix; boundary adds near-threshold cases and rule collisions",
    )
    p.add_argument(
        "--task",
        choices=list(TASKS),
        default="agent",
        help="agent = the model calls the lookups and decides; extract = the model reads the invoice "
        "facts in one call and the rules engine decides (mccp.rules.invoice_triage)",
    )
    p.add_argument("--messy-n", type=int, default=0, help="messy-invoice evaluation records (0 = none)")
    p.add_argument("--messy-seed", type=int, default=4242)
    p.add_argument("--test-n", type=int, default=0, help="fixed, separately generated test set size (0 = split)")
    p.add_argument("--test-seed", type=int, default=7777)
    p.add_argument("--attempts-url", help="vLLM URL serving the base model, for on-policy RGT attempts")
    p.add_argument("--attempts-key", default="", help="vLLM API key")
    p.add_argument("--attempts-model", default="", help="served model name that answers the attempts")
    p.add_argument("--attempts-cache", help="JSONL cache of attempts, shared across training sizes")
    p.add_argument("--attempts-concurrency", type=int, default=8)

    e = sub.add_parser("evaluate")
    e.add_argument("--workdir", required=True)
    e.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    e.add_argument("--vllm-api-key", required=True)
    e.add_argument("--targets", default="adapter,base", help="comma list: adapter,base,frontier")
    e.add_argument("--sets", default="in_dist,ood", help="comma list of test sets: in_dist,ood,messy")
    e.add_argument(
        "--report-name",
        default="report.json",
        help="file name in the workdir; use another name to score an extra set without touching report.json",
    )
    e.add_argument("--adapter-name", help="vLLM serving name; default derived from the workdir")
    e.add_argument("--max-items", type=int, default=None, help="cap items per set (default: all)")
    e.add_argument(
        "--save-outputs",
        action="store_true",
        help="write every scored answer to <report>.<set>.<target>.outputs.jsonl in the workdir",
    )
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
    e.add_argument(
        "--baseline-extra-body", help='JSON merged into every frontier request, e.g. {"thinking": {"type": "disabled"}}'
    )
    e.add_argument("--baseline-drop", help="comma list of params the provider rejects (temperature,seed,...)")

    c = sub.add_parser("compare")
    c.add_argument("reports", nargs="+", help="report.json files")
    c.add_argument("--out", help="directory for comparison.json and comparison.md")

    am = sub.add_parser("add-messy", help="add the messy-invoice set to an existing workdir")
    am.add_argument("--workdir", required=True)
    am.add_argument("--n", type=int, default=300)
    am.add_argument("--seed", type=int, default=4242)

    cv = sub.add_parser("curve", help="learning curve across methods and training sizes")
    cv.add_argument("reports", nargs="+", help="report.json files")
    cv.add_argument("--threshold", type=float, default=0.95, help="in-dist decision accuracy to reach")
    cv.add_argument("--out", help="directory for curve.json and curve.md")

    pe = sub.add_parser("profile-env", help="print shell exports for a profile")
    pe.add_argument("--profile", default=DEFAULT_PROFILE)

    sub.add_parser("new-key", help="print a random vLLM API key")

    args = ap.parse_args()
    if args.cmd == "prepare":
        return prepare(args)
    if args.cmd == "evaluate":
        return asyncio.run(_evaluate(args))
    if args.cmd == "compare":
        return compare(args)
    if args.cmd == "curve":
        return curve(args)
    if args.cmd == "add-messy":
        return add_messy(args)
    if args.cmd == "profile-env":
        return profile_env(args)
    print(secrets.token_urlsafe(32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
