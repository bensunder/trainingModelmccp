"""`mccp-route`: run the gateway, connect developer tools, operate it.

  mccp-route serve                       run the gateway (settings from ROUTE_* env)
  mccp-route setup --gateway URL         connect Claude Code / OpenCode / Codex (key asked hidden)
  mccp-route remove                      undo `setup`
  mccp-route doctor --gateway URL        check reachability and key, no model call
  mccp-route admin bootstrap ...         create org + team + first key (admin token asked hidden)
  mccp-route admin routing               what each profile routes to today, and why
  mccp-route eval import|estimate|run    build tasks from benchmarks, price a run, measure models

Secrets are never accepted as command-line arguments (shell history, process
lists): keys and tokens come from a hidden prompt or an environment variable.
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from typing import Annotated

import httpx
import typer

from . import setup as setup_mod

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Route gateway: serve, connect tools, operate.")
admin_app = typer.Typer(no_args_is_help=True, help="Operator commands (need the admin token).")
app.add_typer(admin_app, name="admin")

Gateway = Annotated[
    str, typer.Option("--gateway", "-g", envvar="ROUTE_GATEWAY", help="e.g. https://inference.whyaidata.com")
]


def _secret(env: str, prompt: str) -> str:
    v = os.environ.get(env) or getpass.getpass(prompt)
    v = v.strip()
    if not v:
        raise typer.BadParameter(f"empty {prompt.strip(': ')}")
    return v


def _get(url: str, token: str, timeout: float = 15.0) -> httpx.Response:
    return httpx.get(url, headers={"authorization": f"Bearer {token}"}, timeout=timeout)


@app.command()
def serve() -> None:
    """Run the gateway (configuration from ROUTE_* environment variables)."""
    from .app import main

    main()


@app.command()
def setup(
    gateway: Gateway,
    harness: Annotated[str, typer.Option(help="comma list: claude-code,opencode,codex")] = "claude-code,opencode,codex",
    profile: Annotated[str | None, typer.Option(help="route profile (default: the gateway's default)")] = None,
    strong_profile: Annotated[str | None, typer.Option(help="profile for Claude Code's opus alias")] = None,
) -> None:
    """Point this machine's coding tools at the gateway. The key is asked hidden (or ROUTE_API_KEY)."""
    key = _secret("ROUTE_API_KEY", "Route API key (hidden): ")
    r = _get(gateway.rstrip("/") + "/v1/models", key)
    if r.status_code != 200:
        typer.echo(f"gateway refused the key: HTTP {r.status_code} {r.text[:200]}", err=True)
        raise typer.Exit(1)
    profiles = [m["id"] for m in r.json()["data"] if str(m["id"]).startswith("auto")]
    chosen = profile or ("auto-balanced" if "auto-balanced" in profiles else profiles[0])
    try:
        changes = setup_mod.apply(Path.home(), [h.strip() for h in harness.split(",") if h.strip()], gateway, key,
                                  profiles, chosen, strong_profile)  # fmt: skip
    except setup_mod.SetupError as e:
        typer.echo(f"not changed: {e}", err=True)
        raise typer.Exit(2) from e
    for c in changes:
        typer.echo(f"  {c.harness:12} {c.action:9} {c.path}" + (f"  (backup {c.backup.name})" if c.backup else ""))
    typer.echo(
        f"done: profile {chosen}. Open a new terminal (OpenCode and Codex read {setup_mod.KEY_ENV} from your shell)."
    )


@app.command()
def remove(harness: Annotated[str, typer.Option()] = "claude-code,opencode,codex") -> None:
    """Undo `setup`: remove exactly the settings it wrote (backups are kept)."""
    for c in setup_mod.remove(Path.home(), [h.strip() for h in harness.split(",") if h.strip()]):
        typer.echo(f"  {c.harness:12} {c.action:9} {c.path}")


@app.command()
def doctor(gateway: Gateway) -> None:
    """Check the gateway is reachable and the key works (no model call, no cost)."""
    base = gateway.rstrip("/")
    try:
        ready = httpx.get(base + "/readyz", timeout=10)
    except httpx.HTTPError as e:
        typer.echo(f"unreachable: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"ready: {ready.status_code} {json.dumps(ready.json().get('providers', {}))}")
    key = _secret("ROUTE_API_KEY", "Route API key (hidden): ")
    r = _get(base + "/v1/models", key)
    typer.echo(
        f"key: HTTP {r.status_code}"
        + (f", {len(r.json()['data'])} models/profiles" if r.status_code == 200 else f" {r.text[:200]}")
    )
    raise typer.Exit(0 if r.status_code == 200 and ready.status_code == 200 else 1)


@admin_app.command()
def bootstrap(
    gateway: Gateway,
    org: Annotated[str, typer.Option(help="organisation name")],
    team: Annotated[str, typer.Option(help="team name")] = "engineering",
    label: Annotated[str, typer.Option(help="who uses the first key")] = "first-developer",
    approved: Annotated[str | None, typer.Option(help="comma list of approved models (default: all)")] = None,
) -> None:
    """Create an organisation, a team and its first API key (printed once)."""
    tok = _secret("ROUTE_ADMIN_TOKEN", "Admin token (hidden): ")
    h = {"authorization": f"Bearer {tok}"}
    base = gateway.rstrip("/") + "/admin/v1"
    with httpx.Client(headers=h, timeout=20) as c:
        o = c.post(base + "/orgs", json={"name": org})
        o.raise_for_status()
        oid = o.json()["id"]
        t = c.post(
            f"{base}/orgs/{oid}/teams",
            json={"name": team, "approved_models": approved.split(",") if approved else None},
        )
        t.raise_for_status()
        k = c.post(f"{base}/orgs/{oid}/teams/{t.json()['id']}/keys", json={"label": label})
        k.raise_for_status()
    typer.echo(f"org {oid}\nteam {t.json()['id']}\nkey (shown once, store it now): {k.json()['key']}")


@admin_app.command()
def routing(gateway: Gateway) -> None:
    """Show what each profile routes each task class to, and why."""
    tok = _secret("ROUTE_ADMIN_TOKEN", "Admin token (hidden): ")
    r = _get(gateway.rstrip("/") + "/admin/v1/routing", tok)
    r.raise_for_status()
    d = r.json()
    typer.echo(f"catalog {d['catalog_version']}; evidence {d['evidence']['suite']} @ {d['evidence']['generated_at']}")
    for pid, p in d["profiles"].items():
        typer.echo(f"\n{pid} ({p['strategy']})")
        for task, v in p["routes"].items():
            typer.echo(f"  {task:11} -> {v['model'] or '-':16} {v['reason']}")


eval_app = typer.Typer(no_args_is_help=True, help="Measure models on executable coding tasks -> routing evidence.")
app.add_typer(eval_app, name="eval")


@eval_app.command("sample")
def eval_sample(
    out: Annotated[Path, typer.Option(help="where to write the starter task file")] = Path("route-tasks.sample.jsonl"),
) -> None:
    """Write the small original starter task set (plumbing checks; too small to move routing)."""
    from .evals import tasks

    tasks.write(tasks.SAMPLE, out)
    typer.echo(f"wrote {len(tasks.SAMPLE)} tasks to {out}")


@eval_app.command("import")
def eval_import(
    source: Annotated[str, typer.Option(help="humaneval | mbpp")],
    path: Annotated[Path, typer.Option(help="local copy of the pinned benchmark file (see docs/route.md)")],
    out: Annotated[Path, typer.Option(help="task file to write (JSONL)")],
    classes: Annotated[str, typer.Option(help="comma list of quick_edit,debug,tests")] = "quick_edit,debug,tests",
    allow_unpinned: Annotated[bool, typer.Option(help="accept a file whose sha256 is not the pinned release")] = False,
    limit: Annotated[int | None, typer.Option(help="only the first N problems")] = None,
    seed: Annotated[int, typer.Option(help="planted-bug selection seed")] = 0,
) -> None:
    """Build scored tasks from a public benchmark. Runs only the benchmark's own code in the
    local sandbox to validate it; makes no model calls and costs nothing."""
    import asyncio

    from .config import TaskClass
    from .evals import importers, tasks
    from .evals.sandbox import LocalSandbox

    try:
        wanted = tuple(TaskClass(c.strip()) for c in classes.split(",") if c.strip())
        ts, rep = asyncio.run(importers.build(source, importers.read(path), LocalSandbox(), wanted,
                                              allow_unpinned=allow_unpinned, limit=limit, seed=seed))  # fmt: skip
    except (importers.SourceError, ValueError) as e:
        typer.echo(f"not imported: {e}", err=True)
        raise typer.Exit(2) from e
    tasks.write(ts, out)
    typer.echo(f"{source}: {rep.problems} problems -> {len(ts)} tasks {json.dumps(rep.tasks_by_class)} -> {out}")
    if rep.dropped_reference_fails:
        typer.echo(f"  dropped (canonical solution fails its own tests here): {', '.join(rep.dropped_reference_fails)}")
    if rep.no_killed_mutant:
        typer.echo(f"  no debug/tests task (no planted bug that the tests catch): {len(rep.no_killed_mutant)} problems")


@eval_app.command("estimate")
def eval_estimate(
    catalog: Annotated[Path, typer.Option()],
    tasks_file: Annotated[Path, typer.Option("--tasks")],
    models: Annotated[str, typer.Option(help="comma list of catalog model ids")],
    max_tokens: Annotated[int, typer.Option()] = 2048,
) -> None:
    """Estimate the API cost of a run before spending anything."""
    from .config import load_catalog
    from .evals import runner, tasks

    cat = load_catalog(catalog)
    ts, _ = tasks.load(tasks_file)
    est = runner.estimate_usd(cat, ts, [m.strip() for m in models.split(",")], max_tokens)
    for m, usd in est.items():
        typer.echo(f"  {m:20} ~${usd:.4f}  ({len(ts)} tasks; price source: {cat.model(m).price_source})")
    typer.echo(f"  total ~${sum(est.values()):.4f} (rough: prompt chars/4 in, half of max_tokens out)")


@eval_app.command("run")
def eval_run(
    gateway: Gateway,
    catalog: Annotated[Path, typer.Option()],
    tasks_file: Annotated[Path, typer.Option("--tasks")],
    models: Annotated[str, typer.Option(help="comma list of catalog model ids")],
    out: Annotated[Path, typer.Option(help="results directory")] = Path("route-eval"),
    sandbox: Annotated[str, typer.Option(help="docker | local")] = "docker",
    allow_local_sandbox: Annotated[bool, typer.Option(help="accept the weaker local sandbox")] = False,
    image: Annotated[str, typer.Option(help="sandbox image (pin a digest in production)")] = "python:3.12-slim",
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 4,
    max_tokens: Annotated[int, typer.Option()] = 2048,
    yes: Annotated[bool, typer.Option("--yes", help="skip the cost confirmation")] = False,
) -> None:
    """Score models on the task file through the gateway; write results.jsonl and evidence.json."""
    import asyncio

    from .config import load_catalog
    from .evals import runner, tasks
    from .evals.sandbox import DockerSandbox, LocalSandbox

    if sandbox == "local" and not allow_local_sandbox:
        typer.echo("refusing the local sandbox (same filesystem view as this machine); use --sandbox docker, "
                   "or --allow-local-sandbox on a disposable machine", err=True)  # fmt: skip
        raise typer.Exit(2)
    cat = load_catalog(catalog)
    ms = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in ms if not cat.has_model(m)]
    if unknown:
        raise typer.BadParameter(f"not in the catalog: {unknown}")
    ts, digest = tasks.load(tasks_file)
    est = runner.estimate_usd(cat, ts, ms, max_tokens)
    typer.echo(f"{len(ts)} tasks x {len(ms)} models; estimated ~${sum(est.values()):.4f} " + json.dumps(est))
    if not yes and not typer.confirm("Run it (paid API calls)?"):
        raise typer.Exit(1)
    key = _secret("ROUTE_API_KEY", "Route API key for the evaluation (hidden): ")
    sb = DockerSandbox(image=image) if sandbox == "docker" else LocalSandbox()
    rows = asyncio.run(runner.run(cat, gateway, key, ts, ms, sb, out, concurrency, max_tokens))
    ev, prov = runner.evidence(rows, suite=f"{tasks_file.name}@{digest[:12]}",
                               provenance={"tasks_sha256": digest, "catalog_version": cat.version, "sandbox": sb.kind, "models": ms})  # fmt: skip
    (out / "evidence.json").write_text(ev.model_dump_json(indent=2), encoding="utf-8")
    for m in ms:
        cells = ev.results.get(m, {})
        typer.echo(
            f"  {m:20} " + "  ".join(f"{t.value} {c.passed}/{c.n} (low {c.lower:.2f})" for t, c in cells.items())
        )
    typer.echo(
        f"wrote {out / 'results.jsonl'} and {out / 'evidence.json'} ({prov['excluded_no_answer']} unanswered excluded)"
    )
