"""`mccp-route`: run the gateway, connect developer tools, operate it.

  mccp-route serve                       run the gateway (settings from ROUTE_* env)
  mccp-route setup --gateway URL         connect Claude Code / OpenCode (key asked hidden)
  mccp-route remove                      undo `setup`
  mccp-route doctor --gateway URL        check reachability and key, no model call
  mccp-route admin bootstrap ...         create org + team + first key (admin token asked hidden)
  mccp-route admin routing               what each profile routes to today, and why

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
    harness: Annotated[str, typer.Option(help="comma list: claude-code,opencode")] = "claude-code,opencode",
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
    typer.echo(f"done: profile {chosen}. Open a new terminal (OpenCode reads {setup_mod.KEY_ENV} from your shell).")


@app.command()
def remove(harness: Annotated[str, typer.Option()] = "claude-code,opencode") -> None:
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
