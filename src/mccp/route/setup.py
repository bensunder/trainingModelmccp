"""Point a developer's coding tools at the gateway, in one command.

Edits are surgical and reversible:
* only the keys this tool owns are written (everything else in the file is kept),
* a timestamped backup is written next to the file before the first change,
* `remove` deletes exactly the keys `apply` wrote,
* files holding the API key are made owner-only (0600).

Harness config formats (checked 2026-11-12):
* Claude Code: `~/.claude/settings.json` -> "env": ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN
  (sent as `Authorization: Bearer`), ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL
  (haiku = background tasks).
* OpenCode: `~/.config/opencode/opencode.json` -> provider.<id> with npm
  "@ai-sdk/openai-compatible", options.baseURL, options.apiKey "{env:ROUTE_API_KEY}".
* Codex: speaks only the OpenAI Responses API (wire_api="responses"); not supported
  until the gateway serves /v1/responses.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROVIDER_ID = "route"
KEY_ENV = "ROUTE_API_KEY"
ENV_FILE_LINE = '[ -f "$HOME/.config/route/route.env" ] && . "$HOME/.config/route/route.env"  # route gateway'
CLAUDE_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
HARNESSES = ("claude-code", "opencode", "codex")


class SetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Change:
    harness: str
    path: Path
    action: str  # written | removed | unchanged
    backup: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SetupError(f"{path} is not valid JSON ({e}); fix or move it, nothing was changed") from e
    if not isinstance(data, dict):
        raise SetupError(f"{path} does not hold a JSON object; nothing was changed")
    return data


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    b = path.with_name(f"{path.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    shutil.copy2(path, b)
    return b


def _write_json(path: Path, data: dict[str, Any], private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if private:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    tmp.replace(path)


def claude_code_apply(home: Path, gateway: str, key: str, profile: str, strong_profile: str | None = None) -> Change:
    path = home / ".claude" / "settings.json"
    data = _read_json(path)
    env = dict(data.get("env") or {})
    want = {
        "ANTHROPIC_BASE_URL": gateway.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": profile,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": profile,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": strong_profile or profile,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": f"{profile}:background",
    }
    if all(env.get(k) == v for k, v in want.items()):
        return Change("claude-code", path, "unchanged")
    backup = _backup(path)
    env.update(want)
    data["env"] = env
    _write_json(path, data, private=True)
    return Change("claude-code", path, "written", backup)


def claude_code_remove(home: Path) -> Change:
    path = home / ".claude" / "settings.json"
    data = _read_json(path)
    env = dict(data.get("env") or {})
    if not any(k in env for k in CLAUDE_KEYS):
        return Change("claude-code", path, "unchanged")
    backup = _backup(path)
    for k in CLAUDE_KEYS:
        env.pop(k, None)
    if env:
        data["env"] = env
    else:
        data.pop("env", None)
    _write_json(path, data, private=True)
    return Change("claude-code", path, "removed", backup)


def opencode_apply(home: Path, gateway: str, profiles: list[str], default_profile: str) -> Change:
    path = home / ".config" / "opencode" / "opencode.json"
    data = _read_json(path)
    providers = dict(data.get("provider") or {})
    want = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Route gateway",
        "options": {"baseURL": gateway.rstrip("/") + "/v1", "apiKey": "{env:" + KEY_ENV + "}"},
        "models": {p: {"name": f"Route {p}"} for p in profiles},
    }
    model = f"{PROVIDER_ID}/{default_profile}"
    if providers.get(PROVIDER_ID) == want and data.get("model") == model:
        return Change("opencode", path, "unchanged")
    backup = _backup(path)
    data.setdefault("$schema", "https://opencode.ai/config.json")
    providers[PROVIDER_ID] = want
    data["provider"] = providers
    data["model"] = model
    _write_json(path, data, private=False)  # holds no key: the key comes from the environment
    return Change("opencode", path, "written", backup)


def opencode_remove(home: Path) -> Change:
    path = home / ".config" / "opencode" / "opencode.json"
    data = _read_json(path)
    providers = dict(data.get("provider") or {})
    if PROVIDER_ID not in providers:
        return Change("opencode", path, "unchanged")
    backup = _backup(path)
    providers.pop(PROVIDER_ID)
    data["provider"] = providers
    if str(data.get("model", "")).startswith(PROVIDER_ID + "/"):
        data.pop("model")
    _write_json(path, data, private=False)
    return Change("opencode", path, "removed", backup)


def env_file_apply(home: Path, key: str, shells: tuple[str, ...] = (".zshrc", ".bashrc")) -> list[Change]:
    """~/.config/route/route.env (0600) exports the key; shell rc files source it
    (one marked line, added once)."""
    envf = home / ".config" / "route" / "route.env"
    envf.parent.mkdir(parents=True, exist_ok=True)
    tmp = envf.with_name("route.env.tmp")
    tmp.write_text(f"export {KEY_ENV}='{key}'\n", encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    tmp.replace(envf)
    out = [Change("env", envf, "written")]
    for rc in shells:
        p = home / rc
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        if ENV_FILE_LINE in text:
            out.append(Change("env", p, "unchanged"))
            continue
        backup = _backup(p)
        p.write_text(text + ("" if text.endswith("\n") or not text else "\n") + ENV_FILE_LINE + "\n", encoding="utf-8")
        out.append(Change("env", p, "written", backup))
    return out


def env_file_remove(home: Path, shells: tuple[str, ...] = (".zshrc", ".bashrc")) -> list[Change]:
    out = []
    envf = home / ".config" / "route" / "route.env"
    if envf.exists():
        envf.unlink()
        out.append(Change("env", envf, "removed"))
    for rc in shells:
        p = home / rc
        if p.exists() and ENV_FILE_LINE in p.read_text(encoding="utf-8"):
            backup = _backup(p)
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines(keepends=True) if ENV_FILE_LINE not in ln]
            p.write_text("".join(lines), encoding="utf-8")
            out.append(Change("env", p, "removed", backup))
    return out


def apply(home: Path, harnesses: list[str], gateway: str, key: str, profiles: list[str], default_profile: str,
          strong_profile: str | None = None) -> list[Change]:  # fmt: skip
    unknown = set(harnesses) - set(HARNESSES)
    if unknown:
        raise SetupError(f"unknown harness {sorted(unknown)}; choose from {', '.join(HARNESSES)}")
    if "codex" in harnesses:
        raise SetupError(
            "Codex speaks only the OpenAI Responses API; the gateway does not serve /v1/responses yet. "
            "Set up claude-code and opencode now; Codex follows when the Responses translator ships."
        )
    if not gateway.startswith("https://") and not gateway.startswith(("http://127.0.0.1", "http://localhost")):
        raise SetupError("gateway URL must be https (the API key travels with every request)")
    if not key.startswith("rk_"):
        raise SetupError("that is not a gateway API key (they start with rk_)")
    if default_profile not in profiles:
        raise SetupError(f"{default_profile} is not one of the gateway's profiles {profiles}")
    out: list[Change] = []
    if "claude-code" in harnesses:
        out.append(claude_code_apply(home, gateway, key, default_profile, strong_profile))
    if "opencode" in harnesses:
        out.append(opencode_apply(home, gateway, profiles, default_profile))
        out.extend(env_file_apply(home, key))
    return out


def remove(home: Path, harnesses: list[str]) -> list[Change]:
    out: list[Change] = []
    if "claude-code" in harnesses:
        out.append(claude_code_remove(home))
    if "opencode" in harnesses:
        out.append(opencode_remove(home))
        out.extend(env_file_remove(home))
    return out
