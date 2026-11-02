"""Promotion policy rules evaluated by the real OPA binary (no database needed)."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.conftest import OPA_BIN, ROOT, needs_opa

pytestmark = needs_opa


def deny_messages(input_doc: dict) -> list[str]:
    out = subprocess.run(
        [OPA_BIN, "eval", "--format", "json", "--stdin-input", "-d", str(ROOT / "policy"), "data.promotion.deny"],
        input=json.dumps(input_doc),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout)["result"]
    return result[0]["expressions"][0]["value"] if result else []


def _input(origin: str | None, allowed: list[str] | None) -> dict:
    doc = {
        "target": "staging",
        "risk_tier": "medium",
        "base_model": {"id": "Qwen/Qwen3-8B", "license": "Apache-2.0", "origin_country": origin},
    }
    if allowed is not None:
        doc["tenant_policy"] = {"allowed_base_model_origins": allowed}
    return doc


def _origin_denials(doc: dict) -> list[str]:
    return [m for m in deny_messages(doc) if m.startswith("base model origin")]


@pytest.mark.parametrize(
    ("origin", "allowed", "denied"),
    [
        ("CN", None, False),  # no tenant policy: unrestricted (previous behaviour)
        ("CN", [], False),  # empty list: unrestricted
        ("US", ["US", "FR"], False),
        ("CN", ["US", "FR"], True),
        (None, ["US"], True),  # pre-profile job: unknown origin is denied under a list
    ],
)
def test_tenant_origin_allow_list(origin, allowed, denied):
    msgs = _origin_denials(_input(origin, allowed))
    assert bool(msgs) is denied, msgs
    if denied:
        assert "Qwen/Qwen3-8B" in msgs[0]
