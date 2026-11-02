#!/usr/bin/env python3
"""Validate a model profile against the real tokenizer (CPU only).

Runs in the trainer environment (transformers + huggingface_hub), like the
trainer entrypoint, and downloads only tokenizer/config files (no weights).

    python trainer/profile_check.py --spec spec.json --out report.json

spec.json (written by `mccp profile validate`):
    {"profile": "<ref>", "hf_id": ..., "revision": <sha or null>,
     "chat": {"tools": {...}, "reasoning": {...} | null},
     "dims": {...}, "fixtures": [<records with messages + tools>]}

Checks (each must pass for the profile to be marked validated):
  1. revision resolves to a commit (pinned if the spec has none);
  2. config.json matches the profile dims;
  3. for every fixture with tools: native `apply_chat_template(tools=...)` is
     token-identical to the profile rendering (`render_tools`);
  4. string tool-call arguments render identically to object arguments;
  5. the reasoning kwarg (if any) is accepted and changes nothing it shouldn't.
On a rendering mismatch the report shows both renderings around the first
differing token so the profile can be corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mccp_trainer_entry import render_tools  # noqa: E402  (same directory; trainer env)

TOKENIZER_FILES = ["*.json", "*.jinja", "tokenizer*", "*.model", "*.tiktoken", "*.txt"]
WEIGHT_FILES = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]


def _stringify_args(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {
                **m,
                "tool_calls": [
                    {
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": tc["function"]["arguments"]
                            if isinstance(tc["function"].get("arguments"), str)
                            else json.dumps(tc["function"].get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                    for tc in m["tool_calls"]
                ],
            }
        out.append(m)
    return out


def _diff(tok: Any, a: list[int], b: list[int]) -> dict[str, Any]:
    i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    lo = max(0, i - 40)
    return {
        "first_diff_token": i,
        "native": tok.decode(a[lo : i + 80]),
        "profile": tok.decode(b[lo : i + 80]),
        "native_len": len(a),
        "profile_len": len(b),
    }


def check(spec: dict[str, Any], cache_dir: str) -> dict[str, Any]:
    """Fetch tokenizer + config (no weights) at the resolved revision, then evaluate."""
    from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    revision = HfApi().model_info(spec["hf_id"], revision=spec.get("revision")).sha
    local = snapshot_download(
        repo_id=spec["hf_id"],
        revision=revision,
        allow_patterns=TOKENIZER_FILES,
        ignore_patterns=WEIGHT_FILES,
        local_dir=os.path.join(cache_dir, "tok"),
    )
    cfg_path = os.path.join(local, "config.json")
    config = None
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)
    return evaluate(AutoTokenizer.from_pretrained(local), config, spec, revision)


def evaluate(tok: Any, config: dict[str, Any] | None, spec: dict[str, Any], revision: str) -> dict[str, Any]:
    """All checks; pure apart from the tokenizer (testable with a stub)."""
    report: dict[str, Any] = {"profile": spec["profile"], "hf_id": spec["hf_id"], "problems": [], "revision": revision}
    if spec.get("revision") and spec["revision"] != revision:
        report["problems"].append(f"pinned revision {spec['revision']} resolved to {revision}")
    template = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    if not template:
        report["problems"].append("tokenizer has no chat template")
        report["ok"] = False
        return report
    report["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()

    # 2. dims vs config.json
    if config is not None:
        expect = spec.get("dims") or {}
        mapping = {
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "num_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "num_kv_heads": "num_key_value_heads",
        }
        dims_diff = {
            k: {"profile": expect[k], "config": config.get(c)}
            for k, c in mapping.items()
            if k in expect and config.get(c) is not None and config.get(c) != expect[k]
        }
        head_dim = config.get("head_dim") or (
            config["hidden_size"] // config["num_attention_heads"]
            if config.get("num_attention_heads") and config.get("hidden_size")
            else None
        )
        if "head_dim" in expect and head_dim and head_dim != expect["head_dim"]:
            dims_diff["head_dim"] = {"profile": expect["head_dim"], "config": head_dim}
        report["config_architectures"] = config.get("architectures")
        report["dims_mismatch"] = dims_diff
        if dims_diff:
            report["problems"].append(f"profile dims differ from config.json: {sorted(dims_diff)}")
    else:
        report["problems"].append("config.json not found")

    chat = spec["chat"]
    reasoning = chat.get("reasoning")
    kwargs = {reasoning["template_kwarg"]: False} if reasoning else {}
    fixtures = spec["fixtures"]

    # 5. the reasoning kwarg must be accepted by the template
    try:
        tok.apply_chat_template(fixtures[0]["messages"][:2], tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception as e:  # noqa: BLE001 - reported
        report["problems"].append(f"template rejects reasoning kwarg {kwargs}: {e}")

    # 3. tool rendering parity
    tool_ok = 0
    mismatch = None
    for rec in fixtures:
        if not rec.get("tools"):
            continue
        native = list(tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=True, **kwargs))
        mine = list(
            tok.apply_chat_template(render_tools(rec["messages"], rec["tools"], chat["tools"]), tokenize=True, **kwargs)
        )
        if native == mine:
            tool_ok += 1
        elif mismatch is None:
            mismatch = _diff(tok, native, mine)
    report["tool_render_checks"] = tool_ok
    if mismatch:
        report["tool_render_mismatch"] = mismatch
        report["problems"].append("profile tool rendering differs from the native template (see tool_render_mismatch)")
    elif tool_ok == 0:
        report["problems"].append("no fixtures with tools were checked")

    # 4. string vs object tool-call arguments
    args_ok = 0
    for rec in fixtures:
        if not any(m.get("tool_calls") for m in rec["messages"]):
            continue
        a = list(tok.apply_chat_template(rec["messages"], tokenize=True, **kwargs))
        b = list(tok.apply_chat_template(_stringify_args(rec["messages"]), tokenize=True, **kwargs))
        if a == b:
            args_ok += 1
        else:
            report["problems"].append("string tool arguments render differently from objects")
            report["string_args_mismatch"] = _diff(tok, a, b)
            break
    report["string_argument_checks"] = args_ok
    report["ok"] = not report["problems"]
    report["checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    with tempfile.TemporaryDirectory(prefix="mccp-profile-") as tmp:
        try:
            report = check(spec, tmp)
        except Exception as e:  # noqa: BLE001 - reported as a failed check
            report = {"profile": spec.get("profile"), "ok": False, "problems": [f"{type(e).__name__}: {e}"]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
