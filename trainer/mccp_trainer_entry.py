#!/usr/bin/env python3
"""MCCP trainer entrypoint. Runs inside the GPU trainer image.

Contract with the control plane (run.json in, events.jsonl out):

1. Download the base model at the *pinned revision* into ./model (Soup takes a
   model id or path but has no revision field, so we pin here).
2. Verify the tokenizer chat-template SHA-256 against the recipe/profile.
3. Render the training data for this model: Soup's `chatml` format drops a
   `tools` column, so tool definitions are folded into the conversation as the
   model profile declares (`run["chat"]`). This is the only place in MCCP that
   renders tools, so the rendering always runs next to the real tokenizer.
4. Render-check: for golden records with tools, the native template rendering
   (`tools=`) must be token-identical to the profile rendering; string tool
   arguments must render identically to objects. Any drift fails the job
   before a GPU hour is spent.
5. Token statistics: p50/p95/p99 sample length; refuse if p99 > max_length
   (truncation would cut the supervised assistant turn); trained_tokens.
6. Write soup.yaml and run `soup train`, forwarding progress as JSON events.

Standalone on purpose: depends only on the stdlib, transformers,
huggingface_hub and soup-cli, so the trainer image does not need the
control-plane package. Python >= 3.10.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

EXIT_OK = 0
EXIT_PREFLIGHT = 2
EXIT_TRAIN = 3
EXIT_INTERRUPTED = 75
RENDERED_DATA = "data/train_and_val.rendered.jsonl"

# Tool rendering used by run.json files written before model profiles existed
# (Qwen3 hermes format). Identical to profiles/qwen3-8b@v1.yaml `tools`.
LEGACY_QWEN_CHAT: dict[str, Any] = {
    "profile": "legacy-qwen3-hermes",
    "tools": {
        "strategy": "system_append",
        "header": (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        ),
        "footer": (
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        ),
        "item_prefix": "\n",
        "item_separator": "",
        "join": "\n\n",
        "role": None,
        "json_indent": None,
        "json_mode": "per_tool",
    },
    "reasoning": {"template_kwarg": "enable_thinking", "open_tag": "<think>", "close_tag": "</think>"},
}
_LEGACY_MARKER = "# Tools\n\nYou may call one or more functions to assist with the user query."


def _tool_json(value: Any, indent: int | None) -> str:
    # Same output as Hugging Face's Jinja `tojson` filter (json.dumps defaults).
    return json.dumps(value, ensure_ascii=False, indent=indent)


def render_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fold `tools` into `messages` as the profile's tools spec declares."""
    if not tools:
        return messages
    strategy = spec["strategy"]
    indent = spec.get("json_indent")
    if spec.get("json_mode", "per_tool") == "list":
        body = _tool_json(tools, indent)
    else:
        body = spec.get("item_separator", "\n").join(spec.get("item_prefix", "") + _tool_json(t, indent) for t in tools)
    if strategy == "system_append":
        block = spec.get("header", "") + body + spec.get("footer", "")
        if messages and messages[0]["role"] == "system":
            first = dict(messages[0])
            first["content"] = (first.get("content") or "") + spec.get("join", "\n\n") + block
            return [first, *messages[1:]]
        return [{"role": "system", "content": block}, *messages]
    if strategy == "role_message":
        msg = {"role": spec["role"], "content": spec.get("header", "") + body + spec.get("footer", "")}
        if messages and messages[0]["role"] == "system":
            return [messages[0], msg, *messages[1:]]
        return [msg, *messages]
    raise ValueError(f"profile tools strategy `{strategy}` cannot render tool definitions")


def prerender_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Legacy Qwen3 hermes rendering (kept for run.json files without `chat`)."""
    return render_tools(messages, tools, LEGACY_QWEN_CHAT["tools"])


def render_row(row: dict[str, Any], chat: dict[str, Any]) -> dict[str, Any]:
    """One training row -> Soup chatml row (`messages` only).

    Rows from current dataset builds carry `tools`; they are rendered here.
    Rows from older builds were pre-rendered for Qwen3 hermes at upload time:
    they pass through only when this run uses that exact rendering."""
    if row.get("tools"):
        return {"messages": render_tools(row["messages"], row["tools"], chat["tools"])}
    sys_msg = row["messages"][0] if row.get("messages") and row["messages"][0]["role"] == "system" else None
    if sys_msg and _LEGACY_MARKER in (sys_msg.get("content") or "") and chat["tools"] != LEGACY_QWEN_CHAT["tools"]:
        raise ValueError(
            f"dataset rows were pre-rendered for Qwen3 hermes tools but this run uses profile "
            f"{chat.get('profile')}; rebuild the dataset version (current builds keep tools unrendered)"
        )
    return {"messages": row["messages"]}


def template_kwargs(run: dict[str, Any]) -> dict[str, Any]:
    if "chat_template_kwargs" in run:
        return dict(run["chat_template_kwargs"])
    return {"enable_thinking": bool(run.get("enable_thinking", False))}


class Events:
    def __init__(self, path: str) -> None:
        # Long-lived append handle for the run; closed in main().
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, kind: str, **data: Any) -> None:
        self._f.write(json.dumps({"type": kind, "ts": time.time(), **data}) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def preflight(run: dict[str, Any], ev: Events) -> dict[str, Any]:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    base = run["base_model"]
    local = os.path.abspath("model")
    snapshot_download(repo_id=base["id"], revision=base["revision"], local_dir=local)
    tok = AutoTokenizer.from_pretrained(local)
    template = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if run["chat_template_sha256"] not in ("*", template_sha):
        raise SystemExit(
            _fail(ev, EXIT_PREFLIGHT, f"chat template sha {template_sha} != recipe {run['chat_template_sha256']}")
        )

    chat = run.get("chat") or LEGACY_QWEN_CHAT
    kwargs = template_kwargs(run)
    checked = 0
    for rec in read_jsonl(run["golden_path"])[:20]:
        if not rec.get("tools"):
            continue
        if chat["tools"]["strategy"] == "none":
            raise SystemExit(
                _fail(ev, EXIT_PREFLIGHT, f"profile {chat.get('profile')} has no tool support but the data uses tools")
            )
        native = tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=True, **kwargs)
        pre = tok.apply_chat_template(
            render_tools(rec["messages"], rec["tools"], chat["tools"]), tokenize=True, **kwargs
        )
        if list(native) != list(pre):
            raise SystemExit(
                _fail(
                    ev,
                    EXIT_PREFLIGHT,
                    f"render-check failed: profile {chat.get('profile')} tool rendering differs from the native template",
                )
            )
        checked += 1

    # Render the Soup input (train + val) for this model; Soup reads only this file.
    try:
        rendered = [render_row(r, chat) for r in read_jsonl(run["soup_yaml"]["data"]["train"])]
        train_rows = [render_row(r, chat) for r in read_jsonl(run["train_path"])]
    except ValueError as e:
        raise SystemExit(_fail(ev, EXIT_PREFLIGHT, str(e))) from e
    with open(RENDERED_DATA, "w", encoding="utf-8") as f:
        for r in rendered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Tool-call arguments must be JSON strings: Arrow-backed loaders pad object
    # arguments to one struct schema (e.g. `"vendor_id": null` on lookup_po) and
    # the adapter learns the nulls. Prove the string form renders token-identical
    # to the object form on the real template, or refuse to train.
    args_checked = 0
    for row in train_rows:
        msgs = row["messages"]
        calls = [tc for m in msgs for tc in (m.get("tool_calls") or [])]
        if not calls:
            continue
        if any(not isinstance(tc["function"].get("arguments"), str) for tc in calls):
            raise SystemExit(
                _fail(
                    ev,
                    EXIT_PREFLIGHT,
                    "tool-call arguments must be JSON strings in training data (rebuild the dataset)",
                )
            )
        if args_checked < 20:
            as_objects = [
                {
                    **m,
                    "tool_calls": [
                        {**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])}}
                        for tc in m["tool_calls"]
                    ],
                }
                if m.get("tool_calls")
                else m
                for m in msgs
            ]
            a = tok.apply_chat_template(msgs, tokenize=True, **kwargs)
            b = tok.apply_chat_template(as_objects, tokenize=True, **kwargs)
            if list(a) != list(b):
                raise SystemExit(
                    _fail(ev, EXIT_PREFLIGHT, "render-check failed: string tool arguments render differently")
                )
            args_checked += 1

    lengths = []
    for row in train_rows:
        lengths.append(len(tok.apply_chat_template(row["messages"], tokenize=True, **kwargs)))
    lengths.sort()
    p = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]  # noqa: E731
    stats = {
        "n": len(lengths),
        "p50": p(0.5),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": lengths[-1],
        "mean": statistics.fmean(lengths),
    }
    max_length = run["soup_yaml"]["data"]["max_length"]
    if stats["p99"] > max_length:
        raise SystemExit(_fail(ev, EXIT_PREFLIGHT, f"p99 sample length {stats['p99']} exceeds max_length {max_length}"))
    epochs = run["soup_yaml"]["training"]["epochs"]
    result = {
        "chat_template_sha256": template_sha,
        "profile": chat.get("profile"),
        "rendering_sha256": chat.get("rendering_sha256"),
        "render_check_records": checked,
        "tool_argument_render_checks": args_checked,
        "token_stats": stats,
        "trained_tokens": int(sum(lengths) * epochs),
        "token_count_method": "tokenizer.apply_chat_template",
    }
    ev.emit("preflight", **result)
    return result


def _fail(ev: Events, code: int, message: str) -> int:
    ev.emit("error", code=code, message=message)
    return code


def _checkpoints(output: str) -> dict[int, str]:
    out = {}
    for d in glob.glob(os.path.join(output, "checkpoint-*")):
        try:
            out[int(d.rsplit("-", 1)[1])] = d
        except ValueError:
            continue
    return out


def train(run: dict[str, Any], ev: Events, preflight_result: dict[str, Any]) -> int:
    import yaml  # type: ignore[import-not-found]

    cfg = dict(run["soup_yaml"])
    cfg["base"] = "./model"
    # Soup trains on the profile-rendered file written by preflight (same row
    # order, so its val_split boundary is unchanged).
    cfg["data"] = {**cfg["data"], "train": "./" + RENDERED_DATA}
    with open("soup.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # Resolve `soup` from the trainer interpreter's own environment so a
    # different venv earlier on PATH can never supply a different Soup.
    soup_bin = os.path.join(os.path.dirname(sys.executable), "soup")
    cmd = [soup_bin if os.path.exists(soup_bin) else "soup", "train", "--config", "soup.yaml", "--yes"]
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cmd += ["--tracker", "mlflow"]
    output = os.path.abspath(cfg["output"])
    if _checkpoints(output):
        cmd += ["--resume", "auto"]
    ev.emit("started", command=cmd)

    with open("train.log", "ab") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    interrupted = {"flag": False}

    def forward(signum, _frame):
        interrupted["flag"] = True
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    seen_ckpt: set[int] = set()
    seen_step = -1
    while child.poll() is None:
        time.sleep(5)
        for step, path in sorted(_checkpoints(output).items()):
            if step in seen_ckpt:
                continue
            seen_ckpt.add(step)
            ev.emit("checkpoint", step=step, path=os.path.relpath(path))
            state = os.path.join(path, "trainer_state.json")
            if os.path.exists(state):
                with open(state, encoding="utf-8") as f:
                    for entry in json.load(f).get("log_history", []):
                        s = int(entry.get("step", -1))
                        if s > seen_step:
                            seen_step = s
                            ev.emit("metrics", step=s, metrics={k: v for k, v in entry.items() if k != "step"})
    rc = child.returncode
    if interrupted["flag"]:
        ev.emit("interrupted", returncode=rc)
        return EXIT_INTERRUPTED
    if rc != 0:
        return _fail(ev, EXIT_TRAIN, f"soup train exited with {rc}; see train.log")
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not os.path.exists(os.path.join(output, required)):
            return _fail(ev, EXIT_TRAIN, f"training finished but {required} is missing in {output}")
    ev.emit("done", adapter_dir=os.path.relpath(output), trained_tokens=preflight_result["trained_tokens"])
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    with open(args.run, encoding="utf-8") as f:
        run = json.load(f)
    ev = Events("events.jsonl")
    # The flock must live as long as this process: it is what stops a second
    # worker from training in the same directory.
    lock = open(".trainer.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return _fail(ev, EXIT_PREFLIGHT, "another trainer holds this work directory")
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        pf = preflight(run, ev)
        return train(run, ev, pf)
    except SystemExit as e:
        return int(e.code or EXIT_PREFLIGHT)
    except Exception as e:  # noqa: BLE001 - report every failure as an event
        return _fail(ev, EXIT_TRAIN, f"{type(e).__name__}: {e}")
    finally:
        ev.close()


if __name__ == "__main__":
    sys.exit(main())
