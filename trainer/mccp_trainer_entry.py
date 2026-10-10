#!/usr/bin/env python3
"""MCCP trainer entrypoint. Runs inside the GPU trainer image.

Contract with the control plane (run.json in, events.jsonl out):

1. Download the base model at the *pinned revision* into ./model (Soup takes a
   model id or path but has no revision field, so we pin here).
2. Verify the tokenizer chat-template SHA-256 against the recipe.
3. Render-check: for golden records with tools, the native template rendering
   (`tools=`) must equal the pre-rendered system block the dataset uses. Any
   drift fails the job before a GPU hour is spent.
4. Token statistics: p50/p95/p99 sample length; refuse if p99 > max_length
   (truncation would cut the supervised assistant turn); trained_tokens.
5. Write soup.yaml and run `soup train`, forwarding progress as JSON events.

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

_TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def prerender_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Must stay identical to mccp.datasets.render.prerender_tools (tested)."""
    if not tools:
        return messages
    block = _TOOLS_HEADER + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + _TOOLS_FOOTER
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0])
        first["content"] = first["content"] + "\n\n" + block
        return [first, *messages[1:]]
    return [{"role": "system", "content": block}, *messages]


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

    kwargs = {"enable_thinking": bool(run["enable_thinking"])}
    checked = 0
    for rec in read_jsonl(run["golden_path"])[:20]:
        if not rec.get("tools"):
            continue
        native = tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=True, **kwargs)
        pre = tok.apply_chat_template(prerender_tools(rec["messages"], rec["tools"]), tokenize=True, **kwargs)
        if list(native) != list(pre):
            raise SystemExit(
                _fail(ev, EXIT_PREFLIGHT, "render-check failed: pre-rendered tools differ from native template")
            )
        checked += 1

    lengths = []
    for row in read_jsonl(run["train_path"]):
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
        "render_check_records": checked,
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
