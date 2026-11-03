"""Diagnose a finished batch whose benchmark looks wrong (read-only on the run).

  train-text  (trainer venv)  show the exact text one training row became after the
                              base model's chat template: what the adapter learned to emit
  live        (control venv)  send a few held-out items to the base model and the adapter
                              on a running vLLM, and one item to the frontier endpoint, and
                              print the raw responses (special tokens kept) and HTTP errors

Run through deploy/gpu/probe.sh, which starts and stops vLLM around `live`.
Prints no secrets: the frontier key is read from the environment, never echoed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


def _rows(path: Path, n: int) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def train_text(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer  # trainer venv only

    w = Path(args.workdir)
    row = _rows(w / "data/train_and_val.rendered.jsonl", 1)[0]
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    text = tok.apply_chat_template(row["messages"], tokenize=False)
    print("== roles:", [m["role"] for m in row["messages"]])
    for i, m in enumerate(row["messages"]):
        if m["role"] == "assistant":
            print(f"== assistant message {i} as stored:", json.dumps(m, ensure_ascii=False)[:600])
    print("== rendered training text (last 2500 chars):")
    print(text[-2500:])
    return 0


async def live(args: argparse.Namespace) -> int:
    import httpx

    from mccp.evals import layers

    w = Path(args.workdir)
    records = _rows(w / "data/golden.jsonl", 4)
    items = layers.golden_items(records, 3)  # tool call, tool call, final of record 0
    xbody = json.loads(args.extra_body) if args.extra_body else {}
    hdr = {"Authorization": f"Bearer {args.vllm_key}"}
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(
            f"{args.vllm_url}/v1/load_lora_adapter",
            json={"lora_name": "probe-adapter", "lora_path": str(w / "output")},
            headers=hdr,
        )
        print("== load adapter:", r.status_code, r.text[:200])
        for model in (args.base_served, "probe-adapter"):
            for k, it in enumerate(items):
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": it.messages,
                    "max_tokens": 400,
                    "temperature": 0,
                    **xbody,
                    "skip_special_tokens": False,
                }
                if it.tools:
                    payload["tools"] = it.tools
                r = await c.post(f"{args.vllm_url}/v1/chat/completions", json=payload, headers=hdr)
                print(f"\n== {model} item {k} (expected {it.kind}) HTTP {r.status_code}")
                if r.status_code >= 400:
                    print(r.text[:600])
                    continue
                msg = r.json()["choices"][0]["message"]
                print("expected:", json.dumps(it.expected, ensure_ascii=False)[:400])
                print("content :", json.dumps(msg.get("content"), ensure_ascii=False)[:900])
                print("tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False)[:400])
                if msg.get("reasoning_content"):
                    print("reasoning:", json.dumps(msg["reasoning_content"], ensure_ascii=False)[:400])
        await c.post(f"{args.vllm_url}/v1/unload_lora_adapter", json={"lora_name": "probe-adapter"}, headers=hdr)

        key = os.environ.get("MCCP_BASELINE_API_KEY", "")
        if args.baseline_url and args.baseline_model and key:
            it = items[0]
            payload = {"model": args.baseline_model, "messages": it.messages, "max_completion_tokens": 300}
            if it.tools:
                payload["tools"] = it.tools
            r = await c.post(
                f"{args.baseline_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
            print(f"\n== frontier {args.baseline_model} HTTP {r.status_code}")
            print(r.text[:900])
        else:
            print("\n== frontier: skipped (no baseline configured)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-text")
    t.add_argument("--workdir", required=True)
    t.add_argument("--model-dir", required=True)
    lv = sub.add_parser("live")
    lv.add_argument("--workdir", required=True)
    lv.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    lv.add_argument("--vllm-key", required=True)
    lv.add_argument("--base-served", required=True)
    lv.add_argument("--extra-body", default="")
    lv.add_argument("--baseline-url", default="")
    lv.add_argument("--baseline-model", default="")
    args = p.parse_args()
    if args.cmd == "train-text":
        return train_text(args)
    return asyncio.run(live(args))


if __name__ == "__main__":
    sys.exit(main())
