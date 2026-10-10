#!/usr/bin/env python3
"""Run lm-evaluation-harness against a vLLM OpenAI endpoint and write
{task: score} JSON, the contract the L3 regression layer expects.

Generation tasks (ifeval, gsm8k) use `local-chat-completions`; loglikelihood
tasks (mmlu, arc) need `local-completions`, which this wrapper selects.
Requires `pip install lm-eval`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LOGLIKELIHOOD = {"mmlu", "arc_challenge", "arc_easy", "hellaswag", "mmlu_pro"}
METRIC = {"ifeval": "prompt_level_strict_acc,none", "gsm8k": "exact_match,strict-match"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    scores: dict[str, float] = {}
    for task in a.tasks.split(","):
        ll = task in LOGLIKELIHOOD
        kind = "local-completions" if ll else "local-chat-completions"
        endpoint = f"{a.base_url.rstrip('/')}/v1/{'completions' if ll else 'chat/completions'}"
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                "-m",
                "lm_eval",
                "--model",
                kind,
                "--model_args",
                f"model={a.model},base_url={endpoint},num_concurrent=4,tokenized_requests=False",
                "--tasks",
                task,
                "--limit",
                str(a.limit),
                "--output_path",
                td,
            ]
            if not ll:
                cmd.append("--apply_chat_template")
            subprocess.run(cmd, check=True)
            result = json.loads(next(Path(td).rglob("results_*.json")).read_text())
        metrics = result["results"][task]
        key = METRIC.get(task) or next(k for k in metrics if k.startswith("acc,"))
        scores[task] = float(metrics[key])
    Path(a.output).write_text(json.dumps(scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
