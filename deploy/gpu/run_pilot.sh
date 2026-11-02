#!/usr/bin/env bash
# MCCP single-GPU pilot: everything after `pilot.py prepare` + trainer start.
#   1. installs vLLM 0.29.0 in its own venv (skipped if present)
#   2. waits for the trainer to finish (or fails fast on a trainer error)
#   3. starts vLLM on the pinned base snapshot with runtime LoRA enabled
#   4. runs `pilot.py evaluate` (L0 publish guard + benchmark, adapter vs base)
#   5. copies the adapter + evidence to /workspace/mccp-pilot-results
#
# Usage (on the GPU box):  bash /root/mccp/deploy/gpu/run_pilot.sh
# For multi-seed / larger runs use run_batch.sh. Safe to re-run: finished steps
# are skipped. No `set -e` on purpose; every step checks its own result.

WORK="${WORK:-/root/mccp-pilot}"
LOG="$WORK/pilot-run.log"
mkdir -p "$WORK"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

[ -f "$WORK/run.json" ] || die "$WORK/run.json missing: run pilot.py prepare first"
command -v uv >/dev/null || die "uv not on PATH (run: source \$HOME/.local/bin/env)"

ensure_vllm_venv || die "vLLM install failed; see: tail -40 $LOG"
ensure_key || die "could not create API key"

say "waiting for training to finish (checks every 60 s)"
wait_training "$WORK" || die "training failed (details above)"
[ -f "$WORK/output/adapter_model.safetensors" ] || die "no adapter in $WORK/output"

ensure_vllm "$WORK/model" "$WORK/vllm.log" || die "vLLM did not start (log above)"

SETS=in_dist
[ -f "$WORK/data/golden_ood.jsonl" ] && SETS=in_dist,ood
say "running evaluation (publish guard + benchmark on $SETS, adapter vs base)"
(cd "$REPO" && uv run python deploy/gpu/pilot.py evaluate --workdir "$WORK" --sets "$SETS" --vllm-api-key "$KEY") 2>&1 | tee -a "$LOG"
EVAL_RC=${PIPESTATUS[0]}

DEST=/workspace/mccp-pilot-results
save_run "$WORK" "$DEST"
say "results saved to $DEST: $(ls "$DEST" | tr '\n' ' ')"

case "$EVAL_RC" in
  0) say "PILOT PASSED" ;;
  1) say "PILOT RAN, verdict FAILED: see the checks above and $DEST/report.json" ;;
  *) say "evaluation did not complete (exit $EVAL_RC): see output above" ;;
esac
say "vLLM is still running (pgrep -f 'vllm serve'). When you are done, Stop the machine in GridShare."
