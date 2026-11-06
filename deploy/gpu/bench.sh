#!/usr/bin/env bash
# Serving throughput + unit-cost benchmark on this GPU (see deploy/gpu/throughput.py).
# Normally started by `mccp-gpu bench`; can be run directly inside tmux:
#   REPO=/root/mccp-releases/<sha> DEST=/workspace/mccp-bench-x bash deploy/gpu/bench.sh
# Env: PROFILE (qwen3-8b@v1), CONCURRENCY (1,5,10), DURATION (120 s per level),
#      OUTPUT_TOKENS (256), ADAPTER_DIR (optional LoRA to measure instead of the base),
#      GPU_RATE_INR (30.5), USD_PER_INR (rate for the $ column), EAGER (1; 0 = CUDA graphs)
# Downloads the base model if missing, starts vLLM, measures, stops vLLM. No `set -e` on purpose.

STARTED=$(date +%s)
PROFILE="${PROFILE:-qwen3-8b@v1}"
CONCURRENCY="${CONCURRENCY:-1,5,10}"
DURATION="${DURATION:-120}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
GPU_RATE_INR="${GPU_RATE_INR:-30.5}"
USD_PER_INR="${USD_PER_INR:-}"
REPO="${REPO:-/root/mccp}"
DEST="${DEST:-/workspace/mccp-bench-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$DEST"
LOG="$DEST/bench.log.txt"   # .txt: GridShare's /workspace save skips *.log
export PATH="$HOME/.local/bin:$PATH"
eval "$(cd "$REPO" && uv run python deploy/gpu/pilot.py profile-env --profile "$PROFILE")" \
  || { echo "STOP: unknown model profile $PROFILE" | tee -a "$LOG"; exit 1; }
MODEL_CACHE="${MODEL_CACHE:-/root/models/$PROFILE_ID}"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

say "throughput bench: $PROFILE_REF ($HF_ID@${HF_REVISION:0:12}), concurrency [$CONCURRENCY], ${DURATION}s per level, ${OUTPUT_TOKENS} output tokens/request, GPU INR $GPU_RATE_INR/h"
[ -z "$ADAPTER_DIR" ] || [ -f "$ADAPTER_DIR/adapter_config.json" ] || die "no adapter at $ADAPTER_DIR"
ensure_vllm_venv || die "vLLM install failed; see $LOG"
ensure_key || die "could not create the vLLM API key"

ensure_model "$MODEL_CACHE" || die "model download failed; see $LOG"

stop_vllm   # a clean server with known settings, never one left over from another job
ensure_vllm "$MODEL_CACHE" "$DEST/vllm.log.txt" || die "vLLM did not start (see $DEST/vllm.log.txt)"

XBODY=""
[ -n "$THINKING_KWARG" ] && XBODY="{\"chat_template_kwargs\": {\"$THINKING_KWARG\": false}}"
ARGS=(--vllm-key "$KEY" --model "$BASE_SERVED" --extra-body "$XBODY" --concurrency "$CONCURRENCY"
      --duration "$DURATION" --output-tokens "$OUTPUT_TOKENS" --gpu-rate-per-hour "$GPU_RATE_INR"
      --currency INR --gpu-name "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
      --bench-started "$STARTED" --out "$DEST"
      --serving-settings "vLLM bf16, enforce-eager=$EAGER, max $MAX_SEQS sequences, $MAX_LEN context, gpu-memory-utilization $VLLM_MEM, LoRA enabled")
[ -n "$USD_PER_INR" ] && ARGS+=(--usd-per-currency "$USD_PER_INR")
[ -n "$ADAPTER_DIR" ] && ARGS+=(--adapter-dir "$ADAPTER_DIR")
(cd "$REPO" && uv run python deploy/gpu/throughput.py "${ARGS[@]}") 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
stop_vllm
if [ "$RC" = 0 ]; then say "BENCH DONE: $DEST/throughput.md"; else say "BENCH FINISHED WITH ERRORS (exit $RC): $DEST/throughput.md"; fi
exit "$RC"
