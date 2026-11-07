#!/usr/bin/env bash
# Score the adapters of a finished run on the messy-invoice set, without retraining.
# Normally started by `mccp-gpu eval-messy [RUN_ID]`. Adds report_messy.json next to each
# report.json (report.json is never touched), re-writes curve.md / comparison.md with a
# messy column, and copies everything to the run's /workspace folder.
# Env: RUN_DIR (required), DEST (required), PROFILE, MESSY_N (300), REPO, RESCORE (0; 1 = re-score
# adapters that already have report_messy.json, e.g. after a scoring change).
# Every answer is saved to report_messy.messy.<target>.outputs.jsonl next to the report.
# No `set -e` on purpose.

PROFILE="${PROFILE:-qwen3-8b@v1}"
MESSY_N="${MESSY_N:-300}"
REPO="${REPO:-/root/mccp}"
GPU_USD_PER_HOUR="${GPU_USD_PER_HOUR:-0.35}"
[ -d "$RUN_DIR" ] || { echo "STOP: RUN_DIR $RUN_DIR not found"; exit 1; }
mkdir -p "$DEST"
LOG="$DEST/eval-messy.log.txt"
export PATH="$HOME/.local/bin:$PATH"
eval "$(cd "$REPO" && uv run python deploy/gpu/pilot.py profile-env --profile "$PROFILE")" \
  || { echo "STOP: unknown model profile $PROFILE" | tee -a "$LOG"; exit 1; }
MODEL_CACHE="${MODEL_CACHE:-/root/models/$PROFILE_ID}"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

mapfile -t WORKDIRS < <(find "$RUN_DIR" -mindepth 2 -maxdepth 2 -name report.json -printf '%h\n' | sort)
[ "${#WORKDIRS[@]}" -gt 0 ] || die "no evaluated workdirs in $RUN_DIR"
say "messy-invoice eval: ${#WORKDIRS[@]} adapters in $RUN_DIR, $MESSY_N records"
ensure_key || die "could not read the vLLM API key"
ensure_model "$MODEL_CACHE" || die "base model download failed; see $LOG"   # a fresh machine has none
ensure_vllm "$MODEL_CACHE" "$DEST/vllm-messy.log.txt" || die "vLLM did not start"
FIRST=1
REPORTS=()
for W in "${WORKDIRS[@]}"; do
  REPORTS+=("$W/report.json")
  [ -f "$W/data/golden_messy.jsonl" ] || \
    (cd "$REPO" && uv run python deploy/gpu/pilot.py add-messy --workdir "$W" --n "$MESSY_N") 2>&1 | tee -a "$LOG"
  if [ -f "$W/report_messy.json" ] && [ "${RESCORE:-0}" != 1 ]; then say "$(basename "$W") already scored on messy"; FIRST=0; continue; fi
  TARGETS=adapter
  [ "$FIRST" = 1 ] && TARGETS=adapter,base
  say "scoring $(basename "$W") ($TARGETS) on messy invoices"
  (cd "$REPO" && uv run python deploy/gpu/pilot.py evaluate --workdir "$W" --vllm-api-key "$KEY" \
    --targets "$TARGETS" --sets messy --report-name report_messy.json --gpu-usd-per-hour "$GPU_USD_PER_HOUR" \
    --save-outputs) 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -le 1 ] || die "messy evaluation failed for $W"
  mkdir -p "$DEST/$(basename "$W")"
  cp "$W/report_messy.json" "$W"/report_messy.*.outputs.jsonl "$DEST/$(basename "$W")/" 2>/dev/null
  FIRST=0
done
if [ "${#WORKDIRS[@]}" -gt 1 ]; then
  (cd "$REPO" && uv run python deploy/gpu/pilot.py curve "${REPORTS[@]}" --out "$DEST") 2>&1 | tee -a "$LOG"
else
  say "single adapter: messy results are in $(basename "${WORKDIRS[0]}")/report_messy.json"
fi
stop_vllm
say "MESSY EVAL DONE: $DEST"
