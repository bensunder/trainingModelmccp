#!/usr/bin/env bash
# Score the adapters of a finished run on an extra test set, without retraining:
# SET=messy (default), docs (rendered PDF -> scan -> OCR) or invcdip (real scanned invoices).
# Normally started by `mccp-gpu eval-messy [RUN_ID]`. Adds report_messy.json next to each
# report.json (report.json is never touched), re-writes curve.md / comparison.md with a
# messy column, and copies everything to the run's /workspace folder.
# Env: RUN_DIR (required), DEST (required), PROFILE, MESSY_N (300), REPO, RESCORE (0; 1 = re-score
# adapters that already have report_messy.json, e.g. after a scoring change).
# Every answer is saved to report_messy.messy.<target>.outputs.jsonl next to the report.
# No `set -e` on purpose.

PROFILE="${PROFILE:-qwen3-8b@v1}"
MESSY_N="${MESSY_N:-300}"
SET="${SET:-messy}"
case "$SET" in messy|docs|invcdip) ;; *) echo "STOP: SET must be messy, docs or invcdip (got $SET)"; exit 1 ;; esac
CACHE="${CACHE:-/workspace/mccp-cache}"   # generated docs and INV-CDIP downloads/OCR, reused across machines
REPO="${REPO:-/root/mccp}"
GPU_USD_PER_HOUR="${GPU_USD_PER_HOUR:-0.35}"
[ -d "$RUN_DIR" ] || { echo "STOP: RUN_DIR $RUN_DIR not found"; exit 1; }
mkdir -p "$DEST"
LOG="$DEST/eval-$SET.log.txt"
export PATH="$HOME/.local/bin:$PATH"
eval "$(cd "$REPO" && uv run python deploy/gpu/pilot.py profile-env --profile "$PROFILE")" \
  || { echo "STOP: unknown model profile $PROFILE" | tee -a "$LOG"; exit 1; }
MODEL_CACHE="${MODEL_CACHE:-/root/models/$PROFILE_ID}"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

mapfile -t WORKDIRS < <(find "$RUN_DIR" -mindepth 2 -maxdepth 2 -name report.json -printf '%h\n' | sort)
[ "${#WORKDIRS[@]}" -gt 0 ] || die "no evaluated workdirs in $RUN_DIR"
say "$SET eval: ${#WORKDIRS[@]} adapters in $RUN_DIR, $MESSY_N records"
ensure_key || die "could not read the vLLM API key"
# Build the test set first: rendering/OCR (docs) and downloads/OCR (invcdip) run on the
# CPU, so the GPU is not held by an idle vLLM meanwhile. Cached in $CACHE for reuse.
for W in "${WORKDIRS[@]}"; do
  # RESCORE=1 rebuilds the set too: prompts and labels come from the current code
  # (the OCR is cached, so this costs seconds), never from an older set file.
  if [ ! -f "$W/data/golden_$SET.jsonl" ] || [ "${RESCORE:-0}" = 1 ]; then
    say "building the $SET set for $(basename "$W") (CPU; cached in $CACHE)"
    (cd "$REPO" && uv run python deploy/gpu/pilot.py add-set --set "$SET" --workdir "$W" --n "$MESSY_N" --cache "$CACHE") 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" = 0 ] || die "could not build the $SET set for $W (log above)"
  fi
done
ensure_model "$MODEL_CACHE" || die "base model download failed; see $LOG"   # a fresh machine has none
ensure_vllm "$MODEL_CACHE" "$DEST/vllm-$SET.log.txt" || die "vLLM did not start"
FIRST=1
REPORTS=()
for W in "${WORKDIRS[@]}"; do
  REPORTS+=("$W/report.json")
  if [ -f "$W/report_$SET.json" ] && [ "${RESCORE:-0}" != 1 ]; then say "$(basename "$W") already scored on $SET"; FIRST=0; continue; fi
  TARGETS=adapter
  [ "$FIRST" = 1 ] && TARGETS=adapter,base
  say "scoring $(basename "$W") ($TARGETS) on $SET"
  (cd "$REPO" && uv run python deploy/gpu/pilot.py evaluate --workdir "$W" --vllm-api-key "$KEY" \
    --targets "$TARGETS" --sets "$SET" --report-name "report_$SET.json" --gpu-usd-per-hour "$GPU_USD_PER_HOUR" \
    --save-outputs) 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -le 1 ] || die "$SET evaluation failed for $W"
  mkdir -p "$DEST/$(basename "$W")"
  cp "$W/report_$SET.json" "$W"/report_"$SET".*.outputs.jsonl "$W/set_$SET.json" "$DEST/$(basename "$W")/" 2>/dev/null
  FIRST=0
done
if [ "$SET" = messy ] && [ "${#WORKDIRS[@]}" -gt 1 ]; then
  (cd "$REPO" && uv run python deploy/gpu/pilot.py curve "${REPORTS[@]}" --out "$DEST") 2>&1 | tee -a "$LOG"
else
  say "results: <run>/report_$SET.json per adapter, every answer in report_$SET.$SET.<target>.outputs.jsonl"
fi
stop_vllm
say "$(printf '%s' "$SET" | tr '[:lower:]' '[:upper:]') EVAL DONE: $DEST"
