#!/usr/bin/env bash
# MCCP benchmark batch on one GPU: N records x EPOCHS, one adapter per seed,
# each scored on the in-distribution held-out split and the OOD set, against
# the base model and (optionally) a prompted frontier model.
#
#   1. checks network reachability (GitHub, PyPI, Hugging Face) and tools
#   2. prepares one workdir per seed (shared base-model cache; no re-download)
#   3. trains the seeds one after another (GPU free of vLLM while training)
#   4. starts vLLM once, evaluates every adapter (base/frontier only once)
#   5. writes comparison.md (mean / min / max across seeds) and saves all
#      evidence to /workspace/mccp-batch-<tag>, then stops vLLM to free the GPU
#
# Usage (GPU box, inside tmux):
#   bash /root/mccp/deploy/gpu/run_batch.sh
# Optional frontier baseline (key stays in the environment, never on a command line):
#   export MCCP_BASELINE_API_KEY=...     # read silently: read -rs MCCP_BASELINE_API_KEY; export MCCP_BASELINE_API_KEY
#   BASELINE_URL=https://api.openai.com BASELINE_MODEL=gpt-4.1 \
#   BASELINE_PRICE_IN=2 BASELINE_PRICE_OUT=8 bash /root/mccp/deploy/gpu/run_batch.sh
#   (Azure OpenAI: BASELINE_URL=https://<resource>.openai.azure.com/openai BASELINE_AUTH=api-key
#    BASELINE_MODEL=<deployment>.  LiteLLM: BASELINE_URL=http://<host>:4000.
#    Moonshot Kimi: BASELINE_URL=https://api.moonshot.ai BASELINE_MODEL=kimi-k2.6 BASELINE_DROP=temperature
#    BASELINE_EXTRA_BODY='{"thinking": {"type": "disabled"}}')
#
# Learning curves (one batch, one vLLM start, base model scored once):
#   SIZES="100 300 1000" METHODS="sft rgt" SEEDS=42 TEST_N=300 bash run_batch.sh
#   METHODS: sft = standard fine-tuning, rgt = Rationale-Guided Training (strategy-explicit
#   copies of each decision sample; with RGT_ATTEMPTS=1 the base model's own answers are
#   graded first so strategies name its actual mistakes). TEST_N fixes one test set for
#   every size (required when SIZES has more than one value). Writes curve.md.
#
# Safe to re-run: finished prepares, trainings and evaluations are skipped.

PROFILE="${PROFILE:-qwen3-8b@v1}"   # model profile (profiles/<ref>.yaml); swap it to swap the base model
N="${N:-3000}"
EPOCHS="${EPOCHS:-1}"
SEEDS="${SEEDS:-42 7}"
OOD_N="${OOD_N:-200}"
SAVE_STEPS="${SAVE_STEPS:-100}"
GPU_USD_PER_HOUR="${GPU_USD_PER_HOUR:-0.35}"
RUNS="${RUNS:-/root/mccp-runs}"
REPO="${REPO:-/root/mccp}"
export PATH="$HOME/.local/bin:$PATH"
eval "$(cd "$REPO" && uv run python deploy/gpu/pilot.py profile-env --profile "$PROFILE")" \
  || { echo "STOP: unknown model profile $PROFILE"; exit 1; }
MODEL_CACHE="${MODEL_CACHE:-/root/models/$PROFILE_ID}"
SIZES="${SIZES:-$N}"
METHODS="${METHODS:-sft}"
TEST_N="${TEST_N:-}"
TEST_SEED="${TEST_SEED:-7777}"
RGT_ATTEMPTS="${RGT_ATTEMPTS:-1}"
TRAIN_MIX="${TRAIN_MIX:-standard}"   # standard | boundary (near-threshold cases + rule collisions)
MESSY_N="${MESSY_N:-300}"        # messy-invoice eval records (OCR noise, emails, tables, distractors); 0 = off
EVAL_SETS="in_dist,ood"
[ "$MESSY_N" -gt 0 ] && EVAL_SETS="in_dist,ood,messy"
read -r -a SIZE_LIST <<<"${SIZES//,/ }"
read -r -a METHOD_LIST <<<"${METHODS//,/ }"
read -r -a SEED_LIST <<<"${SEEDS//,/ }"
CURVE=0
[ "${#SIZE_LIST[@]}" -gt 1 ] || [ "$METHODS" != sft ] && CURVE=1
TAG="n${N}-e${EPOCHS}"
[ "$CURVE" = 1 ] && TAG="curve-e${EPOCHS}"
[ "$TRAIN_MIX" = standard ] || TAG="$TRAIN_MIX-$TAG"
[ "$PROFILE_REF" = "qwen3-8b@v1" ] || TAG="$PROFILE_ID-$TAG"
LOG="$RUNS/batch-$TAG.log"
DEST="${DEST:-/workspace/mccp-batch-$TAG}"
mkdir -p "$RUNS"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

say "batch $TAG (training mix: $TRAIN_MIX): profile $PROFILE_REF ($PROFILE_STATUS, origin $PROFILE_ORIGIN, $HF_ID), methods [${METHOD_LIST[*]}], sizes [${SIZE_LIST[*]}], seeds [${SEED_LIST[*]}], test ${TEST_N:-split}, OOD $OOD_N, runs in $RUNS"
for M in "${METHOD_LIST[@]}"; do [ "$M" = sft ] || [ "$M" = rgt ] || die "unknown method $M (sft|rgt)"; done
for S in "${SIZE_LIST[@]}"; do [[ "$S" =~ ^[0-9]+$ ]] || die "size $S is not a number"; done
[ "${#SIZE_LIST[@]}" -le 1 ] || [ -n "$TEST_N" ] || die "several SIZES need TEST_N (one fixed test set), or the sizes are scored on different data"
DRAFT_ARG=()
if [ "$PROFILE_STATUS" != "validated" ]; then
  [ "${ALLOW_DRAFT:-0}" = 1 ] || die "profile $PROFILE_REF is $PROFILE_STATUS: run 'cd $REPO && uv run mccp profile validate $PROFILE_REF --trainer-python $TRAINER_PY' first (CPU only), or ALLOW_DRAFT=1 for an exploratory run"
  DRAFT_ARG=(--allow-draft)
  say "WARNING: draft profile; results are exploratory, not evidence"
fi
command -v uv >/dev/null || die "uv not on PATH (run: source \$HOME/.local/bin/env)"
[ -x "$TRAINER_PY" ] || die "trainer venv missing at $TRAINER_PY (see the pilot install steps)"
check_network || die "this machine cannot reach the internet; Stop it and pick another GPU"
if [ -n "$BASELINE_URL" ]; then
  [ -n "$BASELINE_MODEL" ] || die "BASELINE_URL set but BASELINE_MODEL is empty"
  [ -n "$MCCP_BASELINE_API_KEY" ] || die "BASELINE_URL set but MCCP_BASELINE_API_KEY is not exported"
  say "frontier baseline: $BASELINE_MODEL at $BASELINE_URL (auth ${BASELINE_AUTH:-bearer})"
else
  say "no frontier baseline (set BASELINE_URL/BASELINE_MODEL/MCCP_BASELINE_API_KEY to include one)"
fi
ensure_vllm_venv || die "vLLM install failed; see: tail -40 $LOG"
ensure_key || die "could not create API key"

# Reuse the base model already downloaded by the Qwen pilot (about 16 GB).
if [ "$PROFILE_REF" = "qwen3-8b@v1" ] && [ ! -d "$MODEL_CACHE" ] && [ -d /root/mccp-pilot/model ] && [ ! -L /root/mccp-pilot/model ]; then
  mkdir -p "$(dirname "$MODEL_CACHE")"
  mv /root/mccp-pilot/model "$MODEL_CACHE" && ln -s "$MODEL_CACHE" /root/mccp-pilot/model
  say "moved the pilot's base model to the shared cache $MODEL_CACHE"
fi
mkdir -p "$MODEL_CACHE"
say "disk: $(df -h / | tail -1 | awk '{print $4 " free of " $2}')"

# Variants: one workdir per (method, size, seed). A plain batch (sft, one size)
# keeps the original workdir names so interrupted runs resume where they stopped.
VARIANTS=()
for M in "${METHOD_LIST[@]}"; do for S in "${SIZE_LIST[@]}"; do for SEED in "${SEED_LIST[@]}"; do
  VARIANTS+=("$M:$S:$SEED")
done; done; done
wdir() {  # method size seed
  if [ "$CURVE" = 0 ]; then echo "$RUNS/$TAG-s$3"; else echo "$RUNS/${TAG%curve-e*}$1-n$2-e$EPOCHS-s$3"; fi
  # (TAG carries the profile and a non-standard TRAIN_MIX, so mixes never share a workdir)
}
timed() {  # workdir stage start_epoch: append the stage's wall time
  printf '{"stage": "%s", "seconds": %s}\n' "$2" "$(( $(date +%s) - $3 ))" >> "$1/timing.jsonl"
}

# 1. prepare every variant (RGT attempts need the base model served once) ----------
NEED_ATTEMPTS=0
for V in "${VARIANTS[@]}"; do
  IFS=: read -r M S SEED <<<"$V"
  W="$(wdir "$M" "$S" "$SEED")"
  [ "$M" = rgt ] && [ "$RGT_ATTEMPTS" = 1 ] && [ ! -f "$W/run.json" ] && NEED_ATTEMPTS=1
done
if [ "$NEED_ATTEMPTS" = 1 ]; then
  ensure_model "$MODEL_CACHE" || die "could not download $HF_ID"
  T0=$(date +%s)
  ensure_vllm "$MODEL_CACHE" "$RUNS/vllm-$TAG.log" || die "vLLM did not start for RGT attempts (log above)"
  ATTEMPT_SERVE_S=$(( $(date +%s) - T0 ))
fi
for V in "${VARIANTS[@]}"; do
  IFS=: read -r M S SEED <<<"$V"
  W="$(wdir "$M" "$S" "$SEED")"
  mkdir -p "$W"
  [ -e "$W/model" ] || ln -s "$MODEL_CACHE" "$W/model"
  [ -f "$W/run.json" ] && continue
  say "preparing $M n=$S seed $SEED"
  PARGS=(prepare --workdir "$W" --n "$S" --epochs "$EPOCHS" --seed "$SEED" --ood-n "$OOD_N"
         --save-steps "$SAVE_STEPS" --gpu-hourly-usd "$GPU_USD_PER_HOUR" --profile "$PROFILE_REF" --method "$M"
         --train-mix "$TRAIN_MIX")
  [ -n "$TEST_N" ] && PARGS+=(--test-n "$TEST_N" --test-seed "$TEST_SEED")
  [ "$MESSY_N" -gt 0 ] && PARGS+=(--messy-n "$MESSY_N")
  if [ "$M" = rgt ] && [ "$RGT_ATTEMPTS" = 1 ]; then
    PARGS+=(--attempts-url "http://127.0.0.1:$PORT" --attempts-key "$KEY" --attempts-model "$BASE_SERVED"
            --attempts-cache "$RUNS/rgt-attempts-$PROFILE_ID.jsonl")
  fi
  T0=$(date +%s)
  (cd "$REPO" && uv run python deploy/gpu/pilot.py "${PARGS[@]}" "${DRAFT_ARG[@]}") 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || die "prepare failed for $M n=$S seed $SEED"
  timed "$W" prepare "$T0"
done
[ "$NEED_ATTEMPTS" = 1 ] && say "vLLM start for RGT attempts took ${ATTEMPT_SERVE_S}s (one-off, shared by all RGT variants)"

# 2. train every variant, one after another (GPU free of vLLM) --------------------
for V in "${VARIANTS[@]}"; do
  IFS=: read -r M S SEED <<<"$V"
  W="$(wdir "$M" "$S" "$SEED")"
  if grep -q '"type": "done"' "$W/events.jsonl" 2>/dev/null; then
    say "$M n=$S seed $SEED already trained"
    continue
  fi
  grep -q '"type": "error"' "$W/events.jsonl" 2>/dev/null && die "$W has a trainer error in events.jsonl; fix it or remove that workdir"
  stop_vllm
  T0=""
  if pgrep -f "mccp_trainer_entry.py" >/dev/null; then
    say "a trainer is already running; waiting for it"
  else
    say "training $M n=$S seed $SEED (resumes from the last checkpoint if interrupted)"
    T0=$(date +%s)
    (cd "$W" && nohup "$TRAINER_PY" "$REPO/trainer/mccp_trainer_entry.py" --run run.json > trainer.out 2>&1 &)
    sleep 20
  fi
  wait_training "$W" || die "training failed for $M n=$S seed $SEED (details above)"
  # Wall time from launch to adapter: model load, render checks and training.
  [ -n "$T0" ] && timed "$W" train "$T0"
done

# 3. evaluate (base and frontier once: every variant shares the test sets) ---------
ensure_vllm "$MODEL_CACHE" "$RUNS/vllm-$TAG.log" || die "vLLM did not start (log above)"
FIRST=1
REPORTS=()
for V in "${VARIANTS[@]}"; do
  IFS=: read -r M S SEED <<<"$V"
  W="$(wdir "$M" "$S" "$SEED")"
  REPORTS+=("$W/report.json")
  if [ -f "$W/report.json" ] && grep -q '"benchmark"' "$W/report.json"; then
    say "$M n=$S seed $SEED already evaluated"
    FIRST=0
    continue
  fi
  if [ "$MESSY_N" -gt 0 ] && [ ! -f "$W/data/golden_messy.jsonl" ]; then  # workdir prepared by older code
    (cd "$REPO" && uv run python deploy/gpu/pilot.py add-messy --workdir "$W" --n "$MESSY_N") 2>&1 | tee -a "$LOG"
  fi
  TARGETS=adapter
  if [ "$FIRST" = 1 ]; then
    TARGETS=adapter,base
    [ -n "$BASELINE_URL" ] && TARGETS=adapter,base,frontier
  fi
  say "evaluating $M n=$S seed $SEED ($TARGETS on $EVAL_SETS)"
  ARGS=(evaluate --workdir "$W" --vllm-api-key "$KEY" --targets "$TARGETS" --sets "$EVAL_SETS"
        --gpu-usd-per-hour "$GPU_USD_PER_HOUR")
  if [ -n "$BASELINE_URL" ]; then
    ARGS+=(--baseline-url "$BASELINE_URL" --baseline-model "$BASELINE_MODEL" --baseline-auth "${BASELINE_AUTH:-bearer}")
    [ -n "$BASELINE_PRICE_IN" ] && ARGS+=(--baseline-price-in "$BASELINE_PRICE_IN")
    [ -n "$BASELINE_PRICE_OUT" ] && ARGS+=(--baseline-price-out "$BASELINE_PRICE_OUT")
    [ -n "$BASELINE_EXTRA_BODY" ] && ARGS+=(--baseline-extra-body "$BASELINE_EXTRA_BODY")
    [ -n "$BASELINE_DROP" ] && ARGS+=(--baseline-drop "$BASELINE_DROP")
  fi
  T0=$(date +%s)
  (cd "$REPO" && uv run python deploy/gpu/pilot.py "${ARGS[@]}") 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  [ "$RC" -le 1 ] || die "evaluation did not complete for $M n=$S seed $SEED (exit $RC)"
  timed "$W" eval "$T0"
  [ "$RC" = 0 ] || say "$M n=$S seed $SEED: verdict FAILED or an endpoint error rate was above 5% (see report)"
  FIRST=0
done

# 4. compare + save ------------------------------------------------------------------
if [ "$CURVE" = 1 ]; then
  say "learning curve across methods and sizes"
  (cd "$REPO" && uv run python deploy/gpu/pilot.py curve "${REPORTS[@]}" --out "$DEST") 2>&1 | tee -a "$LOG"
else
  say "comparison across seeds"
  (cd "$REPO" && uv run python deploy/gpu/pilot.py compare "${REPORTS[@]}" --out "$DEST") 2>&1 | tee -a "$LOG"
fi
for V in "${VARIANTS[@]}"; do
  IFS=: read -r M S SEED <<<"$V"
  W="$(wdir "$M" "$S" "$SEED")"
  save_run "$W" "$DEST/$(basename "$W")"
done
cp "$LOG" "$DEST/batch.log.txt" 2>/dev/null
cp "$RUNS/vllm-$TAG.log" "$DEST/vllm.log.txt" 2>/dev/null
stop_vllm
if [ "$CURVE" = 1 ]; then
  say "BATCH DONE. Results in $DEST (curve.md + one folder per run). Stop the machine in GridShare now."
else
  say "BATCH DONE. Results in $DEST (comparison.md + one folder per seed). Stop the machine in GridShare now."
fi
