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
TAG="n${N}-e${EPOCHS}"
[ "$PROFILE_REF" = "qwen3-8b@v1" ] || TAG="$PROFILE_ID-$TAG"
LOG="$RUNS/batch-$TAG.log"
DEST="${DEST:-/workspace/mccp-batch-$TAG}"
mkdir -p "$RUNS"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

say "batch $TAG: profile $PROFILE_REF ($PROFILE_STATUS, origin $PROFILE_ORIGIN, $HF_ID), seeds [$SEEDS], OOD $OOD_N, runs in $RUNS"
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

# 1. prepare + train each seed -----------------------------------------------------
for SEED in $SEEDS; do
  W="$RUNS/$TAG-s$SEED"
  mkdir -p "$W"
  [ -e "$W/model" ] || ln -s "$MODEL_CACHE" "$W/model"
  if [ ! -f "$W/run.json" ]; then
    say "preparing seed $SEED"
    (cd "$REPO" && uv run python deploy/gpu/pilot.py prepare --workdir "$W" --n "$N" --epochs "$EPOCHS" \
      --seed "$SEED" --ood-n "$OOD_N" --save-steps "$SAVE_STEPS" --gpu-hourly-usd "$GPU_USD_PER_HOUR" \
      --profile "$PROFILE_REF" "${DRAFT_ARG[@]}") 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" = 0 ] || die "prepare failed for seed $SEED"
  fi
  if grep -q '"type": "done"' "$W/events.jsonl" 2>/dev/null; then
    say "seed $SEED already trained"
    continue
  fi
  grep -q '"type": "error"' "$W/events.jsonl" 2>/dev/null && die "seed $SEED has a trainer error in $W/events.jsonl; fix it or remove that workdir"
  stop_vllm
  if pgrep -f "mccp_trainer_entry.py" >/dev/null; then
    say "a trainer is already running; waiting for it"
  else
    say "training seed $SEED (resumes from the last checkpoint if interrupted)"
    (cd "$W" && nohup "$TRAINER_PY" "$REPO/trainer/mccp_trainer_entry.py" --run run.json > trainer.out 2>&1 &)
    sleep 20
  fi
  wait_training "$W" || die "training failed for seed $SEED (details above)"
done

# 2. evaluate ------------------------------------------------------------------------
ensure_vllm "$MODEL_CACHE" "$RUNS/vllm-$TAG.log" || die "vLLM did not start (log above)"
FIRST=1
REPORTS=()
for SEED in $SEEDS; do
  W="$RUNS/$TAG-s$SEED"
  REPORTS+=("$W/report.json")
  if [ -f "$W/report.json" ] && grep -q '"benchmark"' "$W/report.json"; then
    say "seed $SEED already evaluated"
    FIRST=0
    continue
  fi
  TARGETS=adapter
  if [ "$FIRST" = 1 ]; then
    TARGETS=adapter,base
    [ -n "$BASELINE_URL" ] && TARGETS=adapter,base,frontier
  fi
  say "evaluating seed $SEED ($TARGETS on in_dist + ood)"
  ARGS=(evaluate --workdir "$W" --vllm-api-key "$KEY" --targets "$TARGETS" --sets "in_dist,ood"
        --gpu-usd-per-hour "$GPU_USD_PER_HOUR")
  if [ -n "$BASELINE_URL" ]; then
    ARGS+=(--baseline-url "$BASELINE_URL" --baseline-model "$BASELINE_MODEL" --baseline-auth "${BASELINE_AUTH:-bearer}")
    [ -n "$BASELINE_PRICE_IN" ] && ARGS+=(--baseline-price-in "$BASELINE_PRICE_IN")
    [ -n "$BASELINE_PRICE_OUT" ] && ARGS+=(--baseline-price-out "$BASELINE_PRICE_OUT")
    [ -n "$BASELINE_EXTRA_BODY" ] && ARGS+=(--baseline-extra-body "$BASELINE_EXTRA_BODY")
    [ -n "$BASELINE_DROP" ] && ARGS+=(--baseline-drop "$BASELINE_DROP")
  fi
  (cd "$REPO" && uv run python deploy/gpu/pilot.py "${ARGS[@]}") 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  [ "$RC" -le 1 ] || die "evaluation did not complete for seed $SEED (exit $RC)"
  [ "$RC" = 0 ] || say "seed $SEED: verdict FAILED or an endpoint error rate was above 5% (see report)"
  FIRST=0
done

# 3. compare + save ------------------------------------------------------------------
say "comparison across seeds"
(cd "$REPO" && uv run python deploy/gpu/pilot.py compare "${REPORTS[@]}" --out "$DEST") 2>&1 | tee -a "$LOG"
for SEED in $SEEDS; do save_run "$RUNS/$TAG-s$SEED" "$DEST/seed-$SEED"; done
cp "$LOG" "$DEST/batch.log.txt" 2>/dev/null
cp "$RUNS/vllm-$TAG.log" "$DEST/vllm.log.txt" 2>/dev/null
stop_vllm
say "BATCH DONE. Results in $DEST (comparison.md + one folder per seed). Stop the machine in GridShare now."
