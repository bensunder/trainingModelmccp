#!/usr/bin/env bash
# MCCP single-GPU pilot: everything after `pilot.py prepare` + trainer start.
#   1. installs vLLM 0.29.0 in its own venv (skipped if present)
#   2. waits for the trainer to finish (or fails fast on a trainer error)
#   3. starts vLLM on the pinned base snapshot with runtime LoRA enabled
#   4. runs `pilot.py evaluate` (L0 publish guard + L1 adapter vs base)
#   5. copies the adapter + evidence to /workspace/mccp-pilot-results
#
# Usage (on the GPU box):  bash /root/mccp/deploy/gpu/run_pilot.sh
# Safe to re-run: finished steps are skipped. No `set -e` on purpose; every
# step checks its own result and stops with a clear message.

WORK=/root/mccp-pilot
REPO=/root/mccp
VENV=/root/venvs/vllm
KEYFILE=/root/vllm.key
PORT=8000
LOG="$WORK/pilot-run.log"
VLLM_MEM="${VLLM_MEM:-0.92}"   # fraction of the 20 GB card; lower if vLLM OOMs at start
MAX_LEN="${MAX_LEN:-4096}"
MAX_SEQS="${MAX_SEQS:-16}"    # eval sends 4 at a time; vLLM default 256 OOMs the warm-up on 20 GB

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORK"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

[ -f "$WORK/run.json" ] || die "$WORK/run.json missing: run pilot.py prepare first"
command -v uv >/dev/null || die "uv not on PATH (run: source \$HOME/.local/bin/env)"

# 1. vLLM venv ---------------------------------------------------------------
if [ -x "$VENV/bin/vllm" ]; then
  say "vLLM already installed"
else
  say "installing vLLM 0.29.0 (several minutes)"
  uv venv "$VENV" --python 3.12 >>"$LOG" 2>&1 || die "could not create $VENV"
  uv pip install --python "$VENV/bin/python" vllm==0.29.0 >>"$LOG" 2>&1 \
    || die "vLLM install failed; see: tail -40 $LOG"
fi
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tee -a "$LOG"
[ -s "$KEYFILE" ] || (cd "$REPO" && uv run python deploy/gpu/pilot.py new-key > "$KEYFILE") \
  || die "could not create API key"
KEY="$(cat "$KEYFILE")"

# 2. wait for training ---------------------------------------------------------
say "waiting for training to finish (checks every 30 s)"
while true; do
  if grep -q '"type": "done"' "$WORK/events.jsonl" 2>/dev/null; then
    say "training done: $(grep '"type": "done"' "$WORK/events.jsonl" | tail -1)"
    break
  fi
  if grep -q '"type": "error"' "$WORK/events.jsonl" 2>/dev/null; then
    grep '"type": "error"' "$WORK/events.jsonl" | tail -1 | tee -a "$LOG"
    tail -30 "$WORK/train.log" | tee -a "$LOG"
    die "training failed (details above)"
  fi
  if ! pgrep -f mccp_trainer_entry.py >/dev/null; then
    tail -30 "$WORK/trainer.out" "$WORK/train.log" 2>/dev/null | tee -a "$LOG"
    die "trainer is not running and reported no result (details above)"
  fi
  printf '  %s\n' "$(tail -c 200 "$WORK/train.log" 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+/[0-9]+' | tail -1)"
  sleep 30
done
[ -f "$WORK/output/adapter_model.safetensors" ] || die "no adapter in $WORK/output"

# 3. vLLM ----------------------------------------------------------------------
up() { curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" | grep -q qwen3-8b-base; }

start_vllm() {  # $1 = "with" | "without" the server-side thinking-off default
  local extra=()
  [ "$1" = with ] && extra=(--default-chat-template-kwargs '{"enable_thinking": false}')
  say "starting vLLM ($1 --default-chat-template-kwargs, mem=$VLLM_MEM, max_len=$MAX_LEN, max_seqs=$MAX_SEQS)"
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True nohup "$VENV/bin/vllm" serve "$WORK/model" \
    --served-model-name qwen3-8b-base --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$VLLM_MEM" --max-num-seqs "$MAX_SEQS" --enforce-eager \
    --enable-lora --max-loras 1 --max-lora-rank 16 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    "${extra[@]}" --host 127.0.0.1 --port "$PORT" --api-key "$KEY" \
    > "$WORK/vllm.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 90); do       # up to 15 minutes
    up && { say "vLLM is up (pid $VLLM_PID)"; return 0; }
    kill -0 "$VLLM_PID" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

if up; then
  say "vLLM already running"
else
  if ! start_vllm with; then
    if grep -qiE 'unrecognized arguments|no such option|default-chat-template-kwargs' "$WORK/vllm.log"; then
      say "this vLLM lacks --default-chat-template-kwargs; retrying without it (eval disables thinking per request)"
      pkill -f "vllm serve" ; sleep 5
      start_vllm without || { tail -40 "$WORK/vllm.log" | tee -a "$LOG"; die "vLLM did not start (log above)"; }
    else
      grep -E "Error|error:|out of memory" "$WORK/vllm.log" | grep -v 'File "' | tail -15 | tee -a "$LOG"
      die "vLLM did not start (log above). If it says out of memory: VLLM_MEM=0.95 MAX_LEN=2048 MAX_SEQS=8 bash $0"
    fi
  fi
fi

# 4. evaluate ------------------------------------------------------------------
say "running evaluation (publish guard + L1, adapter vs base)"
(cd "$REPO" && uv run python deploy/gpu/pilot.py evaluate --workdir "$WORK" --vllm-api-key "$KEY") 2>&1 | tee -a "$LOG"
EVAL_RC=${PIPESTATUS[0]}

# 5. save evidence to /workspace (autosaved by GridShare) ------------------------
DEST=/workspace/mccp-pilot-results
mkdir -p "$DEST"
cp "$WORK"/output/adapter_config.json "$WORK"/output/adapter_model.safetensors "$DEST"/ 2>/dev/null
cp "$WORK"/{events.jsonl,pilot.json,run.json,report.json,soup.yaml,train.log,vllm.log,pilot-run.log} "$DEST"/ 2>/dev/null
say "results saved to $DEST: $(ls "$DEST" | tr '\n' ' ')"

case "$EVAL_RC" in
  0) say "PILOT PASSED" ;;
  1) say "PILOT RAN, verdict FAILED: see the checks above and $DEST/report.json" ;;
  *) say "evaluation did not complete (exit $EVAL_RC): see output above" ;;
esac
say "vLLM is still running (pid file: pgrep -f 'vllm serve'). When you are done, Stop the machine in GridShare."
