# shellcheck shell=bash
# Shared helpers for the single-GPU pilot scripts (sourced, not executed).
# Callers set LOG before sourcing. No `set -e`: every helper returns a status
# and callers decide whether to stop.

REPO="${REPO:-/root/mccp}"
VENV="${VENV:-/root/venvs/vllm}"
TRAINER_PY="${TRAINER_PY:-/root/venvs/trainer/bin/python}"
KEYFILE="${KEYFILE:-/root/vllm.key}"
PORT="${PORT:-8000}"
VLLM_MEM="${VLLM_MEM:-0.92}"   # fraction of a 20-24 GB card; lower if vLLM OOMs at start
MAX_LEN="${MAX_LEN:-4096}"
MAX_SEQS="${MAX_SEQS:-16}"     # vLLM's default (256) OOMs the warm-up next to a 16 GB bf16 model
EAGER="${EAGER:-1}"            # 1 = --enforce-eager (verified); 0 = CUDA graphs (faster decode, more memory, untested here)
BASE_SERVED="${BASE_SERVED:-qwen3-8b-base}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"   # empty = model has no reasoning parser
THINKING_KWARG="${THINKING_KWARG-enable_thinking}"  # chat_template kwarg that switches reasoning; empty = none

export PATH="$HOME/.local/bin:$PATH"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

check_network() {  # the first GridShare host had no outbound HTTPS; fail in seconds, not hours
  local u code bad=0
  for u in https://github.com https://pypi.org https://huggingface.co; do
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u")
    if [ "$code" = "000" ]; then say "network: $u UNREACHABLE"; bad=1; else say "network: $u $code"; fi
  done
  return $bad
}

ensure_vllm_venv() {
  if [ ! -x "$VENV/bin/vllm" ]; then
    say "installing vLLM 0.29.0 (several minutes)"
    uv venv "$VENV" --python 3.12 >>"$LOG" 2>&1 || return 1
    uv pip install --python "$VENV/bin/python" vllm==0.29.0 >>"$LOG" 2>&1 || return 1
  fi
  # vLLM JIT-compiles some kernels at start-up: needs ninja (+ nvcc/gcc from the CUDA devel image).
  if [ ! -x "$VENV/bin/ninja" ]; then
    say "installing ninja (vLLM kernel build tool)"
    uv pip install --python "$VENV/bin/python" ninja >>"$LOG" 2>&1 || return 1
  fi
  export PATH="$VENV/bin:$PATH"
  { command -v nvcc >/dev/null || [ ! -x /usr/local/cuda/bin/nvcc ]; } || export PATH="$PATH:/usr/local/cuda/bin"
  say "vLLM $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?'); build tools: ninja=$(command -v ninja || echo MISSING) nvcc=$(command -v nvcc || echo MISSING) gcc=$(command -v gcc || echo MISSING)"
}

ensure_key() {
  [ -s "$KEYFILE" ] || (cd "$REPO" && uv run python deploy/gpu/pilot.py new-key > "$KEYFILE") || return 1
  chmod 600 "$KEYFILE"
  KEY="$(cat "$KEYFILE")"
}

vllm_up() { curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" | grep -q "$BASE_SERVED"; }

stop_vllm() {
  if pgrep -f "vllm serve" >/dev/null; then
    say "stopping vLLM (frees the GPU)"
    pkill -f "vllm serve"
    for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || return 0; sleep 2; done
    pkill -9 -f "vllm serve"
  fi
  return 0
}

_start_vllm() {  # $1 model dir, $2 vlog, $3 = with|without server-side thinking-off default
  local extra=()
  [ "$3" = with ] && [ -n "$THINKING_KWARG" ] && extra=(--default-chat-template-kwargs "{\"$THINKING_KWARG\": false}")
  [ -n "$REASONING_PARSER" ] && extra+=(--reasoning-parser "$REASONING_PARSER")
  [ "$EAGER" = 1 ] && extra+=(--enforce-eager)
  say "starting vLLM on $1 (mem=$VLLM_MEM max_len=$MAX_LEN max_seqs=$MAX_SEQS eager=$EAGER, thinking default: $3)"
  # FlashInfer's JIT sampling kernel fails to build against host CUDA 12 toolkits;
  # the PyTorch-native sampler gives identical outputs.
  VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True nohup "$VENV/bin/vllm" serve "$1" \
    --served-model-name "$BASE_SERVED" --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$VLLM_MEM" --max-num-seqs "$MAX_SEQS" \
    --enable-lora --max-loras 1 --max-lora-rank 16 \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    "${extra[@]}" --host 127.0.0.1 --port "$PORT" --api-key "$KEY" \
    > "$2" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do  # up to 15 minutes
    vllm_up && { say "vLLM is up (pid $pid)"; return 0; }
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

ensure_vllm() {  # $1 model dir, $2 vllm log path
  vllm_up && { say "vLLM already running"; return 0; }
  _start_vllm "$1" "$2" with && return 0
  if grep -qiE 'unrecognized arguments|no such option' "$2"; then
    say "this vLLM rejects --default-chat-template-kwargs; retrying without it (eval disables thinking per request)"
    stop_vllm
    _start_vllm "$1" "$2" without && return 0
  fi
  grep -E "Error|error:|out of memory|FAILED" "$2" | grep -v 'File "' | grep -v "Engine core initialization failed" | tail -15 | tee -a "$LOG"
  say "if it says out of memory, retry with: VLLM_MEM=0.95 MAX_LEN=2048 MAX_SEQS=8"
  return 1
}

wait_training() {  # $1 workdir; returns 0 when done, 1 on trainer error / death
  local w="$1"
  while true; do
    if grep -q '"type": "done"' "$w/events.jsonl" 2>/dev/null; then
      say "training done ($w): $(grep '"type": "done"' "$w/events.jsonl" | tail -1)"
      return 0
    fi
    if grep -q '"type": "error"' "$w/events.jsonl" 2>/dev/null; then
      grep '"type": "error"' "$w/events.jsonl" | tail -1 | tee -a "$LOG"
      tail -30 "$w/train.log" 2>/dev/null | tee -a "$LOG"
      return 1
    fi
    if ! pgrep -f "mccp_trainer_entry.py" >/dev/null; then
      tail -30 "$w/trainer.out" "$w/train.log" 2>/dev/null | tee -a "$LOG"
      say "trainer for $w is not running and reported no result"
      return 1
    fi
    printf '  %s\n' "$(tail -c 300 "$w/train.log" 2>/dev/null | tr '\r' '\n' | grep -aE '[0-9]+/[0-9]+' | tail -1)"
    sleep 60
  done
}

ensure_model() {  # $1 model dir: download the profile's pinned base model if missing
  [ -f "$1/config.json" ] && return 0
  check_network || return 1
  say "downloading $HF_ID (about 16 GB for an 8B model)"
  mkdir -p "$1"
  "$TRAINER_PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(repo_id=sys.argv[1], revision=sys.argv[2] or None, local_dir=sys.argv[3])" \
    "$HF_ID" "${HF_REVISION:-}" "$1" >>"$LOG" 2>&1
}

save_run() {  # $1 workdir, $2 destination (under /workspace, which GridShare keeps)
  mkdir -p "$2"
  cp "$1"/output/adapter_config.json "$1"/output/adapter_model.safetensors "$2"/ 2>/dev/null
  cp "$1"/{events.jsonl,pilot.json,run.json,report.json,soup.yaml,timing.jsonl} "$2"/ 2>/dev/null
  cp "$1"/data/rgt_strategies.jsonl "$2"/ 2>/dev/null
  cp "$1"/*.outputs.jsonl "$2"/ 2>/dev/null   # every scored answer (evaluate --save-outputs)
  # GridShare's /workspace save skips *.log, so keep logs as .txt.
  local f
  for f in train vllm pilot-run batch trainer; do
    [ -f "$1/$f.log" ] && cp "$1/$f.log" "$2/$f.log.txt"
    [ -f "$1/$f.out" ] && cp "$1/$f.out" "$2/$f.out.txt"
  done
  return 0
}

drop_mirror() {  # $1 workdir, $2 saved run dir: delete the checkpoint mirror once the adapter is safely saved
  local ptr="$1/.ckpt-mirror" dir
  [ -f "$ptr" ] || return 0
  dir="$(head -1 "$ptr")"
  # Only once the saved adapter is byte-identical to the trained one.
  cmp -s "$1/output/adapter_model.safetensors" "$2/adapter_model.safetensors" \
    || { say "kept checkpoint mirror $dir (adapter not confirmed saved in $2)"; return 0; }
  case "$dir" in
    */mccp-ckpt/[0-9a-f]*) ;;   # never delete anything outside a mirror root
    *) say "refusing to delete unexpected mirror path '$dir'"; return 0 ;;
  esac
  rm -rf -- "$dir" && rm -f -- "$ptr" && say "deleted checkpoint mirror $dir (adapter saved)"
  return 0
}
