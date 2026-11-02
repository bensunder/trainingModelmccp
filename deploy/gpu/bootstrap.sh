#!/usr/bin/env bash
# One-command setup of a fresh GPU box (Ubuntu + CUDA) for MCCP runs.
# Idempotent: re-running only does what is missing. No `set -e` on purpose.
#
#   git clone https://github.com/bensunder/trainingModelmccp /root/mccp   # first time only
#   bash /root/mccp/deploy/gpu/bootstrap.sh
#
# Installs: uv, the control-plane env (uv.lock), the hash-locked trainer venv,
# and the vLLM venv (+ ninja). Verifies network, GPU, driver, disk and torch.

REPO="${REPO:-/root/mccp}"
LOG="${LOG:-/root/bootstrap.log}"
BRANCH="${BRANCH:-main}"
# shellcheck source=deploy/gpu/lib.sh
. "$(dirname "$0")/lib.sh"

say "bootstrap: repo $REPO ($BRANCH)"
check_network || die "this machine cannot reach GitHub / PyPI / Hugging Face; Stop it and pick another GPU"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found: pick the Ubuntu + CUDA template"
say "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader)"
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRIVER_MAJOR:-0}" -ge 580 ] || die "driver $DRIVER_MAJOR < 580: the locked torch (CUDA 13) will not run here"
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
say "disk free: ${FREE_GB} GB"
[ "${FREE_GB:-0}" -ge 45 ] || die "need >= 45 GB free (model 16 GB + venvs ~15 GB + checkpoints)"

if ! command -v uv >/dev/null; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 || die "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -d "$REPO/.git" ]; then
  say "updating repo"
  git -C "$REPO" fetch -q origin "$BRANCH" || die "git fetch failed"
  if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    die "$REPO has local changes; commit or discard them, then re-run"
  fi
  git -C "$REPO" checkout -q "$BRANCH" 2>/dev/null || git -C "$REPO" checkout -q -b "$BRANCH" "origin/$BRANCH"
  git -C "$REPO" reset -q --hard "origin/$BRANCH"
else
  die "$REPO is not a git clone (git clone https://github.com/bensunder/trainingModelmccp $REPO)"
fi
say "repo at $(git -C "$REPO" log -1 --format='%h %s')"

say "control-plane environment (uv.lock)"
(cd "$REPO" && uv sync --frozen --no-dev >>"$LOG" 2>&1) || die "uv sync failed; see $LOG"

if [ ! -x "$TRAINER_PY" ]; then
  say "trainer venv (hash-locked; ~2.5 GB of CUDA wheels)"
  uv venv "$(dirname "$(dirname "$TRAINER_PY")")" --python 3.12 >>"$LOG" 2>&1 || die "trainer venv failed"
fi
uv pip install --python "$TRAINER_PY" --require-hashes -r "$REPO/trainer/requirements.lock.txt" >>"$LOG" 2>&1 \
  || die "trainer install failed; see $LOG"
"$TRAINER_PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else '-'); sys.exit(0 if ok else 1)" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || die "torch cannot see the GPU"

ensure_vllm_venv || die "vLLM install failed; see $LOG"
ensure_key || die "could not create the vLLM API key"
say "BOOTSTRAP OK. Next: bash $REPO/deploy/gpu/run_batch.sh   (inside tmux)"
