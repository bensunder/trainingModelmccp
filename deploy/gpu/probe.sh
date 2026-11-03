#!/usr/bin/env bash
# Diagnose the latest finished mccp-gpu run (or RUN_ID=<id>): training text, raw
# base/adapter responses with special tokens, and the frontier endpoint's HTTP error.
# Starts vLLM for a few requests and stops it again (about 3-5 minutes of GPU).
#   bash /root/mccp-src/deploy/gpu/probe.sh
# No `set -e` on purpose.

MCCP_HOME="${MCCP_HOME:-/root/.mccp}"
SRC="${MCCP_SRC:-/root/mccp-src}"
RUN_ID="${RUN_ID:-$(tail -1 "$MCCP_HOME/done.tsv" 2>/dev/null | cut -f1)}"
[ -n "$RUN_ID" ] || { echo "STOP: no finished run in $MCCP_HOME/done.tsv"; exit 1; }
PROFILE="$(awk -F'\t' -v id="$RUN_ID" '$1==id{print $2}' "$MCCP_HOME/done.tsv" | tail -1)"
REPO="$(find /root/mccp-releases -mindepth 2 -maxdepth 2 -name .venv -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$REPO" ] || { echo "STOP: no installed release under /root/mccp-releases"; exit 1; }
W="$(find "/root/mccp-runs/$RUN_ID" -mindepth 2 -maxdepth 2 -name report.json -printf '%h\n' 2>/dev/null | sort | head -1)"
[ -n "$W" ] || { echo "STOP: no evaluated workdir under /root/mccp-runs/$RUN_ID"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
eval "$(cd "$REPO" && uv run python deploy/gpu/pilot.py profile-env --profile "$PROFILE")" || { echo "STOP: profile $PROFILE"; exit 1; }
MODEL_CACHE="/root/models/$PROFILE_ID"
LOG="/root/mccp-runs/$RUN_ID/probe.log"
# shellcheck source=deploy/gpu/lib.sh
. "$REPO/deploy/gpu/lib.sh"
ensure_key || exit 1
XBODY=""
[ -n "$THINKING_KWARG" ] && XBODY="{\"chat_template_kwargs\": {\"$THINKING_KWARG\": false}}"

say "probe: run $RUN_ID, profile $PROFILE_REF, workdir $W"
echo "################ 1. training text"
"$TRAINER_PY" "$SRC/deploy/gpu/probe.py" train-text --workdir "$W" --model-dir "$MODEL_CACHE" 2>&1 | grep -v Warning

echo "################ 2. live responses"
ensure_vllm "$MODEL_CACHE" "/root/mccp-runs/$RUN_ID/vllm-probe.log" || { echo "STOP: vLLM did not start"; exit 1; }
(
  set -a
  # shellcheck disable=SC1090,SC1091
  [ -f "$MCCP_HOME/config.env" ] && . "$MCCP_HOME/config.env"
  # shellcheck disable=SC1090,SC1091
  [ -f "$MCCP_HOME/secrets.env" ] && . "$MCCP_HOME/secrets.env"
  set +a
  "$REPO/.venv/bin/python" "$SRC/deploy/gpu/probe.py" live --workdir "$W" --vllm-key "$KEY" \
    --base-served "$BASE_SERVED" --extra-body "$XBODY" \
    --baseline-url "${BASELINE_URL:-}" --baseline-model "${BASELINE_MODEL:-}"
)
stop_vllm
say "probe done"
