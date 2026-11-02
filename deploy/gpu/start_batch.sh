#!/usr/bin/env bash
# Interactive launcher for run_batch.sh with an optional OpenAI frontier baseline.
#   bash /root/mccp/deploy/gpu/start_batch.sh
# Asks for the API key (hidden, read from the terminal so a multi-line paste
# cannot be swallowed as the key), verifies it with a free /v1/models call,
# lets you pick a model by number, then starts the batch. Press Enter at the
# key prompt to run without a baseline. No `set -e` on purpose.

BASE_URL="${BASELINE_URL:-https://api.openai.com}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -t 0 ] && [ ! -r /dev/tty ]; then
  echo "STOP: run this in an interactive terminal"; exit 1
fi

KEY=""
while true; do
  printf 'Paste your OpenAI API key and press Enter (hidden; just Enter = no baseline): '
  IFS= read -rs KEY </dev/tty
  echo
  KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"
  if [ -z "$KEY" ]; then
    echo "No key: running WITHOUT a frontier baseline."
    break
  fi
  case "$KEY" in
    sk-*) ;;
    *) echo "That does not look like an OpenAI key (should start with sk-). Try again."; continue ;;
  esac
  MODELS_JSON="$(curl -s -m 20 "$BASE_URL/v1/models" -H "Authorization: Bearer $KEY")"
  if ! printf '%s' "$MODELS_JSON" | grep -q '"data"'; then
    echo "Key rejected or endpoint unreachable: $(printf '%s' "$MODELS_JSON" | head -c 200)"
    echo "Try again, or press Enter alone to skip the baseline."
    continue
  fi
  mapfile -t MODELS < <(printf '%s' "$MODELS_JSON" | grep -o '"id": *"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' \
    | grep -E '^(gpt-|o[0-9])' | grep -viE 'audio|realtime|transcribe|tts|image|search|embedding|instruct|codex' | sort -r)
  if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Key works but no chat models are listed for it. Press Enter alone to skip, or use another key."
    continue
  fi
  echo "Key OK. Chat models this key can use:"
  for i in "${!MODELS[@]}"; do printf '  %2d) %s\n' "$((i + 1))" "${MODELS[$i]}"; done
  while true; do
    printf 'Pick the model a customer would compare against (number): '
    IFS= read -r PICK </dev/tty
    if [[ "$PICK" =~ ^[0-9]+$ ]] && [ "$PICK" -ge 1 ] && [ "$PICK" -le "${#MODELS[@]}" ]; then
      MODEL="${MODELS[$((PICK - 1))]}"
      break
    fi
    echo "Enter a number from 1 to ${#MODELS[@]}."
  done
  export MCCP_BASELINE_API_KEY="$KEY"
  export BASELINE_URL="$BASE_URL" BASELINE_MODEL="$MODEL"
  echo "Frontier baseline: $MODEL (paid calls only during evaluation, about \$3-10 in total)."
  break
done
unset KEY MODELS_JSON

echo "Starting the batch. Training takes ~4.6 h; you can disconnect (tmux keeps it running)."
exec bash "$DIR/run_batch.sh"
