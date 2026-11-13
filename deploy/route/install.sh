#!/usr/bin/env bash
# First-time setup of the Route gateway on this host. Safe to re-run: existing
# secrets are kept; only missing values are asked for. Secrets are read with
# hidden prompts and written to deploy/route/.env (mode 600), never echoed.
#
#   cd /opt/mccp-git && ./deploy/route/install.sh
cd "$(dirname "$0")" || exit 1
umask 077
ENV_FILE=.env
touch "$ENV_FILE" && chmod 600 "$ENV_FILE"

have() { grep -q "^$1=" "$ENV_FILE"; }
put() { have "$1" || printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; }
rand_b64() { python3 -c "import base64,os;print(base64.b64encode(os.urandom($1)).decode())"; }

command -v docker >/dev/null || { echo "docker is required"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose v2 is required"; exit 1; }
[ -f route.yaml ] || { cp route.example.yaml route.yaml && echo "created route.yaml from the example: set real prices and model names before production"; }
mkdir -p evidence

put ROUTE_PG_PASSWORD "$(rand_b64 24 | tr -d '/+=')"
put ROUTE_KEY_PEPPER "$(rand_b64 32)"
if ! have ROUTE_ADMIN_TOKEN_SHA256; then
  TOKEN="adm_$(rand_b64 32 | tr -d '/+=')"
  put ROUTE_ADMIN_TOKEN_SHA256 "$(printf '%s' "$TOKEN" | sha256sum | cut -d' ' -f1)"
  echo "ADMIN TOKEN (shown once; store it in your password manager now):"
  echo "  $TOKEN"
  unset TOKEN
fi

for VAR in DEEPSEEK_API_KEY MOONSHOT_API_KEY ZAI_API_KEY ANTHROPIC_API_KEY; do
  if have "$VAR"; then echo "$VAR: kept"; continue; fi
  printf '%s (hidden; Enter to skip): ' "$VAR"
  IFS= read -rs VAL </dev/tty; echo
  VAL="$(printf '%s' "$VAL" | tr -d '[:space:]')"
  [ -n "$VAL" ] && put "$VAR" "$VAL" && echo "$VAR: saved"
  unset VAL
done

if grep -q "UNVERIFIED" route.yaml && ! grep -q "^ROUTE_ENV=" "$ENV_FILE"; then
  echo "route.yaml still has UNVERIFIED prices: starting in development mode (ROUTE_ENV=development)."
  echo "Set real prices and price_source, then add ROUTE_ENV=production to .env and re-run."
  put ROUTE_ENV development
fi

docker compose -f compose.yaml --env-file "$ENV_FILE" up -d --build || exit 1
printf 'waiting for the gateway'
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${ROUTE_PORT:-8090}/healthz" >/dev/null 2>&1; then echo " up"; break; fi
  printf '.'; sleep 2
done
curl -sS "http://127.0.0.1:${ROUTE_PORT:-8090}/readyz"; echo
