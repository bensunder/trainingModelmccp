#!/usr/bin/env bash
# Install / upgrade the MCCP control plane on a CPU-only VPS (Docker + Compose v2).
# Idempotent: secrets are generated once and kept; re-running rebuilds and restarts.
#
#   bash deploy/vps/install.sh                  # from the repo root on the VPS
#   MCCP_CADDY_SITE=api.whyaidata.com bash ...  # also add a /mccp route to Caddy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VPS="$ROOT/deploy/vps"
ENV_FILE="$VPS/.env"
SECRETS="$VPS/secrets"
COMPOSE=(docker compose -f "$VPS/compose.vps.yaml" --env-file "$ENV_FILE")
CADDY_SITE="${MCCP_CADDY_SITE:-api.whyaidata.com}"
CADDYFILE="${MCCP_CADDYFILE:-/etc/caddy/Caddyfile}"

log() { printf '\033[1;34m[mccp]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[mccp] %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"
docker compose version >/dev/null 2>&1 || die "docker compose v2 is required"
command -v openssl >/dev/null || die "openssl is required"

for port in 8095 55432 5055; do
  if ss -ltnH "( sport = :$port )" 2>/dev/null | grep -q . && ! docker ps --format '{{.Names}}' | grep -q '^mccp-'; then
    die "port $port is already in use on this host; free it or edit compose.vps.yaml"
  fi
done

# ---- secrets (generated once) -------------------------------------------------
rnd() { openssl rand -hex 24; }
if [[ ! -f "$ENV_FILE" ]]; then
  log "generating $ENV_FILE"
  umask 077
  cat >"$ENV_FILE" <<EOF
POSTGRES_PASSWORD=$(rnd)
MCCP_OWNER_DB_PASSWORD=$(rnd)
MCCP_APP_DB_PASSWORD=$(rnd)
MLFLOW_DB_PASSWORD=$(rnd)
MCCP_AUTH_MODE=dev
MCCP_DEV_JWT_SECRET=$(openssl rand -hex 32)
# Bind Postgres to a private/allow-listed address when the GPU worker runs elsewhere.
MCCP_PG_BIND=127.0.0.1
# Private vLLM endpoint on the GPU host (used by evaluation and the publish guard).
MCCP_VLLM_BASE_URL=http://127.0.0.1:9
# Switch to Entra ID: MCCP_AUTH_MODE=oidc and set these.
MCCP_OIDC_ISSUER=
MCCP_OIDC_AUDIENCE=
MCCP_OIDC_JWKS_URL=
EOF
fi
set -a; source "$ENV_FILE"; set +a

if [[ ! -f "$VPS/postgres-init.sql" ]]; then
  umask 077
  cat >"$VPS/postgres-init.sql" <<EOF
CREATE ROLE mccp_owner LOGIN PASSWORD '${MCCP_OWNER_DB_PASSWORD}';
CREATE ROLE mccp_app LOGIN PASSWORD '${MCCP_APP_DB_PASSWORD}' NOBYPASSRLS;
CREATE DATABASE mccp OWNER mccp_owner;
GRANT CONNECT ON DATABASE mccp TO mccp_app;
CREATE ROLE mlflow LOGIN PASSWORD '${MLFLOW_DB_PASSWORD}';
CREATE DATABASE mlflow OWNER mlflow;
EOF
fi
# Readable only by the postgres container user (uid 999 in the official
# Debian-based image), never by other local users: it contains DB passwords.
chown 999:999 "$VPS/postgres-init.sql"
chmod 600 "$VPS/postgres-init.sql"

mkdir -p "$SECRETS"
if [[ ! -f "$SECRETS/signing-key.pem" ]]; then
  log "generating adapter signing keypair (ECDSA P-256)"
  openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out "$SECRETS/signing-key.pem"
  openssl ec -in "$SECRETS/signing-key.pem" -pubout -out "$SECRETS/signing-pub.pem" 2>/dev/null
fi
# The API container runs as uid 10001.
chown 10001:10001 "$SECRETS/signing-key.pem" "$SECRETS/signing-pub.pem"
chmod 400 "$SECRETS/signing-key.pem"; chmod 444 "$SECRETS/signing-pub.pem"

# ---- build and start ------------------------------------------------------------
log "building and starting containers"
"${COMPOSE[@]}" up -d --build

log "waiting for the API"
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8095/readyz >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:8095/readyz >/dev/null || { "${COMPOSE[@]}" logs --tail=80 api migrate; die "API did not become ready"; }

# ---- first tenant (idempotent) ---------------------------------------------------
if ! "${COMPOSE[@]}" exec -T postgres psql -U postgres -d mccp -tAc "SELECT 1 FROM tenants WHERE idp_tenant_id='whyaidata'" | grep -q 1; then
  log "creating tenant whyaidata"
  "${COMPOSE[@]}" exec -T api mccp tenant create whyaidata --idp-tenant-id whyaidata >/dev/null
fi

# ---- optional Caddy route -----------------------------------------------------------
if [[ -f "$CADDYFILE" ]] && command -v caddy >/dev/null; then
  if grep -q "handle_path /mccp/\*" "$CADDYFILE"; then
    log "Caddy route /mccp already present"
  elif grep -qE "^${CADDY_SITE//./\\.}[[:space:]]*\{" "$CADDYFILE"; then
    cp "$CADDYFILE" "$CADDYFILE.bak.mccp.$(date +%s)"
    awk -v site="$CADDY_SITE" '
      $0 ~ "^"site"[[:space:]]*\\{" && !done { print; print "    handle_path /mccp/* {"; print "        reverse_proxy 127.0.0.1:8095"; print "    }"; done=1; next }
      { print }' "$CADDYFILE" >"$CADDYFILE.new"
    if caddy validate --config "$CADDYFILE.new" --adapter caddyfile >/dev/null 2>&1; then
      mv "$CADDYFILE.new" "$CADDYFILE" && systemctl reload caddy && log "Caddy: https://$CADDY_SITE/mccp/ -> MCCP API"
    else
      rm -f "$CADDYFILE.new"; log "Caddy config did not validate with the route; left unchanged"
    fi
  else
    log "Caddy site $CADDY_SITE not found in $CADDYFILE; API is on 127.0.0.1:8095 only"
  fi
fi

# ---- tokens for the demo users ----------------------------------------------------
mint() { "${COMPOSE[@]}" exec -T -e MCCP_AUTH_MODE=dev api mccp dev-token "$1" --tenant whyaidata "${@:2}" --ttl 86400; }
umask 077
{
  echo "ADMIN_TOKEN=$(mint ben-admin --role Customizer.Admin)"
  echo "SUBMIT_TOKEN=$(mint ben --role Customizer.Submit)"
  echo "APPROVE_TOKEN=$(mint risk-owner --role Customizer.Approve)"
} >"$VPS/tokens.env"

log "done."
log "  API (local):  http://127.0.0.1:8095/docs"
log "  Tokens (24h): $VPS/tokens.env   (submitter and approver are different users by design)"
log "  Status:       docker compose -f $VPS/compose.vps.yaml --env-file $ENV_FILE ps"
