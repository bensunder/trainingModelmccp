# Route: control layer for AI coding traffic

One gateway in front of every model provider your developers' coding tools use.
Admins approve models per team, set budgets, apply guardrails and see every dollar;
each request goes to the cheapest model whose **measured** quality clears the bar for
that kind of task.

```
Claude Code ──/v1/messages──────────┐
OpenCode ─────/v1/chat/completions──┼─▶ Route gateway ──▶ DeepSeek / Moonshot / Z.ai / Anthropic ...
Codex ────────/v1/responses─────────┘    auth · rate limit · budget · guardrails
                                         classify · choose (evidence) · failover · meter
```

## Request path

1. **Authenticate**: `x-api-key` or `Authorization: Bearer rk_...`; keys are stored as
   HMAC-SHA256 under a server pepper; revocation is immediate (cache tombstone).
2. **Rate limit**: token bucket per key (`rpm_per_key`, org setting; atomic Lua in Redis).
3. **Budget**: hard budgets (org / team / key, day / month) refuse with 429 and
   `Retry-After` until the period ends; soft budgets add an `x-route-warning` header.
4. **Guardrails**: secrets (cloud keys, tokens, private keys, credentials in URLs and
   assignments; also inside tool arguments), PII, prompt-injection phrases. Per-org
   action: `off | flag | redact | block`. Findings record the rule and a count, never
   the matched text. A blocked request never reaches a provider.
5. **Classify**: `background | quick_edit | explain | tests | refactor | debug | agentic`,
   deterministic rules with reasons (no model call on the hot path).
6. **Choose**: see *Routing* below.
7. **Call**: the client's body is forwarded as-is (only `model` is replaced) in the
   client's own wire format; the provider must serve that format. Failover to the next
   model happens only before the first byte (connection error, timeout, 408/409/425/429/
   5xx/529, open circuit breaker, empty or refused answer); a started stream is never
   retried.
8. **Meter**: tokens from the provider's own usage report (JSON or SSE; OpenAI streams
   are asked for a final usage chunk), cost from the catalog price, and **baseline cost**
   (same tokens on the catalog's baseline model) for savings. Written in batches off the
   request path. Responses with no usage report are counted, not silently billed as 0.

Response headers on every answer: `x-route-request-id`, `x-route-model`,
`x-route-profile`, `x-route-task` (and `x-route-warning`, `x-route-guardrail` when set).

## Routing

Clients pick a **profile** by model name (`auto-cost`, `auto-balanced`,
`auto-intelligence`), a concrete approved model (pinned, no routing), or anything else
(the key's / team's / org's default profile; Claude Code's own model names land here).

* **No evidence, no downgrade.** A cheaper model is chosen for a task class only when the
  evaluation shows its pass rate clears the profile's bar, judged by the **Wilson lower
  bound** (9/10 does not clear 0.85; 190/200 does). Otherwise the profile's fallback
  (strong model) serves it.
* **Background requests** (titles, summaries, Claude Code's haiku-alias traffic) go to
  the profile's cheap background model. This is the safe early saving.
* **Strategies**: `cost` = cheapest clearing the bar; `balanced` = cheapest within
  `balanced_margin` of the best; `intelligence` = best measured, cost as tie-break.
* **Sticky sessions**: an agent session (Claude Code `metadata.user_id`, `x-session-id`,
  OpenAI `user`, or the conversation's first turn) stays on its model; switching mid-
  session breaks tool context and discards the provider's prompt cache.

`GET /admin/v1/routing` (or `mccp-route admin routing`) shows, per profile and task
class, the model chosen today and why.

## Evidence: measuring models

```
mccp-route eval sample --out tasks.jsonl                  # small original starter set
mccp-route eval estimate --catalog route.yaml --tasks tasks.jsonl --models deepseek-flash,glm,kimi,claude-sonnet
mccp-route eval run -g https://inference.whyaidata.com --catalog route.yaml --tasks tasks.jsonl \
    --models deepseek-flash,glm,kimi,claude-sonnet --out eval-2026-11 --sandbox docker
cp eval-2026-11/evidence.json deploy/route/evidence/evidence.json   # gateway reads ROUTE_EVIDENCE
```

### Real task sets from public benchmarks (no model calls, no cost)

```
git clone https://github.com/openai/human-eval && git -C human-eval checkout b9453e8f1865df0a0057a1383767caa400acfbcf
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research \
    && git -C google-research sparse-checkout set mbpp
mccp-route eval import --source humaneval --path human-eval/data/HumanEval.jsonl.gz --out he.jsonl
mccp-route eval import --source mbpp --path google-research/mbpp/sanitized-mbpp.json --out mbpp.jsonl
cat he.jsonl mbpp.jsonl > tasks.jsonl
```

The importer refuses a file whose SHA-256 is not the pinned release (`importers.PINS`),
and runs only the benchmark's own code (local sandbox) to validate each task. Per
problem it makes a `quick_edit` task (implement from the spec), a `debug` task (the
canonical solution with one planted bug: a single comparison / arithmetic / boolean /
constant mutation, kept only if the benchmark's tests catch it) and a `tests` task
(write asserts that pass on the canonical solution and catch every planted bug; an
answer that redefines the function is rejected). Measured on the pinned files:

| source | licence | problems | quick_edit | debug | tests | dropped |
|---|---|---|---|---|---|---|
| HumanEval (164) | MIT | 164 | 164 | 147 | 147 | 0 |
| MBPP sanitized (427) | repo Apache-2.0; dataset card CC-BY-4.0 (card not re-checked here) | 427 | 426 | 293 | 293 | 1 (canonical fails its own tests) |

Every one of the 1,470 tasks was checked: the canonical answer passes, the planted bug
fails, the benchmark's own tests pass as a `tests` answer. Caveats: these benchmarks
are public and old, so models may have trained on them (absolute pass rates are
optimistic; the comparison on identical tasks is what routing uses), and the tasks
are single functions, not multi-file repository edits. `refactor` gets no tasks yet.

Tasks are scored by running code in a sandbox, never by judging prose: quick_edit /
debug / refactor answers must pass hidden tests; a `tests` answer must pass on the
reference implementation and fail on every planted bug. `explain` and `agentic` have no
mechanical score yet, so they stay on the fallback model. Requests go through the
gateway with each model pinned, so measured cost per task is what real traffic pays.
Unanswered requests (HTTP errors) are excluded from `n`: an outage is not quality
evidence. The evidence file records the task-file SHA-256, catalog version and sandbox.

The Docker sandbox: no network, read-only root, 64 MB noexec tmpfs, all capabilities
dropped, no-new-privileges, uid 65534, memory/CPU/pids limits, per-run container. The
local sandbox (network namespace + rlimits) is for development; `eval run` refuses it
unless `--allow-local-sandbox`.

## Connecting developer tools

```
mccp-route setup -g https://inference.whyaidata.com            # key asked hidden
mccp-route remove                                               # undo exactly that
mccp-route doctor -g https://inference.whyaidata.com            # reachability + key, no model call
```

| Tool | File | What is written |
|---|---|---|
| Claude Code | `~/.claude/settings.json` (0600) | `env`: ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL (haiku = `<profile>:background`) |
| OpenCode | `~/.config/opencode/opencode.json` | provider `route` (`@ai-sdk/openai-compatible`, baseURL `/v1`, apiKey `{env:ROUTE_API_KEY}`), default model |
| Codex | `~/.codex/config.toml` | `model`, `model_provider = "route"`, `[model_providers.route]` base_url, env_key, `wire_api = "responses"` |

OpenCode and Codex read the key from `ROUTE_API_KEY`, exported by
`~/.config/route/route.env` (0600), sourced from `~/.zshrc` / `~/.bashrc` by one marked
line. Every edit keeps the rest of the file, writes a timestamped backup first, and is
idempotent.

**Codex**: Codex speaks only the Responses API; `/v1/responses` translates to Chat
Completions (messages, function calls and outputs, freeform `custom` tools such as
apply_patch, streaming events). `previous_response_id` and hosted tools (web_search,
file_search, ...) are refused with a clear 400.

## Admin API

`/admin/v1` with `Authorization: Bearer <admin token>` (only its SHA-256 is stored):
orgs (create, patch `default_profile` / `rpm_per_key`), `PUT orgs/{id}/guardrails`,
teams (create, patch approved models / default profile), keys (create: plaintext shown
once; list; revoke), budgets (upsert, list, delete), `GET orgs/{id}/usage` (spend,
baseline, saved, saved %, tokens, cache hit rate, by model / day / team / task; optional
raw events), `GET routing`. Errors are RFC 9457 `application/problem+json`. Every org-
scoped call filters by org in SQL; cross-org access returns 404 (tested).

## Deployment (srv1507394)

```
cd /opt/mccp-git && git pull && ./deploy/route/install.sh
```

The installer creates `deploy/route/.env` (0600): Postgres password, key pepper, admin
token hash (the token is printed once), provider keys via hidden prompts. It starts
Postgres + Redis on an internal network (no published ports) and the gateway on
`127.0.0.1:8090`; add `deploy/route/Caddyfile.snippet` to the host Caddyfile for TLS at
`inference.whyaidata.com`. Until `route.yaml` has verified prices the gateway runs with
`ROUTE_ENV=development`; production mode refuses `UNVERIFIED` prices.

## Status (2026-11-12)

Verified by tests (in-process, fake providers; real Postgres for the store): the full
request path for all three APIs, routing rules, failover, sticky sessions, budgets,
rate limits, revocation, guardrails, metering from JSON and SSE, tenant isolation,
setup edits for all three tools, the Responses translation, sandboxed scoring and
evidence generation.

Not yet verified: the Docker image build and compose stack on the VPS, Caddy, real
provider endpoints and model names (`route.example.yaml` marks every price and the
Anthropic model id `UNVERIFIED`), live Claude Code / OpenCode / Codex sessions through
the gateway, the Docker sandbox on a host with a daemon. Not built yet: org-admin (non-operator) accounts, mechanical scoring for `explain` and
`agentic`, `refactor` tasks from real code, hosted open models on GridShare as a
route.
