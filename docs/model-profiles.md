# Model profiles: switchable base models

A **model profile** (`profiles/<id>@v<N>.yaml`) holds everything that is specific to a base model family. Recipes name a profile; nothing else in MCCP hard-codes a model.

| Profile section | What it controls | Used by |
|---|---|---|
| `base` | Hugging Face id, pinned revision, license, **publisher country of origin**, fleet architecture, chat-template hash | recipe consistency checks, evidence bundle, governance gate |
| `tools` | How tool definitions are folded into the conversation (`system_append` or `role_message`), as the model's own template does | the trainer (the only place that renders) |
| `reasoning` | The chat-template kwarg that switches reasoning (e.g. Qwen `enable_thinking`, Granite `thinking`) and its tags | trainer, eval requests, vLLM default, canary smoke test |
| `serving` | vLLM tool-call and reasoning parsers, LoRA modules vLLM applies, VRAM to serve | vLLM launch, publish guard |
| `training` | LoRA target modules, QLoRA support, VRAM to train | recipe checks |
| `validation` | Evidence that the rendering is token-identical to the native template | profile status |

## Status: draft vs validated

A profile is **validated** only when its validation evidence matches its pinned revision, chat-template hash and rendering. Any edit to `tools` or `reasoning` changes the rendering hash and drops it back to **draft**. Jobs refuse draft profiles unless `MCCP_ALLOW_DRAFT_PROFILES=true` (rehearsal only); never edit a validated profile in place, publish `@v<N+1>` instead.

Shipped profiles:

| Profile | Origin | License | Status |
|---|---|---|---|
| `qwen3-8b@v1` | CN | Apache-2.0 | validated (GPU run 2026-11-01; byte-identical to the rendering that run proved) |
| `granite-3.3-8b-instruct@v1` | US | Apache-2.0 | draft until validated (expected Granite tool format; not yet proven on its tokenizer) |

## Validate a profile (CPU only, no weights, no GPU spend)

On any box with the trainer environment (e.g. the GPU box after `bootstrap.sh`):

```bash
cd /root/mccp && uv run mccp profile validate granite-3.3-8b-instruct@v1 --trainer-python /root/venvs/trainer/bin/python
```

It downloads only tokenizer and config files, then checks: the revision resolves; `config.json` matches the profile dims; for 24 fixtures (training and OOD phrasings) the profile rendering is **token-identical** to `apply_chat_template(tools=...)`; string tool arguments render like objects; the reasoning kwarg is accepted. On success it pins the revision and template hash into the YAML and writes `profiles/validations/<ref>.json`; on failure it changes nothing and prints both renderings around the first differing token so the profile can be fixed.

## Train and benchmark another base model

```bash
PROFILE=granite-3.3-8b-instruct@v1 bash /root/mccp/deploy/gpu/run_batch.sh
```

The batch takes the vLLM served name, parsers and thinking switch from the profile, caches weights per profile and tags results `/workspace/mccp-batch-<profile>-n<N>-e<E>`.

## Datasets are model-neutral

Dataset versions keep a `tools` column (format `chatml+tools/string-args/v2`); the trainer renders them with the job's profile, so one dataset serves every base model. Dataset versions built before profiles were pre-rendered for Qwen3; they still train Qwen jobs and are refused, with a "rebuild the dataset" error, for any other profile.

## Governance: origin allow-lists

```bash
uv run mccp tenant set-origins <tenant-uuid> US FR     # restrict
uv run mccp tenant set-origins <tenant-uuid>           # remove the restriction
```

The evidence bundle records the base model's profile, publisher and origin, and the tenant's allow-list. The promotion policy denies an adapter whose base model origin is not on the list (an unknown origin, from jobs created before profiles, is denied too). Tenants without a list are unaffected.
