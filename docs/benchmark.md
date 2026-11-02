# Benchmark kit: generalization, reproducibility, frontier baseline

`deploy/gpu/run_batch.sh` turns the single pilot into evidence a buyer can check:

| Question | How the kit answers it |
|---|---|
| Does the adapter generalize, or did it memorize the generator? | Every adapter is scored on the held-out **in-distribution** split *and* an **out-of-distribution (OOD)** set: unseen vendors, items and ID ranges, eight new request phrasings, cases within a fraction of each policy threshold, and rule-precedence collisions. Labels come from the same policy function, so they are correct by construction. Prepare refuses to run if any OOD input shares a non-boilerplate 13-gram with training inputs. |
| Is the result reproducible? | One adapter per training seed (`SEEDS="42 7"`). The data and held-out split are fixed by `--data-seed`, so seeds differ only in data order and LoRA initialization. `comparison.md` reports mean, min and max across seeds. |
| Why not just prompt a frontier model? | The same teacher-forced items go to any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, LiteLLM) with identical tools and system prompt. |
| What does it cost and how fast is it? | Per target: p50/p95 latency, tokens, and USD per 1,000 triage decisions (token prices for hosted models; GPU $/h x eval wall time for self-hosted, an upper bound because eval concurrency is not tuned serving). |

## Run it (GPU box, Ubuntu + CUDA, >= 20 GB VRAM)

```bash
git clone https://github.com/bensunder/trainingModelmccp /root/mccp     # first time only
bash /root/mccp/deploy/gpu/bootstrap.sh                                # network, driver, disk, venvs
tmux new -s batch
bash /root/mccp/deploy/gpu/run_batch.sh                                # 3,000 records x 1 epoch x seeds 42 and 7
```

With a frontier baseline (the key is read from the environment, never from the command line):

```bash
read -rs MCCP_BASELINE_API_KEY && export MCCP_BASELINE_API_KEY
BASELINE_URL=https://api.openai.com BASELINE_MODEL=<model> BASELINE_PRICE_IN=<usd/1M in> BASELINE_PRICE_OUT=<usd/1M out> \
  bash /root/mccp/deploy/gpu/run_batch.sh
```

- Azure OpenAI: `BASELINE_URL=https://<resource>.openai.azure.com/openai BASELINE_AUTH=api-key BASELINE_MODEL=<deployment>`.
- LiteLLM: `BASELINE_URL=http://<host>:<port>`.
- The client retries 429/5xx with backoff, adapts once to providers that reject `max_tokens`, `temperature`, `seed` or `parallel_tool_calls`, and opens a circuit breaker after repeated failures. A target with more than 5% endpoint errors is marked `valid: false` and must not be quoted.

Knobs (environment): `N` (3000), `EPOCHS` (1), `SEEDS` ("42 7"), `OOD_N` (200), `SAVE_STEPS` (100), `GPU_USD_PER_HOUR` (0.35), `VLLM_MEM`, `MAX_LEN`, `MAX_SEQS`.

Timing on an RTX 4000 Ada (20 GB): about 2.3 h of training per seed at 3,000 records x 1 epoch, plus about 30-45 min of evaluation for the batch. The script is resumable: re-running skips finished prepares, trainings and evaluations, and an interrupted training resumes from its last checkpoint.

## Outputs

`/workspace/mccp-batch-n<N>-e<EPOCHS>/` (kept by GridShare after Stop):

- `comparison.md` / `comparison.json`: headline table per test set (adapter mean and range across seeds, base, frontier).
- `seed-<s>/report.json`: guard results, per-target L1, `by_scenario` and `by_ood_kind` accuracy, latency, cost, frontier parameter adaptations.
- `seed-<s>/adapter_model.safetensors` + `adapter_config.json`, training events and logs (`*.log.txt`).

## Reading the numbers

- **decision_accuracy** (decision and reason_code both right) is the business metric. `task_score` also includes the two tool-call steps.
- **field_accuracy_lenient** strips markdown fences before parsing. Compare it with strict JSON validity: a frontier model that wraps valid JSON in fences is not wrong about the invoice, but it would break a strict integration.
- **by_ood_kind** shows *where* errors are, e.g. `boundary_price_under` (1.1-1.85% variance must still be approved).
- Items are teacher-forced (gold tool results are supplied), so the numbers measure each decision step, not a free-running agent loop.
