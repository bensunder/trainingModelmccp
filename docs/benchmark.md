# Benchmark kit: generalization, reproducibility, frontier baseline

`deploy/gpu/run_batch.sh` turns the single pilot into evidence a buyer can check:

| Question | How the kit answers it |
|---|---|
| Does the adapter generalize, or did it memorize the generator? | Every adapter is scored on the held-out **in-distribution** split *and* an **out-of-distribution (OOD)** set: unseen vendors, items and ID ranges, eight new request phrasings, cases within a fraction of each policy threshold, and rule-precedence collisions. Labels come from the same policy function, so they are correct by construction. Prepare refuses to run if any OOD input shares a non-boilerplate 13-gram with training inputs. |
| Is the result reproducible? | One adapter per training seed (`SEEDS="42 7"`). The data and held-out split are fixed by `--data-seed`, so seeds differ only in data order and LoRA initialization. `comparison.md` reports mean, min and max across seeds. |
| Why not just prompt a frontier model? | The same teacher-forced items go to any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, LiteLLM) with identical tools and system prompt. |
| What does it cost and how fast is it? | Per target: p50/p95 latency, tokens, and USD per 1,000 triage decisions (token prices for hosted models; GPU $/h x eval wall time for self-hosted, an upper bound because eval concurrency is not tuned serving). |

## Run it: one command (GPU box, Ubuntu + CUDA, >= 20 GB VRAM, as root)

```bash
curl -fsSL https://raw.githubusercontent.com/bensunder/trainingModelmccp/main/deploy/gpu/mccp-gpu | bash -s -- up --queue qwen3-8b@v1
```

That one line: installs the `mccp-gpu` command, checks network/driver/disk, installs the environments, asks once for
the frontier provider (OpenAI or Moonshot Kimi) and its key (hidden; checked against `/v1/models` plus one tiny test
completion, a fraction of a cent, so an account without credit is caught at setup; Enter alone = no frontier baseline;
default model `gpt-4.1` or `kimi-k2.6`), and starts a background runner in tmux that works through the queue. You can close the terminal.

| Command | What it does |
|---|---|
| `mccp-gpu status` | Runner state, current run, training step and ETA, GPU use, rupees spent so far, queue, finished runs |
| `mccp-gpu queue add granite-3.3-8b-instruct@v1` | Queue another batch (options `--n 3000 --epochs 1 --seeds "42 7"`); runs after the current one |
| `mccp-gpu results` | Prints each finished run's `comparison.md` |
| `mccp-gpu logs` | Follows the runner log (Ctrl+C stops following, not the run) |
| `mccp-gpu up` | Resumes the queue after a failure or `stop` (training continues from its last checkpoint) |
| `mccp-gpu stop` | Stops the runner, training and vLLM so the box can be stopped; the queue is kept |
| `mccp-gpu baseline` | Re-enter the key or change the baseline model |
| `mccp-gpu update` | Refreshes the code used by future runs |

Guarantees: one runner per box (lock), runs execute one after another, each run uses a frozen copy of the commit it
started from (updating `main` never changes a run in progress), a failed run pauses the queue so no more GPU time is
spent, GPU work started by hand is waited for, and results land in `/workspace/mccp-<run id>/` (kept by GridShare after
Stop). The key is stored in `/root/.mccp/secrets.env` (mode 600) and never appears on a command line or in a log.
State lives in `/root/.mccp` (queue, finished runs, runner log).

## Run it by hand (advanced)

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
- Moonshot Kimi: `BASELINE_URL=https://api.moonshot.ai` (mainland keys: `https://api.moonshot.cn`) `BASELINE_MODEL=kimi-k2.6`
  `BASELINE_DROP=temperature` `BASELINE_EXTRA_BODY='{"thinking": {"type": "disabled"}}'`. Kimi fixes temperature itself
  (other values are rejected), and thinking is switched off so it answers within the same token budget as the other
  targets. `mccp-gpu baseline` sets all of this when you pick Kimi.
- Account-level failures (bad key, no credit, suspended account) are not retried: one preflight call before the
  benchmark detects them, the frontier row is skipped with the provider's message in `report.json`
  (`frontier_skipped`), and the adapter/base evaluation continues.
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

## Serving throughput and cost per 1M output tokens (one rented GPU)

Before buying GPUs or committing to 24/7 rental, measure what one rented GPU can serve:

```bash
mccp-gpu bench --rate <your GridShare INR per hour>          # Qwen3-8B, concurrency 1,5,10, 120 s each
mccp-gpu bench --adapter /workspace/<run>/  --rate 30.5        # measure a trained LoRA adapter instead of the base
mccp-gpu bench --eager 0                                       # CUDA graphs instead of --enforce-eager (faster, untested here)
mccp-gpu results                                               # prints throughput.md
```

For each concurrency level C the benchmark keeps exactly C requests in flight for the set time (closed loop), streams
every response and records: output tokens/s across all requests, per-request decode speed, time to first token and
latency (p50/p95), GPU utilization/memory/power (nvidia-smi, 0.5 s samples), vLLM KV-cache use, queued requests and
preemptions (vLLM `/metrics`), hours and cost to generate 100M output tokens, and

    cost per 1M output tokens = GPU cost per hour / (output tokens per second x 3600) x 1,000,000

Prompts are real invoice-triage requests (system prompt, tool schemas, invoice text) from the same generator as the
training data. Output length is fixed per request (vLLM `ignore_eos`) so every level does the same work. Results land in
`/workspace/mccp-bench-<id>/` (`throughput.md`, `throughput.json`, `bench.log.txt`, `vllm.log.txt`). The command refuses
to start while training runs, and the training queue waits for a running benchmark. No API calls; only GPU time.

Reading it: GPU memory is mostly vLLM's pre-allocation (`gpu-memory-utilization`), so the KV-cache percentage is the real
headroom signal; queued requests or preemptions mean the level is past what the GPU (or the 16-sequence limit) sustains.

## Messy-invoice test set

Every batch now also scores on 300 **messy invoices** (`MESSY_N`, 0 = off): the same cases and labels as the
training distribution, but the text a real AP queue produces. Six kinds, round-robin, about 30% combining two:

| Kind | What changes |
|---|---|
| `ocr_noise` | letter swaps, dropped letters, OCR confusions (rn->m, l->I), random caps and spaces; never touches a digit |
| `email_forward` | the request arrives as a forwarded email with headers and quoted lines |
| `table_layout` | invoice header and a line-item table instead of a sentence |
| `distractor_note` | a human note arguing for the wrong outcome ("vendor confirmed the price, please approve") |
| `id_format` | PO / vendor ids in house formats (`po#4500012345`, `V 100023`); tool calls must use canonical ids |
| `currency_format` | `USD 1,234.50`, `1,234.50 USD`, `US$1234.50`, `$1,234.50/ea` |

Labels come from the policy (`decide`) on the PO and vendor records, so they are correct by construction; a test
re-derives every label from the messy text plus the tool results. Results break down per kind (`by_ood_kind`).

Score an already finished run without retraining (adapters must still be on the box):

```bash
mccp-gpu eval-messy            # latest successful run; writes report_messy.json per adapter and refreshes curve.md
```
