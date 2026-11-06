# Rationale-Guided Training (RGT) in MCCP

RGT (O'Neill, Partridge, Jayasekara, *Upweight the strategy, not the tokens*, Baseten Research,
2025-10-29) changes **what** is supervised, not the training algorithm. For every decision sample the
training set keeps the standard pair and adds a strategy-explicit copy:

| Pair | Final assistant turn |
|---|---|
| A (standard) | `<think>` *(empty)* `</think>` answer, rendered exactly as the model is served |
| B (strategy) | `<think>` strategy `</think>` answer |

At inference the model is served with an empty reasoning block (thinking off), so latency is unchanged.
The paper reports reaching the same accuracy with substantially fewer examples than standard SFT on two
medical-scribing tasks (Qwen3-32B, LoRA rank 256, LLM grader). It is a vendor research post, not an
independent replication; MCCP measures the effect on its own task instead of assuming it.

## Phase 1: deterministic RGT on invoice triage (implemented)

* **Grader** (`mccp.rgt.invoices.InvoiceRuleGrader`): the policy is an ordered rule list, so grading
  needs no model. It reads the PO and vendor tool results in the conversation, the reference answer
  and (optionally) the model's own attempt, and explains why the reference is right and what the
  attempt got wrong (wrong precedence, wrong decision mapping, wrong variance formula, invalid JSON).
* **Distiller** (`InvoiceTemplateDistiller`): the strategy is the rule chain evaluated on that invoice,
  e.g. `Check in order: PO-4500040495 found ok; invoice not yet paid on this PO ok; bank details
  changed 1466 days ago (>= 30) ok; billed qty <= received qty 148 ok; price variance +20.81% exceeds
  +/-2% -> hold / PRICE_VARIANCE. variance_pct = ... = +20.81.` plus `Mistake to avoid: ...` when the
  model's attempt was wrong. About 60-90 tokens.
* **On-policy attempts** (optional, default on in batches): the base model answers each decision prompt
  first (greedy, served by vLLM, cached by prompt hash), so strategies name the mistakes it actually makes.
* **Only decision turns get pair B.** Tool-lookup turns follow a fixed procedure, so RGT adds one sample
  per record (about 4 samples per record vs 3 for SFT, ~33% more training compute per record).
* **Render check against the real tokenizer** (trainer preflight, before any GPU training): pair B's
  strategy must appear once, inside the final turn's reasoning block, before the answer; pair A must
  render byte-for-byte as the serving prompt (thinking off) followed by the answer. A template that drops
  reasoning, or renders training and serving differently, fails the job.
* **Audit trail:** `pilot.json` records the grader and distiller ids, attempt pass rate, rule counts and a
  hash of all strategies; `data/rgt_strategies.jsonl` holds every strategy that was trained on.

Grading and distilling cost nothing (no API calls). The only cost is GPU time.

## Running the learning curve

```bash
mccp-gpu queue add qwen3-8b@v1 --n 100,300,1000 --methods sft,rgt --seeds 42 --test-n 300
mccp-gpu results      # prints curve.md when done
```

Every run is scored on the same separately generated 300-record test set (disjoint ids, 13-gram
decontaminated against training) and the fixed OOD set. The base model is scored once. `curve.md`
reports, per method and size: training samples, minutes for data preparation and training, decision
accuracy (in-distribution and OOD), task score, and the fewest records at which each method reaches
95% decision accuracy, with the SFT/RGT ratio.

Directly, without the queue:
`SIZES="100 300 1000" METHODS="sft rgt" SEEDS=42 TEST_N=300 bash deploy/gpu/run_batch.sh`
(`RGT_ATTEMPTS=0` skips the base model's attempts.)

## Limits

* Synthetic, rule-generated data: the speed numbers are real, but the accuracy result says how fast a
  model learns this rule set, not how it performs on real invoices.
* One seed per point unless you add seeds; small sizes are noisy.
* Phase 2 (LLM grader, iterative refinement and LLM distiller for tasks without a rule-based reference)
  is not built. It involves paid model calls and is decided on Phase 1's results.
