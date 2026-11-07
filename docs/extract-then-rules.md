# Extract-then-rules: the model reads, code decides

`--task extract` (default `agent`) changes what the model is trained and scored on:

| | `agent` (original) | `extract` |
|---|---|---|
| Model calls per invoice | 3 (lookup PO, lookup vendor, decide) | **1** |
| Model output | the decision JSON | the invoice facts: `invoice_id, po_number, vendor_id, billed_qty, unit_price_billed` |
| Who looks up PO / vendor | the model, via tool calls | code, by the extracted ids |
| Who applies thresholds and precedence | the model's weights | `mccp.rules.invoice_triage` (`invoice-triage-rules@v1`) |
| Training samples per record | 3 | 1 |
| Explanation of a decision | none | rule trace (each rule, the facts it saw, whether it fired) |

## Why: the evidence (Qwen3-8B QLoRA, 1,000 records, 2026-11-06)

Two `agent`-task runs on the same fixed test, OOD and messy sets (`standard` vs `boundary`
training mix). Near the policy thresholds the model guesses, and denser boundary training only
moved the guess:

| OOD case (n=20 each) | standard | boundary |
|---|---|---|
| price variance just over 2% (should hold) | 90% | 35% |
| price variance just under 2% (should approve) | 25% | 75% |
| *both together* | *57.5%* | *55%* |
| bank change 25-29 days ago (should escalate) | 20% | 10% |
| bank change 30-34 days ago (should not) | 85% | 75% |
| bank change + quantity problem (bank wins) | 0% | 0% |

In-distribution bank-change accuracy fell from 100% to 27.8% with the boundary mix, which accounts
for the whole in-distribution drop (98.0% to 89.7%). 59 training rows with exactly the bank +
quantity collision still left that case at 0/20. Learning a numeric cutoff and a rule order from
examples gives a fuzzy boundary; more or denser data shifts it and does not remove it. The policy
is six deterministic rules, so it moved out of the model.

## The policy as code

`src/mccp/rules/invoice_triage.py` is the single source of truth. The data generator labels with
it (agent-task datasets are byte-identical to before the move, tested), evaluation scores with it,
and the serving path decides with it. First match wins:

1. PO not found -> hold `MISSING_PO`
2. vendor not found -> hold `VENDOR_NOT_FOUND` (production only: an extracted id that does not resolve)
3. already paid -> reject `DUPLICATE_INVOICE`
4. bank details changed < 30 days ago -> escalate `VENDOR_BANK_CHANGE`
5. billed qty > received qty -> hold `QTY_MISMATCH`
6. |price variance| > 2% -> hold `PRICE_VARIANCE`
7. otherwise -> approve `WITHIN_TOLERANCE`

## How it is scored (end to end, not teacher-forced)

Each extract-task test record carries the system-of-record data in `meta.lookups` (keyed by id)
and the policy answer in `meta.expected`. Scoring runs the real pipeline on the model's output:
lookups by the **extracted** ids, then the policy. A wrong PO number finds nothing and becomes
`MISSING_PO`, a wrong price can flip a variance decision, invalid JSON is wrong. So
`decision_accuracy` means the same thing as for the agent task and the two compare directly in
`curve.md`. `field_accuracy_by_field` shows which fact the model misread. The pass/fail gate has
no tool-call checks for this task (there are no tool calls), instead of failing them.

Both tasks are generated from identical cases: same seeds give the same invoices, texts and labels
(tested), only the supervision differs.

## Serving

`mccp.agent.extract.triage_extract(chat, text, model=..., lookup_po=..., lookup_vendor=...)`: one
model call, strict schema validation, lookups (sync or async), policy, trace. Any failure to decide
on verified facts (endpoint error, invalid output, a lookup that raises) returns
`route: human_review` with the reason and no decision; nothing is approved on a failure. A lookup
that answers "not found" is a policy outcome (hold), not a failure.

## Run it

```bash
mccp-gpu queue add qwen3-8b@v1 --n 1000 --seeds 42 --test-n 300 --task extract
```

Then compare with the agent-task run on the same test sets:
`uv run python deploy/gpu/pilot.py curve <agent report.json> <extract report.json> --out <dir>`.

## Limits

* The facts are read from synthetic text. Real invoices (PDF/OCR, multi-line, other currencies)
  need their own extraction set before any accuracy claim.
* The policy is only as good as the lookups: the vendor-id / PO-vendor consistency check, credit
  memos and multi-line invoices are not in v1 of the rules.
