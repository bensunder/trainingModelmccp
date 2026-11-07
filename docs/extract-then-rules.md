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

## ID canonicalization (`invoice-extract-ids@v1`)

First extract run (1k records): 100% in-distribution and OOD, 93.3% on messy invoices, with the base
model identical to the adapter. Every miss was a schema-invalid answer (JSON validity 93.3% = decision
accuracy), concentrated in house id formats (`id_format` 66%): the model copied `P.O. 4500...` /
`vendor no. ...` instead of writing `PO-...` / `V-...`. Those answers route to human review, so they
were deferrals, not wrong decisions.

Canonical form is a formatting rule, so code applies it (`mccp.rules.extraction`), identically in
scoring (suite `output_normalizer`) and serving (`triage_extract`): one run of digits with a
recognised prefix for that field becomes `PO-<digits>` / `V-<digits>` / `INV-<digits>`. Digits are
never changed; another field's prefix, split digits or letters among digits are left as written and
still go to a human; a canonical id that does not exist finds nothing and the policy holds the invoice.
`json_validity_raw` keeps reporting what the model wrote before canonicalization, and
`normalized_answers` how many answers were rewritten, so the step cannot hide model behaviour.

`evaluate --save-outputs` (on by default in batches and `eval-messy`) writes every scored answer to
`<report>.<set>.<target>.outputs.jsonl`, so misses can be read. Re-score a finished run after a scoring
change: `mccp-gpu eval-messy <RUN_ID> --rescore`.

## Documents, not text: `docs` and `invcdip` sets

Every set above hands the model clean text. Production reads PDFs and scans through OCR, so two
more sets measure that path (`mccp-gpu eval-messy --set docs|invcdip`, scoring a finished run):

* **docs** (`mccp.examples.documents`): the same cases rendered as one-page PDF invoices in four
  layouts (classic, modern, compact, remittance with a freight line), rasterized, scan-degraded at
  three levels (clean, light: slight skew/blur/JPEG, heavy: 2 degree skew, blur, speckle,
  downscaling) and OCR'd with RapidOCR (CPU, pip-only, models bundled). The pages carry what real
  invoices carry and the text sets never did: a remit-to `PO Box` near the PO number, bank account
  and routing digits, subtotal/tax/total beside the unit price, a freight line, terms, dates and
  phone numbers. Labels are exact (same `make_case`), so the full decision is scored end to end.
  OCR artifacts are kept on purpose (merged words, `$35.805.60` for `$35,805.60`); the engine's
  angle classifier is off because it flipped short skewed boxes 180 degrees (`90` read as `06`), an
  engine artifact rather than scan damage.
* **invcdip** (`mccp.examples.invcdip`): the 350 labelled **real** scanned invoices of INV-CDIP
  (Gao et al., 2022; documents from the UCSF Industry Documents Library, downloaded from its public
  AWS Open Data mirror). There are no PO or vendor records for them, so this set scores extraction
  only (`examples/suite-invoice-realdoc.yaml`): invoice number, PO number (labelled on 85) and
  amount payable, matched ignoring case/spaces/punctuation, amounts to the cent. Unlabelled fields
  are not scored; a label that PII redaction removed from the input is dropped and counted; a value
  the OCR lost stays scored (OCR is part of the pipeline). `set_invcdip.json` records the **OCR
  ceiling** per field (share of labels present in the OCR text), so a miss can be attributed to the
  OCR or to the model. **License: annotations CC BY-NC 4.0, documents under UCSF terms: internal
  evidence only**; nothing is redistributed.

Both are built on the CPU before vLLM starts and cached in `/workspace/mccp-cache`, so the slow part
(about 2-3 s of OCR per page) is paid once across runs and machines.

## Amounts as printed (`invoice-extract@v2`, `realdoc-amounts@v1`)

On the real scans the model dropped decimal points in about half of the wrong totals: the OCR text
read `GRAND TOTAL $1,533.54`, `316.77`, `$22.25` and the model wrote `153354`, `31677`, `2225`. The
prompt asked for "a number without currency symbols or thousands separators", and the model stripped
the decimal point with the separators. So number formatting moved to code, like id formatting:

* the model copies the amount exactly as printed: `unit_price_raw` (extract task),
  `total_amount_raw` (real scans);
* `mccp.rules.extraction.parse_money` turns it into the canonical `unit_price_billed` /
  `total_amount`: currency symbol or code before or after, a per-unit suffix, thousands commas in the
  right places, up to two decimals, stray OCR marks around it (`8,333.00~`);
* it refuses anything a person would have to interpret (`$35.805.60`, `316:77`, `2:851475`,
  `3l6.77`): no number is produced, the answer fails the schema and serving routes the invoice to
  human review; scoring counts these as `unparseable_amounts`;
* the gold label goes through the same parser, so answer and label are compared as numbers
  (`USD 1,234.50` and `$1,234.50` match); an annotation it cannot read is not scored;
* `json_validity_raw` is measured against the model's own schema (`raw_output_schema`), so it keeps
  reporting what the model wrote.

Suites moved to version 2. OCR text is cached by rendering version (`RENDER_VERSION`); records are
always rebuilt by the current code, so a prompt or label change never needs a new OCR pass.

## Financial-field validation (`invoice-extract-validation@v1`, `realdoc-validation@v1`)

Between extraction and the rules (`mccp.rules.validation`), identically in serving and scoring. It
never corrects a value; it confirms it against the text the model read or blocks the invoice:

* **grounding (blocks):** every extracted value must be printed on the page: an amount in one of its
  printed forms (`1,533.54`, `1533.54`, and `1,534`/`1534` for whole amounts) as a whole number, an id
  by its digits (free-form real-document ids by letters and digits). `153354` for a page that says
  `$1,533.54` is not on the page, so it is blocked, never "repaired".
* **line arithmetic (blocks when contradicted):** on the line that prints the unit price, if other
  amounts with cents are printed, one must equal qty x price; taking the line amount as the unit
  price contradicts it. No printed line amount is not a contradiction; corroboration is reported.
* **total ambiguity (flags):** amounts on total-labelled lines (not SUBTOTAL); several of them, or an
  answer that is on the page but not on such a line, is reported so the policy can be set from rates.

Checked on every gold answer of the test, OOD and messy sets: none is blocked.

**Outcomes** (what a customer buys, in every report and in `mccp-gpu summary`): `auto_correct`
(decided without a person, and right), `human_review` (invalid output or blocked by validation) and
`auto_wrong` (passed everything and was still wrong: the number to drive to zero).

**Failure classification:** `mccp-gpu failures [FOLDER] [--set S] [--examples N]` gives every wrong
field one cause, comparing the label, the answer and the exact text the model read: decimal dropped,
unparseable copy, missing, wrong amount on the page (with the kind of line it came from: net, gross,
tax, subtotal, ...), invented or misread, or label not on the page (OCR lost it or the label
disagrees with the page). Each cause shows how many validation caught and how many would have been
decided wrongly. Rows saved before validation existed are re-validated, so old runs are judged by the
current pipeline.

## Review policies, the auto-wrong split and the label audit

**Findings vs policy.** The validator reports findings; a review policy
(`mccp.rules.review_policy`) decides which flags also route an invoice to a person.
Flags a policy does not block on are passed to the reviewer as hints. Blocking
findings (a value not on the page) always route to a person, whatever the policy.

* `realdoc-validation@v2` = v1 plus `missing_required_field` (invoice number or
  total null; which ones in `missing`). v1 is kept unchanged, and v2 with no
  blocking policy decides exactly what v1 decided (pinned by a test).
* `realdoc-validation@v3` = v2, and a total printed with one decimal place (`48.1`
  for 48.10) counts as on the page; v2 sent those to a person although label,
  answer and page agreed. v1/v2 and the extract validator keep two-decimal forms.
* Policies: `realdoc-review-flags-none@v1`,
  `realdoc-review-missing@v1`, `realdoc-review-several-totals@v1`,
  `realdoc-review-missing-several-totals@v1`. A policy may only name flags its
  validator raises.
* `mccp-gpu failures RUN --set invcdip --policy ID` reports outcomes under a
  policy; `--block-on f,g` measures an unregistered one. The simulation always
  starts from validation alone, so its rows compare across policies; `--policies all`
  prints every combination.

Measured on the 348 INV-CDIP scans (Qwen3-8B adapter, flags-none): auto_correct
61.8%, human 15.5%, auto_wrong 22.7% (79). Blocking on `several_total_amounts`
stops 21 wrong for 28 extra reviews; `invoice_number_not_near_label` 18 for 40;
`total_not_on_total_line` 2 for 9; the first two together 34 for 63 (union, not
sum). No flag combination is precise enough to block on by itself.

The realdoc suite (v4) decides with `realdoc-validation@v3` and
`realdoc-review-missing-several-totals@v1`. `failures --validation realdoc-validation@v1
--policy none --keep-redacted-labels` reproduces the measurement above exactly;
`deploy/gpu/measure_policies.sh` does that and then measures each fix as its own step.

**Redaction placeholders are never labels.** PII redaction runs over the label as
well as the text, so a PO number that looks like a card number became the label
`<CREDIT_CARD_1>` and was scored. The set builder now drops any label redaction
changed (`drop_redacted_labels`), and re-scoring treats a placeholder label in an
older set as unlabelled (`redacted_labels_unscored`).

**The split (C).** Every auto-wrong answer is `amount_only`, `id_only` or
`amount_and_id` (plus per-field counts), for the reported outcome and for every
simulation row (`wrong amt/id/both`). A wrong amount costs money directly; a wrong
id mostly breaks duplicate detection.

**Label audit (B).** `mccp-gpu audit-labels RUN --set invcdip` lists every scored
field whose label is not in the text the model read and writes, to
`/workspace/mccp-audit/<run>/<set>/` (never into the run folder; only cached PDFs and
annotations are read): `audit.csv`, `audit.jsonl`, `index.html` and page images with
the annotator's box. The `hint` column (`ocr_near_miss`,
`answer_on_page_label_absent`, `absent`) is a guess and changes nothing. A person
fills in `verdict` (`label_wrong`, `ocr_lost`, `model_wrong`) and, for
`label_wrong`, optionally `corrected_label` as printed (parsed by the same code;
an unreadable amount is an error). Re-running keeps typed verdicts.
`failures --adjudications audit.csv` then reports both: as measured, and excluding
verified label errors (re-scored against the corrected label, or unscored).
`ocr_lost` stays scored: OCR is part of the pipeline being measured.

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
