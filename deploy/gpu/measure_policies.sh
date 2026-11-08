#!/usr/bin/env bash
# Offline review-policy measurement on a saved run (no GPU, no inference, no API
# calls, no downloads; nothing in the run folder is written).
#
#   curl -fsSL https://raw.githubusercontent.com/bensunder/trainingModelmccp/main/deploy/gpu/measure_policies.sh | bash
#   ... | bash -s -- /workspace/mccp-<RUN_ID> invcdip
#
# 1. updates the code (mccp-gpu update)
# 2. failures, one step per fix (+ your audit.csv verdicts on the last, if filled in)
# 3. the label audit sheet and page images
# 5. prints a short summary (fits one screen); full outputs are kept in the audit folder
#
# On the 2026-11-07 run it first reproduces the original baseline exactly (v1,
# 215/54/79; 21/28, 18/40, 2/9, 34/63, 34/66), then measures each fix as its own
# step: redaction-label fix, one-decimal totals (v3), A+D blocking (the default).

RUN="${1:-/workspace/mccp-20261107-043803-qwen3-8b-v1}"
SET="${2:-invcdip}"
RUN="${RUN%/}"
OUT="$(dirname "$RUN")/mccp-audit/$(basename "$RUN")/$SET"

[ -d "$RUN" ] || { echo "no run folder $RUN"; exit 1; }
command -v mccp-gpu >/dev/null || { echo "mccp-gpu not installed: run the curl ... mccp-gpu | bash -s -- update line first"; exit 1; }

echo "updating code..."
mccp-gpu update > /tmp/mccp-measure-update.txt 2>&1 || { tail -5 /tmp/mccp-measure-update.txt; exit 1; }
tail -1 /tmp/mccp-measure-update.txt
mkdir -p "$OUT"

# verdicts typed into audit.csv (any non-empty verdict column) are applied as a second view
ADJ=()
if [ -s "$OUT/audit.csv" ] && python3 -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8')))
sys.exit(0 if any((r.get('verdict') or '').strip() for r in rows) else 1)
" "$OUT/audit.csv"; then
  ADJ=(--adjudications "$OUT/audit.csv")
fi

V1=realdoc-validation@v1
V3=realdoc-validation@v3
step() {  # name, then failures options; full output in $OUT/step_<name>.txt, JSON in step_<name>.json
  local name="$1"; shift
  echo "measuring: $name..."
  mccp-gpu failures "$RUN" --set "$SET" --policies all --json-out "$OUT/step_$name.json" "$@" > "$OUT/step_$name.txt" 2>&1 \
    || { tail -8 "$OUT/step_$name.txt"; exit 1; }
}
# each fix on its own, in order, so each one's effect is measured separately
step 0_baseline_v1      --validation "$V1" --policy none --keep-redacted-labels
step 1_redaction_fix    --validation "$V1" --policy none
step 2_one_decimal_v3   --validation "$V3" --policy none
step 3_default_a_plus_d "${ADJ[@]}"
echo "building the label audit..."
mccp-gpu audit-labels "$RUN" --set "$SET" > "$OUT/audit_labels.txt" 2>&1 || { tail -8 "$OUT/audit_labels.txt"; exit 1; }

python3 - "$OUT" "$(basename "$RUN")" "$SET" <<'PY'
import csv
import glob
import json
import os
import sys

out, run, set_name = sys.argv[1], sys.argv[2], sys.argv[3]
def load(name):
    return json.load(open(os.path.join(out, "step_%s.json" % name)))


steps = [("0_baseline_v1", "baseline (v1, as reported before)"), ("1_redaction_fix", "+ redaction-label fix"),
         ("2_one_decimal_v3", "+ one-decimal totals (v3)"), ("3_default_a_plus_d", "+ A+D blocking (new default)")]
base = load("0_baseline_v1")["measured"]
cur_doc = load("3_default_a_plus_d")
m, adj = cur_doc["measured"], cur_doc.get("adjusted")
n = m["rows"]
MISS, SEV, NEAR, OFF = "missing_required_field", "several_total_amounts", "invoice_number_not_near_label", "total_not_on_total_line"

print("")
print("== %s / %s: %d invoices" % (run, set_name, n))
if run == "mccp-20261107-043803-qwen3-8b-v1" and set_name == "invcdip":
    o0 = base["outcomes"]
    rows0 = {tuple(p["block_on"]): p for p in base["policies"]}
    want = {(): (0, 0), (SEV,): (21, 28), (NEAR,): (18, 40), (OFF,): (2, 9),
            (NEAR, SEV): (34, 63), (NEAR, SEV, OFF): (34, 66)}
    bad = []
    if (o0.get("auto_correct"), o0.get("human_review"), o0.get("auto_wrong")) != (215, 54, 79):
        bad.append("outcomes %s" % o0)
    for k, v in want.items():
        r = rows0.get(tuple(sorted(k)))
        got = (r["stops_wrong"], r["extra_reviews_of_correct"]) if r else None
        if got != v:
            bad.append("%s: %s != %s" % (" + ".join(k) or "baseline", got, v))
    print("   baseline check: " + ("PASS (v1 reproduces 215/54/79 and every flag row)" if not bad else "FAIL " + "; ".join(bad)))
print("   %-34s %12s %12s %12s   %s" % ("step", "auto_correct", "human", "auto_wrong", "wrong amt/id/both"))
for name, label in steps:
    r = load(name)["measured"]
    o = r["outcomes"]
    k = r["auto_wrong_by_kind"]
    print("   %-34s %5d %5.1f%% %5d %5.1f%% %5d %5.1f%%   %d/%d/%d" % (
        label, o.get("auto_correct", 0), 100.0 * o.get("auto_correct", 0) / n, o.get("human_review", 0),
        100.0 * o.get("human_review", 0) / n, o.get("auto_wrong", 0), 100.0 * o.get("auto_wrong", 0) / n,
        k.get("amount_only", 0), k.get("id_only", 0), k.get("amount_and_id", 0)))
rows = {tuple(p["block_on"]): p for p in m["policies"]}
print("   policy options on v3 (from validation alone):")


def line(label, r):
    k = r["auto_wrong_by_kind"]
    return "   %-26s %5d %6d %8.1f%% %7.1f%% %8.1f%%   %d/%d/%d" % (
        label, r["stops_wrong"], r["extra_reviews_of_correct"], 100 * r["auto_correct"], 100 * r["human_review"],
        100 * r["auto_wrong"], k.get("amount_only", 0), k.get("id_only", 0), k.get("amount_and_id", 0))


print("   %-26s %5s %6s %9s %8s %9s   %s" % ("block on", "stops", "extra", "auto_ok", "human", "auto_wrong", "wrong amt/id/both"))
for label, k in (("nothing", ()), ("A missing field", (MISS,)), ("D several totals", (SEV,)),
                 ("A + D", (MISS, SEV)), ("A + D + near-label", (MISS, SEV, NEAR))):
    r = rows.get(tuple(sorted(k)))
    if r:
        print(line(label, r))
print("   auto-wrong by field (new default): %s" % m["auto_wrong_by_field"])
cf = m.get("confusable_ids") or {}
if cf:
    print("   confusable-tolerant id match (l/1 O/0 S/5 B/8 G/6 T/7 Z/2; measurement only):")
    print("     would fix %d of %d id-only auto-wrong (%s); ids also fixed on %d of %d amount+id" % (
        cf["id_only_recoverable"], cf["id_only_auto_wrong"],
        ", ".join("%s %d/%d" % (f, c.get("recoverable", 0), c.get("wrong", 0)) for f, c in cf["by_field"].items()),
        cf["amount_and_id_ids_recoverable"], cf["amount_and_id_auto_wrong"]))
    print("     risk: labelled ids that collide with another after folding: %s" % cf["labelled_ids_colliding_after_folding"])
if m.get("redacted_labels_unscored"):
    print("   PII-placeholder labels no longer scored: %d" % m["redacted_labels_unscored"])

audit = [json.loads(x) for x in open(os.path.join(out, "audit.jsonl")) if x.strip()]
by = {}
for it in audit:
    key = "%s/%s" % (it["field"], it["outcome"])
    by[key] = by.get(key, 0) + 1
hints = {}
for it in audit:
    hints[it["hint"]] = hints.get(it["hint"], 0) + 1
typed = sum(1 for r in csv.DictReader(open(os.path.join(out, "audit.csv"), newline="", encoding="utf-8")) if (r.get("verdict") or "").strip())
print("   label audit: %d labels not in the text the model read; %s" % (len(audit), by))
print("   hints (guesses only): %s   verdicts filled in: %d" % (hints, typed))
if adj:
    a = adj["outcomes"]
    print("   with your verdicts (%d label errors excluded): auto_correct=%d human=%d auto_wrong=%d (%.1f%%)" % (
        adj["label_overrides_applied"], a.get("auto_correct", 0), a.get("human_review", 0), a.get("auto_wrong", 0),
        100.0 * a.get("auto_wrong", 0) / adj["rows"]))
print("   full outputs: %s" % out)
PY
