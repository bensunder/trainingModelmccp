#!/usr/bin/env bash
# Offline review-policy measurement on a saved run (no GPU, no inference, no API
# calls, no downloads; nothing in the run folder is written).
#
#   curl -fsSL https://raw.githubusercontent.com/bensunder/trainingModelmccp/main/deploy/gpu/measure_policies.sh | bash
#   ... | bash -s -- /workspace/mccp-<RUN_ID> invcdip
#
# 1. updates the code (mccp-gpu update)
# 2. failures with every flag combination (+ your audit.csv verdicts, if any filled in)
# 3. the registered policies: missing, several-totals, missing+several-totals
# 4. the label audit sheet and page images
# 5. prints a short summary (fits one screen); full outputs are kept in the audit folder
#
# On the 2026-11-07 run it also checks the baseline against the numbers measured
# before this change (79 auto-wrong; 21/28, 18/40, 2/9, 34/63, 34/66).

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

echo "measuring every flag combination..."
mccp-gpu failures "$RUN" --set "$SET" --policies all --json "${ADJ[@]}" > "$OUT/failures_all.txt" 2>&1 \
  || { tail -8 "$OUT/failures_all.txt"; exit 1; }
for p in missing several-totals missing-several-totals; do
  echo "policy realdoc-review-$p@v1..."
  mccp-gpu failures "$RUN" --set "$SET" --policy "realdoc-review-$p@v1" > "$OUT/policy_$p.txt" 2>&1 \
    || { tail -8 "$OUT/policy_$p.txt"; exit 1; }
done
echo "building the label audit..."
mccp-gpu audit-labels "$RUN" --set "$SET" > "$OUT/audit_labels.txt" 2>&1 || { tail -8 "$OUT/audit_labels.txt"; exit 1; }

python3 - "$OUT" "$(basename "$RUN")" "$SET" <<'PY'
import csv
import glob
import json
import os
import sys

out, run, set_name = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(out, "failures_*_adapter.json")), key=os.path.getmtime)
data = json.load(open(files[-1]))
m, adj = data["measured"], data.get("adjusted")
rows = {tuple(p["block_on"]): p for p in m["policies"]}
n = m["rows"]
o = m["outcomes"]
MISS, SEV, NEAR, OFF = "missing_required_field", "several_total_amounts", "invoice_number_not_near_label", "total_not_on_total_line"


def pct(k):
    return "%s=%d (%.1f%%)" % (k, o.get(k, 0), 100.0 * o.get(k, 0) / n)


print("")
print("== %s / %s: %d invoices   %s" % (run, set_name, n, "  ".join(pct(k) for k in ("auto_correct", "human_review", "auto_wrong"))))

if run == "mccp-20261107-043803-qwen3-8b-v1" and set_name == "invcdip":
    want = {(): (0, 0), (SEV,): (21, 28), (NEAR,): (18, 40), (OFF,): (2, 9),
            (NEAR, SEV): (34, 63), (NEAR, SEV, OFF): (34, 66)}
    bad = []
    if o.get("auto_wrong") != 79:
        bad.append("auto_wrong %s != 79" % o.get("auto_wrong"))
    for k, (s, e) in want.items():
        r = rows.get(tuple(sorted(k)))
        got = (r["stops_wrong"], r["extra_reviews_of_correct"]) if r else None
        if got != (s, e):
            bad.append("%s: %s != %s" % (" + ".join(k) or "baseline", got, (s, e)))
    print("   baseline check: " + ("PASS (identical to the numbers before this change)" if not bad else "FAIL " + "; ".join(bad)))


def line(label, r):
    k = r["auto_wrong_by_kind"]
    return "   %-26s %5d %6d %8.1f%% %7.1f%% %8.1f%%   %d/%d/%d" % (
        label, r["stops_wrong"], r["extra_reviews_of_correct"], 100 * r["auto_correct"], 100 * r["human_review"],
        100 * r["auto_wrong"], k.get("amount_only", 0), k.get("id_only", 0), k.get("amount_and_id", 0))


print("   %-26s %5s %6s %9s %8s %9s   %s" % ("block on", "stops", "extra", "auto_ok", "human", "auto_wrong", "wrong amt/id/both"))
for label, k in (("nothing (current)", ()), ("A missing field", (MISS,)), ("D several totals", (SEV,)),
                 ("A + D", (MISS, SEV)), ("A + D + near-label", (MISS, SEV, NEAR))):
    r = rows.get(tuple(sorted(k)))
    if r:
        print(line(label, r))
print("   auto-wrong by field (current): %s" % m["auto_wrong_by_field"])

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
