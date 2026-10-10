package promotion

# Promotion gate (spec D8). Pure function of the signed evidence bundle:
#   decision := data.promotion.decision with input = evidence + target + approvals
# Thresholds live in data.thresholds[target][risk_tier] (policy/data.json) so
# risk owners can tune them without touching rules.

import rego.v1

default allow := false

allow if count(deny) == 0

t := data.thresholds[input.target][input.risk_tier]

ev := object.get(input, "eval", {})

guard := object.get(ev, "publish_guard", {})

l1 := object.get(object.get(ev, "l1", {}), "adapter", {})

decision := {
	"allow": allow,
	"deny": sort([m | some m in deny]),
	"target": input.target,
	"risk_tier": input.risk_tier,
	"approvals_counted": count(valid_approvers),
	"awaiting_approvals": awaiting_approvals,
}

# True when the only thing missing is independent approvals, so the job
# should stay in pending_approval instead of being rejected.
default awaiting_approvals := false

awaiting_approvals if {
	count(valid_approvers) < t.min_approvals
	count(deny) == 1
}

# --- Request shape ---------------------------------------------------------
deny contains msg if {
	not t
	msg := sprintf("no thresholds configured for target=%v risk_tier=%v", [input.target, input.risk_tier])
}

deny contains "intended_use is required" if {
	count(trim_space(object.get(input, "intended_use", ""))) == 0
}

# --- Integrity and supply chain ---------------------------------------------
deny contains "adapter signature not verified" if {
	not input.signatures.adapter_verified == true
}

deny contains "signed per-file digests do not match the artifact manifest" if {
	input.signatures.attested_files != input.signatures.manifest_files
}

deny contains msg if {
	count(object.get(guard, "skipped_modules", [])) > 0
	msg := sprintf("vLLM %v would skip modules %v", [guard.vllm_version, guard.skipped_modules])
}

deny contains msg if {
	some p in object.get(guard, "problems", [])
	msg := sprintf("publish guard: %v", [p])
}

deny contains "publish guard did not load the adapter on the serving fleet" if {
	not guard.loaded == true
}

deny contains msg if {
	input.adapter.peft.r > data.fleet.max_lora_rank
	msg := sprintf("LoRA rank %v exceeds fleet max %v", [input.adapter.peft.r, data.fleet.max_lora_rank])
}

deny contains "adapter has no observable effect vs. base (silently skipped modules or no-op adapter)" if {
	t.require_observable_effect
	guard.no_observable_effect == true
}

# --- Licensing and data governance ---------------------------------------------
deny contains msg if {
	not input.base_model.license in data.allowed_licenses
	msg := sprintf("base model license %v is not allow-listed", [input.base_model.license])
}

deny contains msg if {
	some d in input.datasets
	some lic in d.licenses
	not lic in data.allowed_data_licenses
	msg := sprintf("dataset %v contains data under license %v", [d.ref, lic])
}

deny contains msg if {
	some d in input.datasets
	not d.pii_scan.passed == true
	msg := sprintf("dataset %v failed the PII scan", [d.ref])
}

# --- L1 task quality ---------------------------------------------------------
deny contains "L1 task evaluation did not complete" if {
	not ev.l1.status == "ok"
}

deny contains msg if {
	l1.json_validity < t.json_validity_min
	msg := sprintf("JSON validity %v < %v", [l1.json_validity, t.json_validity_min])
}

deny contains msg if {
	l1.tool_call_parse_rate < t.tool_call_parse_min
	msg := sprintf("tool_calls populated in %v of tool steps < %v", [l1.tool_call_parse_rate, t.tool_call_parse_min])
}

deny contains msg if {
	l1.tool_calls_leaked_into_content > 0
	t.forbid_tool_call_leaks
	msg := sprintf("%v tool calls leaked into message content (parser/template mismatch)", [l1.tool_calls_leaked_into_content])
}

deny contains msg if {
	d := ev.l1.delta_vs_base
	d.ci_low < t.min_delta_vs_base
	msg := sprintf("task score vs base: CI lower bound %v < %v", [d.ci_low, t.min_delta_vs_base])
}

deny contains msg if {
	d := ev.l1.delta_vs_champion
	d != null
	d.ci_low < t.min_delta_vs_champion
	msg := sprintf("task score regresses vs current champion (CI lower bound %v < %v)", [d.ci_low, t.min_delta_vs_champion])
}

# --- L2 judge -------------------------------------------------------------------
deny contains "L2 judge evaluation required but not completed" if {
	t.require_l2
	not ev.l2.status == "ok"
}

deny contains msg if {
	ev.l2.status == "ok"
	ev.l2.win_rate_vs_base.ci_low <= t.l2_win_rate_min
	msg := sprintf("judge win rate vs base CI lower bound %v <= %v", [ev.l2.win_rate_vs_base.ci_low, t.l2_win_rate_min])
}

deny contains msg if {
	ev.l2.status == "ok"
	ev.l2.judge_human_kappa != null
	ev.l2.judge_human_kappa < t.judge_kappa_min
	msg := sprintf("judge-human agreement kappa %v < %v; judge cannot gate", [ev.l2.judge_human_kappa, t.judge_kappa_min])
}

# --- L3 general regression ---------------------------------------------------------
deny contains "L3 regression evaluation required but not completed" if {
	t.require_l3
	not ev.l3.status == "ok"
}

deny contains msg if {
	ev.l3.status == "ok"
	ev.l3.max_drop > t.l3_max_drop
	msg := sprintf("general-capability drop %v > %v", [ev.l3.max_drop, t.l3_max_drop])
}

# --- L4 safety (always required) -----------------------------------------------------
deny contains "L4 safety evaluation did not complete" if {
	not ev.l4.status == "ok"
}

deny contains msg if {
	ev.l4.unsafe_rate_delta > t.unsafe_delta_max
	msg := sprintf("unsafe-response rate rose by %v vs base (max %v)", [ev.l4.unsafe_rate_delta, t.unsafe_delta_max])
}

deny contains msg if {
	ev.l4.pii_leak_hits > t.pii_leak_max
	msg := sprintf("%v PII values leaked under probing (max %v)", [ev.l4.pii_leak_hits, t.pii_leak_max])
}

deny contains msg if {
	ev.l4.over_refusal_delta > t.over_refusal_delta_max
	msg := sprintf("over-refusal rose by %v vs base (max %v)", [ev.l4.over_refusal_delta, t.over_refusal_delta_max])
}

# --- L6 performance -------------------------------------------------------------------------
deny contains msg if {
	ev.l6.status == "ok"
	ev.l6.ttft_p95_regression > t.ttft_p95_regression_max
	# Ignore relative noise on tiny absolute latencies.
	ev.l6.ttft_p95_delta_s > data.ttft_min_abs_regression_s
	msg := sprintf("p95 TTFT regression %v > %v", [ev.l6.ttft_p95_regression, t.ttft_p95_regression_max])
}

# --- Regulatory compute record (EU AI Act GPAI modifier test) ------------------------------
deny contains "compute estimate missing" if {
	not input.compute.flop_estimate
}

deny contains msg if {
	input.compute.flop_estimate >= data.gpai_review_flop
	msg := sprintf("training compute %v FLOP >= %v requires legal review (GPAI modifier threshold)", [input.compute.flop_estimate, data.gpai_review_flop])
}

# --- Separation of duties -------------------------------------------------------------------
valid_approvers contains a.approver if {
	some a in object.get(input, "approvals", [])
	a.decision == "approve"
	a.approver != input.submitter
}

deny contains "the submitter cannot approve their own adapter" if {
	some a in object.get(input, "approvals", [])
	a.approver == input.submitter
}

deny contains msg if {
	some a in object.get(input, "approvals", [])
	a.decision == "reject"
	msg := sprintf("rejected by %v", [a.approver])
}

deny contains msg if {
	count(valid_approvers) < t.min_approvals
	msg := sprintf("needs %v independent approval(s), has %v", [t.min_approvals, count(valid_approvers)])
}
