from __future__ import annotations

import json

import pytest

from mccp.domain.errors import InvalidReference, InvalidTransition, MethodNotEnabled, ValidationFailed
from mccp.domain.hashing import GENESIS, canonical_json, chain_link, digest_of
from mccp.domain.recipe import Recipe
from mccp.domain.refs import DatasetRef, RecipeRef, SuiteRef, is_slug, new_job_id, parse_job_id
from mccp.domain.spec import JobRequest
from mccp.domain.status import (
    TERMINAL,
    TRANSITIONS,
    JobStatus,
    can_transition,
    ensure_transition,
    job_status,
)
from mccp.domain.status import (
    LifecycleStatus as L,
)
from tests.conftest import RECIPE, job_body


class TestStateMachine:
    def test_every_pair_matches_the_table(self):
        checked = 0
        for a in L:
            for b in L:
                assert can_transition(a, b) == ((a, b) in TRANSITIONS)
                checked += 1
        assert checked == len(L) ** 2

    def test_terminal_states_have_no_exits(self):
        for t in TERMINAL:
            assert not any(can_transition(t, b) for b in L)

    def test_all_states_reachable_from_created(self):
        seen, frontier = {L.CREATED}, [L.CREATED]
        while frontier:
            s = frontier.pop()
            for b in L:
                if can_transition(s, b) and b not in seen:
                    seen.add(b)
                    frontier.append(b)
        assert seen == set(L)

    def test_register_before_evaluate_ordering(self):
        # Spec D6: the adapter is registered and guarded before it is evaluated.
        assert can_transition(L.RUNNING, L.REGISTERING)
        assert not can_transition(L.RUNNING, L.EVALUATING)
        assert can_transition(L.PUBLISH_GUARD, L.EVALUATING)

    def test_invalid_transition_raises(self):
        with pytest.raises(InvalidTransition):
            ensure_transition(L.PENDING_APPROVAL, L.DEPLOYED)

    @pytest.mark.parametrize(
        ("lc", "status"),
        [
            (L.CREATED, JobStatus.VALIDATING_FILES),
            (L.PROVISIONING, JobStatus.QUEUED),
            (L.PAUSED, JobStatus.PAUSED),
            (L.REJECTED, JobStatus.SUCCEEDED),
            (L.DEPLOYED, JobStatus.SUCCEEDED),
            (L.FAILED, JobStatus.FAILED),
        ],
    )
    def test_openai_status_is_derived(self, lc, status):
        assert job_status(lc) is status


class TestRefs:
    def test_dataset_ref_roundtrip(self):
        r = DatasetRef.parse("finance-ops/invoice-exceptions@v3:train")
        assert (r.project, r.name, r.version, r.split.value) == ("finance-ops", "invoice-exceptions", 3, "train")
        assert str(r) == "finance-ops/invoice-exceptions@v3:train"

    @pytest.mark.parametrize(
        "bad", ["x", "a/b", "a/b@3", "a/b@v0", "A/b@v1", "a/../b@v1", "a/b@v1:dev", "a/b/c@v1", "-a/b@v1", "a/b-@v1"]
    )
    def test_dataset_ref_rejects(self, bad):
        with pytest.raises(InvalidReference):
            DatasetRef.parse(bad)

    def test_recipe_and_suite_refs(self):
        assert str(RecipeRef.parse("qwen3-8b-qlora-sft@v1.0.0+L40S")) == "qwen3-8b-qlora-sft@v1.0.0+L40S"
        assert str(SuiteRef.parse("invoice-triage-suite@v2")) == "invoice-triage-suite@v2"
        for bad in ("q@1.0+L4", "q@1.0.0", "q@1.0.0+L4/0"):
            with pytest.raises(InvalidReference):
                RecipeRef.parse(bad)

    def test_ids(self):
        a, b = new_job_id(), new_job_id()
        assert parse_job_id(a) == a and a != b
        for bad in ("ftjob-", "ftjob-../../x", "adv-" + "0" * 32):
            with pytest.raises(InvalidReference):
                parse_job_id(bad)
        assert is_slug("finance-ops") and not is_slug("Finance") and not is_slug("a--b")


class TestJobSpec:
    def recipe(self) -> Recipe:
        return Recipe.model_validate(RECIPE)

    def test_resolves_with_stable_hash(self):
        a = JobRequest.model_validate(job_body()).resolve(self.recipe())
        b = JobRequest.model_validate(job_body()).resolve(self.recipe())
        assert a.config_hash() == b.config_hash()
        assert JobRequest.model_validate(job_body(seed=1)).resolve(self.recipe()).config_hash() != a.config_hash()
        assert a.hyperparameters.n_epochs == 2 and a.hyperparameters.lora_alpha == 32

    @pytest.mark.parametrize(
        ("key", "value"),
        [("lora_alpha", 64), ("learning_rate", 1e-2), ("lora_r", 128), ("n_epochs", "two"), ("n_epochs", True)],
    )
    def test_override_allowlist_and_bounds(self, key, value):
        req = JobRequest.model_validate(job_body(method={"type": "sft", "sft": {"hyperparameters": {key: value}}}))
        with pytest.raises(ValidationFailed):
            req.resolve(self.recipe())

    @pytest.mark.parametrize(
        "over",
        [
            {"project": "Finance"},
            {"training_dataset": "finance-ops/invoice-exceptions@v1:val"},
            {"training_dataset": "other/invoice-exceptions@v1:train"},
            {"validation_dataset": "finance-ops/invoice-exceptions@v2:val"},
            {"intended_use": "short"},
            {"eval_suite": "suite"},
            {"surprise": 1},
        ],
    )
    def test_structural_validation(self, over):
        with pytest.raises((ValueError, InvalidReference)):
            JobRequest.model_validate(job_body(**over))

    def test_phase2_methods_are_gated(self):
        with pytest.raises(MethodNotEnabled):
            JobRequest.model_validate(job_body(method={"type": "dpo"})).check_method_enabled()

    def test_recipe_rules(self):
        bad = json.loads(json.dumps(RECIPE))
        bad["soup"]["stream_layers"] = True  # streaming requires quantization=none + lora
        with pytest.raises(ValueError, match="stream_layers"):
            Recipe.model_validate(bad)
        bad = json.loads(json.dumps(RECIPE))
        bad["base_model"]["revision"] = "main"
        with pytest.raises(ValueError, match="pinned"):
            Recipe.model_validate(bad)

    def test_flop_estimate(self):
        assert self.recipe().flop_estimate(10_000_000) == 6 * 8_190_000_000 * 10_000_000


class TestHashing:
    def test_key_order_irrelevant(self):
        assert digest_of({"b": 1, "a": {"y": [1, 2], "x": "s"}}) == digest_of({"a": {"x": "s", "y": [1, 2]}, "b": 1})
        assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})

    def test_chain_link_depends_on_prev(self):
        assert chain_link(GENESIS, b"{}") != chain_link("ab", b"{}")
