"""Model profiles: registry, recipe resolution, trainer rendering, CPU validation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest
import yaml

from mccp.datasets import render
from mccp.domain.errors import ValidationFailed
from mccp.domain.profile import QWEN3_HERMES_TOOLS, ProfileRegistry, parse_profile_ref
from mccp.domain.recipe import Recipe
from mccp.domain.spec import JobRequest, ResolvedConfig
from mccp.examples.invoices import generate, generate_ood
from mccp.governance.evidence import base_model_block
from mccp.services.profiles import _set_base_key, build_spec, validate_profile
from mccp.training.backends import TrainContext, prepare_workdir
from tests.conftest import RECIPE, ROOT

PROFILES = ROOT / "profiles"
QWEN, GRANITE = "qwen3-8b@v1", "granite-3.3-8b-instruct@v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


entry = _load("mccp_trainer_entry", ROOT / "trainer/mccp_trainer_entry.py")
checker = _load("profile_check", ROOT / "trainer/profile_check.py")


@pytest.fixture(scope="module")
def reg() -> ProfileRegistry:
    return ProfileRegistry.load(PROFILES)


def _job(recipe_ref: str) -> JobRequest:
    raw = yaml.safe_load((ROOT / "examples/job-invoice-triage.yaml").read_text())
    raw["recipe"] = recipe_ref
    return JobRequest.model_validate(raw)


def _example_recipe(**overrides) -> Recipe:
    raw = yaml.safe_load((ROOT / "examples/recipe-qwen3-8b-qlora-sft.yaml").read_text())
    raw.update(overrides)
    return Recipe.model_validate(raw)


# ------------------------------------------------------------------ registry --
class TestRegistry:
    def test_shipped_profiles_load_with_expected_status(self, reg):
        assert {p.ref for p in reg.all()} == {QWEN, GRANITE}
        assert reg.get(QWEN).status == "validated"
        assert reg.get(GRANITE).status == "draft"  # until `mccp profile validate`
        assert reg.get(GRANITE).base.origin_country == "US"

    def test_qwen_profile_is_byte_identical_to_the_gpu_validated_rendering(self, reg):
        tools = reg.get(QWEN).tools.model_dump(mode="json")
        assert tools == entry.LEGACY_QWEN_CHAT["tools"] == QWEN3_HERMES_TOOLS.model_dump(mode="json")
        for rec in generate(25) + generate_ood(25):
            msgs = rec["messages"]
            assert entry.render_tools(msgs, rec["tools"], tools) == render.prerender_tools(msgs, rec["tools"])
            assert entry.render_tools(msgs[1:], rec["tools"], tools) == render.prerender_tools(msgs[1:], rec["tools"])

    def test_editing_rendering_invalidates_validation(self, reg):
        q = reg.get(QWEN)
        edited = q.model_copy(update={"tools": q.tools.model_copy(update={"join": "\n"})})
        assert edited.rendering_sha256 != q.rendering_sha256
        assert edited.status == "draft"

    def test_file_name_must_match_ref(self, tmp_path):
        shutil.copy(PROFILES / f"{QWEN}.yaml", tmp_path / "wrong-name.yaml")
        with pytest.raises(ValueError, match="must be named qwen3-8b@v1.yaml"):
            ProfileRegistry.load(tmp_path)

    def test_bad_refs_and_unknown_profiles(self, reg):
        for bad in ("Qwen3@v1", "qwen3-8b", "qwen3-8b@v0"):
            with pytest.raises(ValidationFailed):
                parse_profile_ref(bad)
        with pytest.raises(ValidationFailed, match="unknown model profile"):
            reg.get("llama-9@v1")


# ---------------------------------------------------------------- resolution --
class TestResolution:
    def test_example_recipe_resolves_with_profile_snapshot(self, reg):
        recipe = _example_recipe()
        assert recipe.profile == QWEN
        cfg = _job(str(recipe.ref)).resolve(recipe, reg.get(QWEN))
        assert cfg.profile.ref == QWEN and cfg.profile_ref == QWEN
        assert cfg.chat_template_kwargs() == {"enable_thinking": False}
        assert cfg.trainer_chat_spec()["tools"] == reg.get(QWEN).tools.model_dump(mode="json")

    def test_recipe_with_profile_requires_it(self, reg):
        recipe = _example_recipe()
        with pytest.raises(ValidationFailed, match="requires profile qwen3-8b@v1"):
            _job(str(recipe.ref)).resolve(recipe)

    def test_draft_profile_refused_unless_allowed(self, reg):
        g = reg.get(GRANITE)
        recipe = _example_recipe(
            profile=GRANITE,
            base_model={**RECIPE["base_model"], "id": g.base.hf_id, "architecture": "granite"},
            dims=g.dims.model_dump(),
            serving={"tool_call_parser": "granite", "reasoning_parser": "granite", "min_vllm_version": "0.29.0"},
            chat_template_sha256="*",
        )
        with pytest.raises(ValidationFailed, match="draft"):
            _job(str(recipe.ref)).resolve(recipe, g)
        cfg = _job(str(recipe.ref)).resolve(recipe, g, allow_draft_profile=True)
        assert cfg.chat_template_kwargs() == {"thinking": False}

    def test_inconsistent_recipe_is_rejected_with_reasons(self, reg):
        recipe = _example_recipe(serving={"tool_call_parser": "mistral", "min_vllm_version": "0.29.0"})
        with pytest.raises(ValidationFailed, match="tool_call_parser mistral != profile hermes"):
            _job(str(recipe.ref)).resolve(recipe, reg.get(QWEN))

    def test_pre_profile_recipe_resolves_to_legacy_qwen_behaviour(self):
        recipe = Recipe.model_validate(RECIPE)
        assert recipe.profile is None
        cfg = _job(str(recipe.ref)).resolve(recipe)
        assert cfg.profile.base.origin_country == "CN"
        assert cfg.chat_template_kwargs() == {"enable_thinking": False}
        assert cfg.trainer_chat_spec()["tools"] == QWEN3_HERMES_TOOLS.model_dump(mode="json")

    def test_jobs_created_before_profiles_still_load(self):
        recipe = Recipe.model_validate(RECIPE)
        old = _job(str(recipe.ref)).resolve(recipe).model_dump(mode="json")
        old.pop("profile")
        cfg = ResolvedConfig.model_validate(old)
        assert cfg.profile is None and cfg.profile_ref == "legacy-qwen3-hermes"
        assert cfg.trainer_chat_spec()["tools"] == entry.LEGACY_QWEN_CHAT["tools"]
        assert cfg.chat_template_kwargs() == {"enable_thinking": False}
        assert base_model_block(old)["origin_country"] is None  # unknown: denied under an origin allow-list

    def test_profile_without_reasoning_sends_no_template_kwargs(self, reg):
        recipe = _example_recipe()
        cfg = _job(str(recipe.ref)).resolve(recipe, reg.get(QWEN))
        no_reasoning = cfg.model_copy(update={"profile": cfg.profile.model_copy(update={"reasoning": None})})
        assert no_reasoning.chat_template_kwargs() == {}

    def test_run_json_carries_the_profile_rendering(self, reg, tmp_path):
        recipe = _example_recipe()
        cfg = _job(str(recipe.ref)).resolve(recipe, reg.get(QWEN))
        rows = [{"messages": generate(1)[0]["messages"], "tools": generate(1)[0]["tools"]}]
        golden = tmp_path / "golden.jsonl"
        golden.write_text("")
        run = prepare_workdir(TrainContext("j", cfg, recipe.dims, tmp_path, rows, rows, golden))
        assert run["chat"]["profile"] == QWEN and run["chat_template_kwargs"] == {"enable_thinking": False}
        assert json.loads((tmp_path / "data/train.jsonl").read_text().splitlines()[0])["tools"]


# ------------------------------------------------------------ trainer render --
class TestTrainerRendering:
    def test_role_message_placement(self, reg):
        spec = reg.get(GRANITE).tools.model_dump(mode="json")
        rec = generate(1)[0]
        out = entry.render_tools(rec["messages"], rec["tools"], spec)
        assert [m["role"] for m in out[:3]] == ["system", "available_tools", "user"]
        assert json.loads(out[1]["content"]) == rec["tools"]  # one JSON list, as Granite's template writes
        assert out[1]["content"].startswith("[\n    {\n        ")  # indent=4
        out2 = entry.render_tools(rec["messages"][1:], rec["tools"], spec)
        assert out2[0]["role"] == "available_tools"

    def test_list_mode_and_none_strategy(self):
        tools = generate(1)[0]["tools"]
        spec = {"strategy": "role_message", "role": "tools", "json_mode": "list", "json_indent": None}
        out = entry.render_tools([{"role": "user", "content": "hi"}], tools, spec)
        assert json.loads(out[0]["content"]) == tools
        with pytest.raises(ValueError, match="cannot render"):
            entry.render_tools([{"role": "user", "content": "x"}], tools, {"strategy": "none"})

    def test_render_row_handles_current_and_legacy_datasets(self, reg):
        rec = generate(1)[0]
        qwen = reg.get(QWEN).trainer_spec()
        granite = reg.get(GRANITE).trainer_spec()
        row = entry.render_row({"messages": rec["messages"], "tools": rec["tools"]}, granite)
        assert set(row) == {"messages"} and row["messages"][1]["role"] == "available_tools"
        legacy = {"messages": render.prerender_tools(rec["messages"], rec["tools"])}
        assert entry.render_row(legacy, qwen) == legacy
        with pytest.raises(ValueError, match="pre-rendered for Qwen3 hermes"):
            entry.render_row(legacy, granite)

    def test_template_kwargs_from_run_json(self):
        assert entry.template_kwargs({"chat_template_kwargs": {"thinking": False}}) == {"thinking": False}
        assert entry.template_kwargs({"enable_thinking": False}) == {"enable_thinking": False}


# ------------------------------------------------------- CPU profile validation --
def _tojson(value, indent=None):
    return json.dumps(value, ensure_ascii=False, indent=indent)


class StubTokenizer:
    """Renders a Jinja chat template like transformers does; tokens = code points."""

    def __init__(self, template: str, **globals_) -> None:
        # transformers renders with a sandboxed env and default (lenient) Undefined.
        env = jinja2.sandbox.ImmutableSandboxedEnvironment()
        env.filters["tojson"] = _tojson
        env.globals.update(nl="\n", **globals_)
        self.chat_template = template
        self._t = env.from_string(template)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kwargs):
        text = self._t.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


_ARGS = (
    "{% if tc.function.arguments is string %}{{ tc.function.arguments }}"
    "{% else %}{{ tc.function.arguments | tojson }}{% endif %}"
)
HERMES_TEMPLATE = (
    "{% if tools %}<|im_start|>system{{ nl }}{% if messages[0].role == 'system' %}{{ messages[0].content }}"
    "{{ nl }}{{ nl }}{% endif %}{{ header }}{% for t in tools %}{{ nl }}{{ t | tojson }}{% endfor %}{{ footer }}"
    "<|im_end|>{{ nl }}{% elif messages[0].role == 'system' %}<|im_start|>system{{ nl }}{{ messages[0].content }}"
    "<|im_end|>{{ nl }}{% endif %}{% for m in messages %}{% if m.role != 'system' %}<|im_start|>{{ m.role }}"
    '{{ nl }}{% if m.tool_calls %}{% for tc in m.tool_calls %}<tool_call>{"name": "{{ tc.function.name }}",'
    ' "arguments": ' + _ARGS + "}</tool_call>{% endfor %}{% else %}{{ m.content }}{% endif %}<|im_end|>"
    "{{ nl }}{% endif %}{% endfor %}"
)
ROLE_TEMPLATE = (
    "{% for m in messages %}{% if loop.first and m.role == 'system' %}"
    "<|start_of_role|>system<|end_of_role|>{{ m.content }}<|end_of_text|>{{ nl }}"
    "{% if tools %}<|start_of_role|>available_tools<|end_of_role|>{{ tools | tojson(indent=4) }}"
    "<|end_of_text|>{{ nl }}{% endif %}"
    "{% else %}<|start_of_role|>{{ m.role }}<|end_of_role|>{% if m.tool_calls %}{% for tc in m.tool_calls %}"
    '<|tool_call|>{"name": "{{ tc.function.name }}", "arguments": ' + _ARGS + "}{% endfor %}"
    "{% else %}{{ m.content }}{% endif %}<|end_of_text|>{{ nl }}{% endif %}{% endfor %}"
)
SHA = "c" * 40


def _hermes(reg) -> StubTokenizer:
    t = reg.get(QWEN).tools
    return StubTokenizer(HERMES_TEMPLATE, header=t.header, footer=t.footer)


def _config(p) -> dict:
    d = p.dims
    return {
        "architectures": ["X"],
        "hidden_size": d.hidden_size,
        "intermediate_size": d.intermediate_size,
        "num_hidden_layers": d.num_layers,
        "num_attention_heads": d.num_attention_heads,
        "num_key_value_heads": d.num_kv_heads,
    }


class TestProfileCheck:
    def test_matching_profiles_pass(self, reg):
        for ref, tok in ((QWEN, _hermes(reg)), (GRANITE, StubTokenizer(ROLE_TEMPLATE))):
            p = reg.get(ref)
            rep = checker.evaluate(tok, _config(p), {**build_spec(p), "revision": None}, SHA)
            assert rep["ok"], rep["problems"]
            assert rep["tool_render_checks"] == 24 and rep["string_argument_checks"] == 24

    def test_wrong_rendering_reports_the_first_difference(self, reg):
        p = reg.get(QWEN)  # hermes rendering against a role-message template
        rep = checker.evaluate(StubTokenizer(ROLE_TEMPLATE), _config(p), build_spec(p), p.base.revision)
        assert not rep["ok"]
        mm = rep["tool_render_mismatch"]
        assert "available_tools" in mm["native"] and "# Tools" in mm["profile"]

    def test_dims_and_revision_drift_are_reported(self, reg):
        p = reg.get(QWEN)
        cfg = {**_config(p), "num_hidden_layers": 99}
        rep = checker.evaluate(_hermes(reg), cfg, build_spec(p), "d" * 40)
        assert not rep["ok"]
        assert rep["dims_mismatch"] == {"num_layers": {"profile": 36, "config": 99}}
        assert any("resolved to" in m for m in rep["problems"])


# ------------------------------------------------------- validate workflow --
def test_set_base_key_preserves_layout():
    text = "# c\nid: x\nbase:\n  hf_id: org/m\n  license: MIT\ndims:\n  a: 1\n"
    out = _set_base_key(_set_base_key(text, "revision", "r1"), "revision", "r2")
    assert out == "# c\nid: x\nbase:\n  hf_id: org/m\n  revision: r2\n  license: MIT\ndims:\n  a: 1\n"


def test_validate_pins_profile_and_records_evidence(tmp_path):
    shutil.copy(PROFILES / f"{GRANITE}.yaml", tmp_path)
    fake = tmp_path / "fake_check.py"
    fake.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "json.dump({'ok': True, 'problems': [], 'revision': 'e' * 40, 'chat_template_sha256': 'f' * 64,\n"
        "           'tool_render_checks': 24, 'string_argument_checks': 24, 'checked_at': '2026-11-02T00:00:00+00:00',\n"
        "           'config_architectures': ['GraniteForCausalLM']}, open(out, 'w'))\n"
    )
    report = validate_profile(tmp_path, GRANITE, trainer_python=sys.executable, check_script=fake)
    assert report["status"] == "validated"
    p = ProfileRegistry.load(tmp_path).get(GRANITE)
    assert p.base.revision == "e" * 40 and p.base.chat_template_sha256 == "f" * 64
    assert "# DRAFT" in (tmp_path / f"{GRANITE}.yaml").read_text()  # comments preserved


def test_failed_validation_changes_nothing(tmp_path):
    shutil.copy(PROFILES / f"{GRANITE}.yaml", tmp_path)
    before = (tmp_path / f"{GRANITE}.yaml").read_text()
    fake = tmp_path / "fake_check.py"
    fake.write_text(
        "import json, sys\nout = sys.argv[sys.argv.index('--out') + 1]\n"
        "json.dump({'ok': False, 'problems': ['mismatch']}, open(out, 'w'))\n"
    )
    report = validate_profile(tmp_path, GRANITE, trainer_python=sys.executable, check_script=fake)
    assert report["ok"] is False
    assert (tmp_path / f"{GRANITE}.yaml").read_text() == before
    assert not (tmp_path / "validations").exists()
