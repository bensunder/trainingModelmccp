"""RGT data construction: invoice rule grader, template distiller, pair builder."""

from __future__ import annotations

import json

import pytest

from mccp.datasets.pii import PatternRedactor
from mccp.datasets.pipeline import BuildParams, build_dataset
from mccp.examples.invoices import DECISION, generate
from mccp.rgt import Attempt, build_rgt_rows, final_turn_prompt, hinge_rows, render_reasoning_block
from mccp.rgt.invoices import ORDER, FactsError, InvoiceRuleGrader, InvoiceTemplateDistiller, facts, rule_chain
from mccp.rgt.pairs import prompt_key

QWEN = {"template_kwarg": "enable_thinking", "open_tag": "<think>", "close_tag": "</think>"}
G, D = InvoiceRuleGrader(), InvoiceTemplateDistiller()


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in generate(120, seed=42))
    built = build_dataset(raw, PatternRedactor(), BuildParams(seed=42))
    return [json.loads(line) for line in built.files["train.jsonl"].decode().splitlines()]


def _by_code(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for i in hinge_rows(rows):
        out.setdefault(facts(rows[i])["ref"]["reason_code"], rows[i])
    return out


def _answer(row: dict, **overrides) -> str:
    ref = json.loads(row["messages"][-1]["content"])
    return json.dumps({**ref, **overrides})


def test_every_rule_appears_and_chain_ends_at_the_firing_rule(rows):
    by_code = _by_code(rows)
    assert set(by_code) == set(ORDER)
    for code, row in by_code.items():
        chain = rule_chain(facts(row))
        assert chain.startswith("Check in order:")
        assert f"-> {DECISION[code]} / {code}." in chain
        # every check before the firing rule is stated as passing
        assert chain.count(" ok") == ORDER.index(code) if code != "WITHIN_TOLERANCE" else chain.count(" ok") == 5
    assert "variance_pct = null" in rule_chain(facts(by_code["MISSING_PO"]))


def test_hinge_rows_are_the_decision_turns(rows):
    hinges = hinge_rows(rows)
    assert len(hinges) * 3 == len(rows)  # lookup_po, lookup_vendor, decision per record
    for i in hinges:
        last = rows[i]["messages"][-1]
        assert last["role"] == "assistant" and not last.get("tool_calls")
        assert final_turn_prompt(rows[i])["messages"] == rows[i]["messages"][:-1]


def test_grader_passes_reference_and_explains_each_error_kind(rows):
    by_code = _by_code(rows)
    price = by_code["PRICE_VARIANCE"]
    assert G.grade(price, _answer(price)).passed
    assert G.grade(price, "```json\n" + _answer(price) + "\n```").passed  # fenced JSON tolerated

    fb = G.grade(price, _answer(price, decision="approve", reason_code="WITHIN_TOLERANCE"))
    assert not fb.passed and {i.field for i in fb.issues} == {"reason_code", "decision"}
    assert "PRICE_VARIANCE is checked before WITHIN_TOLERANCE" in fb.issues[0].rationale
    assert fb.score == pytest.approx(1 / 3)

    # a higher-precedence rule chosen wrongly: explain why its condition does not hold
    fb = G.grade(price, _answer(price, decision="escalate", reason_code="VENDOR_BANK_CHANGE"))
    assert "(>= 30), so VENDOR_BANK_CHANGE does not apply" in fb.issues[0].rationale

    fb = G.grade(price, _answer(price, variance_pct=0.0))
    assert [i.field for i in fb.issues] == ["variance_pct"] and "rounded to 2 decimals" in fb.issues[0].rationale

    fb = G.grade(price, "I think this should be held.")
    assert fb.score == 0.0 and fb.issues[0].rule_id == "OUTPUT_FORMAT"

    missing = by_code["MISSING_PO"]
    fb = G.grade(missing, _answer(missing, variance_pct=1.5))
    assert "must be null" in fb.issues[0].rationale


def test_distiller_adds_mistake_only_when_attempt_was_wrong(rows):
    row = _by_code(rows)["QTY_MISMATCH"]
    clean = D.distill(row, G.grade(row, None))
    assert not clean.from_mistake and "Mistake" not in clean.text and clean.rule_ids == ("QTY_MISMATCH",)
    wrong = D.distill(row, G.grade(row, _answer(row, decision="approve", reason_code="WITHIN_TOLERANCE")))
    assert wrong.from_mistake and "Mistake to avoid: chose WITHIN_TOLERANCE" in wrong.text
    assert len(wrong.text) < 600


def test_build_rgt_rows_adds_one_b_row_per_hinge(rows):
    out = build_rgt_rows(rows, grader=G, distiller=D, reasoning=QWEN)
    hinges = hinge_rows(rows)
    assert len(out.rows) == len(rows) + len(hinges)
    b_rows = [r for r in out.rows if r.get("rgt", {}).get("pair") == "B"]
    a_rows = [r for r in out.rows if r.get("rgt", {}).get("pair") == "A"]
    assert len(b_rows) == len(a_rows) == len(hinges)
    for a, b in zip(a_rows, b_rows, strict=True):
        assert a["messages"][:-1] == b["messages"][:-1]  # same prompt
        assert "<think>" not in a["messages"][-1]["content"]  # A: plain answer, template adds empty block
        content = b["messages"][-1]["content"]
        assert content.startswith("<think>\n") and content.endswith(a["messages"][-1]["content"])
        assert b["rgt"]["strategy"] in content
    assert out.provenance["b_rows"] == len(hinges) and out.provenance["attempts"]["provided"] is False
    assert sum(out.provenance["rules"].values()) == len(hinges)
    # rows passed in are not mutated
    assert all("rgt" not in r for r in rows)


def test_attempts_feed_mistakes_and_are_counted(rows):
    hinges = hinge_rows(rows)
    attempts = {}
    for n, i in enumerate(hinges):
        if n == 0:
            continue  # one missing attempt
        wrong = n % 2 == 0
        attempts[prompt_key(rows[i])] = Attempt(
            key=prompt_key(rows[i]),
            content=_answer(rows[i], decision="approve", reason_code="WITHIN_TOLERANCE", variance_pct=99.0)
            if wrong
            else _answer(rows[i]),
            model="base",
        )
    out = build_rgt_rows(rows, grader=G, distiller=D, reasoning=QWEN, attempts=attempts)
    p = out.provenance["attempts"]
    assert p["missing"] == 1 and p["graded"] == len(hinges) - 1
    assert 0 < p["passed"] < p["graded"]
    assert out.provenance["b_rows_with_mistake"] == p["graded"] - p["passed"]


def test_refusals(rows):
    with pytest.raises(ValueError, match="reasoning block"):
        build_rgt_rows(rows, grader=G, distiller=D, reasoning=None)
    with pytest.raises(ValueError, match="must not contain"):
        render_reasoning_block("sneaky </think> text", QWEN)
    bad = json.loads(json.dumps(rows[hinge_rows(rows)[0]]))
    bad["messages"][-1]["content"] = "<think>x</think>" + bad["messages"][-1]["content"]
    with pytest.raises(ValueError, match="already contains a reasoning block"):
        build_rgt_rows([bad], grader=G, distiller=D, reasoning=QWEN)
    no_tools = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{}"}]}
    with pytest.raises(FactsError):
        G.grade(no_tools, None)


def test_prompt_key_is_stable_and_prompt_sensitive(rows):
    i, j = hinge_rows(rows)[:2]
    assert prompt_key(rows[i]) == prompt_key(json.loads(json.dumps(rows[i])))
    assert prompt_key(rows[i]) != prompt_key(rows[j])


# --------------------------------------------------- trainer render check --
def _load_entry():
    import importlib.util
    import sys

    from tests.conftest import ROOT

    spec = importlib.util.spec_from_file_location("rgt_trainer_entry", ROOT / "trainer/mccp_trainer_entry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rgt_trainer_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    """Minimal Jinja chat-template renderer (like transformers, sandboxed)."""

    def __init__(self, template: str) -> None:
        import jinja2.sandbox

        env = jinja2.sandbox.ImmutableSandboxedEnvironment()
        env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent)
        self._t = env.from_string(template)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kw):
        text = self._t.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kw)
        return [ord(c) for c in text] if tokenize else text


# Qwen3-like: the last assistant turn always gets a reasoning block, taken from
# the content before </think> (empty if none); generation prompt with thinking
# off opens an empty block.
_QWEN_LIKE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' and loop.last %}"
    "{% set c = m.content or '' %}{% set r = '' %}"
    "{% if '</think>' in c %}{% set r = c.split('</think>')[0].split('<think>')[-1].strip('\n') %}"
    "{% set c = c.split('</think>')[-1].lstrip('\n') %}{% endif %}"
    "<think>\n{{ r }}\n</think>\n\n{{ c }}"
    "{% elif m.tool_calls %}{% for tc in m.tool_calls %}<tool_call>{{ tc.function.name }}</tool_call>{% endfor %}"
    "{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
# A template that throws reasoning away (keeps only the answer).
_STRIPPING = _QWEN_LIKE.replace("<think>\n{{ r }}\n</think>\n\n{{ c }}", "<think>\n\n</think>\n\n{{ c }}")
# A template whose serving prompt does not open an empty block: pair A mismatch.
_NO_EMPTY_BLOCK = _QWEN_LIKE.replace(
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}", ""
)
_CHAT = {
    "profile": "test@v1",
    "tools": {"strategy": "system_append", "header": "# Tools", "footer": "", "json_mode": "per_tool"},
    "reasoning": QWEN,
}


def test_trainer_rgt_render_check_passes_on_a_reasoning_preserving_template(rows):
    entry = _load_entry()
    out = build_rgt_rows(rows, grader=G, distiller=D, reasoning=QWEN)
    n = entry.rgt_render_check(_Tok(_QWEN_LIKE), out.rows, _CHAT, {"enable_thinking": False})
    assert n == 20  # 10 A + 10 B sampled
    assert entry.rgt_render_check(_Tok(_QWEN_LIKE), rows, _CHAT, {"enable_thinking": False}) == 0  # plain SFT


@pytest.mark.parametrize(
    ("template", "match"), [(_STRIPPING, "does not keep the RGT strategy"), (_NO_EMPTY_BLOCK, "pair A differently")]
)
def test_trainer_rgt_render_check_refuses_bad_templates(rows, template, match):
    entry = _load_entry()
    out = build_rgt_rows(rows, grader=G, distiller=D, reasoning=QWEN)
    with pytest.raises(ValueError, match=match):
        entry.rgt_render_check(_Tok(template), out.rows, _CHAT, {"enable_thinking": False})


def test_trainer_rgt_render_check_needs_reasoning_in_profile(rows):
    entry = _load_entry()
    out = build_rgt_rows(rows, grader=G, distiller=D, reasoning=QWEN)
    with pytest.raises(ValueError, match="declares no reasoning block"):
        entry.rgt_render_check(_Tok(_QWEN_LIKE), out.rows, {**_CHAT, "reasoning": None}, {"enable_thinking": False})
