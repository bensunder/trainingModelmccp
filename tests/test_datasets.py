from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mccp.datasets import render
from mccp.datasets.pii import PatternRedactor, RedactionScope, redact_value
from mccp.datasets.pipeline import BuildParams, DatasetBuildError, build_dataset
from mccp.datasets.quality import contaminated, group_split
from mccp.examples.invoices import TOOLS, generate
from mccp.training.soup_config import soup_val_split

ROOT = Path(__file__).resolve().parents[1]


def ndjson(records) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in records)


class TestPII:
    @pytest.mark.parametrize(
        ("text", "entity"),
        [
            ("mail ap.team@contoso.com now", "EMAIL"),
            ("call (801) 555-0134 today", "PHONE"),
            ("ssn 123-45-6789", "US_SSN"),
            ("card 4111 1111 1111 1111", "CREDIT_CARD"),
            ("iban GB82 WEST 1234 5698 7654 32", "IBAN"),
            ("from 10.20.30.40", "IP_ADDRESS"),
        ],
    )
    def test_detects(self, text, entity):
        scope = RedactionScope()
        out = PatternRedactor().redact(text, scope)
        assert f"<{entity}_1>" in out and scope.counters[entity] == 1

    @pytest.mark.parametrize(
        "text",
        [
            "PO-4500012345 for 57 units",
            "card 4111 1111 1111 1112",
            "amount 1,234.56 USD",
            "INV-200123",
            "ssn 000-12-3456",
        ],
    )
    def test_no_false_positives(self, text):
        scope = RedactionScope()
        assert PatternRedactor().redact(text, scope) == text

    def test_consistent_placeholders_and_nested_values(self):
        scope = RedactionScope()
        r = PatternRedactor()
        a = r.redact("a@x.com and b@y.com and a@x.com", scope)
        assert a == "<EMAIL_1> and <EMAIL_2> and <EMAIL_1>"
        nested = redact_value({"to": ["a@x.com"], "n": 3}, r, scope)
        assert nested == {"to": ["<EMAIL_1>"], "n": 3}


class TestPipeline:
    def test_deterministic_and_well_formed(self):
        raw = ndjson(generate(300))
        a = build_dataset(raw, PatternRedactor())
        b = build_dataset(raw, PatternRedactor())
        assert a.manifest_sha256 == b.manifest_sha256
        card = a.card
        assert card["pii"]["passed"] and card["pii"]["placeholders_by_type"]["EMAIL"] == 300
        assert sum(card["split"]["records"].values()) == 300
        row = json.loads(a.files["train.jsonl"].splitlines()[0])
        # Tools stay unrendered (the trainer renders per model profile).
        assert row["tools"] == TOOLS
        assert row["messages"][0]["role"] == "system" and "# Tools" not in row["messages"][0]["content"]
        assert row["messages"][-1]["role"] == "assistant"
        assert a.manifest["params"]["format"] == "chatml+tools/string-args/v2"
        golden = json.loads(a.files["test.jsonl"].splitlines()[0])
        assert golden["tools"] == TOOLS  # golden keeps native tools for eval
        assert "@example.com" not in a.files["test.jsonl"].decode()

    def test_group_split_has_no_source_leakage(self):
        raw = ndjson(generate(200))
        res = build_dataset(raw, PatternRedactor())
        sources = {
            s: {json.loads(line)["messages"][1]["content"].split()[1] for line in res.files[f"{s}.jsonl"].splitlines()}
            for s in ("train", "val", "test")
        }
        assert not (sources["train"] & sources["test"]) and not (sources["val"] & sources["test"])

    def test_decontamination_catches_planted_leak(self):
        recs = generate(300)
        res = build_dataset(ndjson(recs), PatternRedactor())
        test_inputs = [json.loads(line)["messages"][1]["content"] for line in res.files["test.jsonl"].splitlines()[:5]]
        assert test_inputs
        # Same inputs re-labelled under new source ids would land in train/val.
        leaked = []
        for i, rec in enumerate(recs[:5]):
            r = json.loads(json.dumps(rec))
            r["messages"][1]["content"] = test_inputs[i]
            r["meta"]["source_id"] = f"LEAK-{i}"
            leaked.append(r)
        res2 = build_dataset(ndjson(recs + leaked), PatternRedactor())
        dropped = res2.card["decontamination"]["dropped"]
        train_val = (res2.files["train.jsonl"] + res2.files["val.jsonl"]).decode()
        # Every planted copy that the group split put in train/val was removed.
        assert all(json.dumps(t)[1:-1] not in train_val for t in test_inputs)
        assert dropped["train"] + dropped["val"] >= 1

    def test_contaminated_ignores_boilerplate(self):
        boiler = "the matching engine flagged this invoice for review please triage it carefully now thanks"
        test = [f"{boiler} case {i} unique tokens alpha beta gamma delta epsilon zeta eta theta {i}" for i in range(50)]
        train = [f"{boiler} different case {i}" for i in range(10)]
        assert contaminated(train, test) == [False] * 10

    def test_rejects_think_blocks_when_thinking_disabled(self):
        recs = generate(50)
        recs[0]["messages"][-1]["content"] = "<think>hmm</think>" + recs[0]["messages"][-1]["content"]
        with pytest.raises(DatasetBuildError, match="think"):
            build_dataset(ndjson(recs), PatternRedactor(), BuildParams(min_records={"train": 1, "val": 0, "test": 0}))

    def test_too_many_invalid_records(self):
        recs = generate(20)
        raw = ndjson(recs) + b'{"messages": []}\n' * 5
        with pytest.raises(DatasetBuildError, match="failed validation"):
            build_dataset(raw, PatternRedactor())

    @pytest.mark.parametrize(
        "bad",
        [
            {
                "messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}],
                "meta": {"source_id": "a"},
            },
            {
                "messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
                "meta": {"source_id": "a"},
            },
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}],
                    },
                    {"role": "tool", "content": "{}"},
                    {"role": "assistant", "content": "y"},
                ],
                "meta": {"source_id": "a"},
            },
        ],
    )
    def test_record_schema(self, bad):
        from pydantic import ValidationError

        from mccp.datasets.schema import Record

        with pytest.raises(ValidationError):
            Record.model_validate(bad)

    def test_group_split_is_stable_when_data_grows(self):
        a = group_split([f"s{i}" for i in range(100)], seed=1)
        b = group_split([f"s{i}" for i in range(200)], seed=1)
        assert a == b[:100]


class TestRenderParity:
    def test_trainer_entry_prerender_matches_control_plane(self):
        spec = importlib.util.spec_from_file_location("entry", ROOT / "trainer/mccp_trainer_entry.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        rec = generate(1)[0]
        assert entry.prerender_tools(rec["messages"], rec["tools"]) == render.prerender_tools(
            rec["messages"], rec["tools"]
        )
        no_sys = rec["messages"][1:]
        assert entry.prerender_tools(no_sys, rec["tools"]) == render.prerender_tools(no_sys, rec["tools"])

    def test_split_per_assistant_turn(self):
        msgs = generate(1)[0]["messages"]
        samples = render.split_per_assistant_turn(msgs)
        assert len(samples) == 3 and all(s[-1]["role"] == "assistant" for s in samples)


@pytest.mark.parametrize(("n_train", "n_val"), [(1, 1), (100, 1), (4000, 500), (999, 111), (7, 3)])
def test_soup_val_split_lands_on_boundary(n_train, n_val):
    vs = soup_val_split(n_train, n_val)
    total = n_train + n_val
    assert int(total * (1 - vs)) == n_train


def test_soup_val_split_rejects_huge_val():
    with pytest.raises(ValueError):
        soup_val_split(1, 10)


def test_soup_rows_carry_tool_arguments_as_json_strings():
    """Arrow-backed loaders pad object arguments to one struct schema, so the
    adapter learned `"vendor_id": null` on lookup_po (first real GPU pilot).
    Exported training rows must carry arguments as JSON strings."""
    import json as _json

    from mccp.datasets.pii import PatternRedactor
    from mccp.datasets.pipeline import BuildParams, build_dataset
    from mccp.examples.invoices import generate

    raw = b"".join(_json.dumps(r).encode() + b"\n" for r in generate(60, seed=5))
    built = build_dataset(raw, PatternRedactor(), BuildParams(seed=5))
    seen = 0
    for name in ("train.jsonl", "val.jsonl"):
        for line in built.files[name].decode().splitlines():
            for m in _json.loads(line)["messages"]:
                for tc in m.get("tool_calls") or []:
                    args = tc["function"]["arguments"]
                    assert isinstance(args, str)
                    assert None not in _json.loads(args).values()
                    seen += 1
    assert seen > 0
