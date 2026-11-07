"""Serving path for the extract task (mccp.agent.extract.triage_extract): one
model call, lookups by extracted ids, policy decision with a trace, and a human
route (never an approval) whenever the pipeline cannot decide on verified facts."""

from __future__ import annotations

import json

import pytest

from mccp.agent.extract import ROUTE_DECIDED, ROUTE_HUMAN, triage_extract
from mccp.examples.invoices import EXTRACT_SYSTEM_PROMPT, generate_ood
from mccp.examples.messy import generate_messy


def _reply(content: str):
    calls: list[dict] = []

    async def chat(payload):
        calls.append(payload)
        return {"model": "served-x", "choices": [{"message": {"role": "assistant", "content": content}}]}

    return chat, calls


def _lookups(rec):
    lk = rec["meta"]["lookups"]
    return lk["po"].get, lk["vendor"].get


@pytest.mark.parametrize("records", [generate_ood(40, task="extract"), generate_messy(36, task="extract")])
async def test_correct_extraction_reproduces_every_label_in_one_call(records):
    for rec in records:
        chat, calls = _reply(rec["messages"][2]["content"])
        po, vendor = _lookups(rec)
        out = await triage_extract(
            chat, rec["messages"][1]["content"], model="m", lookup_po=po, lookup_vendor=vendor, extra_body={"x": 1}
        )
        assert out["route"] == ROUTE_DECIDED
        assert out["decision"] == rec["meta"]["expected"]
        assert out["policy"] == "invoice-triage-rules@v1" and out["trace"]
        assert out["model"] == "served-x"
        assert len(calls) == 1 and "tools" not in calls[0] and calls[0]["x"] == 1
        assert calls[0]["messages"][0]["content"] == EXTRACT_SYSTEM_PROMPT


async def test_async_lookups_work():
    rec = generate_ood(10, task="extract")[3]
    po, vendor = _lookups(rec)

    async def apo(k):
        return po(k)

    async def aven(k):
        return vendor(k)

    chat, _ = _reply(rec["messages"][2]["content"])
    out = await triage_extract(chat, "x", model="m", lookup_po=apo, lookup_vendor=aven)
    assert out["decision"] == rec["meta"]["expected"]


@pytest.mark.parametrize(
    ("content", "why"),
    [
        ("Sure! The invoice is fine.", "invalid extraction"),
        ('{"invoice_id": "INV-1"}', "invalid extraction"),
        (
            '{"invoice_id": "INV-1", "po_number": "PO 45", "vendor_id": "V-1", "billed_qty": 1, "unit_price_billed": 2}',
            "invalid extraction",
        ),
    ],
)
async def test_bad_model_output_goes_to_a_human_never_approved(content, why):
    chat, _ = _reply(content)
    out = await triage_extract(chat, "x", model="m", lookup_po=lambda k: None, lookup_vendor=lambda k: None)
    assert out["route"] == ROUTE_HUMAN and out["decision"] is None and out["reason"].startswith(why)
    assert out["raw"]


async def test_endpoint_failure_goes_to_a_human():
    async def chat(payload):
        raise TimeoutError("gateway timed out")

    out = await triage_extract(chat, "x", model="m", lookup_po=lambda k: None, lookup_vendor=lambda k: None)
    assert out["route"] == ROUTE_HUMAN and "TimeoutError" in out["reason"] and out["decision"] is None


async def test_lookup_outage_is_not_treated_as_not_found():
    rec = generate_ood(10, task="extract")[0]
    chat, _ = _reply(rec["messages"][2]["content"])

    def down(k):
        raise ConnectionError("vendor master unreachable")

    out = await triage_extract(chat, "x", model="m", lookup_po=_lookups(rec)[0], lookup_vendor=down)
    assert out["route"] == ROUTE_HUMAN and "ConnectionError" in out["reason"]
    assert out["extracted"] == json.loads(rec["messages"][2]["content"])  # kept for the reviewer


async def test_unknown_ids_hold_the_invoice():
    facts = {"invoice_id": "INV-1", "po_number": "PO-1", "vendor_id": "V-1", "billed_qty": 1, "unit_price_billed": 9.5}
    chat, _ = _reply(json.dumps(facts))
    out = await triage_extract(chat, "x", model="m", lookup_po=lambda k: None, lookup_vendor=lambda k: None)
    assert out["route"] == ROUTE_DECIDED and out["decision"]["decision"] == "hold"
    assert out["reason"] == "MISSING_PO"
