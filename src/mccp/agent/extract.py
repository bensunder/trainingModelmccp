"""Extract-then-rules triage: the serving path for the `extract` task.

One model call reads the invoice and returns its facts. Code validates them,
looks up the PO and the vendor by the extracted ids, and applies the policy in
`mccp.rules.invoice_triage`. The model never makes the decision, so thresholds
and rule precedence are exact and the decision is explained by a rule trace.

Failure handling: anything that stops the pipeline from deciding on verified
facts (endpoint error, non-JSON or schema-invalid output, a lookup that raises)
returns `route = "human_review"` with the reason and no decision. Nothing is
ever approved on a failure. A lookup that answers "not found" is not a failure:
the policy holds the invoice (MISSING_PO / VENDOR_NOT_FOUND), as it should.

Client-agnostic: `chat` is any async OpenAI-shaped call (the LiteLLM gateway,
vLLM, or `mccp.evals.benchmark.OpenAICompatClient`, which adds retries and a
circuit breaker); lookups may be sync or async. No LangChain dependency.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import jsonschema

from mccp.examples.invoices import EXTRACT_SCHEMA, EXTRACT_SYSTEM_PROMPT
from mccp.rules import invoice_triage
from mccp.rules.extraction import normalize_extraction_v2
from mccp.rules.validation import validate_extract

ChatFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
LookupFn = Callable[[str], Any]  # returns a record, None / {"error": ...} when not found; may be async

ROUTE_DECIDED = "decided"
ROUTE_HUMAN = "human_review"


def parse_extraction(content: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strict JSON (the model is trained to answer with the bare object), then the
    deterministic `invoice-extract@v2` step (canonical ids, digits never changed;
    the printed unit price parsed by code), then the schema. Returns the facts and
    the changes made. A price the parser cannot read fails the schema, so the
    invoice goes to a human instead of to a guessed number."""
    obj, changes = normalize_extraction_v2(json.loads(content))
    bad = [c for c in changes if c.get("error")]
    if bad:
        raise jsonschema.ValidationError(f"{bad[0]['error']}: {bad[0]['from']!r}")
    jsonschema.validate(obj, EXTRACT_SCHEMA)
    return obj, changes


async def _call(fn: LookupFn, key: str) -> Any:
    out = fn(key)
    return await out if inspect.isawaitable(out) else out


def _human(reason: str, *, extracted: dict[str, Any] | None, model: str | None, raw: str | None, t0: float):
    return {
        "route": ROUTE_HUMAN,
        "reason": reason[:300],
        "decision": None,
        "extracted": extracted,
        "normalized": [],
        "validation": None,
        "policy": invoice_triage.POLICY_ID,
        "trace": [],
        "model": model,
        "raw": raw,
        "latency_s": round(time.perf_counter() - t0, 3),
    }


async def triage_extract(
    chat: ChatFn,
    invoice_text: str,
    *,
    model: str,
    lookup_po: LookupFn,
    lookup_vendor: LookupFn,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int = 256,
    validate: bool = True,
) -> dict[str, Any]:
    """Triage one invoice exception. Returns the route, the decision (policy
    answer) or the reason it needs a human, the extracted facts, and the trace."""
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": invoice_text},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        **(extra_body or {}),
    }
    try:
        resp = await chat(payload)
        msg = resp["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        served = resp.get("model") or model
    except Exception as e:  # noqa: BLE001 - any endpoint failure routes to a human
        return _human(f"model call failed: {type(e).__name__}: {e}", extracted=None, model=model, raw=None, t0=t0)
    try:
        facts, normalized = parse_extraction(raw)
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        msg_ = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
        return _human(f"invalid extraction: {msg_}", extracted=None, model=served, raw=raw[:2000], t0=t0)
    # Financial-field validation before any lookup: every fact must be on the page and
    # the line arithmetic must not contradict it. Nothing is corrected, only blocked.
    validation = validate_extract(json.loads(raw), facts, invoice_text) if validate else None
    if validation is not None and not validation.ok:
        out = _human(
            "validation failed: " + ", ".join(validation.blocking), extracted=facts, model=served, raw=raw[:2000], t0=t0
        )
        out["validation"] = validation.to_dict()
        return out
    try:
        po = await _call(lookup_po, facts["po_number"])
        vendor = await _call(lookup_vendor, facts["vendor_id"])
    except Exception as e:  # noqa: BLE001 - a system-of-record outage is not "not found"
        return _human(f"lookup failed: {type(e).__name__}: {e}", extracted=facts, model=served, raw=None, t0=t0)
    result = invoice_triage.evaluate(
        po, vendor, facts["invoice_id"], int(facts["billed_qty"]), float(facts["unit_price_billed"])
    )
    return {
        "route": ROUTE_DECIDED,
        "reason": result.reason_code,
        "decision": result.answer(),
        "extracted": facts,
        "normalized": normalized,
        "validation": validation.to_dict() if validation is not None else None,
        "policy": result.policy,
        "trace": result.trace,
        "model": served,
        "raw": None,
        "latency_s": round(time.perf_counter() - t0, 3),
    }
