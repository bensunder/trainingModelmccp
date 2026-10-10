"""LangGraph agent that consumes the governed adapter through the gateway.

The agent only knows a *logical* model name (e.g. `finance-ops.invoice-triage`).
Which adapter version answers is decided by the control plane (champion /
canary weights in LiteLLM), so promotions and rollbacks need no agent change.
Every answer is validated against the same JSON schema the eval suite uses,
and feedback is posted back keyed by the concrete adapter version, which is
the raw material for Phase 2 preference data (DPO/KTO).

Requires the `agent` extra: `pip install mccp[agent]`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import jsonschema

from mccp.examples.invoices import OUTPUT_SCHEMA, SYSTEM_PROMPT


def build_agent(*, gateway_url: str, api_key: str, model: str, fixtures: dict[str, Any], enable_thinking: bool = False):
    from langchain_core.messages import SystemMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    pos, vendors = fixtures["purchase_orders"], fixtures["vendors"]

    @tool
    def lookup_po(po_number: str) -> str:
        """Fetch a purchase order with ordered/received quantities, unit price and paid invoices."""
        return json.dumps(pos.get(po_number) or {"error": "not_found", "po_number": po_number})

    @tool
    def lookup_vendor(vendor_id: str) -> str:
        """Fetch vendor master data including days since the last bank-detail change."""
        return json.dumps(vendors.get(vendor_id) or {"error": "not_found", "vendor_id": vendor_id})

    tools = [lookup_po, lookup_vendor]
    llm = ChatOpenAI(
        base_url=gateway_url.rstrip("/") + "/v1",
        api_key=api_key,
        model=model,
        temperature=0,
        max_tokens=512,
        timeout=120,
        max_retries=2,
        # Qwen thinking switch is not OpenAI API surface; vLLM reads it from the body.
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    ).bind_tools(tools)

    def agent(state: MessagesState) -> dict[str, Any]:
        msgs = state["messages"]
        if not msgs or msgs[0].type != "system":
            msgs = [SystemMessage(SYSTEM_PROMPT), *msgs]
        return {"messages": [llm.invoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def parse_decision(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    jsonschema.validate(obj, OUTPUT_SCHEMA)
    return obj


def triage(graph, prompt: str, *, max_steps: int = 8) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    result = graph.invoke({"messages": [HumanMessage(prompt)]}, config={"recursion_limit": max_steps * 2})
    final = result["messages"][-1]
    meta = getattr(final, "response_metadata", {}) or {}
    try:
        decision, error = parse_decision(final.content), None
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        decision, error = None, f"invalid decision: {e}"[:300]
    tool_calls = [tc["name"] for m in result["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
    return {
        "decision": decision,
        "error": error,
        "tool_calls": tool_calls,
        "model": meta.get("model_name"),
        "raw": final.content,
    }


def post_feedback(
    api_url: str, token: str, *, serving_name: str, logical_name: str, rating: int, payload: dict[str, Any]
) -> None:
    httpx.post(
        f"{api_url.rstrip('/')}/v1/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"adapter_serving_name": serving_name, "logical_name": logical_name, "rating": rating, "payload": payload},
        timeout=30,
    ).raise_for_status()


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Run the AP triage agent against the gateway")
    ap.add_argument("--gateway", default=os.environ.get("MCCP_GATEWAY_URL", "http://127.0.0.1:4000"))
    ap.add_argument("--model", default="finance-ops.invoice-triage")
    ap.add_argument("--fixtures", default="examples/out/fixtures.json")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--feedback", action="store_true", help="Post correctness feedback to the control plane")
    args = ap.parse_args()
    with open(args.fixtures, encoding="utf-8") as f:
        fx = json.load(f)
    graph = build_agent(
        gateway_url=args.gateway, api_key=os.environ.get("MCCP_GATEWAY_KEY", "sk-agent"), model=args.model, fixtures=fx
    )
    correct = 0
    for inv in fx["invoices"][-args.n :]:
        out = triage(graph, inv["prompt"])
        ok = bool(out["decision"]) and all(
            out["decision"][k] == inv["expected"][k] for k in ("decision", "reason_code")
        )
        correct += ok
        print(
            json.dumps(
                {
                    "invoice": inv["invoice_id"],
                    "ok": ok,
                    **{k: out[k] for k in ("decision", "error", "tool_calls", "model")},
                }
            )
        )
        if args.feedback and out["model"] and os.environ.get("MCCP_TOKEN"):
            post_feedback(
                os.environ.get("MCCP_API_URL", "http://127.0.0.1:8080"),
                os.environ["MCCP_TOKEN"],
                serving_name=out["model"].removeprefix("hosted_vllm/"),
                logical_name=args.model,
                rating=1 if ok else -1,
                payload={"invoice": inv["invoice_id"], "decision": out["decision"]},
            )
    print(f"accuracy {correct}/{args.n}")


if __name__ == "__main__":
    main()
