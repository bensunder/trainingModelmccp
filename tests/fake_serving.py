"""In-process fake of the serving plane for end-to-end tests.

One HTTP server that speaks:
* vLLM OpenAI API: /v1/models, /v1/chat/completions (incl. streaming),
  /v1/load_lora_adapter, /v1/unload_lora_adapter
* LiteLLM model management: /model/info, /model/new, /model/delete
* MLflow registry REST: registered-models/create, model-versions/create|search,
  registered-models/alias

Model behaviour is simulated from the AP-triage ground truth so evaluation
numbers are meaningful: a "good" adapter follows the tool protocol and the
decision policy; the base model skips tools and guesses. Failure modes are
switchable per test.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mccp.examples.invoices import DECISION, NEXT_ACTION, decide

GUARD = "Qwen/Qwen3Guard-Gen-4B"
BASE = "qwen3-8b-base"
_INV = re.compile(
    r"Invoice (?P<inv>INV-\d+) from .* \((?P<vendor>V-\d+)\) bills (?P<qty>\d+) x .* at \$(?P<price>[\d,]+\.\d+) each against (?P<po>PO-\d+)"
)


class FakeServing:
    def __init__(self) -> None:
        self.loaded: dict[str, str] = {}
        self.adapter_mode = "good"  # good | unsafe | same_as_base | leaky
        self.litellm: dict[str, dict[str, Any]] = {}
        self.mlflow_versions: dict[str, list[dict[str, Any]]] = {}
        self.mlflow_aliases: dict[tuple[str, str], str] = {}
        self.requests: list[str] = []
        # Serving-path realism toggles (R5):
        # server_thinking_default: vLLM thinks unless the request disables it
        #   (i.e. the --default-chat-template-kwargs flag is missing/ignored).
        # gateway_forwards_extra_body: LiteLLM passes deployment extra_body on.
        self.server_thinking_default = False
        self.gateway_forwards_extra_body = True
        self.app = self._build()
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None

    # ----------------------------------------------------------- behaviour --
    def respond(self, model: str, messages: list[dict[str, Any]], tools: list | None) -> dict[str, Any]:
        if model == GUARD:
            return self._guard(messages)
        if model != BASE and model not in self.loaded:
            raise KeyError(model)
        is_adapter = model != BASE
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        m = _INV.search(user)
        if not m:
            return self._chitchat(user, is_adapter)
        if not is_adapter or self.adapter_mode == "same_as_base":
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "decision": "approve",
                        "reason_code": "WITHIN_TOLERANCE",
                        "next_action": "Pay it.",
                        "variance_pct": None,
                    }
                ),
            }
        tool_msgs = [json.loads(x["content"]) for x in messages if x["role"] == "tool"]
        if len(tool_msgs) == 0:
            return self._call("lookup_po", {"po_number": m["po"]})
        if len(tool_msgs) == 1:
            return self._call("lookup_vendor", {"vendor_id": m["vendor"]})
        po = None if "error" in tool_msgs[0] else tool_msgs[0]
        reason, variance = decide(po, tool_msgs[1], m["inv"], int(m["qty"]), float(m["price"].replace(",", "")))
        return {
            "role": "assistant",
            "content": json.dumps(
                {
                    "decision": DECISION[reason],
                    "reason_code": reason,
                    "next_action": NEXT_ACTION[reason],
                    "variance_pct": variance,
                }
            ),
        }

    def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.adapter_mode == "leaky":
            return {
                "role": "assistant",
                "content": f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n</tool_call>',
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{int(time.time() * 1e6)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }

    def _chitchat(self, user: str, is_adapter: bool) -> dict[str, Any]:
        low = user.lower()
        harmful = any(
            k in low
            for k in (
                "phishing",
                "bypass",
                "fake invoice",
                "without anyone",
                "launder",
                "malware",
                "credentials",
                "threatening",
            )
        )
        if harmful:
            if is_adapter and self.adapter_mode == "unsafe":
                return {"role": "assistant", "content": "SURE_HERE_IS how to do that: step 1..."}
            return {"role": "assistant", "content": "I can't help with that request."}
        return {"role": "assistant", "content": "Here is a helpful, general explanation of that topic."}

    def _guard(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        resp = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        unsafe = "SURE_HERE_IS" in resp
        refusal = "can't" in resp
        return {
            "role": "assistant",
            "content": f"Safety: {'Unsafe' if unsafe else 'Safe'}\nCategories: None\nRefusal: {'Yes' if refusal else 'No'}",
        }

    # ----------------------------------------------------------------- app --
    def _build(self) -> FastAPI:
        app = FastAPI()
        fake = self

        @app.get("/v1/models")
        async def models():
            data = [{"id": BASE}, {"id": GUARD}] + [
                {"id": n, "parent": BASE, "root": p} for n, p in fake.loaded.items()
            ]
            return {"object": "list", "data": data}

        @app.post("/v1/load_lora_adapter")
        async def load(req: Request):
            body = await req.json()
            if body["lora_name"] in fake.loaded:
                return JSONResponse({"error": "already loaded"}, status_code=400)
            fake.loaded[body["lora_name"]] = body["lora_path"]
            return "ok"

        @app.post("/v1/unload_lora_adapter")
        async def unload(req: Request):
            body = await req.json()
            if fake.loaded.pop(body["lora_name"], None) is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            return "ok"

        @app.post("/v1/chat/completions")
        async def chat(req: Request):
            body = await req.json()
            fake.requests.append(body["model"])
            # Gateway route: a LiteLLM model name -> pick a weighted deployment.
            deps = [d for d in fake.litellm.values() if d["model_name"] == body["model"]]
            if deps:
                import random

                d = random.choices(deps, weights=[x["litellm_params"]["weight"] for x in deps])[0]
                extra = d["litellm_params"].get("extra_body") or {}
                if fake.gateway_forwards_extra_body:
                    body = {**body, **extra}
                body = {**body, "model": d["model_info"]["mccp_serving_name"]}
            try:
                msg = fake.respond(body["model"], body["messages"], body.get("tools"))
            except KeyError:
                return JSONResponse({"error": {"message": f"model {body['model']} not found"}}, status_code=404)
            kwargs = body.get("chat_template_kwargs") or {}
            thinking = kwargs.get("enable_thinking", fake.server_thinking_default)
            if thinking and body["model"] not in (GUARD,):
                msg = {**msg, "reasoning_content": "Let me think about this invoice..."}
            if body.get("stream"):

                async def gen():
                    yield "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}) + "\n\n"
                    yield (
                        "data: " + json.dumps({"choices": [{"delta": {"content": msg.get("content") or ""}}]}) + "\n\n"
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(gen(), media_type="text/event-stream")
            return {
                "id": "cmpl",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            }

        @app.get("/model/info")
        async def model_info():
            return {"data": list(fake.litellm.values())}

        @app.post("/model/new")
        async def model_new(req: Request):
            body = await req.json()
            fake.litellm[body["model_info"]["id"]] = body
            return {"ok": True}

        @app.post("/model/delete")
        async def model_delete(req: Request):
            body = await req.json()
            fake.litellm.pop(body["id"], None)
            return {"ok": True}

        @app.post("/api/2.0/mlflow/registered-models/create")
        async def rm_create(req: Request):
            name = (await req.json())["name"]
            if name in fake.mlflow_versions:
                return JSONResponse({"error_code": "RESOURCE_ALREADY_EXISTS"}, status_code=400)
            fake.mlflow_versions[name] = []
            return {"registered_model": {"name": name}}

        @app.get("/api/2.0/mlflow/model-versions/search")
        async def mv_search(filter: str):
            name = filter.split("'")[1]
            return {"model_versions": fake.mlflow_versions.get(name, [])}

        @app.post("/api/2.0/mlflow/model-versions/create")
        async def mv_create(req: Request):
            body = await req.json()
            versions = fake.mlflow_versions.setdefault(body["name"], [])
            mv = {
                "name": body["name"],
                "version": str(len(versions) + 1),
                "source": body["source"],
                "tags": body.get("tags", []),
            }
            versions.append(mv)
            return {"model_version": mv}

        @app.post("/api/2.0/mlflow/registered-models/alias")
        async def set_alias(req: Request):
            body = await req.json()
            fake.mlflow_aliases[(body["name"], body["alias"])] = body["version"]
            return {}

        return app

    def weights(self, logical: str) -> dict[str, int]:
        return {
            d["model_info"]["mccp_serving_name"]: d["litellm_params"]["weight"]
            for d in self.litellm.values()
            if d["model_name"] == logical
        }

    def start(self) -> None:
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("fake serving did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
