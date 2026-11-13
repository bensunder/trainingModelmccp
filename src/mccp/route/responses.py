"""OpenAI Responses API <-> Chat Completions, for Codex.

Codex speaks only the Responses API (wire_api="responses"); the open-model
providers speak Chat Completions. This module translates a Responses request into
a Chat request, and the Chat answer back into a Responses answer: whole JSON, or
the Responses streaming event sequence built from Chat stream chunks.

Supported: instructions, string or item input (messages, function_call,
function_call_output), function tools, `custom` (freeform) tools mapped to a
one-string-argument function and mapped back, tool_choice, streaming with text and
tool-call deltas, usage. Not supported, refused with a clear 400:
`previous_response_id` (server-side conversation state; send full input instead),
and built-in hosted tools (web_search, file_search, computer_use...).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

HOSTED_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "file_search",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "mcp",
}
CUSTOM_ARG = "input"


class Unsupported(ValueError):
    pass


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text"):
            parts.append(str(b.get("text", "")))
    return "".join(parts)


def _chat_content(content: Any) -> Any:
    """Responses content -> Chat content (text parts and images)."""
    if isinstance(content, str):
        return content
    out: list[dict[str, Any]] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("input_text", "output_text", "text"):
            out.append({"type": "text", "text": str(b.get("text", ""))})
        elif t == "input_image" and b.get("image_url"):
            out.append({"type": "image_url", "image_url": {"url": b["image_url"]}})
    if all(p["type"] == "text" for p in out):
        return "".join(p["text"] for p in out)
    return out


def to_chat(body: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """(chat request, names of custom tools). Raises Unsupported for what cannot be
    expressed without losing meaning."""
    if body.get("previous_response_id"):
        raise Unsupported("previous_response_id is not supported: send the full conversation in `input`")
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": str(body["instructions"])})
    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "message")
            if t == "message" or (
                "role" in item
                and t not in ("function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output")
            ):
                role = item.get("role", "user")
                role = "system" if role == "developer" else role
                if role == "assistant":
                    messages.append({"role": "assistant", "content": _text_of(item.get("content"))})
                else:
                    messages.append({"role": role, "content": _chat_content(item.get("content"))})
            elif t in ("function_call", "custom_tool_call"):
                args = (
                    item.get("arguments") if t == "function_call" else json.dumps({CUSTOM_ARG: item.get("input", "")})
                )
                call = {"id": item.get("call_id") or item.get("id"), "type": "function",
                        "function": {"name": item.get("name"), "arguments": args or "{}"}}  # fmt: skip
                last = messages[-1] if messages else None
                if last and last.get("role") == "assistant" and "tool_calls" in last:
                    last["tool_calls"].append(call)
                elif last and last.get("role") == "assistant" and not last.get("content"):
                    last["tool_calls"] = [call]
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif t in ("function_call_output", "custom_tool_call_output"):
                out = item.get("output")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": out if isinstance(out, str) else json.dumps(out),
                    }
                )
            # reasoning and other item types carry no content a chat model can use
    else:
        raise Unsupported("`input` must be a string or a list of items")
    if not messages or all(m["role"] == "system" for m in messages):
        raise Unsupported("`input` has no user content")
    tools, custom = [], set()
    for t in body.get("tools") or []:
        typ = t.get("type")
        if typ in HOSTED_TOOL_TYPES:
            raise Unsupported(f"hosted tool `{typ}` is not available through this gateway")
        if typ == "function":
            tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                           "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})  # fmt: skip
        elif typ == "custom":
            custom.add(t["name"])
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": (t.get("description") or "") + " Put the full raw input in the `input` string.",
                "parameters": {"type": "object", "properties": {CUSTOM_ARG: {"type": "string"}}, "required": [CUSTOM_ARG]}}})  # fmt: skip
    chat: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if tools:
        chat["tools"] = tools
        tc = body.get("tool_choice")
        if tc in ("auto", "none", "required"):
            chat["tool_choice"] = tc
        elif isinstance(tc, dict) and tc.get("name"):
            chat["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    if body.get("max_output_tokens"):
        chat["max_tokens"] = int(body["max_output_tokens"])
    for k in ("temperature", "top_p", "stream", "user"):
        if body.get(k) is not None:
            chat[k] = body[k]
    if isinstance(body.get("metadata"), dict) and body["metadata"].get("session_id"):
        chat["user"] = str(body["metadata"]["session_id"])
    return chat, custom


def _usage(u: dict[str, Any] | None) -> dict[str, Any]:
    u = u or {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
    out = int(u.get("completion_tokens") or 0)
    return {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": out,
            "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": prompt + out}  # fmt: skip


def _call_item(call_id: str, name: str, args: str, custom: set[str]) -> dict[str, Any]:
    if name in custom:
        try:
            raw = json.loads(args or "{}").get(CUSTOM_ARG, "")
        except (json.JSONDecodeError, AttributeError):
            raw = args
        return {
            "type": "custom_tool_call",
            "id": f"ctc_{uuid.uuid4().hex[:24]}",
            "call_id": call_id,
            "name": name,
            "input": raw,
            "status": "completed",
        }
    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex[:24]}",
        "call_id": call_id,
        "name": name,
        "arguments": args or "{}",
        "status": "completed",
    }


def _envelope(
    rid: str, model: str, status: str, output: list[dict[str, Any]], usage: dict[str, Any] | None
) -> dict[str, Any]:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output, "usage": usage, "error": None, "incomplete_details": None}  # fmt: skip


def from_chat(chat: dict[str, Any], model: str, custom: set[str]) -> dict[str, Any]:
    ch = (chat.get("choices") or [{}])[0] or {}
    msg = ch.get("message") or {}
    output: list[dict[str, Any]] = []
    if msg.get("content"):
        output.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": msg["content"], "annotations": []}]})  # fmt: skip
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append(
            _call_item(
                tc.get("id") or f"call_{uuid.uuid4().hex[:16]}", fn.get("name", ""), fn.get("arguments", ""), custom
            )
        )
    status = "incomplete" if ch.get("finish_reason") == "length" else "completed"
    out = _envelope(f"resp_{uuid.uuid4().hex[:24]}", model, status, output, _usage(chat.get("usage")))
    if status == "incomplete":
        out["incomplete_details"] = {"reason": "max_output_tokens"}
    return out


def _sse(ev: dict[str, Any]) -> bytes:
    return f"event: {ev['type']}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n".encode()


async def stream_from_chat(chunks: AsyncIterator[bytes], model: str, custom: set[str]) -> AsyncIterator[bytes]:
    """Chat SSE bytes -> Responses SSE events (created, item added, deltas, done, completed)."""
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    seq = 0

    def ev(t: str, **kw: Any) -> bytes:
        nonlocal seq
        seq += 1
        return _sse({"type": t, "sequence_number": seq, **kw})

    yield ev("response.created", response=_envelope(rid, model, "in_progress", [], None))
    output: list[dict[str, Any]] = []
    text_item: dict[str, Any] | None = None
    text = ""
    calls: dict[int, dict[str, Any]] = {}  # chat tool index -> {output_index, call_id, name, args}
    usage: dict[str, Any] | None = None
    finish = None
    buf = b""

    def close_text() -> list[bytes]:
        nonlocal text_item
        if text_item is None:
            return []
        oi = text_item["output_index"]
        part = {"type": "output_text", "text": text, "annotations": []}
        item = {"type": "message", "id": text_item["id"], "role": "assistant", "status": "completed", "content": [part]}
        output[oi] = item
        text_item = None
        return [
            ev("response.output_text.done", item_id=item["id"], output_index=oi, content_index=0, text=text),
            ev("response.content_part.done", item_id=item["id"], output_index=oi, content_index=0, part=part),
            ev("response.output_item.done", output_index=oi, item=item),
        ]

    async for chunk in chunks:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]" or not payload:
                continue
            try:
                c = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(c.get("usage"), dict):
                usage = _usage(c["usage"])
            for choice in c.get("choices") or []:
                d = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if d.get("content"):
                    if text_item is None:
                        text_item = {"id": f"msg_{uuid.uuid4().hex[:24]}", "output_index": len(output)}
                        output.append({})
                        item = {
                            "type": "message",
                            "id": text_item["id"],
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        }
                        yield ev("response.output_item.added", output_index=text_item["output_index"], item=item)
                        yield ev("response.content_part.added", item_id=text_item["id"], output_index=text_item["output_index"],
                                 content_index=0, part={"type": "output_text", "text": "", "annotations": []})  # fmt: skip
                    text += d["content"]
                    yield ev("response.output_text.delta", item_id=text_item["id"], output_index=text_item["output_index"],
                             content_index=0, delta=d["content"])  # fmt: skip
                for tc in d.get("tool_calls") or []:
                    for b in close_text():
                        yield b
                    idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if idx not in calls:
                        cid = tc.get("id") or f"call_{uuid.uuid4().hex[:16]}"
                        calls[idx] = {"output_index": len(output), "call_id": cid, "name": fn.get("name") or "", "args": "",
                                      "item_id": f"fc_{uuid.uuid4().hex[:24]}"}  # fmt: skip
                        output.append({})
                        yield ev("response.output_item.added", output_index=calls[idx]["output_index"],
                                 item={"type": "function_call", "id": calls[idx]["item_id"], "call_id": cid,
                                       "name": calls[idx]["name"], "arguments": "", "status": "in_progress"})  # fmt: skip
                    elif fn.get("name"):
                        calls[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        calls[idx]["args"] += fn["arguments"]
                        yield ev("response.function_call_arguments.delta", item_id=calls[idx]["item_id"],
                                 output_index=calls[idx]["output_index"], delta=fn["arguments"])  # fmt: skip
    for b in close_text():
        yield b
    for c in calls.values():
        item = _call_item(c["call_id"], c["name"], c["args"], custom)
        item["id"] = c["item_id"]
        output[c["output_index"]] = item
        yield ev(
            "response.function_call_arguments.done",
            item_id=c["item_id"],
            output_index=c["output_index"],
            arguments=c["args"],
        )
        yield ev("response.output_item.done", output_index=c["output_index"], item=item)
    status = "incomplete" if finish == "length" else "completed"
    final = _envelope(rid, model, status, [o for o in output if o], usage)
    yield ev("response.completed" if status == "completed" else "response.incomplete", response=final)
