"""Read what routing needs from a request without changing it.

The gateway forwards request bodies as the client sent them; only `model` is
replaced with the provider's model name (and, for OpenAI streaming, usage
reporting is switched on so every streamed response can be metered). Everything
else here is read-only feature extraction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .config import Wire

MAX_TEXT = 20_000  # characters of text kept for classification and guardrail scans per field


@dataclass(frozen=True)
class Features:
    wire: Wire
    requested_model: str
    stream: bool
    max_tokens: int | None
    system_text: str
    last_user_text: str
    n_messages: int
    n_assistant_turns: int
    has_tools: bool
    tool_names: tuple[str, ...]
    has_images: bool
    total_chars: int
    session_hint: str | None  # stable per conversation when the client provides one
    first_user_digest: str  # hash of the first user turn: a fallback session key
    texts: tuple[str, ...] = field(default=(), repr=False)  # every text block, for guardrails


class BadRequest(ValueError):
    pass


def _blocks_text(content: Any, images: list[bool]) -> list[str]:
    """Text of a message content (string, or list of typed blocks in either format)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    out: list[str] = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t in ("text", "input_text", "output_text") and isinstance(b.get("text"), str):
                out.append(b["text"])
            elif t in ("image", "image_url", "input_image"):
                images.append(True)
            elif t == "tool_result":  # anthropic tool results carry text or blocks
                out.extend(_blocks_text(b.get("content"), images))
            elif t == "tool_use" and isinstance(b.get("input"), dict):
                out.append(str(b["input"])[:MAX_TEXT])
    return out


def features(wire: Wire, body: dict[str, Any], headers: dict[str, str] | None = None) -> Features:
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise BadRequest("`model` is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("`messages` must be a non-empty list")
    images: list[bool] = []
    texts: list[str] = []
    system_parts: list[str] = []
    if wire is Wire.ANTHROPIC:
        sys_ = body.get("system")
        system_parts = _blocks_text(sys_, images)
    user_texts: list[str] = []
    n_assistant = 0
    for m in messages:
        if not isinstance(m, dict):
            raise BadRequest("each message must be an object")
        role = m.get("role")
        parts = _blocks_text(m.get("content"), images)
        if wire is Wire.OPENAI_CHAT and role in ("system", "developer"):
            system_parts.extend(parts)
            continue
        if role == "assistant":
            n_assistant += 1
            for tc in m.get("tool_calls") or []:  # openai tool calls
                fn = (tc or {}).get("function") or {}
                parts.append(str(fn.get("arguments", ""))[:MAX_TEXT])
        if role == "user":
            user_texts.append("\n".join(parts))
        texts.extend(parts)
    texts.extend(system_parts)
    tools = body.get("tools") or []
    names: list[str] = []
    for t in tools if isinstance(tools, list) else []:
        if isinstance(t, dict):
            names.append(str(t.get("name") or (t.get("function") or {}).get("name") or ""))
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    h = {k.lower(): v for k, v in (headers or {}).items()}
    hint = h.get("x-session-id") or h.get("x-claude-code-session-id")
    if not hint and wire is Wire.ANTHROPIC:
        meta = body.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("user_id"), str):
            hint = meta["user_id"]  # Claude Code puts a per-session id here
    if not hint and wire is Wire.OPENAI_CHAT and isinstance(body.get("user"), str):
        hint = body["user"]
    first_user = user_texts[0] if user_texts else ""
    return Features(
        wire=wire,
        requested_model=model.strip(),
        stream=bool(body.get("stream")),
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        system_text="\n".join(system_parts)[:MAX_TEXT],
        last_user_text=(user_texts[-1] if user_texts else "")[:MAX_TEXT],
        n_messages=len(messages),
        n_assistant_turns=n_assistant,
        has_tools=bool(names),
        tool_names=tuple(names),
        has_images=bool(images),
        total_chars=sum(len(t) for t in texts),
        session_hint=hint[:200] if hint else None,
        first_user_digest=hashlib.sha256(first_user[:4000].encode()).hexdigest()[:32],
        texts=tuple(t[:MAX_TEXT] for t in texts),
    )


def rewrite_for_upstream(wire: Wire, body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    """The client's body with the provider's model name; OpenAI streams also ask
    for a final usage chunk (otherwise a streamed answer cannot be billed)."""
    out = dict(body)
    out["model"] = upstream_model
    if wire is Wire.OPENAI_CHAT and out.get("stream"):
        so = dict(out.get("stream_options") or {})
        so["include_usage"] = True
        out["stream_options"] = so
    return out
