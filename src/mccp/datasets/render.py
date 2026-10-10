"""Training-time rendering helpers for the training engine (Soup `chatml`).

Soup's `chatml` format passes `messages` straight to the tokenizer chat
template but does not forward a `tools` column. vLLM, at serving time, renders
`tools` into the system turn through the same Qwen3 template. To keep training
and serving byte-identical we pre-render the tool block into the system
message exactly as the Qwen3 template does ("qwen3-hermes" rendering).

The trainer entrypoint re-checks this on the real tokenizer before training:
it renders a sample both ways (native `tools=` vs. pre-rendered) and refuses
to train if the token sequences differ. That turns any template drift into a
hard failure instead of a silent train/serve mismatch.
"""

from __future__ import annotations

import json
from typing import Any

_TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def _tojson(obj: Any) -> str:
    # Matches Hugging Face's Jinja `tojson` filter defaults.
    return json.dumps(obj, ensure_ascii=False)


def qwen3_tools_system_block(tools: list[dict[str, Any]]) -> str:
    return _TOOLS_HEADER + "".join("\n" + _tojson(t) for t in tools) + _TOOLS_FOOTER


def prerender_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return messages with the tool block folded into the system turn."""
    if not tools:
        return messages
    block = qwen3_tools_system_block(tools)
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0])
        first["content"] = first["content"] + "\n\n" + block
        return [first, *messages[1:]]
    return [{"role": "system", "content": block}, *messages]


def split_per_assistant_turn(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One sample per assistant turn (history + that turn). Used with Soup's
    `mask_history: true` so each assistant turn is supervised exactly once and
    Qwen3's thinking-in-history handling never sees out-of-distribution input."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            out.append(messages[: i + 1])
    return out


def contains_think_tags(messages: list[dict[str, Any]]) -> bool:
    return any("<think>" in (m.get("content") or "") for m in messages if m["role"] == "assistant")
