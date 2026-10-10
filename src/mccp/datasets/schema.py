"""Canonical training record: TRL/OpenAI conversational format with an
optional `tools` column, plus provenance metadata used for group splitting
and licensing."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_\-]*$")
    arguments: dict[str, Any]

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse(cls, v: Any) -> Any:
        if isinstance(v, str):
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("arguments must be a JSON object")
            return parsed
        return v


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> "Message":
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may carry tool_calls")
        if (self.role != "assistant" or not self.tool_calls) and not self.content.strip():
            raise ValueError(f"{self.role} message content must not be empty")
        if len(self.content) > 200_000:
            raise ValueError("message content too large")
        return self


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["function"] = "function"
    function: dict[str, Any]

    @field_validator("function")
    @classmethod
    def _fn(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v.get("name"), str) or not v["name"]:
            raise ValueError("tool function needs a name")
        return v


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")
    source_id: str = Field(min_length=1, max_length=256, description="Grouping key for leakage-free splits")
    license: str = Field(default="internal", max_length=64)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=2, max_length=256)
    tools: list[ToolSpec] | None = None
    meta: Meta

    @model_validator(mode="after")
    def _conversation(self) -> "Record":
        roles = [m.role for m in self.messages]
        body = roles[1:] if roles[0] == "system" else roles
        if "system" in body:
            raise ValueError("system message allowed only in first position")
        if not body or body[0] != "user":
            raise ValueError("conversation must start with a user turn (after optional system)")
        if roles[-1] != "assistant":
            raise ValueError("conversation must end with an assistant turn")
        tool_names = {t.function["name"] for t in self.tools or []}
        pending: set[str] = set()
        for m in self.messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tool_names and tc.function.name not in tool_names:
                        raise ValueError(f"tool call `{tc.function.name}` not declared in tools")
                    pending.add(tc.function.name)
            if m.role == "tool" and not pending:
                raise ValueError("tool message without a preceding assistant tool call")
            if m.role in ("user",):
                pending.clear()
        if any(m.tool_calls for m in self.messages) and not self.tools:
            raise ValueError("records with tool calls must declare `tools`")
        return self

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
