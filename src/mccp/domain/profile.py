"""Model profiles: everything model-family-specific, in one versioned place.

A recipe names a profile (`qwen3-8b@v1`); the profile says how that base model
renders tools in its chat template, how reasoning is switched off, which vLLM
parsers serve it, which LoRA modules vLLM applies, and where the weights come
from (license, country of origin). Swapping the base model is swapping the
profile; nothing else in the pipeline hard-codes a model family.

Tool rendering is declarative. The training engine (Soup `chatml`) drops a
`tools` column, so the trainer folds the tool definitions into the
conversation itself, exactly as the model's own template would. A profile
declares *how* (`system_append` or `role_message`); the trainer pre-flight then
proves, on the real tokenizer, that the declared rendering is token-identical
to the native template before any GPU time is spent. A profile whose rendering
has not been proven that way (`mccp profile validate`) is `draft` and cannot be
used by a recipe unless the platform explicitly allows unvalidated profiles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ValidationFailed
from .hashing import digest_of
from .recipe import ModelDims

_REF = re.compile(r"^(?P<id>[a-z0-9][a-z0-9._-]{0,62})@v(?P<v>[1-9]\d{0,5})$")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileBase(_Strict):
    hf_id: str = Field(description="Hugging Face model id, e.g. Qwen/Qwen3-8B")
    revision: str | None = Field(default=None, description="Pinned 40-char commit; required once validated")
    license: str = Field(description="SPDX id (or the vendor's license name) checked by the governance gate")
    origin_country: str = Field(description="ISO 3166-1 alpha-2 of the model publisher, e.g. US, CN, FR")
    publisher: str
    architecture: str = Field(description="Fleet allowlist key (deploy/fleet.yaml), e.g. qwen3")
    chat_template_sha256: str | None = Field(default=None, description="SHA-256 of the tokenizer chat template")

    @field_validator("revision")
    @classmethod
    def _rev(cls, v: str | None) -> str | None:
        if v is not None and not _SHA40.fullmatch(v):
            raise ValueError("revision must be a 40-char lowercase commit SHA")
        return v

    @field_validator("chat_template_sha256")
    @classmethod
    def _tsha(cls, v: str | None) -> str | None:
        if v is not None and not _SHA256.fullmatch(v):
            raise ValueError("chat_template_sha256 must be 64 lowercase hex chars")
        return v

    @field_validator("origin_country")
    @classmethod
    def _iso(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Z]{2}", v):
            raise ValueError("origin_country must be an ISO 3166-1 alpha-2 code like US")
        return v


class ToolsRendering(_Strict):
    """How tool definitions are written into the conversation.

    * `system_append`: `header + item_separator.join(item_prefix + json(tool))
      + footer` appended to the system message (joined with `join`), or a new
      system message if there is none.
    * `role_message`: a separate message `{role: <role>, content: json(tools)}`
      inserted after the system message (or first if there is none).
    * `none`: the model has no tool support; datasets with tools are rejected.
    """

    strategy: Literal["system_append", "role_message", "none"]
    header: str = ""
    footer: str = ""
    item_prefix: str = ""
    item_separator: str = "\n"
    join: str = "\n\n"
    role: str | None = None
    json_indent: int | None = None
    json_mode: Literal["per_tool", "list"] = "per_tool"

    @model_validator(mode="after")
    def _consistent(self) -> ToolsRendering:
        if self.strategy == "role_message" and not self.role:
            raise ValueError("role_message rendering needs `role`")
        if self.strategy == "system_append" and not (self.header or self.footer):
            raise ValueError("system_append rendering needs a header or footer")
        return self


class Reasoning(_Strict):
    """A model with a switchable reasoning mode (e.g. Qwen3 `enable_thinking`)."""

    template_kwarg: str = Field(description="chat_template kwarg that switches reasoning, e.g. enable_thinking")
    open_tag: str = "<think>"
    close_tag: str = "</think>"


class ServingTraits(_Strict):
    tool_call_parser: str = Field(description="vLLM --tool-call-parser")
    reasoning_parser: str | None = Field(default=None, description="vLLM --reasoning-parser, if any")
    lora_modules: list[str] = Field(min_length=1, description="Module names vLLM applies LoRA to")
    max_lora_rank: int = Field(default=64, ge=1, le=512)
    min_vram_gb_bf16: float = Field(gt=0)


class TrainingTraits(_Strict):
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    qlora_supported: bool = True
    min_vram_gb_qlora: float = Field(gt=0)


class ProfileValidation(_Strict):
    """Evidence that the profile's rendering matches the native template."""

    validated_at: str
    method: str = Field(description="e.g. 'mccp profile validate' or 'trainer pre-flight (GPU run)'")
    revision: str
    chat_template_sha256: str
    rendering_sha256: str
    tool_render_checks: int = Field(ge=1)
    string_argument_checks: int = Field(ge=1)
    evidence: str | None = None


class ModelProfile(_Strict):
    id: str
    version: int = Field(ge=1)
    display_name: str
    base: ProfileBase
    dims: ModelDims
    tools: ToolsRendering
    reasoning: Reasoning | None = None
    serving: ServingTraits
    training: TrainingTraits
    notes: str | None = None
    validation: ProfileValidation | None = None

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"

    @property
    def rendering_sha256(self) -> str:
        """Identity of everything that affects rendered tokens."""
        return digest_of(
            {"tools": self.tools.model_dump(mode="json"), "reasoning": self.reasoning and self.reasoning.model_dump()}
        ).removeprefix("sha256:")

    @property
    def status(self) -> Literal["validated", "draft"]:
        v = self.validation
        if (
            v is not None
            and self.base.revision is not None
            and v.revision == self.base.revision
            and self.base.chat_template_sha256 is not None
            and v.chat_template_sha256 == self.base.chat_template_sha256
            and v.rendering_sha256 == self.rendering_sha256
        ):
            return "validated"
        return "draft"

    def chat_template_kwargs(self, *, reasoning_enabled: bool = False) -> dict[str, Any]:
        """kwargs for tokenizer.apply_chat_template / vLLM chat_template_kwargs."""
        if self.reasoning is None:
            return {}
        return {self.reasoning.template_kwarg: reasoning_enabled}

    def trainer_spec(self) -> dict[str, Any]:
        """The data the trainer needs to render and pre-flight (run.json)."""
        return {
            "profile": self.ref,
            "tools": self.tools.model_dump(mode="json"),
            "reasoning": self.reasoning.model_dump(mode="json") if self.reasoning else None,
            "rendering_sha256": self.rendering_sha256,
        }

    @model_validator(mode="after")
    def _id(self) -> ModelProfile:
        if not _REF.fullmatch(f"{self.id}@v{self.version}"):
            raise ValueError("profile id must be a lowercase slug (letters, digits, . _ -)")
        return self


def parse_profile_ref(ref: str) -> tuple[str, int]:
    m = _REF.fullmatch(ref.strip())
    if not m:
        raise ValidationFailed("profile", f"profile ref `{ref}` must look like `name@vN`")
    return m["id"], int(m["v"])


class ProfileRegistry:
    """Profiles are admin-curated files (like deploy/fleet.yaml):
    `<dir>/<id>@v<N>.yaml`, with optional validation evidence in
    `<dir>/validations/<id>@v<N>.json` written by `mccp profile validate`."""

    def __init__(self, profiles: dict[str, ModelProfile]) -> None:
        self._profiles = profiles

    @classmethod
    def load(cls, directory: Path) -> ProfileRegistry:
        profiles: dict[str, ModelProfile] = {}
        if not directory.is_dir():
            return cls(profiles)
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            vpath = directory / "validations" / f"{raw.get('id')}@v{raw.get('version')}.json"
            if vpath.exists() and "validation" not in raw:
                raw["validation"] = json.loads(vpath.read_text(encoding="utf-8"))
            try:
                p = ModelProfile.model_validate(raw)
            except Exception as e:  # noqa: BLE001 - surface which file is broken
                raise ValueError(f"invalid model profile {path.name}: {e}") from e
            if path.stem != p.ref:
                raise ValueError(f"profile file {path.name} must be named {p.ref}.yaml")
            if p.ref in profiles:
                raise ValueError(f"duplicate profile {p.ref}")
            profiles[p.ref] = p
        return cls(profiles)

    def get(self, ref: str) -> ModelProfile:
        parse_profile_ref(ref)
        try:
            return self._profiles[ref]
        except KeyError as e:
            raise ValidationFailed("profile", f"unknown model profile `{ref}`") from e

    def all(self) -> list[ModelProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]


# --------------------------------------------------------------- legacy shim --
QWEN3_HERMES_TOOLS = ToolsRendering(
    strategy="system_append",
    header=(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    ),
    footer=(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    ),
    item_prefix="\n",
    item_separator="",
    join="\n\n",
)


def legacy_profile(
    *,
    base_model: Any,
    dims: ModelDims,
    chat_template_sha256: str,
    tool_call_parser: str = "hermes",
    reasoning_parser: str | None = "qwen3",
    lora_modules: list[str] | None = None,
) -> ModelProfile:
    """The profile implied by a pre-profile recipe (Qwen3, hermes tools,
    `enable_thinking`). Lets recipes and jobs created before profiles existed
    keep resolving to byte-identical behaviour."""
    return ModelProfile(
        id="legacy-" + re.sub(r"[^a-z0-9._-]+", "-", base_model.id.lower()).strip("-")[:50],
        version=1,
        display_name=f"{base_model.id} (pre-profile recipe)",
        base=ProfileBase(
            hf_id=base_model.id,
            revision=base_model.revision if _SHA40.fullmatch(base_model.revision) else None,
            license=base_model.license,
            origin_country="CN" if base_model.id.lower().startswith("qwen/") else "ZZ",
            publisher=base_model.id.split("/")[0],
            architecture=base_model.architecture,
            chat_template_sha256=chat_template_sha256 if _SHA256.fullmatch(chat_template_sha256) else None,
        ),
        dims=dims,
        tools=QWEN3_HERMES_TOOLS,
        reasoning=Reasoning(template_kwarg="enable_thinking"),
        serving=ServingTraits(
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            lora_modules=lora_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            min_vram_gb_bf16=20.0,
        ),
        training=TrainingTraits(min_vram_gb_qlora=12.0),
        notes="Synthesized from a recipe without `profile`; Qwen3 hermes tool rendering.",
    )
