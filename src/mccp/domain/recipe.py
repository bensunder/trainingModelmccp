"""Admin-curated training recipes (spec FR-3). A job may override only the
hyperparameters a recipe allows, within the recipe's bounds."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .refs import RecipeRefField


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseModelRef(Strict):
    id: str = Field(description="Hugging Face model id, e.g. Qwen/Qwen3-8B")
    revision: str = Field(description="Pinned commit SHA")
    license: str = Field(description="SPDX id, checked by the governance gate")
    architecture: str = Field(description="Publish-guard allowlist key, e.g. qwen3")

    @field_validator("revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{7,64}", v):
            raise ValueError("must be a pinned commit SHA (hex, >= 7 chars)")
        return v

    @field_validator("id")
    @classmethod
    def _hf_id(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", v):
            raise ValueError("must look like org/name")
        return v


class ModelDims(Strict):
    parameters: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)


class SftHyperparameters(Strict):
    n_epochs: int = Field(ge=1, le=20)
    learning_rate: float = Field(gt=0, le=1e-2)
    lora_r: int = Field(ge=1, le=512)
    lora_alpha: int = Field(ge=1, le=1024)
    lora_dropout: float = Field(ge=0, le=0.5)
    per_device_batch_size: int = Field(ge=1, le=256)
    gradient_accumulation_steps: int = Field(ge=1, le=256)
    max_length: int = Field(ge=128, le=131072)
    warmup_ratio: float = Field(ge=0, le=0.5)

    @field_validator("lora_r")
    @classmethod
    def _pow2(cls, v: int) -> int:
        if v & (v - 1):
            raise ValueError("lora_r must be a power of two")
        return v


class Bounds(Strict):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> "Bounds":
        if self.min > self.max:
            raise ValueError("min must be <= max")
        return self


class SoupSettings(Strict):
    """Engine-specific knobs passed through to Soup's YAML (training engine)."""

    backend: Literal["transformers", "unsloth"] = "transformers"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    optimizer: str = "paged_adamw_8bit"
    scheduler: str = "cosine"
    stream_layers: bool = Field(default=False, description="Layer streaming for small GPUs; slower")
    save_steps: int = Field(default=100, ge=10)
    logging_steps: int = Field(default=10, ge=1)
    version: str = Field(description="Exact soup-cli version the trainer image pins")


class ServingProfile(Strict):
    serving_dtype: str = "bfloat16"
    tool_call_parser: str = "hermes"
    reasoning_parser: str | None = "qwen3"
    min_vllm_version: str


class Recipe(Strict):
    ref: RecipeRefField
    trainer_image: str = Field(description="Immutable trainer image reference (digest)")
    profile: str | None = Field(
        default=None,
        description="Model profile ref (`profiles/<id>@v<N>.yaml`). Omitted only by pre-profile recipes, "
        "which resolve to the Qwen3 hermes behaviour they were written for.",
    )
    base_model: BaseModelRef
    dims: ModelDims
    finetuning_types: list[Literal["qlora", "lora"]] = Field(min_length=1)
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    enable_thinking: bool = False
    tools_rendering: Literal["qwen3-hermes"] = "qwen3-hermes"
    chat_template_sha256: str = Field(description="SHA-256 of the tokenizer chat template; trainer verifies")
    engine: dict[str, str] = Field(default_factory=dict)
    soup: SoupSettings
    defaults: SftHyperparameters
    allowed_overrides: dict[str, Bounds] = Field(default_factory=dict)
    max_lora_rank: int = Field(ge=1, le=512)
    gpu_hourly_usd: float = Field(ge=0)
    serving: ServingProfile

    @model_validator(mode="after")
    def _consistent(self) -> "Recipe":
        unknown = set(self.allowed_overrides) - set(SftHyperparameters.model_fields)
        if unknown:
            raise ValueError(f"allowed_overrides has unknown fields: {sorted(unknown)}")
        if self.defaults.lora_r > self.max_lora_rank:
            raise ValueError("defaults.lora_r exceeds max_lora_rank")
        if self.soup.stream_layers:
            # Soup's layer streaming supports sft + quantization=none, batch 1,
            # no gradient accumulation only.
            if self.soup.quantization != "none" or "qlora" in self.finetuning_types:
                raise ValueError("stream_layers requires quantization=none and finetuning_types=['lora']")
            if self.defaults.per_device_batch_size != 1 or self.defaults.gradient_accumulation_steps != 1:
                raise ValueError("stream_layers requires per_device_batch_size=1 and gradient_accumulation_steps=1")
        if "qlora" in self.finetuning_types and self.soup.quantization != "4bit":
            raise ValueError("qlora recipes must use soup.quantization=4bit")
        return self

    def flop_estimate(self, trained_tokens: int) -> float:
        """6·N·D full-model estimate recorded for the EU AI Act modifier test."""
        return 6.0 * self.dims.parameters * trained_tokens
