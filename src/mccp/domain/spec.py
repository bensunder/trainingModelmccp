"""Customization job request (OpenAI/NeMo-shaped, spec D4) and its resolution
against a recipe into the immutable, hashable run configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import MethodNotEnabled, ValidationFailed
from .hashing import digest_of
from .recipe import BaseModelRef, Recipe, SftHyperparameters, SoupSettings
from .refs import DatasetRefField, RecipeRefField, Split, SuiteRefField, is_slug

RiskTier = Literal["low", "medium", "high"]


class SftMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class Method(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sft", "dpo", "grpo"]
    sft: SftMethod | None = None
    # Reserved for Phase 2. Accepted by the schema so clients get a precise
    # "not enabled" error instead of a generic validation failure.
    dpo: dict[str, Any] | None = None
    grpo: dict[str, Any] | None = None


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    recipe: RecipeRefField
    method: Method
    finetuning_type: Literal["qlora", "lora"]
    training_dataset: DatasetRefField
    validation_dataset: DatasetRefField
    eval_suite: SuiteRefField
    output_adapter: str
    risk_tier: RiskTier
    intended_use: str = Field(min_length=10, max_length=2000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("project", "output_adapter")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not is_slug(v):
            raise ValueError("must be a lowercase slug")
        return v

    @field_validator("metadata")
    @classmethod
    def _meta(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 16:
            raise ValueError("at most 16 keys")
        for k, val in v.items():
            if not k or len(k) > 64 or len(val) > 512:
                raise ValueError("keys <= 64 chars, values <= 512 chars")
        return v

    @model_validator(mode="after")
    def _structure(self) -> "JobRequest":
        for field, ref, want in (
            ("training_dataset", self.training_dataset, Split.TRAIN),
            ("validation_dataset", self.validation_dataset, Split.VAL),
        ):
            if ref.project != self.project:
                raise ValueError(f"{field} must belong to project `{self.project}`")
            if ref.split is not want:
                raise ValueError(f"{field} must reference the `{want.value}` split")
        if self.training_dataset.version_ref != self.validation_dataset.version_ref:
            raise ValueError("validation_dataset must be the same dataset version as training_dataset")
        return self

    def check_method_enabled(self, enabled: frozenset[str] = frozenset({"sft"})) -> None:
        if self.method.type not in enabled:
            raise MethodNotEnabled(self.method.type)
        if self.method.type == "sft" and (self.method.dpo is not None or self.method.grpo is not None):
            raise ValidationFailed("method", "only the `sft` section is allowed for type sft")

    def resolve(self, recipe: Recipe) -> "ResolvedConfig":
        self.check_method_enabled()
        if recipe.ref != self.recipe:
            raise ValidationFailed("recipe", "recipe mismatch")
        if self.finetuning_type not in recipe.finetuning_types:
            raise ValidationFailed("finetuning_type", f"recipe {recipe.ref} does not allow `{self.finetuning_type}`")
        overrides = self.method.sft.hyperparameters if self.method.sft else {}
        merged = recipe.defaults.model_dump()
        for key, value in overrides.items():
            bounds = recipe.allowed_overrides.get(key)
            if bounds is None:
                raise ValidationFailed(f"hyperparameters.{key}", "override not allowed by recipe")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationFailed(f"hyperparameters.{key}", "must be a number")
            if not (bounds.min <= value <= bounds.max):
                raise ValidationFailed(f"hyperparameters.{key}", f"must be within [{bounds.min}, {bounds.max}]")
            merged[key] = value
        try:
            hp = SftHyperparameters.model_validate(merged)
        except ValidationError as e:
            err = e.errors()[0]
            raise ValidationFailed("hyperparameters." + ".".join(map(str, err["loc"])), err["msg"]) from e
        if hp.lora_r > recipe.max_lora_rank:
            raise ValidationFailed("hyperparameters.lora_r", f"exceeds recipe max_lora_rank {recipe.max_lora_rank}")
        return ResolvedConfig(
            recipe=self.recipe,
            trainer_image=recipe.trainer_image,
            base_model=recipe.base_model,
            finetuning_type=self.finetuning_type,
            method=self.method.type,
            hyperparameters=hp,
            target_modules=recipe.target_modules,
            enable_thinking=recipe.enable_thinking,
            tools_rendering=recipe.tools_rendering,
            chat_template_sha256=recipe.chat_template_sha256,
            engine=recipe.engine,
            soup=recipe.soup,
            training_dataset=self.training_dataset,
            validation_dataset=self.validation_dataset,
            seed=self.seed,
        )


class ResolvedConfig(BaseModel):
    """Immutable run config; its digest is the job's `config_hash`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe: RecipeRefField
    trainer_image: str
    base_model: BaseModelRef
    finetuning_type: Literal["qlora", "lora"]
    method: Literal["sft", "dpo", "grpo"]
    hyperparameters: SftHyperparameters
    target_modules: list[str] | Literal["all-linear"]
    enable_thinking: bool
    tools_rendering: str
    chat_template_sha256: str
    engine: dict[str, str]
    soup: SoupSettings
    training_dataset: DatasetRefField
    validation_dataset: DatasetRefField
    seed: int

    def config_hash(self) -> str:
        return digest_of(self.model_dump(mode="json"))
