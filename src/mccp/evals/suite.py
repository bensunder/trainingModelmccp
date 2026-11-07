"""Evaluation suite definition (versioned, stored per project)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mccp.domain.refs import DatasetRefField, Split


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    seed: int = 0


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(description="Judge model id; use a non-Qwen family to avoid self-preference bias")
    rubric: str = Field(min_length=20)
    max_items: int = Field(default=100, ge=1)
    calibration: list[dict[str, str]] = Field(
        default_factory=list, description="Human-labelled pairs {prompt, a, b, human: A|B|tie} for judge-human kappa"
    )


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    guard_model: str = "Qwen/Qwen3Guard-Gen-4B"
    guard_base_url: str | None = Field(default=None, description="Defaults to the serving vLLM")
    harmful_prompts: list[str] | None = None
    benign_prompts: list[str] | None = None
    pii_probes: list[str] | None = None


class RegressionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: list[str] = Field(
        min_length=1,
        description="Argv run per model; placeholders {base_url} {model} {output}. Must write {task: score} JSON to {output}",
    )
    tasks: list[str] = Field(min_length=1)
    timeout_s: int = Field(default=7200, ge=60)


class PerformanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: int = Field(default=16, ge=1, le=1000)
    concurrency: int = Field(default=4, ge=1, le=64)


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: int = Field(ge=1)
    golden: DatasetRefField
    output_schema: dict[str, Any]
    compare_fields: list[str] = Field(min_length=1)
    max_items: int = Field(default=500, ge=1)
    effect_probes: int = Field(default=5, ge=1, le=50)
    sampling: Sampling = Sampling()
    judge: JudgeConfig | None = None
    safety: SafetyConfig = SafetyConfig()
    regression: RegressionConfig | None = None
    performance: PerformanceConfig = PerformanceConfig()
    # Extract-then-rules tasks: the model returns facts, this policy turns them into
    # the decision, and decision accuracy is scored on the policy's output.
    decision_rules: str | None = None
    # Deterministic canonicalization applied to the parsed output before the schema
    # check (e.g. id formats). Raw format validity is still reported separately.
    output_normalizer: str | None = None
    # exact: every compare field must equal the label. normalized (real documents):
    # ids ignore case/spaces/punctuation, amounts match to the cent, and a field the
    # label leaves null is not scored (unlabelled is not the same as absent).
    field_match: Literal["exact", "normalized"] = "exact"

    @model_validator(mode="after")
    def _known_rules(self) -> "EvalSuite":
        from mccp.rules import NORMALIZERS, RULESETS

        if self.decision_rules is not None and self.decision_rules not in RULESETS:
            raise ValueError(f"unknown decision_rules {self.decision_rules!r} (known: {', '.join(RULESETS)})")
        if self.output_normalizer is not None and self.output_normalizer not in NORMALIZERS:
            raise ValueError(f"unknown output_normalizer {self.output_normalizer!r} (known: {', '.join(NORMALIZERS)})")
        return self

    @model_validator(mode="after")
    def _golden_is_test(self) -> "EvalSuite":
        if self.golden.split is not Split.TEST:
            raise ValueError("golden must reference the frozen `test` split")
        return self
