"""Route catalog: providers, models, prices and route profiles.

One YAML file (`route.yaml`) describes everything the gateway may call and how it
chooses. It is validated as a whole at load time: a profile that names an unknown
model, a model on an unknown provider, or a model with no wire format its provider
serves is a load error, never a runtime surprise.

Wire formats are the API dialects a client speaks and a provider accepts:
`anthropic` (Messages API: Claude Code) and `openai_chat` (Chat Completions:
OpenCode and most SDKs). A request is only sent to a model whose provider accepts
the request's own wire format, so requests are forwarded as-is, never translated
(translation is where gateways lose tool calls and cache hints).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Wire(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai_chat"


class TaskClass(StrEnum):
    """What a coding request is doing. Routing evidence is kept per class."""

    BACKGROUND = "background"  # titles, summaries, short side requests a harness sends
    QUICK_EDIT = "quick_edit"  # small, single-shot code change or completion
    EXPLAIN = "explain"  # questions about code
    TESTS = "tests"  # write or fix tests
    REFACTOR = "refactor"
    DEBUG = "debug"  # errors, tracebacks, failing behaviour
    AGENTIC = "agentic"  # multi-turn tool-using sessions


class Strategy(StrEnum):
    COST = "cost"  # cheapest model that meets the quality bar
    BALANCED = "balanced"  # cheapest model within `balanced_margin` of the best
    INTELLIGENCE = "intelligence"  # highest measured quality, cost as tie-break


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provider(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,40}$")
    name: str
    # Base URL per wire format. The gateway appends `/v1/messages` to an anthropic
    # base, and `/chat/completions` to an openai_chat base, which therefore carries
    # its own version prefix (`https://api.deepseek.com/v1`, `https://api.z.ai/api/paas/v4`).
    base_urls: dict[Wire, str]
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    auth: Literal["bearer", "x-api-key"] = "bearer"
    timeout_s: float = Field(default=600.0, gt=0, le=3600)
    connect_timeout_s: float = Field(default=10.0, gt=0, le=120)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_urls")
    @classmethod
    def _https(cls, v: dict[Wire, str]) -> dict[Wire, str]:
        if not v:
            raise ValueError("a provider needs at least one base URL")
        for url in v.values():
            if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
                raise ValueError(f"provider URL must be https (got {url})")
        return {k: u.rstrip("/") for k, u in v.items()}


class Price(_Strict):
    """USD per 1M tokens. Cache prices default to the input price when unknown
    (so savings are never overstated)."""

    input_per_m: float = Field(ge=0)
    output_per_m: float = Field(ge=0)
    cache_read_per_m: float | None = Field(default=None, ge=0)
    cache_write_per_m: float | None = Field(default=None, ge=0)

    def cost(self, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
        cr = self.input_per_m if self.cache_read_per_m is None else self.cache_read_per_m
        cw = self.input_per_m if self.cache_write_per_m is None else self.cache_write_per_m
        return (
            input_tokens * self.input_per_m + output_tokens * self.output_per_m + cache_read * cr + cache_write * cw
        ) / 1_000_000


class Model(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")  # what clients and admins see
    provider: str
    upstream: str  # the model name the provider expects
    wires: list[Wire]
    price: Price
    price_source: str  # where the price came from and when (prices change; this is audited)
    context_window: int = Field(gt=0)
    supports_tools: bool = True
    tier: Literal["frontier", "strong", "fast"]


class Profile(_Strict):
    """A routing policy clients select by model name (e.g. `auto-cost`)."""

    id: str = Field(pattern=r"^auto(-[a-z0-9]+)*$")
    name: str
    strategy: Strategy
    candidates: list[str] = Field(min_length=1)  # models this profile may route to
    fallback: str  # used when no candidate has evidence meeting the bar ("no evidence, no downgrade")
    background_model: str | None = None  # cheap model for harness side requests
    escalation: list[str] = Field(default_factory=list)  # tried in order when the chosen model fails
    quality_bar: float = Field(default=0.85, ge=0, le=1)  # min Wilson lower bound of the pass rate
    quality_bar_by_class: dict[TaskClass, float] = Field(default_factory=dict)
    balanced_margin: float = Field(default=0.03, ge=0, le=0.5)
    sticky_ttl_s: int = Field(default=3600, ge=60, le=86400)

    def bar(self, task: TaskClass) -> float:
        return self.quality_bar_by_class.get(task, self.quality_bar)


class Catalog(_Strict):
    version: str
    baseline_model: str  # "what this traffic would have cost" for savings reporting
    providers: list[Provider]
    models: list[Model]
    profiles: list[Profile]
    default_profile: str

    @model_validator(mode="after")
    def _references(self) -> Catalog:
        prov = {p.id: p for p in self.providers}
        if len(prov) != len(self.providers):
            raise ValueError("duplicate provider id")
        mods = {m.id: m for m in self.models}
        if len(mods) != len(self.models):
            raise ValueError("duplicate model id")
        for m in self.models:
            if m.provider not in prov:
                raise ValueError(f"model {m.id}: unknown provider {m.provider}")
            missing = set(m.wires) - set(prov[m.provider].base_urls)
            if missing:
                raise ValueError(f"model {m.id}: provider {m.provider} serves no {sorted(missing)} endpoint")
        if self.baseline_model not in mods:
            raise ValueError(f"baseline_model {self.baseline_model} is not a model")
        ids = [p.id for p in self.profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate profile id")
        if self.default_profile not in ids:
            raise ValueError(f"default_profile {self.default_profile} is not a profile")
        for p in self.profiles:
            for ref in [
                *p.candidates,
                p.fallback,
                *p.escalation,
                *([p.background_model] if p.background_model else []),
            ]:
                if ref not in mods:
                    raise ValueError(f"profile {p.id}: unknown model {ref}")
            if p.fallback not in p.candidates:
                raise ValueError(f"profile {p.id}: fallback {p.fallback} must be one of its candidates")
        if set(ids) & set(mods):
            raise ValueError("profile ids and model ids must not overlap")
        return self

    def model(self, model_id: str) -> Model:
        return next(m for m in self.models if m.id == model_id)

    def provider(self, provider_id: str) -> Provider:
        return next(p for p in self.providers if p.id == provider_id)

    def profile(self, profile_id: str) -> Profile | None:
        return next((p for p in self.profiles if p.id == profile_id), None)

    def has_model(self, model_id: str) -> bool:
        return any(m.id == model_id for m in self.models)


UNVERIFIED = "UNVERIFIED"


def load_catalog(path: str | Path, production: bool = False) -> Catalog:
    """Load and validate. In production every price must name a verified source:
    savings and budgets are computed from these numbers."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cat = Catalog.model_validate(data)
    if production:
        bad = [m.id for m in cat.models if m.price_source.upper().startswith(UNVERIFIED)]
        if bad:
            raise ValueError(f"unverified prices for {bad}: set price and price_source before production use")
    return cat
