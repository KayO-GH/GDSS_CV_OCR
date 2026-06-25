"""Catalog of supported extraction models and runtime selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .config import settings
from .costing import ModelPricing


BackendKind = Literal["cohere_direct", "openai_direct", "hf_router"]
CredentialKind = Literal["cohere_api_key", "OPENAI_KEY", "hf_token"]


@dataclass(frozen=True)
class ModelProfile:
    """Description of a selectable extraction model."""

    key: str
    label: str
    backend: BackendKind
    model_id: str
    credential_kind: CredentialKind
    visible_when_unconfigured: bool = False
    pricing: ModelPricing = field(default_factory=ModelPricing)

    @property
    def provider_label(self) -> str:
        if self.backend == "cohere_direct":
            return "Cohere"
        if self.backend == "openai_direct":
            return "OpenAI"
        return "Hugging Face"

    @property
    def credential_value(self) -> str | None:
        return getattr(settings, self.credential_kind)

    @property
    def is_available(self) -> bool:
        return bool(self.credential_value)

    @property
    def status_text(self) -> str:
        return "key detected" if self.is_available else "missing key"

    @property
    def pricing_status_text(self) -> str:
        return "pricing available" if self.pricing.is_known else "pricing unavailable"


SUPPORTED_MODELS: tuple[ModelProfile, ...] = (
    ModelProfile(
        key="hf-glm-4-6v-flash",
        label="Hugging Face GLM-4.6V-Flash",
        backend="hf_router",
        model_id="zai-org/GLM-4.6V-Flash",
        credential_kind="hf_token",
        pricing=ModelPricing(
            source_label="Hugging Face Inference Providers pricing",
            source_url="https://huggingface.co/docs/inference-providers/pricing",
            note="Hugging Face router costs vary by provider; no stable per-token rate is embedded for this model.",
        ),
    ),
    ModelProfile(
        key="hf-qwen3-vl-235b-a22b-instruct",
        label="Hugging Face Qwen3-VL-235B-A22B-Instruct",
        backend="hf_router",
        model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
        credential_kind="hf_token",
        pricing=ModelPricing(
            source_label="Hugging Face Inference Providers pricing",
            source_url="https://huggingface.co/docs/inference-providers/pricing",
            note="Hugging Face router costs vary by provider; no stable per-token rate is embedded for this model.",
        ),
    ),
    ModelProfile(
        key="cohere-command-a-vision-07-2025",
        label="Cohere command-a-vision-07-2025",
        backend="cohere_direct",
        model_id="command-a-vision-07-2025",
        credential_kind="cohere_api_key",
        visible_when_unconfigured=True,
        pricing=ModelPricing(
            source_label="Cohere pricing",
            source_url="https://cohere.com/pricing",
            note="Cohere public pricing did not list command-a-vision-07-2025 token rates when checked.",
        ),
    ),
    ModelProfile(
        key="openai-gpt-4o-mini",
        label="OpenAI gpt-4o-mini",
        backend="openai_direct",
        model_id="gpt-4o-mini",
        credential_kind="OPENAI_KEY",
        pricing=ModelPricing(
            source_label="OpenAI API pricing",
            source_url="https://openai.com/api/pricing/",
            note="OpenAI API pricing did not expose a stable text-token rate for gpt-4o-mini when checked.",
        ),
    ),
)

MODEL_BY_KEY = {profile.key: profile for profile in SUPPORTED_MODELS}

LEGACY_PROVIDER_DEFAULTS = {
    "cohere": "cohere-command-a-vision-07-2025",
    "openai": "openai-gpt-4o-mini",
}


def get_model_profile(model_key: str | None) -> ModelProfile:
    """Return the configured profile for a key, with legacy-provider fallback."""

    candidate = (model_key or "").strip()
    if candidate in MODEL_BY_KEY:
        return MODEL_BY_KEY[candidate]

    legacy_key = LEGACY_PROVIDER_DEFAULTS.get(candidate.lower())
    if legacy_key:
        return MODEL_BY_KEY[legacy_key]

    if not candidate:
        return MODEL_BY_KEY[resolve_default_model_key()]

    msg = f"Unsupported model key: {model_key}"
    raise ValueError(msg)


def get_model_profile_for_backend_model(backend: BackendKind, model_id: str) -> ModelProfile | None:
    for profile in SUPPORTED_MODELS:
        if profile.backend == backend and profile.model_id == model_id:
            return profile
    return None


def available_model_profiles(*, include_unavailable_visible: bool = False) -> list[ModelProfile]:
    """Return models that should be shown in the UI."""

    profiles: list[ModelProfile] = []
    for profile in SUPPORTED_MODELS:
        if profile.is_available:
            profiles.append(profile)
        elif include_unavailable_visible and profile.visible_when_unconfigured:
            profiles.append(profile)
    return profiles


def resolve_default_model_key() -> str:
    """Pick the UI default, honoring explicit config when valid."""

    configured = (settings.default_model_key or "").strip()
    if configured in MODEL_BY_KEY:
        return configured

    legacy_provider = (settings.vlm_provider or "").strip().lower()
    if legacy_provider in LEGACY_PROVIDER_DEFAULTS:
        return LEGACY_PROVIDER_DEFAULTS[legacy_provider]

    return "cohere-command-a-vision-07-2025"


def selected_or_first_available(model_key: str | None) -> ModelProfile:
    """Resolve a selected profile and fall back to the first visible option."""

    available = available_model_profiles(include_unavailable_visible=True)
    if not available:
        return MODEL_BY_KEY["cohere-command-a-vision-07-2025"]

    try:
        selected = get_model_profile(model_key)
    except ValueError:
        selected = MODEL_BY_KEY[resolve_default_model_key()]

    if selected.is_available or selected.visible_when_unconfigured:
        return selected
    return available[0]
