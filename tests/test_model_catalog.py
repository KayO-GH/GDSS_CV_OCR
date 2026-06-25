from __future__ import annotations

from decimal import Decimal

from imdb_app.model_catalog import available_model_profiles, get_model_profile, resolve_default_model_key


def test_default_model_key_prefers_command_a_vision(monkeypatch):
    monkeypatch.setattr("imdb_app.model_catalog.settings.default_model_key", None)
    monkeypatch.setattr("imdb_app.model_catalog.settings.vlm_provider", "cohere")

    assert resolve_default_model_key() == "cohere-command-a-vision-07-2025"


def test_available_model_profiles_hide_unconfigured_models(monkeypatch):
    monkeypatch.setattr("imdb_app.model_catalog.settings.cohere_api_key", "cohere-key")
    monkeypatch.setattr("imdb_app.model_catalog.settings.hf_token", None)
    monkeypatch.setattr("imdb_app.model_catalog.settings.OPENAI_KEY", None)

    profiles = available_model_profiles()

    assert [profile.key for profile in profiles] == ["cohere-command-a-vision-07-2025"]


def test_model_profile_lookup_resolves_hf_router_model():
    profile = get_model_profile("hf-qwen3-vl-235b-a22b-instruct")

    assert profile.backend == "hf_router"
    assert profile.model_id == "Qwen/Qwen3-VL-235B-A22B-Instruct"
    assert profile.credential_kind == "hf_token"
    assert profile.pricing.is_known is True
    assert profile.pricing.input_per_million == Decimal("0.30")
    assert profile.pricing.output_per_million == Decimal("1.50")
    assert profile.pricing.source_url == "https://huggingface.co/inference/models"


def test_model_profiles_expose_pricing_status():
    profile = get_model_profile("hf-glm-4-6v-flash")

    assert profile.pricing.is_known is True
    assert profile.pricing_status_text == "pricing available"
    assert profile.pricing.input_per_million == Decimal("0.30")
    assert profile.pricing.output_per_million == Decimal("0.90")
    assert "Hugging Face Inference Providers supported models" in profile.pricing.source_label
