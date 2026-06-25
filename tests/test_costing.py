from __future__ import annotations

from decimal import Decimal

from imdb_app.costing import (
    ModelPricing,
    build_model_usage_metadata,
    calculate_cost_usd,
    extract_token_usage,
    summarize_model_usage,
)
from imdb_app.models import ProductRecord


def test_extract_token_usage_supports_common_provider_shapes():
    assert extract_token_usage({"usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}) == {
        "is_estimated": False,
        "usage_available": True,
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    assert extract_token_usage({"usage": {"prompt_tokens": 11, "completion_tokens": 22}})["total_tokens"] == 33
    assert extract_token_usage({"usage": {"tokens": {"input_tokens": 12, "output_tokens": 24}}})["total_tokens"] == 36


def test_calculate_cost_uses_per_million_rates_and_cached_tokens():
    pricing = ModelPricing(
        input_per_million=Decimal("1.00"),
        cached_input_per_million=Decimal("0.10"),
        output_per_million=Decimal("2.00"),
    )
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 100_000, "output_tokens": 500_000}

    assert calculate_cost_usd(usage, pricing) == Decimal("1.910000")


def test_build_model_usage_metadata_marks_unknown_pricing_and_estimated_usage():
    metadata = build_model_usage_metadata(
        model_key="cohere-command-a-vision-07-2025",
        provider_label="Cohere",
        model_id="command-a-vision-07-2025",
        pricing=ModelPricing(note="No public pricing."),
        response={},
        image_count=2,
    )

    assert metadata["cost_available"] is False
    assert metadata["cost_is_estimated"] is True
    assert metadata["usage"]["total_tokens"] > 0
    assert metadata["pricing"]["note"] == "No public pricing."


def test_summarize_model_usage_aggregates_records():
    first = ProductRecord(
        id="one",
        metadata={
            "model_usage": {
                "request_count": 1,
                "image_count": 2,
                "model_id": "priced-model",
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                "cost_usd": "0.000500",
                "cost_available": True,
                "cost_is_estimated": False,
            }
        },
    )
    second = ProductRecord(
        id="two",
        metadata={
            "model_usage": {
                "request_count": 1,
                "image_count": 1,
                "model_id": "unknown-model",
                "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
                "cost_usd": None,
                "cost_available": False,
                "cost_is_estimated": True,
                "pricing": {"note": "Pricing unavailable."},
            }
        },
    )

    summary = summarize_model_usage([first, second])

    assert summary["request_count"] == 2
    assert summary["image_count"] == 3
    assert summary["total_tokens"] == 450
    assert summary["partial_cost_available"] is True
    assert summary["total_cost_usd"] == "0.000500"
    assert summary["estimated_count"] == 1
    assert summary["models"] == ["priced-model", "unknown-model"]
