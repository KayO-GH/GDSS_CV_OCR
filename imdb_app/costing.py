"""Token usage and model cost helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: Decimal | None = None
    output_per_million: Decimal | None = None
    cached_input_per_million: Decimal | None = None
    source_label: str = "Pricing unavailable"
    source_url: str | None = None
    checked_on: str = "2026-06-25"
    note: str = "No reliable public token pricing found for this configured model."

    @property
    def is_known(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None


def _decimal(value: object | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _int_value(*values: object | None) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_token_usage(response: dict[str, Any]) -> dict[str, int | bool]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"is_estimated": False, "usage_available": False}

    tokens = usage.get("tokens") if isinstance(usage.get("tokens"), dict) else {}
    input_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}

    input_tokens = _int_value(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
        tokens.get("input_tokens"),
        tokens.get("prompt_tokens"),
    )
    output_tokens = _int_value(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        tokens.get("output_tokens"),
        tokens.get("completion_tokens"),
    )
    total_tokens = _int_value(
        usage.get("total_tokens"),
        tokens.get("total_tokens"),
        (input_tokens or 0) + (output_tokens or 0) if input_tokens is not None or output_tokens is not None else None,
    )
    cached_input_tokens = _int_value(
        usage.get("cached_input_tokens"),
        input_details.get("cached_tokens"),
        prompt_details.get("cached_tokens"),
    )

    normalized: dict[str, int | bool] = {
        "is_estimated": False,
        "usage_available": any(value is not None for value in [input_tokens, output_tokens, total_tokens, cached_input_tokens]),
    }
    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    if cached_input_tokens is not None:
        normalized["cached_input_tokens"] = cached_input_tokens
    return normalized


def estimate_token_usage(*, image_count: int) -> dict[str, int | bool | str]:
    image_count = max(image_count, 1)
    input_tokens = 800 + (300 * image_count)
    output_tokens = 700
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "is_estimated": True,
        "usage_available": False,
        "estimate_note": "Heuristic estimate used because provider usage was unavailable.",
    }


def calculate_cost_usd(usage: dict[str, int | bool | str], pricing: ModelPricing) -> Decimal | None:
    if not pricing.is_known:
        return None

    input_tokens = _decimal(usage.get("input_tokens"))
    output_tokens = _decimal(usage.get("output_tokens"))
    cached_input_tokens = _decimal(usage.get("cached_input_tokens"))
    input_billable = input_tokens or Decimal("0")
    if cached_input_tokens and pricing.cached_input_per_million is not None:
        input_billable = max(input_billable - cached_input_tokens, Decimal("0"))
        cached_cost = cached_input_tokens * pricing.cached_input_per_million / Decimal("1000000")
    else:
        cached_cost = Decimal("0")

    input_cost = input_billable * (pricing.input_per_million or Decimal("0")) / Decimal("1000000")
    output_cost = (output_tokens or Decimal("0")) * (pricing.output_per_million or Decimal("0")) / Decimal("1000000")
    return (input_cost + cached_cost + output_cost).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def build_model_usage_metadata(
    *,
    model_key: str,
    provider_label: str,
    model_id: str,
    pricing: ModelPricing,
    response: dict[str, Any],
    image_count: int,
) -> dict[str, Any]:
    usage = extract_token_usage(response)
    if not usage.get("usage_available"):
        usage = estimate_token_usage(image_count=image_count)
    cost = calculate_cost_usd(usage, pricing)
    return {
        "model_key": model_key,
        "provider": provider_label,
        "model_id": model_id,
        "request_count": 1,
        "image_count": image_count,
        "usage": usage,
        "pricing": {
            "is_known": pricing.is_known,
            "input_per_million": str(pricing.input_per_million) if pricing.input_per_million is not None else None,
            "cached_input_per_million": str(pricing.cached_input_per_million) if pricing.cached_input_per_million is not None else None,
            "output_per_million": str(pricing.output_per_million) if pricing.output_per_million is not None else None,
            "source_label": pricing.source_label,
            "source_url": pricing.source_url,
            "checked_on": pricing.checked_on,
            "note": pricing.note,
        },
        "cost_usd": str(cost) if cost is not None else None,
        "cost_available": cost is not None,
        "cost_is_estimated": bool(usage.get("is_estimated")),
    }


def summarize_model_usage(records: Iterable[Any]) -> dict[str, Any]:
    summary = {
        "request_count": 0,
        "image_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "known_cost_count": 0,
        "unknown_cost_count": 0,
        "estimated_count": 0,
        "total_cost_usd": Decimal("0"),
        "models": set(),
        "pricing_notes": set(),
    }
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        usage_metadata = metadata.get("model_usage")
        if not isinstance(usage_metadata, dict):
            continue
        summary["request_count"] += int(usage_metadata.get("request_count") or 0)
        summary["image_count"] += int(usage_metadata.get("image_count") or 0)
        if usage_metadata.get("model_id"):
            summary["models"].add(str(usage_metadata["model_id"]))
        usage = usage_metadata.get("usage") if isinstance(usage_metadata.get("usage"), dict) else {}
        summary["input_tokens"] += int(usage.get("input_tokens") or 0)
        summary["output_tokens"] += int(usage.get("output_tokens") or 0)
        summary["total_tokens"] += int(usage.get("total_tokens") or 0)
        if usage_metadata.get("cost_is_estimated"):
            summary["estimated_count"] += 1
        cost = _decimal(usage_metadata.get("cost_usd"))
        if cost is not None:
            summary["known_cost_count"] += 1
            summary["total_cost_usd"] += cost
        else:
            summary["unknown_cost_count"] += 1
            pricing = usage_metadata.get("pricing") if isinstance(usage_metadata.get("pricing"), dict) else {}
            if pricing.get("note"):
                summary["pricing_notes"].add(str(pricing["note"]))

    total_cost = summary["total_cost_usd"].quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    request_count = summary["request_count"]
    return {
        **summary,
        "models": sorted(summary["models"]),
        "pricing_notes": sorted(summary["pricing_notes"]),
        "total_cost_usd": str(total_cost),
        "cost_available": summary["known_cost_count"] > 0 and summary["unknown_cost_count"] == 0,
        "partial_cost_available": summary["known_cost_count"] > 0 and summary["unknown_cost_count"] > 0,
        "cost_per_request_usd": str((total_cost / Decimal(request_count)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        if request_count
        else None,
    }


def format_cost_usd(value: object | None) -> str:
    decimal_value = _decimal(value)
    if decimal_value is None:
        return "Pricing unavailable"
    return f"${decimal_value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)}"
