"""Dataset-aware pack, promotion, and weight parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


UNIT_PATTERN = r"KG|G|MG|ML|L|LB|OZ"
WEIGHT_PATTERN = re.compile(rf"(?P<quantity>\d+(?:[\.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})", re.IGNORECASE)
MULTIPACK_PATTERN = re.compile(
    rf"(?P<each>\d+(?:[\.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*[Xx×]\s*(?P<count>\d+)\s*(?:PCS?|PIECES?|SACHETS?|BAGS?)?",
    re.IGNORECASE,
)
PROMO_PATTERN = re.compile(r"(?P<paid>\d+)\s*\+\s*(?P<free>\d+)\s*FREE", re.IGNORECASE)
PCS_WEIGHT_PATTERN = re.compile(
    rf"(?P<count>\d+)\s*PCS?\s*(?P<each>\d+(?:[\.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PackParseResult:
    raw_text: str
    normalized_weight: str | None = None
    quantity: str | None = None
    unit: str | None = None
    pack_count: int | None = None
    promotion: str | None = None
    addons: str | None = None
    notes: str | None = None


def _compact_decimal(value: str | Decimal) -> str:
    try:
        decimal = Decimal(str(value).replace(",", "."))
    except InvalidOperation:
        return str(value).replace(",", "")

    if decimal == decimal.to_integral():
        return str(decimal.quantize(Decimal("1")))

    quantized = decimal.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP).normalize()
    return format(quantized, "f").rstrip("0").rstrip(".")


def _weight(quantity: str | Decimal, unit: str) -> str:
    return f"{_compact_decimal(quantity)}{unit.upper()}"


def parse_pack_text(*values: object | None) -> PackParseResult:
    raw_text = " ".join(str(value) for value in values if value)
    text = re.sub(r"\s+", " ", raw_text.upper()).strip()
    if not text:
        return PackParseResult(raw_text="")

    multipack = MULTIPACK_PATTERN.search(text)
    if multipack:
        quantity = _compact_decimal(multipack.group("each"))
        unit = multipack.group("unit").upper()
        count = int(multipack.group("count"))
        return PackParseResult(
            raw_text=text,
            normalized_weight=_weight(quantity, unit),
            quantity=quantity,
            unit=unit,
            pack_count=count,
            addons=f"{count}PCS {_weight(quantity, unit)}",
            notes="Parsed multi-pack weight syntax",
        )

    pcs_weight = PCS_WEIGHT_PATTERN.search(text)
    if pcs_weight:
        quantity = _compact_decimal(pcs_weight.group("each"))
        unit = pcs_weight.group("unit").upper()
        count = int(pcs_weight.group("count"))
        return PackParseResult(
            raw_text=text,
            normalized_weight=_weight(quantity, unit),
            quantity=quantity,
            unit=unit,
            pack_count=count,
            addons=f"{count}PCS {_weight(quantity, unit)}",
            notes="Parsed piece-count weight syntax",
        )

    promo = PROMO_PATTERN.search(text)
    if promo:
        paid = int(promo.group("paid"))
        free = int(promo.group("free"))
        total_count = paid + free
        weight_match = WEIGHT_PATTERN.search(text)
        quantity = None
        unit = None
        normalized_weight = None
        if weight_match:
            unit = weight_match.group("unit").upper()
            try:
                total_weight = Decimal(weight_match.group("quantity").replace(",", "."))
                quantity_decimal = total_weight / Decimal(total_count)
                quantity = _compact_decimal(quantity_decimal)
                normalized_weight = _weight(quantity, unit)
            except (InvalidOperation, ZeroDivisionError):
                quantity = _compact_decimal(weight_match.group("quantity"))
                normalized_weight = _weight(quantity, unit)

        addons = "ENVELOPE" if "ENVELOPE" in text else None
        return PackParseResult(
            raw_text=text,
            normalized_weight=normalized_weight,
            quantity=quantity,
            unit=unit,
            pack_count=total_count,
            promotion=f"{free} FREE",
            addons=addons,
            notes="Parsed promotional pack syntax",
        )

    weight_match = WEIGHT_PATTERN.search(text)
    if not weight_match:
        return PackParseResult(raw_text=text)

    quantity = _compact_decimal(weight_match.group("quantity"))
    unit = weight_match.group("unit").upper()
    return PackParseResult(
        raw_text=text,
        normalized_weight=_weight(quantity, unit),
        quantity=quantity,
        unit=unit,
        notes="Parsed single weight",
    )
