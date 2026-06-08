"""Helpers for cleaning and normalizing extracted attributes."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

from .models import ProductRecord


COUNTRY_NORMALIZATION = {
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
}

PACKAGING_CANONICAL = {
    "box": {"box", "boxed", "carton"},
    "bottle": {"bottle", "bottled"},
    "jar": {"jar", "glass jar"},
    "bag": {"bag", "pouch", "sachet"},
}

WEIGHT_PATTERN = re.compile(r"(?P<quantity>\d+[\d\.,]*)\s*(?P<unit>kg|g|mg|lb|oz|ml|l)", re.IGNORECASE)


def normalize_country(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = value.strip().lower()
    return COUNTRY_NORMALIZATION.get(key, value.strip())


def normalize_packaging(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = value.strip().lower()
    for canonical, variants in PACKAGING_CANONICAL.items():
        if key in variants:
            return canonical.title()
    return value.strip().title()


def normalize_weight(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None

    match = WEIGHT_PATTERN.search(value)
    if not match:
        return None, None

    quantity = match.group("quantity").replace(",", "")
    unit = match.group("unit").lower()
    return quantity, unit


def barcode_is_valid(value: Optional[str]) -> bool:
    if not value:
        return False
    stripped = re.sub(r"[^0-9]", "", value)
    return len(stripped) in {8, 12, 13, 14}


def normalize_record(record: ProductRecord) -> None:
    if record.country_of_origin.value:
        record.country_of_origin.value = normalize_country(record.country_of_origin.value)

    if record.packaging_type.value:
        record.packaging_type.value = normalize_packaging(record.packaging_type.value)

    if record.weight_and_unit.value:
        quantity, unit = normalize_weight(record.weight_and_unit.value)
        if quantity and unit:
            record.metadata.setdefault("weight_parsed", {"quantity": quantity, "unit": unit})

    if record.barcode.value and not barcode_is_valid(record.barcode.value):
        record.barcode.notes = "Failed validation"


def export_lookup_tables() -> Dict[str, Iterable[str]]:
    return {
        "countries": COUNTRY_NORMALIZATION.values(),
        "packaging": PACKAGING_CANONICAL.keys(),
    }

