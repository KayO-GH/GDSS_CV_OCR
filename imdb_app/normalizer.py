"""Helpers for cleaning and normalizing extracted hackathon attributes."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

from .models import IMDB_ATTRIBUTES, ProductRecord


COUNTRY_NORMALIZATION = {
    "ghana": "GHANA",
    "gh": "GHANA",
    "china": "CHINA",
    "prc": "CHINA",
    "nigeria": "NIGERIA",
    "indonesia": "INDONESIA",
    "cote d'ivoire": "COTE D'IVOIRE",
    "côte d'ivoire": "COTE D'IVOIRE",
    "ivory coast": "COTE D'IVOIRE",
    "vietnam": "VIETNAM",
    "viet nam": "VIETNAM",
    "sri lanka": "SRI LANKA",
    "india": "INDIA",
}

PACKAGING_CANONICAL = {
    "BOX": {"box", "boxed", "carton", "cardboard", "cardboard box", "crtn", "cardbrd"},
    "PLASTIC BOTTLE": {"plastic bottle", "plastic bttl", "plastic bttle", "bottle plastic", "bttl plst", "bttl", "pb"},
    "BOTTLE": {"bottle", "bottled"},
    "GLASS JAR": {"glass jar", "glass tub bottle", "glass tub", "glss jar", "jar"},
    "SACHET": {"sachet", "plstc sachet", "plst sachet", "envelope", "wrapper sachet"},
    "TIN": {"tin", "canister"},
    "TUB": {"tub", "plastic tub"},
    "TETRA PAK": {"tetra pak", "tetrapak", "carton drink"},
    "CAN": {"can"},
    "PLASTIC BAG": {"plastic bag", "bag"},
    "WRAPPED": {"wrapped", "wrap"},
    "POUCH": {"pouch"},
}

WEIGHT_PATTERN = re.compile(r"(?P<quantity>\d+(?:[\.,]\d+)?)\s*(?P<unit>kg|g|mg|lb|oz|ml|l)", re.IGNORECASE)


def normalize_text(value: Optional[str], *, uppercase: bool = True) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned:
        return None
    return cleaned.upper() if uppercase else cleaned


def normalize_country(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = normalize_text(value)
    if not key:
        return None
    return COUNTRY_NORMALIZATION.get(key.lower(), key)


def normalize_packaging(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = normalize_text(value)
    if not key:
        return None
    for canonical, variants in PACKAGING_CANONICAL.items():
        if key.lower() in variants or key == canonical:
            return canonical
    return key


def normalize_weight(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None

    match = WEIGHT_PATTERN.search(value)
    if not match:
        return None, None

    quantity = match.group("quantity").replace(",", "")
    unit = match.group("unit").upper()
    return quantity, unit


def normalize_weight_value(value: Optional[str]) -> Optional[str]:
    quantity, unit = normalize_weight(value)
    if not quantity or not unit:
        return normalize_text(value)
    compact_quantity = quantity.rstrip("0").rstrip(".") if "." in quantity else quantity
    return f"{compact_quantity}{unit}"


def normalize_barcode(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    stripped = re.sub(r"[^0-9]", "", str(value))
    return stripped or None


def barcode_is_valid(value: Optional[str]) -> bool:
    stripped = normalize_barcode(value)
    if not stripped:
        return False
    return len(stripped) in {8, 12, 13, 14}


def normalize_record(record: ProductRecord) -> None:
    for attr in IMDB_ATTRIBUTES:
        attribute = getattr(record, attr)
        if attr not in {"barcode", "weight", "packaging_type", "country"}:
            attribute.value = normalize_text(attribute.value)

    record.barcode.value = normalize_barcode(record.barcode.value)
    record.weight.value = normalize_weight_value(record.weight.value)
    record.packaging_type.value = normalize_packaging(record.packaging_type.value)
    record.country.value = normalize_country(record.country.value)

    if record.weight.value:
        quantity, unit = normalize_weight(record.weight.value)
        if quantity and unit:
            record.metadata.setdefault("weight_parsed", {"quantity": quantity, "unit": unit})

    if record.barcode.value and not barcode_is_valid(record.barcode.value):
        record.barcode.notes = "Failed validation"


def export_lookup_tables() -> Dict[str, Iterable[str]]:
    return {
        "countries": COUNTRY_NORMALIZATION.values(),
        "packaging": PACKAGING_CANONICAL.keys(),
    }
