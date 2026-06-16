"""Helpers for cleaning and normalizing extracted hackathon attributes."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

from .models import IMDB_ATTRIBUTES, ProductRecord
from .pack_parser import parse_pack_text
from .validators import normalize_barcode_digits, validate_barcode


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
    parsed = parse_pack_text(value)
    if parsed.quantity and parsed.unit:
        return parsed.quantity, parsed.unit
    if not value:
        return None, None

    match = WEIGHT_PATTERN.search(value)
    if not match:
        return None, None

    quantity = match.group("quantity").replace(",", "")
    unit = match.group("unit").upper()
    return quantity, unit


def normalize_weight_value(value: Optional[str]) -> Optional[str]:
    parsed = parse_pack_text(value)
    if parsed.normalized_weight:
        return parsed.normalized_weight

    quantity, unit = normalize_weight(value)
    if not quantity or not unit:
        return normalize_text(value)
    compact_quantity = quantity.rstrip("0").rstrip(".") if "." in quantity else quantity
    return f"{compact_quantity}{unit}"


def normalize_barcode(value: Optional[str]) -> Optional[str]:
    return normalize_barcode_digits(value)


def barcode_is_valid(value: Optional[str]) -> bool:
    return validate_barcode(value).is_valid


def normalize_record(record: ProductRecord) -> None:
    for attr in IMDB_ATTRIBUTES:
        attribute = getattr(record, attr)
        if attr not in {"barcode", "weight", "packaging_type", "country"}:
            attribute.value = normalize_text(attribute.value)

    record.barcode.value = normalize_barcode(record.barcode.value)

    pack_parse = parse_pack_text(
        record.item_name.value,
        record.weight.value,
        record.promotion.value,
        record.addons.value,
    )
    if record.weight.source != "manual_edit":
        record.weight.value = pack_parse.normalized_weight or normalize_weight_value(record.weight.value)
    else:
        record.weight.value = normalize_weight_value(record.weight.value)

    record.packaging_type.value = normalize_packaging(record.packaging_type.value)
    record.country.value = normalize_country(record.country.value)

    if pack_parse.normalized_weight:
        record.metadata["pack_parsed"] = {
            "raw_text": pack_parse.raw_text,
            "weight": pack_parse.normalized_weight,
            "quantity": pack_parse.quantity,
            "unit": pack_parse.unit,
            "pack_count": pack_parse.pack_count,
            "promotion": pack_parse.promotion,
            "addons": pack_parse.addons,
            "notes": pack_parse.notes,
        }
        record.metadata["weight_parsed"] = {"quantity": pack_parse.quantity, "unit": pack_parse.unit}
        can_enrich_pack_fields = not record.metadata.get("demo_fixture")
        if can_enrich_pack_fields and pack_parse.promotion and not record.promotion.value and record.promotion.source != "manual_edit":
            record.promotion.value = pack_parse.promotion
            record.promotion.source = "pack_parser"
            record.promotion.confidence = max(record.promotion.confidence or 0, 0.8)
        if can_enrich_pack_fields and pack_parse.addons and not record.addons.value and record.addons.source != "manual_edit":
            record.addons.value = pack_parse.addons
            record.addons.source = "pack_parser"
            record.addons.confidence = max(record.addons.confidence or 0, 0.8)
    elif record.weight.value:
        quantity, unit = normalize_weight(record.weight.value)
        if quantity and unit:
            record.metadata["weight_parsed"] = {"quantity": quantity, "unit": unit}

    if record.barcode.value:
        validation = validate_barcode(record.barcode.value)
        record.metadata["barcode_validation"] = {
            "value": validation.value,
            "type": validation.barcode_type,
            "is_valid": validation.is_valid,
            "reason": validation.reason,
            "expected_check_digit": validation.expected_check_digit,
        }
        if validation.is_valid:
            if record.barcode.notes == "Failed validation":
                record.barcode.notes = None
        else:
            record.barcode.notes = "Failed validation"


def export_lookup_tables() -> Dict[str, Iterable[str]]:
    return {
        "countries": COUNTRY_NORMALIZATION.values(),
        "packaging": PACKAGING_CANONICAL.keys(),
    }
