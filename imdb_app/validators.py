"""Validation helpers for catalog fields."""

from __future__ import annotations

import re
from dataclasses import dataclass


BARCODE_TYPES = {
    8: "EAN-8",
    12: "UPC-A",
    13: "EAN-13",
    14: "GTIN-14",
}


@dataclass(frozen=True)
class BarcodeValidation:
    value: str | None
    barcode_type: str | None
    is_valid: bool
    reason: str
    expected_check_digit: int | None = None


def normalize_barcode_digits(value: object | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return digits or None


def calculate_gtin_check_digit(body: str) -> int:
    total = 0
    for index, character in enumerate(reversed(body), start=1):
        weight = 3 if index % 2 else 1
        total += int(character) * weight
    return (10 - (total % 10)) % 10


def validate_barcode(value: object | None) -> BarcodeValidation:
    digits = normalize_barcode_digits(value)
    if not digits:
        return BarcodeValidation(None, None, False, "Missing barcode")

    barcode_type = BARCODE_TYPES.get(len(digits))
    if barcode_type is None:
        return BarcodeValidation(digits, None, False, f"Unsupported barcode length {len(digits)}")

    expected = calculate_gtin_check_digit(digits[:-1])
    actual = int(digits[-1])
    if actual != expected:
        return BarcodeValidation(
            digits,
            barcode_type,
            False,
            f"Invalid {barcode_type} check digit",
            expected_check_digit=expected,
        )

    return BarcodeValidation(digits, barcode_type, True, f"Valid {barcode_type} barcode", expected_check_digit=expected)
