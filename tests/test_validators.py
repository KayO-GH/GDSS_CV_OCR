from __future__ import annotations

from imdb_app.validators import validate_barcode


def test_validate_barcode_supports_common_gtin_lengths():
    assert validate_barcode("54013988").barcode_type == "EAN-8"
    assert validate_barcode("036000291452").barcode_type == "UPC-A"
    assert validate_barcode("6034000482027").barcode_type == "EAN-13"
    assert validate_barcode("10012345678902").barcode_type == "GTIN-14"
    assert validate_barcode("6034000482027").is_valid is True


def test_validate_barcode_rejects_bad_check_digit():
    validation = validate_barcode("6034000482028")

    assert validation.is_valid is False
    assert validation.expected_check_digit == 7
    assert "check digit" in validation.reason
