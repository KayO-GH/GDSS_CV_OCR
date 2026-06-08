from __future__ import annotations

from imdb_app.models import Attribute, ProductRecord
from imdb_app.normalizer import normalize_record


def test_normalize_record_matches_hackathon_export_style():
    record = ProductRecord(
        id="1",
        item_name=Attribute(value="Fizz cola"),
        barcode=Attribute(value="6034-0004 82027"),
        manufacturer=Attribute(value="Acme Ltd"),
        brand=Attribute(value="Fizz"),
        weight=Attribute(value="330 ml"),
        packaging_type=Attribute(value="plastic bttl"),
        country=Attribute(value="gh"),
        fragrance_flavor=Attribute(value="strawberry"),
    )

    normalize_record(record)

    assert record.item_name.value == "FIZZ COLA"
    assert record.barcode.value == "6034000482027"
    assert record.manufacturer.value == "ACME LTD"
    assert record.weight.value == "330ML"
    assert record.packaging_type.value == "PLASTIC BOTTLE"
    assert record.country.value == "GHANA"
    assert record.fragrance_flavor.value == "STRAWBERRY"


def test_invalid_barcode_is_flagged():
    record = ProductRecord(id="1", barcode=Attribute(value="123"))

    normalize_record(record)

    assert record.barcode.notes == "Failed validation"
