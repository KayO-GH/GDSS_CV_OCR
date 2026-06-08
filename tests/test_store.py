from __future__ import annotations

from imdb_app.models import Attribute, ProductRecord
from imdb_app.store import ProductStore


def make_record(record_id: str, barcode: str = "123456789012", confidence: float = 0.8) -> ProductRecord:
    return ProductRecord(
        id=record_id,
        barcode=Attribute(value=barcode, confidence=confidence, source="test"),
        brand=Attribute(value="Fizz", confidence=confidence, source="test"),
        weight_and_unit=Attribute(value="330ml", confidence=confidence, source="test"),
    )


def test_upsert_prefers_higher_confidence():
    store = ProductStore()
    store.upsert(make_record("item", confidence=0.3))
    store.upsert(make_record("item", confidence=0.9))

    record = store.all()[0]
    assert record.barcode.confidence == 0.9


def test_merge_suggestions_scores_similar_records():
    store = ProductStore()
    store.upsert(make_record("catalogue-1"))

    suggestions = store.merge_suggestions(
        [
            {
                "id": "incoming",
                "barcode": {"value": "123456789012"},
                "brand": {"value": "Fizz"},
                "weight_and_unit": {"value": "330ml"},
            }
        ]
    )

    assert suggestions
    assert suggestions[0]["candidates"]
    assert suggestions[0]["candidates"][0]["score"] >= 1.0
