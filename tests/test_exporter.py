from __future__ import annotations

from imdb_app.exporter import Exporter
from imdb_app.models import Attribute, ProductRecord


def make_record(record_id: str = "1") -> ProductRecord:
    return ProductRecord(
        id=record_id,
        product_name=Attribute(value="Fizz Cola", confidence=0.9, source="test"),
        brand=Attribute(value="Fizz", confidence=0.8, source="test"),
    )


def test_exporter_creates_csv(tmp_path):
    exporter = Exporter(base_dir=tmp_path)
    path = exporter.export([make_record()], format="csv")

    assert path.exists()
    assert path.suffix == ".csv"


def test_exporter_rejects_unknown_format(tmp_path):
    exporter = Exporter(base_dir=tmp_path)

    try:
        exporter.export([make_record()], format="pdf")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a ValueError for unsupported format")
