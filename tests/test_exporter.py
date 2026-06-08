from __future__ import annotations

from imdb_app.exporter import Exporter
from imdb_app.models import EXPORT_COLUMNS, Attribute, ProductRecord


def make_record(record_id: str = "1") -> ProductRecord:
    return ProductRecord(
        id=record_id,
        item_name=Attribute(value="FIZZ COLA", confidence=0.9, source="test"),
        brand=Attribute(value="Fizz", confidence=0.8, source="test"),
    )


def test_exporter_creates_csv(tmp_path):
    exporter = Exporter(base_dir=tmp_path)
    path = exporter.export([make_record()], format="csv")

    assert path.exists()
    assert path.name == "predictions.csv"
    assert path.suffix == ".csv"
    assert path.read_text().splitlines()[0].split(",") == EXPORT_COLUMNS
    assert ",,,," in path.read_text()


def test_exporter_rejects_unknown_format(tmp_path):
    exporter = Exporter(base_dir=tmp_path)

    try:
        exporter.export([make_record()], format="pdf")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a ValueError for unsupported format")
