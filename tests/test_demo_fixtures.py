from __future__ import annotations

from imdb_app.demo_fixtures import DEMO_GROUP_IDS, load_demo_records
from imdb_app.models import EXPORT_COLUMNS


def test_load_demo_records_returns_workbook_shaped_rows():
    records = load_demo_records()

    assert [record.id for record in records] == list(DEMO_GROUP_IDS)
    assert all(record.metadata["demo_fixture"] is True for record in records)
    assert all(record.filenames for record in records)
    assert all(list(record.values_for_export()) == EXPORT_COLUMNS for record in records)
    assert {record.brand.value for record in records} == {"BAMA", "TAPOK", "ZESTA"}
    zesta = next(record for record in records if record.brand.value == "ZESTA")
    assert zesta.promotion.value is None
    assert zesta.addons.value == "7 FREE ENVELOPE"
