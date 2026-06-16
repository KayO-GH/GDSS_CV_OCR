"""Curated offline records for the hackathon demo flow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .evaluator import GROUND_TRUTH_PATH, load_ground_truth
from .models import COLUMN_TO_ATTRIBUTE, EXPORT_COLUMNS, Attribute, ProductRecord
from .normalizer import normalize_record


PRODUCT_IMAGE_DIR = Path("hackathon_material/Hackathon Materials/product images")
DEMO_GROUP_IDS = ("S230912494", "S227303151", "S230690915")

DEMO_MATCH_TERMS = {
    "S230912494": "BAMA MAYONNAISE",
    "S227303151": "TAPOK PREMIUM BLACK TEA",
    "S230690915": "ZESTA STRAWBERRY",
}


def _filenames_for_group(group_id: str, image_dir: Path = PRODUCT_IMAGE_DIR) -> list[str]:
    return [path.name for path in sorted(image_dir.glob(f"{group_id}_*.jpg"))]


def load_demo_records(group_ids: Iterable[str] | None = None, ground_truth_path: Path = GROUND_TRUTH_PATH) -> list[ProductRecord]:
    frame = load_ground_truth(ground_truth_path)
    records: list[ProductRecord] = []

    for group_id in group_ids or DEMO_GROUP_IDS:
        term = DEMO_MATCH_TERMS[group_id]
        matches = frame[frame["ITEM_NAME"].str.contains(term, case=False, regex=False, na=False)]
        if matches.empty:
            continue

        row = matches.iloc[0]
        kwargs = {
            "id": group_id,
            "filename": group_id,
            "filenames": _filenames_for_group(group_id),
            "metadata": {
                "group_id": group_id,
                "image_count": len(_filenames_for_group(group_id)),
                "demo_fixture": True,
                "fixture_label": "curated demo data",
            },
        }
        for column in EXPORT_COLUMNS:
            attr = COLUMN_TO_ATTRIBUTE[column]
            value = str(row[column]).strip() or None
            kwargs[attr] = Attribute(value=value, confidence=1.0 if value else 0.0, source="curated_demo_fixture")

        record = ProductRecord(**kwargs)
        normalize_record(record)
        records.append(record)

    return records
