"""Image extraction pipeline that powers the Streamlit experience."""

from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

from .models import ProductRecord
from .normalizer import normalize_record
from .validators import validate_barcode
from .vlm_client import SupportsVLMExtraction, get_vlm_client, normalize_provider
from .barcode import extract_barcode
from .exporter import Exporter
from .grouping import ImageGroup


def preprocess(image_bytes: bytes) -> bytes:
    if Image is None or ImageOps is None:
        return image_bytes

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        return buffered.getvalue()


class ExtractionPipeline:
    """Coordinates VLM extraction, barcode scan, and normalization."""

    def __init__(self, client: SupportsVLMExtraction, exporter: Exporter | None = None) -> None:
        self.client = client
        self.exporter = exporter or Exporter()

    @staticmethod
    def _apply_barcode_scan(record: ProductRecord, barcode_value: str | None) -> None:
        if not barcode_value:
            return

        scanner_validation = validate_barcode(barcode_value)
        model_value = record.barcode.value
        model_validation = validate_barcode(model_value)

        if model_value and model_validation.value != scanner_validation.value:
            record.metadata["barcode_conflict"] = {
                "model": model_validation.value,
                "model_valid": model_validation.is_valid,
                "scanner": scanner_validation.value,
                "scanner_valid": scanner_validation.is_valid,
            }

        should_use_scanner = scanner_validation.is_valid or not model_validation.is_valid
        if should_use_scanner:
            record.barcode.value = scanner_validation.value
            record.barcode.source = "barcode_scan"
            record.barcode.confidence = 0.95 if scanner_validation.is_valid else 0.5

        if "barcode_conflict" in record.metadata:
            record.barcode.notes = "Barcode scanner and model output disagreed; review required"

    async def process_image(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        preprocessed = preprocess(image_bytes)

        vlm_record, barcode_value = await asyncio.gather(
            self.client.extract(preprocessed, filename=filename),
            asyncio.to_thread(extract_barcode, preprocessed),
        )

        if not vlm_record.id:
            vlm_record.id = uuid.uuid4().hex

        self._apply_barcode_scan(vlm_record, barcode_value)

        normalize_record(vlm_record)
        return vlm_record

    async def process_group(self, group: ImageGroup) -> ProductRecord:
        preprocessed = [(image.filename, preprocess(image.image_bytes)) for image in group.images]

        vlm_result, barcode_values = await asyncio.gather(
            self.client.extract_group(preprocessed, group_id=group.group_id),
            asyncio.gather(*(asyncio.to_thread(extract_barcode, image_bytes) for _, image_bytes in preprocessed)),
            return_exceptions=True,
        )

        if isinstance(vlm_result, Exception):
            raise vlm_result

        vlm_record = vlm_result

        if not vlm_record.id:
            vlm_record.id = group.group_id or uuid.uuid4().hex

        vlm_record.filename = group.group_id
        vlm_record.filenames = group.filenames
        vlm_record.metadata.setdefault("group_id", group.group_id)
        vlm_record.metadata.setdefault("image_count", len(group.images))

        if isinstance(barcode_values, Exception):
            vlm_record.metadata["barcode_error"] = str(barcode_values)
            barcode_candidates: Sequence[str | None] = []
        else:
            barcode_candidates = barcode_values

        barcode_value = next((value for value in barcode_candidates if value), None)
        self._apply_barcode_scan(vlm_record, barcode_value)

        normalize_record(vlm_record)
        return vlm_record

    def export(self, records: Sequence[ProductRecord], *, format: str) -> Path:
        return self.exporter.export(records, format=format)


_pipeline: ExtractionPipeline | None = None
_pipeline_provider: str | None = None


def get_pipeline(provider: str | None = None) -> ExtractionPipeline:
    global _pipeline, _pipeline_provider
    normalized_provider = normalize_provider(provider)
    if _pipeline is None or _pipeline_provider != normalized_provider:
        _pipeline = ExtractionPipeline(client=get_vlm_client(normalized_provider))
        _pipeline_provider = normalized_provider
    return _pipeline


def run_pipeline_sync(pipeline: ExtractionPipeline, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
    """Convenience helper for sync contexts such as Streamlit callbacks."""

    return asyncio.run(pipeline.process_image(image_bytes, filename=filename))


def run_group_pipeline_sync(pipeline: ExtractionPipeline, group: ImageGroup) -> ProductRecord:
    """Convenience helper for grouped Streamlit callbacks."""

    return asyncio.run(pipeline.process_group(group))
