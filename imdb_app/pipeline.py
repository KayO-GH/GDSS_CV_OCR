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

from .models import Attribute, IMDB_ATTRIBUTES, ProductRecord
from .normalizer import normalize_record
from .vlm_client import VLMClient, get_vlm_client
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

    def __init__(self, client: VLMClient, exporter: Exporter | None = None) -> None:
        self.client = client
        self.exporter = exporter or Exporter()

    async def process_image(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        preprocessed = preprocess(image_bytes)

        vlm_record, barcode_value = await asyncio.gather(
            self.client.extract(preprocessed, filename=filename),
            asyncio.to_thread(extract_barcode, preprocessed),
        )

        if not vlm_record.id:
            vlm_record.id = uuid.uuid4().hex

        if barcode_value:
            vlm_record.barcode.value = barcode_value
            vlm_record.barcode.source = "barcode_scan"
            vlm_record.barcode.confidence = 0.95

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
            vlm_record = self._fallback_record(group, vlm_result)
        else:
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
        if barcode_value:
            vlm_record.barcode.value = barcode_value
            vlm_record.barcode.source = "barcode_scan"
            vlm_record.barcode.confidence = 0.95

        normalize_record(vlm_record)
        return vlm_record

    @staticmethod
    def _fallback_record(group: ImageGroup, error: Exception) -> ProductRecord:
        attributes = {
            attr: Attribute(value=None, confidence=0.0, source="api_error", notes=str(error))
            for attr in IMDB_ATTRIBUTES
        }
        return ProductRecord(
            id=group.group_id,
            filename=group.group_id,
            filenames=group.filenames,
            metadata={"group_id": group.group_id, "image_count": len(group.images), "api_error": str(error)},
            **attributes,
        )

    def export(self, records: Sequence[ProductRecord], *, format: str) -> Path:
        return self.exporter.export(records, format=format)


_pipeline: ExtractionPipeline | None = None


def get_pipeline() -> ExtractionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ExtractionPipeline(client=get_vlm_client())
    return _pipeline


def run_pipeline_sync(pipeline: ExtractionPipeline, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
    """Convenience helper for sync contexts such as Streamlit callbacks."""

    return asyncio.run(pipeline.process_image(image_bytes, filename=filename))


def run_group_pipeline_sync(pipeline: ExtractionPipeline, group: ImageGroup) -> ProductRecord:
    """Convenience helper for grouped Streamlit callbacks."""

    return asyncio.run(pipeline.process_group(group))
