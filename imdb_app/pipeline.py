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
from .vlm_client import VLMClient, get_vlm_client
from .barcode import extract_barcode
from .exporter import Exporter


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

