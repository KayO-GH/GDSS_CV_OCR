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
from .grouping import ImageEvidence, ImageGroup, ImagePayload, hash_image_bytes


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

    @staticmethod
    def _record_to_evidence(record: ProductRecord, filename: str, image_hash: str, barcode_value: str | None) -> ImageEvidence:
        scanner_validation = validate_barcode(barcode_value)
        record_barcode_validation = validate_barcode(record.barcode.value)
        barcode_validation = scanner_validation if scanner_validation.is_valid else record_barcode_validation
        barcode = barcode_validation.value
        barcode_source = "barcode_scan" if scanner_validation.is_valid else record.barcode.source

        field_confidences = [
            getattr(record, attr).confidence or 0.0
            for attr in ["item_name", "brand", "weight", "packaging_type", "type"]
            if getattr(record, attr).value
        ]
        confidence = max(field_confidences, default=0.0)
        if barcode_validation.is_valid:
            confidence = max(confidence, 0.95)

        return ImageEvidence(
            payload_id=image_hash,
            filename=filename,
            image_hash=image_hash,
            barcode=barcode,
            barcode_is_valid=barcode_validation.is_valid,
            barcode_type=barcode_validation.barcode_type,
            item_name=record.item_name.value,
            brand=record.brand.value,
            weight=record.weight.value,
            packaging_type=record.packaging_type.value,
            type=record.type.value,
            confidence=round(confidence, 2),
            source=barcode_source or record.item_name.source,
            notes=record.item_name.notes or record.barcode.notes,
        )

    async def analyze_image_for_grouping(self, image: ImagePayload) -> ImageEvidence:
        image_hash = hash_image_bytes(image.image_bytes)
        preprocessed = preprocess(image.image_bytes)

        vlm_record, barcode_value = await asyncio.gather(
            self.client.extract(preprocessed, filename=image.filename),
            asyncio.to_thread(extract_barcode, preprocessed),
        )

        normalize_record(vlm_record)
        self._apply_barcode_scan(vlm_record, barcode_value)
        normalize_record(vlm_record)
        return self._record_to_evidence(vlm_record, image.filename, image_hash, barcode_value)

    async def analyze_images_for_grouping(
        self,
        payloads: Sequence[ImagePayload],
        evidence_cache: dict[str, ImageEvidence] | None = None,
    ) -> list[ImageEvidence]:
        cache = evidence_cache if evidence_cache is not None else {}
        evidence: list[ImageEvidence] = []
        missing: list[ImagePayload] = []

        for payload in payloads:
            image_hash = hash_image_bytes(payload.image_bytes)
            cached = cache.get(image_hash)
            if cached is not None:
                evidence.append(
                    ImageEvidence(
                        payload_id=payload.payload_id,
                        filename=payload.filename,
                        image_hash=image_hash,
                        barcode=cached.barcode,
                        barcode_is_valid=cached.barcode_is_valid,
                        barcode_type=cached.barcode_type,
                        item_name=cached.item_name,
                        brand=cached.brand,
                        weight=cached.weight,
                        packaging_type=cached.packaging_type,
                        type=cached.type,
                        confidence=cached.confidence,
                        source=cached.source,
                        notes=cached.notes,
                    )
                )
            else:
                missing.append(payload)

        if missing:
            analyzed = await asyncio.gather(*(self.analyze_image_for_grouping(payload) for payload in missing))
            for payload, item in zip(missing, analyzed):
                cache[item.image_hash] = item
                evidence.append(
                    ImageEvidence(
                        payload_id=payload.payload_id,
                        filename=item.filename,
                        image_hash=item.image_hash,
                        barcode=item.barcode,
                        barcode_is_valid=item.barcode_is_valid,
                        barcode_type=item.barcode_type,
                        item_name=item.item_name,
                        brand=item.brand,
                        weight=item.weight,
                        packaging_type=item.packaging_type,
                        type=item.type,
                        confidence=item.confidence,
                        source=item.source,
                        notes=item.notes,
                    )
                )

        return sorted(evidence, key=lambda item: item.filename)

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
