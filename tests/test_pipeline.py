from __future__ import annotations

import pytest

from imdb_app.models import Attribute, ProductRecord
import imdb_app.pipeline as pipeline_module
from imdb_app.pipeline import ExtractionPipeline
from imdb_app.grouping import ImageGroup, ImagePayload
from imdb_app.vlm_client import ProviderConfigurationError


class StubClient:
    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        del image_bytes
        return ProductRecord(
            id="",
            filename=filename,
            barcode=Attribute(value="123456789012", confidence=0.6, source="stub"),
            manufacturer=Attribute(value="Acme", confidence=0.4, source="stub"),
            brand=Attribute(value="Fizz", confidence=0.7, source="stub"),
            item_name=Attribute(value="Fizz Cola", confidence=0.8, source="stub"),
            weight=Attribute(value="330ml", confidence=0.3, source="stub"),
            packaging_type=Attribute(value="carton", confidence=0.3, source="stub"),
            country=Attribute(value="ghana", confidence=0.2, source="stub"),
            type=Attribute(value="soft drink", confidence=0.5, source="stub"),
            promotion=Attribute(value="Limited edition", confidence=0.1, source="stub"),
        )

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        record = await self.extract(images[0][1], filename=group_id)
        record.filenames = [filename for filename, _ in images]
        return record


class FailingClient:
    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        raise RuntimeError("api unavailable")

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        raise RuntimeError("api unavailable")


@pytest.mark.asyncio
async def test_pipeline_normalizes_attributes(sample_image_bytes: bytes):
    pipeline = ExtractionPipeline(client=StubClient())
    record = await pipeline.process_image(sample_image_bytes, filename="item.png")

    assert record.id  # pipeline assigns UUID when missing
    assert record.packaging_type.value == "BOX"
    assert record.country.value == "GHANA"
    assert record.weight.value == "330ML"
    assert record.metadata["weight_parsed"] == {"quantity": "330", "unit": "ML"}


@pytest.mark.asyncio
async def test_pipeline_processes_image_group(sample_image_bytes: bytes):
    pipeline = ExtractionPipeline(client=StubClient())
    group = ImageGroup(
        group_id="S123",
        images=[
            ImagePayload(filename="S123_1.jpg", image_bytes=sample_image_bytes),
            ImagePayload(filename="S123_2.jpg", image_bytes=sample_image_bytes),
        ],
    )

    record = await pipeline.process_group(group)

    assert record.filename == "S123"
    assert record.filenames == ["S123_1.jpg", "S123_2.jpg"]
    assert record.metadata["image_count"] == 2


@pytest.mark.asyncio
async def test_pipeline_surfaces_barcode_conflicts(monkeypatch, sample_image_bytes: bytes):
    class BarcodeClient(StubClient):
        async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
            record = await super().extract(image_bytes, filename)
            record.barcode = Attribute(value="8410300363439", confidence=0.7, source="stub")
            return record

    monkeypatch.setattr(pipeline_module, "extract_barcode", lambda _image_bytes: "8901035064345")

    pipeline = ExtractionPipeline(client=BarcodeClient())
    record = await pipeline.process_image(sample_image_bytes, filename="item.png")

    assert record.barcode.value == "8901035064345"
    assert record.barcode.source == "barcode_scan"
    assert record.metadata["barcode_conflict"]["model"] == "8410300363439"
    assert record.metadata["barcode_conflict"]["scanner"] == "8901035064345"


@pytest.mark.asyncio
async def test_pipeline_raises_error_when_vlm_fails(sample_image_bytes: bytes):
    pipeline = ExtractionPipeline(client=FailingClient())
    group = ImageGroup(
        group_id="S123",
        images=[ImagePayload(filename="S123_1.jpg", image_bytes=sample_image_bytes)],
    )

    with pytest.raises(RuntimeError, match="api unavailable"):
        await pipeline.process_group(group)


class MissingConfigClient:
    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        raise ProviderConfigurationError("Missing API key for selected provider 'cohere'. Set COHERE_API_KEY or switch providers.")

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        raise ProviderConfigurationError("Missing API key for selected provider 'cohere'. Set COHERE_API_KEY or switch providers.")


@pytest.mark.asyncio
async def test_pipeline_raises_provider_config_error_for_group(sample_image_bytes: bytes):
    pipeline = ExtractionPipeline(client=MissingConfigClient())
    group = ImageGroup(
        group_id="S123",
        images=[ImagePayload(filename="S123_1.jpg", image_bytes=sample_image_bytes)],
    )

    with pytest.raises(ProviderConfigurationError, match="Missing API key for selected provider 'cohere'"):
        await pipeline.process_group(group)
