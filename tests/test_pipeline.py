from __future__ import annotations

import pytest

from imdb_app.models import Attribute, ProductRecord
from imdb_app.pipeline import ExtractionPipeline


class StubClient:
    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        del image_bytes
        return ProductRecord(
            id="",
            filename=filename,
            barcode=Attribute(value="123456789012", confidence=0.6, source="stub"),
            category_type=Attribute(value="beverages", confidence=0.5, source="stub"),
            segment_type=Attribute(value="soda", confidence=0.5, source="stub"),
            manufacturer=Attribute(value="Acme", confidence=0.4, source="stub"),
            brand=Attribute(value="Fizz", confidence=0.7, source="stub"),
            product_name=Attribute(value="Fizz Cola", confidence=0.8, source="stub"),
            weight_and_unit=Attribute(value="330ml", confidence=0.3, source="stub"),
            packaging_type=Attribute(value="carton", confidence=0.3, source="stub"),
            country_of_origin=Attribute(value="usa", confidence=0.2, source="stub"),
            promo_messages=Attribute(value="Limited edition", confidence=0.1, source="stub"),
        )


@pytest.mark.asyncio
async def test_pipeline_normalizes_attributes(sample_image_bytes: bytes):
    pipeline = ExtractionPipeline(client=StubClient())
    record = await pipeline.process_image(sample_image_bytes, filename="item.png")

    assert record.id  # pipeline assigns UUID when missing
    assert record.packaging_type.value == "Box"
    assert record.country_of_origin.value == "United States"
    assert record.metadata["weight_parsed"] == {"quantity": "330", "unit": "ml"}
