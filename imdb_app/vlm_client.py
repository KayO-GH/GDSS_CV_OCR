"""Client for multimodal extraction models."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import httpx

from .config import settings
from .models import Attribute, ProductRecord, IMDB_ATTRIBUTES


PROMPT_TEMPLATE = """
You are a retail product catalog expert competing in an image-to-item-master-data hackathon.
Extract the required catalog fields from all provided images of the same product.

Rules:
- Use package label text, barcode text, brand marks, manufacturer text, and the image tag/name printed at the bottom when visible.
- Combine evidence across all sides/angles. One image may show the front while another shows barcode or manufacturer.
- Return uppercase catalog values.
- BARCODE must be digits only.
- WEIGHT must be compact, such as 250G, 430G, 1L, 500ML, or 2.2KG.
- If a field cannot be confidently extracted, return null. Do not guess.
- Keep promotions, add-ons, and tagline separate when possible.

Fields (JSON keys with output column intent):
- item_name -> ITEM_NAME
- barcode -> BARCODE
- manufacturer -> MANUFACTURER
- brand -> BRAND
- weight -> WEIGHT
- packaging_type -> PACKAGING  TYPE
- country -> COUNTRY
- variant -> VARIANT
- type -> TYPE
- fragrance_flavor -> FRAGRANCE_FLAVOR
- promotion -> PROMOTION
- addons -> ADDONS
- tagline -> TAGLINE

Respond strictly with JSON where each field has:
- value: string or null
- confidence: float 0-1
- source: short string describing evidence or heuristic used
"""


class VLMClient:
    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.vlm_api_key
        self.api_url = str(api_url or settings.vlm_api_url or "https://api.openai.com/v1/responses")
        self.model = model or settings.vlm_model

    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        return await self.extract_group([(filename or "unknown", image_bytes)], group_id=filename)

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        if not self.api_key:
            return self._fallback_record(filename=group_id, filenames=[filename for filename, _ in images])

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(images=images, group_id=group_id)

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data, filename=group_id, filenames=[filename for filename, _ in images])

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> Dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": f"Product group: {group_id or 'unknown'}"},
            {"type": "input_text", "text": "Inspect all images together and produce one product-level row."},
        ]
        for filename, image_bytes in images:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            content = "data:image/jpeg;base64," + encoded
            user_content.extend(
                [
                    {"type": "input_text", "text": f"Image filename: {filename}"},
                    {"type": "input_image", "image_url": content},
                ]
            )

        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATE,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "imdb_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            name: {
                                "type": "object",
                                "properties": {
                                    "value": {"type": ["string", "null"]},
                                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                                    "source": {"type": ["string", "null"]},
                                    "notes": {"type": ["string", "null"]},
                                },
                                "required": ["value", "confidence", "source", "notes"],
                                "additionalProperties": False,
                            }
                            for name in IMDB_ATTRIBUTES
                        },
                        "required": IMDB_ATTRIBUTES,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        }

    def _parse_response(self, response: Dict[str, Any], filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        record_id = response.get("id") or os.urandom(8).hex()
        payload: Dict[str, Any] = {}
        raw_output = response.get("output") or response.get("content")

        if isinstance(raw_output, str):
            payload = json.loads(raw_output)
        elif isinstance(raw_output, list):
            for block in reversed(raw_output):
                if isinstance(block, dict):
                    content = block.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") in {"output_text", "text"}:
                                text = item.get("text")
                                if isinstance(text, str):
                                    payload = json.loads(text)
                                    break
                        if payload:
                            break
                    elif isinstance(content, str):
                        payload = json.loads(content)
                        break

        payload = payload or {}

        attributes = {name: Attribute(**payload.get(name, {})) for name in IMDB_ATTRIBUTES}
        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes)

    def _fallback_record(self, filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        dummy_attributes = {name: Attribute(value=None, confidence=0.0, source="fallback") for name in IMDB_ATTRIBUTES}
        return ProductRecord(id=os.urandom(8).hex(), filename=filename, filenames=filenames or [], **dummy_attributes)


def get_vlm_client() -> VLMClient:
    return VLMClient()
