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
You are a retail product catalog expert. Extract the following structured data fields from the product described in the image context. Use the JSON schema provided.

Fields (keys):
- barcode
- category_type
- segment_type
- manufacturer
- brand
- product_name
- weight_and_unit
- packaging_type
- country_of_origin
- promo_messages

Respond strictly with JSON where each field has:
- value: string or null
- confidence: float 0-1
- source: short string describing evidence or heuristic used
"""


class VLMClient:
    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.vlm_api_key
        self.api_url = api_url or settings.vlm_api_url or "https://api.openai.com/v1/responses"
        self.model = model or settings.vlm_model

    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        if not self.api_key:
            return self._fallback_record(image_bytes=image_bytes, filename=filename)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(image_bytes=image_bytes, filename=filename)

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data, filename=filename)

    def _build_payload(self, image_bytes: bytes, filename: str | None) -> Dict[str, Any]:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content = "data:image/png;base64," + encoded

        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATE,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"Filename: {filename or 'unknown'}"},
                        {"type": "input_image", "image_url": content},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
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
                                "required": ["value", "confidence", "source"],
                            }
                            for name in IMDB_ATTRIBUTES
                        },
                        "required": IMDB_ATTRIBUTES,
                        "additionalProperties": False,
                    },
                },
            },
        }

    def _parse_response(self, response: Dict[str, Any], filename: str | None) -> ProductRecord:
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
        return ProductRecord(id=record_id, filename=filename, **attributes)

    def _fallback_record(self, image_bytes: bytes, filename: str | None) -> ProductRecord:
        del image_bytes
        dummy_attributes = {name: Attribute(value=None, confidence=0.0, source="fallback") for name in IMDB_ATTRIBUTES}
        return ProductRecord(id=os.urandom(8).hex(), filename=filename, **dummy_attributes)


def get_vlm_client() -> VLMClient:
    return VLMClient()

