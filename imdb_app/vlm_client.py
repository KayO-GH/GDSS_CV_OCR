"""Provider-aware clients for multimodal extraction models."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional, Protocol

import httpx

from .config import settings
from .models import Attribute, IMDB_ATTRIBUTES, ProductRecord


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
- notes: string or null
"""


class SupportsVLMExtraction(Protocol):
    provider: str
    model: str
    api_key: str | None

    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        ...

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        ...


class ProviderConfigurationError(RuntimeError):
    """Raised when selected VLM provider is missing required configuration."""


def build_attribute_schema(*, include_numeric_bounds: bool = True) -> dict[str, Any]:
    confidence_schema: dict[str, Any] = {"type": ["number", "null"]}
    if include_numeric_bounds:
        confidence_schema["minimum"] = 0
        confidence_schema["maximum"] = 1

    return {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"]},
            "confidence": confidence_schema,
            "source": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": ["value", "confidence", "source", "notes"],
        "additionalProperties": False,
    }


def build_imdb_schema(*, include_numeric_bounds: bool = True) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: build_attribute_schema(include_numeric_bounds=include_numeric_bounds) for name in IMDB_ATTRIBUTES},
        "required": IMDB_ATTRIBUTES,
        "additionalProperties": False,
    }


def image_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def attributes_from_payload(payload: dict[str, Any]) -> dict[str, Attribute]:
    return {name: Attribute(**(payload.get(name, {}) or {})) for name in IMDB_ATTRIBUTES}


class BaseVLMClient:
    provider = "base"

    def __init__(self, *, api_key: Optional[str], api_url: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.api_url = str(api_url)
        self.model = model

    async def extract(self, image_bytes: bytes, filename: str | None = None) -> ProductRecord:
        return await self.extract_group([(filename or "unknown", image_bytes)], group_id=filename)

    async def extract_group(self, images: list[tuple[str, bytes]], group_id: str | None = None) -> ProductRecord:
        if not self.api_key:
            raise ProviderConfigurationError(self._missing_api_key_message())

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(images=images, group_id=group_id)

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text.strip()
                if detail:
                    raise RuntimeError(
                        f"{self.provider.title()} API error {response.status_code}: {detail}"
                    ) from exc
                raise
            data = response.json()

        return self._parse_response(data, filename=group_id, filenames=[filename for filename, _ in images])

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_response(self, response: dict[str, Any], filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        raise NotImplementedError

    def _missing_api_key_message(self) -> str:
        return f"Missing API key for selected provider '{self.provider}'."

    def _fallback_record(self, filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        dummy_attributes = {name: Attribute(value=None, confidence=0.0, source=f"{self.provider}_fallback") for name in IMDB_ATTRIBUTES}
        return ProductRecord(id=os.urandom(8).hex(), filename=filename, filenames=filenames or [], **dummy_attributes)


class OpenAIVLMClient(BaseVLMClient):
    provider = "openai"

    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.openai_api_key,
            api_url=api_url or str(settings.openai_api_url),
            model=model or settings.openai_model,
        )

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> Dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": f"Product group: {group_id or 'unknown'}"},
            {"type": "input_text", "text": "Inspect all images together and produce one product-level row."},
        ]
        for filename, image_bytes in images:
            user_content.extend(
                [
                    {"type": "input_text", "text": f"Image filename: {filename}"},
                    {"type": "input_image", "image_url": image_data_url(image_bytes)},
                ]
            )

        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "imdb_schema",
                    "schema": build_imdb_schema(),
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

        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes_from_payload(payload or {}))

    def _missing_api_key_message(self) -> str:
        return "Missing API key for selected provider 'openai'. Set OPENAI_API_KEY or switch providers."


class CohereVLMClient(BaseVLMClient):
    provider = "cohere"

    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.cohere_api_key,
            api_url=api_url or str(settings.cohere_api_url),
            model=model or settings.cohere_model,
        )

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> Dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": PROMPT_TEMPLATE},
            {"type": "text", "text": f"Product group: {group_id or 'unknown'}"},
            {"type": "text", "text": "Inspect all images together and generate one JSON object matching the schema."},
        ]
        for filename, image_bytes in images:
            user_content.extend(
                [
                    {"type": "text", "text": f"Image filename: {filename}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_bytes), "detail": "high"}},
                ]
            )

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "response_format": {
                "type": "json_object",
                "schema": build_imdb_schema(include_numeric_bounds=False),
            },
        }

    def _parse_response(self, response: Dict[str, Any], filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        record_id = response.get("id") or os.urandom(8).hex()
        payload: Dict[str, Any] = {}
        message = response.get("message") or {}
        content = message.get("content")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    payload = json.loads(item["text"])
                    break
        elif isinstance(content, str):
            payload = json.loads(content)
        elif isinstance(response.get("text"), str):
            payload = json.loads(response["text"])

        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes_from_payload(payload or {}))

    def _missing_api_key_message(self) -> str:
        return "Missing API key for selected provider 'cohere'. Set COHERE_API_KEY or switch providers."


VLMClient = OpenAIVLMClient


def normalize_provider(provider: str | None) -> str:
    normalized = (provider or settings.vlm_provider or "cohere").strip().lower()
    if normalized not in {"cohere", "openai"}:
        msg = f"Unsupported VLM provider: {provider}"
        raise ValueError(msg)
    return normalized


def get_vlm_client(provider: str | None = None) -> SupportsVLMExtraction:
    normalized = normalize_provider(provider)
    if normalized == "cohere":
        return CohereVLMClient()
    return OpenAIVLMClient()
