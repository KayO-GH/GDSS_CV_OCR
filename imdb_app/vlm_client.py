"""Provider-aware clients for multimodal extraction models."""

from __future__ import annotations

import base64
import asyncio
import json
import os
import re
from typing import Any, Dict, Optional, Protocol

import httpx

from .config import settings
from .costing import ModelPricing, build_model_usage_metadata
from .model_catalog import BackendKind, ModelProfile, get_model_profile, get_model_profile_for_backend_model
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


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
HF_ROUTER_JSON_OBJECT_ONLY_MODEL_MARKERS = ("glm-4.6v", "glm-4-6v")


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


def hf_router_supports_json_schema_response_format(model: str) -> bool:
    normalized_model = model.lower()
    return not any(marker in normalized_model for marker in HF_ROUTER_JSON_OBJECT_ONLY_MODEL_MARKERS)


def hf_router_response_format(model: str, *, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not hf_router_supports_json_schema_response_format(model):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }


def normalize_imdb_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap common provider/schema wrappers around the expected attribute payload."""

    properties = payload.get("properties")
    if isinstance(properties, dict) and any(name in properties for name in IMDB_ATTRIBUTES):
        return properties
    return payload


def normalize_attribute_payload(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return {
            "value": raw_value.get("value"),
            "confidence": raw_value.get("confidence"),
            "source": raw_value.get("source"),
            "notes": raw_value.get("notes"),
        }

    if raw_value is None:
        return {"value": None, "confidence": None, "source": None, "notes": None}

    if isinstance(raw_value, str):
        return {"value": raw_value, "confidence": None, "source": "provider_string_fallback", "notes": None}

    return {"value": json.dumps(raw_value), "confidence": None, "source": "provider_value_fallback", "notes": None}


def attributes_from_payload(payload: dict[str, Any]) -> dict[str, Attribute]:
    normalized_payload = normalize_imdb_payload(payload)
    return {name: Attribute(**normalize_attribute_payload(normalized_payload.get(name))) for name in IMDB_ATTRIBUTES}


def _load_provider_json(text: str, *, provider: str, model: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Provider returned invalid JSON for {provider}/{model}: "
            f"trailing or malformed content at line {exc.lineno}, column {exc.colno}. "
            "Try retrying or switching to a stricter JSON-schema-capable model."
        ) from exc


def compact_error_detail(detail: str) -> str:
    collapsed = re.sub(r"<[^>]+>", " ", detail)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed or detail.strip()


class BaseVLMClient:
    provider = "base"
    backend_kind: BackendKind | None = None

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
            response = await self._post_with_retries(client, headers=headers, payload=payload, group_id=group_id)
            data = response.json()

        record = self._parse_response(data, filename=group_id, filenames=[filename for filename, _ in images])
        record.metadata["model_usage"] = self._model_usage_metadata(data, image_count=len(images))
        return record

    def _model_profile(self) -> ModelProfile | None:
        if self.backend_kind is None:
            return None
        return get_model_profile_for_backend_model(self.backend_kind, self.model)

    def _model_usage_metadata(self, response: dict[str, Any], *, image_count: int) -> dict[str, Any]:
        profile = self._model_profile()
        if profile is not None:
            return build_model_usage_metadata(
                model_key=profile.key,
                provider_label=profile.provider_label,
                model_id=profile.model_id,
                pricing=profile.pricing,
                response=response,
                image_count=image_count,
            )
        return build_model_usage_metadata(
            model_key=f"{self.provider}:{self.model}",
            provider_label=self.provider.title(),
            model_id=self.model,
            pricing=ModelPricing(note=f"No pricing metadata configured for {self.provider} model {self.model}."),
            response=response,
            image_count=image_count,
        )

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_response(self, response: dict[str, Any], filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        raise NotImplementedError

    def _missing_api_key_message(self) -> str:
        return f"Missing API key for selected provider '{self.provider}'."

    async def _post_with_retries(
        self,
        client: httpx.AsyncClient,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        group_id: str | None,
    ) -> httpx.Response:
        attempts = max(1, settings.request_retry_attempts)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if not self._is_retryable_status(exc.response.status_code) or attempt == attempts:
                    raise RuntimeError(self._build_http_error_message(exc.response, group_id=group_id, attempts=attempt)) from exc
                last_error = exc
                await asyncio.sleep(self._retry_delay_seconds(exc.response, attempt))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == attempts:
                    raise RuntimeError(
                        f"{self.provider.title()} API temporary failure after {attempt} attempt(s) for {group_id or 'unknown group'}: {exc}"
                    ) from exc
                last_error = exc
                await asyncio.sleep(settings.request_retry_base_delay_seconds * attempt)

        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError(f"{self.provider.title()} API request failed without response for {group_id or 'unknown group'}")

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in TRANSIENT_STATUS_CODES

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return settings.request_retry_base_delay_seconds * attempt

    def _build_http_error_message(self, response: httpx.Response, *, group_id: str | None, attempts: int) -> str:
        detail = compact_error_detail(response.text)
        prefix = f"{self.provider.title()} API error {response.status_code}"
        if self._is_retryable_status(response.status_code):
            prefix += f" after {attempts} attempt(s)"
        if group_id:
            prefix += f" for {group_id}"
        return f"{prefix}: {detail}" if detail else prefix

    def _fallback_record(self, filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        dummy_attributes = {name: Attribute(value=None, confidence=0.0, source=f"{self.provider}_fallback") for name in IMDB_ATTRIBUTES}
        return ProductRecord(id=os.urandom(8).hex(), filename=filename, filenames=filenames or [], **dummy_attributes)


class OpenAIVLMClient(BaseVLMClient):
    provider = "openai"
    backend_kind = "openai_direct"

    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.OPENAI_KEY,
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
            payload = _load_provider_json(raw_output, provider=self.provider, model=self.model)
        elif isinstance(raw_output, list):
            for block in reversed(raw_output):
                if isinstance(block, dict):
                    content = block.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") in {"output_text", "text"}:
                                text = item.get("text")
                                if isinstance(text, str):
                                    payload = _load_provider_json(text, provider=self.provider, model=self.model)
                                    break
                        if payload:
                            break
                    elif isinstance(content, str):
                        payload = _load_provider_json(content, provider=self.provider, model=self.model)
                        break

        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes_from_payload(payload or {}))

    def _missing_api_key_message(self) -> str:
        return "Missing API key for selected provider 'openai'. Set OPENAI_KEY or switch providers."


class CohereVLMClient(BaseVLMClient):
    provider = "cohere"
    backend_kind = "cohere_direct"

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
                    payload = _load_provider_json(item["text"], provider=self.provider, model=self.model)
                    break
        elif isinstance(content, str):
            payload = _load_provider_json(content, provider=self.provider, model=self.model)
        elif isinstance(response.get("text"), str):
            payload = _load_provider_json(response["text"], provider=self.provider, model=self.model)

        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes_from_payload(payload or {}))

    def _missing_api_key_message(self) -> str:
        return "Missing API key for selected provider 'cohere'. Set COHERE_API_KEY or switch providers."


class HuggingFaceRouterVLMClient(BaseVLMClient):
    provider = "huggingface"
    backend_kind = "hf_router"

    def __init__(self, *, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.hf_token,
            api_url=api_url or str(settings.hf_api_url),
            model=model,
        )

    def _build_payload(self, images: list[tuple[str, bytes]], group_id: str | None) -> Dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Product group: {group_id or 'unknown'}"},
            {"type": "text", "text": "Inspect all images together and produce one product-level row."},
        ]
        for filename, image_bytes in images:
            user_content.extend(
                [
                    {"type": "text", "text": f"Image filename: {filename}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_bytes)}},
                ]
            )

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": user_content},
            ],
            "response_format": hf_router_response_format(
                self.model,
                schema_name="imdb_schema",
                schema=build_imdb_schema(),
            ),
        }

    def _parse_response(self, response: Dict[str, Any], filename: str | None, filenames: list[str] | None = None) -> ProductRecord:
        record_id = response.get("id") or os.urandom(8).hex()
        payload: Dict[str, Any] = {}
        choices = response.get("choices")

        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    payload = _load_provider_json(content, provider=self.provider, model=self.model)
                    break
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
                            payload = _load_provider_json(item["text"], provider=self.provider, model=self.model)
                            break
                    if payload:
                        break

        return ProductRecord(id=record_id, filename=filename, filenames=filenames or [], **attributes_from_payload(payload or {}))

    def _missing_api_key_message(self) -> str:
        return "Missing API key for selected provider 'huggingface'. Set HF_TOKEN or switch models."


VLMClient = OpenAIVLMClient


def normalize_provider(provider: str | None) -> str:
    return get_model_profile(provider).key


def _client_for_profile(profile: ModelProfile) -> SupportsVLMExtraction:
    if profile.backend == "cohere_direct":
        return CohereVLMClient(model=profile.model_id)
    if profile.backend == "openai_direct":
        return OpenAIVLMClient(model=profile.model_id)
    if profile.backend == "hf_router":
        return HuggingFaceRouterVLMClient(model=profile.model_id)
    msg = f"Unsupported backend: {profile.backend}"
    raise ValueError(msg)


def get_vlm_client(provider: str | None = None) -> SupportsVLMExtraction:
    return _client_for_profile(get_model_profile(provider))
