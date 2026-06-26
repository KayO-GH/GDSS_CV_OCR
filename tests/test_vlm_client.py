from __future__ import annotations

import json

import pytest
import httpx

from imdb_app.models import IMDB_ATTRIBUTES
import imdb_app.vlm_client as vlm_client
from imdb_app.vlm_client import (
    CohereVLMClient,
    HuggingFaceRouterVLMClient,
    OpenAIVLMClient,
    ProviderConfigurationError,
    VLMClient,
    get_vlm_client,
    normalize_provider,
)


def test_responses_payload_uses_text_format_schema():
    client = VLMClient(api_key="test", api_url="https://api.openai.com/v1/responses")

    payload = client._build_payload(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert "response_format" not in payload
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["required"] == IMDB_ATTRIBUTES

    field_schema = payload["text"]["format"]["schema"]["properties"]["item_name"]
    assert field_schema["required"] == ["value", "confidence", "source", "notes"]
    assert field_schema["additionalProperties"] is False
    confidence_schema = payload["text"]["format"]["schema"]["properties"]["item_name"]["properties"]["confidence"]
    assert confidence_schema["minimum"] == 0
    assert confidence_schema["maximum"] == 1


def test_cohere_payload_uses_chat_messages_and_response_format_schema():
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")

    payload = client._build_payload(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert payload["model"] == "command-a-vision-07-2025"
    assert payload["messages"][0]["role"] == "user"
    assert payload["response_format"]["type"] == "json_object"
    assert payload["response_format"]["schema"]["required"] == IMDB_ATTRIBUTES
    assert any(item["type"] == "image_url" for item in payload["messages"][0]["content"])
    confidence_schema = payload["response_format"]["schema"]["properties"]["item_name"]["properties"]["confidence"]
    assert "minimum" not in confidence_schema
    assert "maximum" not in confidence_schema


def test_cohere_response_parses_product_record():
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    payload = {
        name: {"value": None, "confidence": 0.0, "source": "test", "notes": None}
        for name in IMDB_ATTRIBUTES
    }
    payload["item_name"]["value"] = "TAPOK PREMIUM BLACK TEA"
    response = {"id": "cohere-id", "message": {"content": [{"type": "text", "text": __import__("json").dumps(payload)}]}}

    record = client._parse_response(response, filename="S1", filenames=["S1_1.jpg"])

    assert record.id == "cohere-id"
    assert record.filename == "S1"
    assert record.filenames == ["S1_1.jpg"]
    assert record.item_name.value == "TAPOK PREMIUM BLACK TEA"


def test_cohere_response_parses_schema_wrapped_payload():
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    wrapped_payload = {
        "type": "object",
        "properties": {
            name: {"value": None, "confidence": 0.0, "source": "test", "notes": None}
            for name in IMDB_ATTRIBUTES
        },
        "required": IMDB_ATTRIBUTES,
        "additionalProperties": False,
    }
    wrapped_payload["properties"]["type"]["value"] = "SEASONING POWDER"
    wrapped_payload["properties"]["item_name"]["value"] = "MUMMY'S KITCHEN STEW SEASONING POWDER"
    response = {"id": "cohere-id", "message": {"content": [{"type": "text", "text": __import__("json").dumps(wrapped_payload)}]}}

    record = client._parse_response(response, filename="S1", filenames=["S1_1.jpg"])

    assert record.item_name.value == "MUMMY'S KITCHEN STEW SEASONING POWDER"
    assert record.type.value == "SEASONING POWDER"


def test_cohere_response_coerces_string_attribute_value():
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    payload = {name: {"value": None, "confidence": 0.0, "source": "test", "notes": None} for name in IMDB_ATTRIBUTES}
    payload["type"] = "BLACK TEA"
    response = {"id": "cohere-id", "message": {"content": [{"type": "text", "text": __import__("json").dumps(payload)}]}}

    record = client._parse_response(response, filename="S1", filenames=["S1_1.jpg"])

    assert record.type.value == "BLACK TEA"
    assert record.type.source == "provider_string_fallback"


def test_provider_factory_defaults_to_supported_clients():
    assert isinstance(get_vlm_client("cohere"), CohereVLMClient)
    assert isinstance(get_vlm_client("openai"), OpenAIVLMClient)
    assert isinstance(get_vlm_client("hf-qwen3-vl-235b-a22b-instruct"), HuggingFaceRouterVLMClient)
    assert isinstance(VLMClient(api_key="test"), OpenAIVLMClient)
    assert normalize_provider("COHERE") == "cohere-command-a-vision-07-2025"


def test_hf_router_payload_uses_openai_compatible_chat_schema():
    client = HuggingFaceRouterVLMClient(
        api_key="test",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    )

    payload = client._build_payload(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert payload["model"] == "Qwen/Qwen3-VL-235B-A22B-Instruct"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert any(item["type"] == "image_url" for item in payload["messages"][1]["content"])


@pytest.mark.parametrize("model", ["zai-org/GLM-4.6V-Flash", "zai-org/glm-4.6v"])
def test_hf_router_glm_payload_uses_json_object_response_format(model: str):
    client = HuggingFaceRouterVLMClient(
        api_key="test",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model=model,
    )

    payload = client._build_payload(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert payload["response_format"] == {"type": "json_object"}


def test_hf_router_glm_grouping_payload_uses_json_object_response_format():
    from scripts.benchmark_grouping_models import _build_batch_payload

    client = HuggingFaceRouterVLMClient(
        api_key="test",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model="zai-org/glm-4.6v",
    )

    payload = _build_batch_payload(client, [("S227303151_569242990.jpg", b"image")], "batch-001")

    assert payload["response_format"] == {"type": "json_object"}


def test_hf_router_response_parses_chat_completion_message():
    client = HuggingFaceRouterVLMClient(
        api_key="test",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model="zai-org/GLM-4.6V-Flash",
    )
    payload = {name: {"value": None, "confidence": 0.0, "source": "test", "notes": None} for name in IMDB_ATTRIBUTES}
    payload["brand"]["value"] = "ZESTA"
    response = {
        "id": "hf-id",
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }

    record = client._parse_response(response, filename="auto-001", filenames=["a.jpg"])

    assert record.id == "hf-id"
    assert record.brand.value == "ZESTA"
    assert record.filename == "auto-001"


def test_hf_router_response_rejects_trailing_json_content():
    client = HuggingFaceRouterVLMClient(
        api_key="test",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model="zai-org/GLM-4.6V-Flash",
    )
    payload = {name: {"value": None, "confidence": 0.0, "source": "test", "notes": None} for name in IMDB_ATTRIBUTES}
    response = {
        "id": "hf-id",
        "choices": [{"message": {"content": f"{json.dumps(payload)}\nextra"}}],
    }

    with pytest.raises(
        RuntimeError,
        match=(
            r"Provider returned invalid JSON for huggingface/zai-org/GLM-4\.6V-Flash: "
            r"trailing or malformed content at line 2, column 1\. "
            r"Try retrying or switching to a stricter JSON-schema-capable model\."
        ),
    ) as exc_info:
        client._parse_response(response, filename="auto-001", filenames=["a.jpg"])

    assert "extra" not in str(exc_info.value)


def test_cohere_response_rejects_trailing_json_content():
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    payload = {name: {"value": None, "confidence": 0.0, "source": "test", "notes": None} for name in IMDB_ATTRIBUTES}
    response = {"id": "cohere-id", "message": {"content": [{"type": "text", "text": f"{json.dumps(payload)}\nextra"}]}}

    with pytest.raises(
        RuntimeError,
        match=(
            r"Provider returned invalid JSON for cohere/command-a-vision-07-2025: "
            r"trailing or malformed content at line 2, column 1\. "
            r"Try retrying or switching to a stricter JSON-schema-capable model\."
        ),
    ) as exc_info:
        client._parse_response(response, filename="S1", filenames=["S1_1.jpg"])

    assert "extra" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_hf_router_client_raises_clear_error_when_api_key_missing():
    client = HuggingFaceRouterVLMClient(
        api_key="",
        api_url="https://router.huggingface.co/v1/chat/completions",
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    )

    with pytest.raises(ProviderConfigurationError, match="Set HF_TOKEN or switch models"):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")


@pytest.mark.asyncio
async def test_openai_client_raises_clear_error_when_api_key_missing():
    client = OpenAIVLMClient(api_key="", api_url="https://api.openai.com/v1/responses")

    with pytest.raises(ProviderConfigurationError, match="Set OPENAI_KEY or switch providers"):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")


@pytest.mark.asyncio
async def test_cohere_client_raises_clear_error_when_api_key_missing():
    client = CohereVLMClient(api_key="", api_url="https://api.cohere.com/v2/chat")

    with pytest.raises(ProviderConfigurationError, match="Set COHERE_API_KEY or switch providers"):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")


@pytest.mark.asyncio
async def test_cohere_client_retries_transient_502_then_succeeds(monkeypatch):
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    calls = {"count": 0}
    payload = {
        name: {"value": None, "confidence": 0.0, "source": "test", "notes": None}
        for name in IMDB_ATTRIBUTES
    }
    payload["item_name"]["value"] = "RECOVERED ITEM"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            calls["count"] += 1
            request = httpx.Request("POST", "https://api.cohere.com/v2/chat")
            if calls["count"] == 1:
                return httpx.Response(
                    502,
                    text="<html><h1>Error: Server Error</h1><h2>Please try again in 30 seconds.</h2></html>",
                    request=request,
                )
            return httpx.Response(
                200,
                json={"id": "cohere-id", "message": {"content": [{"type": "text", "text": json.dumps(payload)}]}},
                request=request,
            )

    monkeypatch.setattr(vlm_client.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(vlm_client.settings, "request_retry_attempts", 2)
    monkeypatch.setattr(vlm_client.settings, "request_retry_base_delay_seconds", 0.0)

    record = await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert calls["count"] == 2
    assert record.item_name.value == "RECOVERED ITEM"
    assert record.metadata["model_usage"]["model_key"] == "cohere-command-a-vision-07-2025"
    assert record.metadata["model_usage"]["cost_available"] is False


@pytest.mark.asyncio
async def test_openai_client_attaches_usage_metadata(monkeypatch):
    client = OpenAIVLMClient(api_key="test", api_url="https://api.openai.com/v1/responses")
    payload = {name: {"value": None, "confidence": 0.0, "source": "test", "notes": None} for name in IMDB_ATTRIBUTES}
    payload["item_name"]["value"] = "USAGE TRACKED ITEM"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://api.openai.com/v1/responses")
            return httpx.Response(
                200,
                json={
                    "id": "openai-id",
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}],
                    "usage": {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168},
                },
                request=request,
            )

    monkeypatch.setattr(vlm_client.httpx, "AsyncClient", FakeAsyncClient)

    record = await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert record.item_name.value == "USAGE TRACKED ITEM"
    assert record.metadata["model_usage"]["model_key"] == "openai-gpt-4o-mini"
    assert record.metadata["model_usage"]["usage"]["input_tokens"] == 123
    assert record.metadata["model_usage"]["usage"]["output_tokens"] == 45
    assert record.metadata["model_usage"]["image_count"] == 1


@pytest.mark.asyncio
async def test_cohere_client_surfaces_compact_error_after_retry_exhaustion(monkeypatch):
    client = CohereVLMClient(api_key="test", api_url="https://api.cohere.com/v2/chat")
    calls = {"count": 0}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            calls["count"] += 1
            request = httpx.Request("POST", "https://api.cohere.com/v2/chat")
            return httpx.Response(
                502,
                text="<html><h1>Error: Server Error</h1><h2>Please try again in 30 seconds.</h2></html>",
                request=request,
            )

    monkeypatch.setattr(vlm_client.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(vlm_client.settings, "request_retry_attempts", 2)
    monkeypatch.setattr(vlm_client.settings, "request_retry_base_delay_seconds", 0.0)

    with pytest.raises(RuntimeError, match="Cohere API error 502 after 2 attempt\\(s\\) for S1: Error: Server Error Please try again in 30 seconds\\."):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")

    assert calls["count"] == 2
