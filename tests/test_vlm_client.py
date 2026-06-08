from __future__ import annotations

import pytest

from imdb_app.models import IMDB_ATTRIBUTES
from imdb_app.vlm_client import (
    CohereVLMClient,
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


def test_provider_factory_defaults_to_supported_clients():
    assert isinstance(get_vlm_client("cohere"), CohereVLMClient)
    assert isinstance(get_vlm_client("openai"), OpenAIVLMClient)
    assert isinstance(VLMClient(api_key="test"), OpenAIVLMClient)
    assert normalize_provider("COHERE") == "cohere"


@pytest.mark.asyncio
async def test_openai_client_raises_clear_error_when_api_key_missing():
    client = OpenAIVLMClient(api_key="", api_url="https://api.openai.com/v1/responses")

    with pytest.raises(ProviderConfigurationError, match="Set OPENAI_API_KEY or switch providers"):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")


@pytest.mark.asyncio
async def test_cohere_client_raises_clear_error_when_api_key_missing():
    client = CohereVLMClient(api_key="", api_url="https://api.cohere.com/v2/chat")

    with pytest.raises(ProviderConfigurationError, match="Set COHERE_API_KEY or switch providers"):
        await client.extract_group(images=[("S1_1.jpg", b"image")], group_id="S1")
