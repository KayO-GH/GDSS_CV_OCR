from __future__ import annotations

from imdb_app.models import IMDB_ATTRIBUTES
from imdb_app.vlm_client import VLMClient


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
