from __future__ import annotations

import base64
import io
from typing import Iterator

import pytest


@pytest.fixture
def sample_image_bytes() -> Iterator[bytes]:
    """Provide a tiny PNG payload for tests without external assets."""

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow optional
        png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        )
        yield base64.b64decode(png_base64)
        return

    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    yield buffer.getvalue()
