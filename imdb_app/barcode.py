"""Barcode extraction utilities."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from pyzbar.pyzbar import decode
except ImportError:  # pragma: no cover
    decode = None  # type: ignore


def extract_barcode(image_bytes: bytes) -> Optional[str]:
    if decode is None or Image is None:
        return None

    with Image.open(BytesIO(image_bytes)) as img:
        barcodes = decode(img)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8").strip()
            if data:
                return data
    return None

