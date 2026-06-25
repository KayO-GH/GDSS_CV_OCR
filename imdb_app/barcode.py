"""Barcode extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from .validators import BarcodeValidation, validate_barcode

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from pyzbar.pyzbar import decode
except ImportError:  # pragma: no cover
    decode = None  # type: ignore


@dataclass(frozen=True)
class BarcodeScanCandidate:
    value: str
    normalized_value: str | None
    barcode_type: str | None
    is_valid: bool
    reason: str
    expected_check_digit: int | None = None
    symbology: str | None = None
    quality: int | None = None
    position: int = 0

    @classmethod
    def from_value(
        cls,
        value: str,
        *,
        symbology: str | None = None,
        quality: int | None = None,
        position: int = 0,
    ) -> "BarcodeScanCandidate":
        validation = validate_barcode(value)
        return cls.from_validation(validation, value=value, symbology=symbology, quality=quality, position=position)

    @classmethod
    def from_validation(
        cls,
        validation: BarcodeValidation,
        *,
        value: str,
        symbology: str | None = None,
        quality: int | None = None,
        position: int = 0,
    ) -> "BarcodeScanCandidate":
        return cls(
            value=value,
            normalized_value=validation.value,
            barcode_type=validation.barcode_type,
            is_valid=validation.is_valid,
            reason=validation.reason,
            expected_check_digit=validation.expected_check_digit,
            symbology=symbology,
            quality=quality,
            position=position,
        )

    def to_metadata(self, *, source_image: str | None = None, selected: bool = False) -> dict[str, object | None]:
        return {
            "value": self.normalized_value or self.value,
            "raw_value": self.value,
            "type": self.barcode_type,
            "is_valid": self.is_valid,
            "reason": self.reason,
            "expected_check_digit": self.expected_check_digit,
            "symbology": self.symbology,
            "quality": self.quality,
            "source_image": source_image,
            "selected": selected,
        }


def _quality(barcode: object) -> int | None:
    quality = getattr(barcode, "quality", None)
    return quality if isinstance(quality, int) else None


def _candidate_rank(candidate: BarcodeScanCandidate) -> tuple[bool, bool, int, int]:
    return (
        candidate.is_valid,
        candidate.barcode_type is not None,
        candidate.quality if candidate.quality is not None else -1,
        -candidate.position,
    )


def choose_barcode_candidate(candidates: list[BarcodeScanCandidate]) -> BarcodeScanCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=_candidate_rank)


def scan_barcodes(image_bytes: bytes) -> list[BarcodeScanCandidate]:
    if decode is None or Image is None:
        return []

    candidates: list[BarcodeScanCandidate] = []
    seen: set[tuple[str | None, str | None]] = set()
    with Image.open(BytesIO(image_bytes)) as img:
        barcodes = decode(img)
        for position, barcode in enumerate(barcodes):
            data = barcode.data.decode("utf-8").strip()
            if not data:
                continue
            candidate = BarcodeScanCandidate.from_value(
                data,
                symbology=getattr(barcode, "type", None),
                quality=_quality(barcode),
                position=position,
            )
            key = (candidate.normalized_value or candidate.value, candidate.symbology)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def extract_barcode(image_bytes: bytes) -> Optional[str]:
    candidate = choose_barcode_candidate(scan_barcodes(image_bytes))
    if candidate is None:
        return None
    return candidate.normalized_value or candidate.value
