from __future__ import annotations

from types import SimpleNamespace

import imdb_app.barcode as barcode_module
from imdb_app.barcode import choose_barcode_candidate, extract_barcode, scan_barcodes


def _decoded(value: str, *, symbology: str = "EAN13", quality: int = 1):
    return SimpleNamespace(data=value.encode("utf-8"), type=symbology, quality=quality)


def test_scan_barcodes_returns_candidates_and_extracts_best_valid(monkeypatch, sample_image_bytes: bytes):
    monkeypatch.setattr(
        barcode_module,
        "decode",
        lambda _img: [
            _decoded("6034000482028", quality=100),
            _decoded("6034000482027", quality=10),
        ],
    )

    candidates = scan_barcodes(sample_image_bytes)
    selected = choose_barcode_candidate(candidates)

    assert [candidate.normalized_value for candidate in candidates] == ["6034000482028", "6034000482027"]
    assert selected is not None
    assert selected.normalized_value == "6034000482027"
    assert extract_barcode(sample_image_bytes) == "6034000482027"


def test_scan_barcodes_prefers_quality_when_candidates_are_valid(monkeypatch, sample_image_bytes: bytes):
    monkeypatch.setattr(
        barcode_module,
        "decode",
        lambda _img: [
            _decoded("6034000482027", quality=10),
            _decoded("8410300363439", quality=90),
        ],
    )

    selected = choose_barcode_candidate(scan_barcodes(sample_image_bytes))

    assert selected is not None
    assert selected.normalized_value == "8410300363439"


def test_extract_barcode_returns_none_when_decoder_is_unavailable(monkeypatch, sample_image_bytes: bytes):
    monkeypatch.setattr(barcode_module, "decode", None)

    assert scan_barcodes(sample_image_bytes) == []
    assert extract_barcode(sample_image_bytes) is None
