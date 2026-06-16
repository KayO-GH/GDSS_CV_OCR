"""In-memory storage utilities for extracted products."""

from __future__ import annotations

import threading
from typing import Dict, Iterable, List

from .models import ProductRecord


class ProductStore:
    def __init__(self) -> None:
        self._records: Dict[str, ProductRecord] = {}
        self._lock = threading.Lock()

    def upsert(self, record: ProductRecord) -> None:
        with self._lock:
            existing = self._records.get(record.id)
            if existing:
                existing.merge_with(record)
            else:
                self._records[record.id] = record

    def all(self) -> List[ProductRecord]:
        with self._lock:
            return list(self._records.values())

    def remove(self, record_id: str) -> ProductRecord | None:
        with self._lock:
            return self._records.pop(record_id, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def merge_suggestions(self, payload: Iterable[dict]) -> List[dict]:
        suggestions: List[dict] = []
        with self._lock:
            for incoming in payload:
                record_id = incoming.get("id")
                best = []
                for existing in self._records.values():
                    if existing.id == record_id:
                        continue
                    score, reasons = self._score(existing, incoming)
                    if score > 0:
                        best.append({"candidate_id": existing.id, "score": score, "reasons": reasons})

                best.sort(key=lambda item: item["score"], reverse=True)
                if best:
                    suggestions.append({"record_id": record_id, "candidates": best[:3]})
        return suggestions

    @staticmethod
    def _score(existing: ProductRecord, incoming: dict) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        incoming_barcode = (incoming.get("barcode") or {}).get("value")
        if incoming_barcode and existing.barcode.value and incoming_barcode == existing.barcode.value:
            score += 0.7
            reasons.append("matching barcode")

        incoming_brand = (incoming.get("brand") or {}).get("value")
        incoming_weight = (incoming.get("weight") or {}).get("value")
        if incoming_brand and existing.brand.value and incoming_brand.lower() == existing.brand.value.lower():
            score += 0.2
            reasons.append("matching brand")

        if incoming_weight and existing.weight.value and incoming_weight.lower() == existing.weight.value.lower():
            score += 0.1
            reasons.append("matching weight")

        return round(score, 2), reasons
