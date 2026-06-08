"""Data models for IMDB product records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


IMDB_ATTRIBUTES = [
    "barcode",
    "category_type",
    "segment_type",
    "manufacturer",
    "brand",
    "product_name",
    "weight_and_unit",
    "packaging_type",
    "country_of_origin",
    "promo_messages",
]


@dataclass
class Attribute:
    value: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductRecord:
    id: str
    filename: Optional[str] = None
    barcode: Attribute = field(default_factory=Attribute)
    category_type: Attribute = field(default_factory=Attribute)
    segment_type: Attribute = field(default_factory=Attribute)
    manufacturer: Attribute = field(default_factory=Attribute)
    brand: Attribute = field(default_factory=Attribute)
    product_name: Attribute = field(default_factory=Attribute)
    weight_and_unit: Attribute = field(default_factory=Attribute)
    packaging_type: Attribute = field(default_factory=Attribute)
    country_of_origin: Attribute = field(default_factory=Attribute)
    promo_messages: Attribute = field(default_factory=Attribute)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {attr: getattr(self, attr).to_dict() for attr in IMDB_ATTRIBUTES}
        data.update({"id": self.id, "filename": self.filename, "metadata": self.metadata})
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProductRecord":
        kwargs: Dict[str, Any] = {"id": payload["id"]}
        for attr in IMDB_ATTRIBUTES:
            attr_payload = payload.get(attr, {}) or {}
            kwargs[attr] = Attribute(**attr_payload)
        kwargs["filename"] = payload.get("filename")
        kwargs["metadata"] = payload.get("metadata", {})
        return cls(**kwargs)

    def values_for_export(self) -> Dict[str, Any]:
        def clean(attribute: Attribute) -> Optional[str]:
            if attribute.value is None:
                return None
            return str(attribute.value).strip() or None

        return {
            "barcode": clean(self.barcode),
            "category_type": clean(self.category_type),
            "segment_type": clean(self.segment_type),
            "manufacturer": clean(self.manufacturer),
            "brand": clean(self.brand),
            "product_name": clean(self.product_name),
            "weight_and_unit": clean(self.weight_and_unit),
            "packaging_type": clean(self.packaging_type),
            "country_of_origin": clean(self.country_of_origin),
            "promo_messages": clean(self.promo_messages),
        }

    def merge_with(self, other: "ProductRecord") -> None:
        for attr in IMDB_ATTRIBUTES:
            current = getattr(self, attr)
            incoming = getattr(other, attr)
            if incoming.value and (current.value is None or (incoming.confidence or 0) > (current.confidence or 0)):
                setattr(self, attr, incoming)
        self.metadata.update(other.metadata)


def to_schema(record: ProductRecord) -> Dict[str, Any]:
    return record.to_dict()


def bulk_to_schema(records: List[ProductRecord]) -> List[Dict[str, Any]]:
    return [to_schema(record) for record in records]

