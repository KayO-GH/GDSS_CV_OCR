"""Data models for hackathon IMDB product records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


IMDB_ATTRIBUTES = [
    "item_name",
    "barcode",
    "manufacturer",
    "brand",
    "weight",
    "packaging_type",
    "country",
    "variant",
    "type",
    "fragrance_flavor",
    "promotion",
    "addons",
    "tagline",
]

EXPORT_COLUMNS = [
    "ITEM_NAME",
    "BARCODE",
    "MANUFACTURER",
    "BRAND",
    "WEIGHT",
    "PACKAGING  TYPE",
    "COUNTRY",
    "VARIANT",
    "TYPE",
    "FRAGRANCE_FLAVOR",
    "PROMOTION",
    "ADDONS",
    "TAGLINE",
]

ATTRIBUTE_TO_COLUMN = dict(zip(IMDB_ATTRIBUTES, EXPORT_COLUMNS))
COLUMN_TO_ATTRIBUTE = {column: attr for attr, column in ATTRIBUTE_TO_COLUMN.items()}


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
    filenames: List[str] = field(default_factory=list)
    item_name: Attribute = field(default_factory=Attribute)
    barcode: Attribute = field(default_factory=Attribute)
    manufacturer: Attribute = field(default_factory=Attribute)
    brand: Attribute = field(default_factory=Attribute)
    weight: Attribute = field(default_factory=Attribute)
    packaging_type: Attribute = field(default_factory=Attribute)
    country: Attribute = field(default_factory=Attribute)
    variant: Attribute = field(default_factory=Attribute)
    type: Attribute = field(default_factory=Attribute)
    fragrance_flavor: Attribute = field(default_factory=Attribute)
    promotion: Attribute = field(default_factory=Attribute)
    addons: Attribute = field(default_factory=Attribute)
    tagline: Attribute = field(default_factory=Attribute)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {attr: getattr(self, attr).to_dict() for attr in IMDB_ATTRIBUTES}
        data.update({"id": self.id, "filename": self.filename, "filenames": self.filenames, "metadata": self.metadata})
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProductRecord":
        kwargs: Dict[str, Any] = {"id": payload["id"]}
        for attr in IMDB_ATTRIBUTES:
            attr_payload = payload.get(attr, {}) or {}
            kwargs[attr] = Attribute(**attr_payload)
        kwargs["filename"] = payload.get("filename")
        kwargs["filenames"] = payload.get("filenames", [])
        kwargs["metadata"] = payload.get("metadata", {})
        return cls(**kwargs)

    def values_for_export(self) -> Dict[str, Any]:
        def clean(attribute: Attribute) -> str:
            if attribute.value is None:
                return ""
            return str(attribute.value).strip()

        return {ATTRIBUTE_TO_COLUMN[attr]: clean(getattr(self, attr)) for attr in IMDB_ATTRIBUTES}

    def merge_with(self, other: "ProductRecord") -> None:
        for attr in IMDB_ATTRIBUTES:
            current = getattr(self, attr)
            incoming = getattr(other, attr)
            if incoming.value and (current.value is None or (incoming.confidence or 0) > (current.confidence or 0)):
                setattr(self, attr, incoming)
        self.filenames = sorted(set(self.filenames + other.filenames))
        if not self.filename:
            self.filename = other.filename
        self.metadata.update(other.metadata)


def to_schema(record: ProductRecord) -> Dict[str, Any]:
    return record.to_dict()


def bulk_to_schema(records: List[ProductRecord]) -> List[Dict[str, Any]]:
    return [to_schema(record) for record in records]
