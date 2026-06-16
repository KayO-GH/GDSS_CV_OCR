"""Utilities for grouping product images into candidate catalog rows."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Protocol


PREFIX_PATTERN = re.compile(r"^(?P<prefix>[^_]+)")
HIGH_CONFIDENCE_THRESHOLD = 0.78


class NamedImage(Protocol):
    name: str


@dataclass(frozen=True)
class ImagePayload:
    filename: str
    image_bytes: bytes
    payload_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class ImageGroup:
    group_id: str
    images: list[ImagePayload]

    @property
    def filenames(self) -> list[str]:
        return [image.filename for image in self.images]


@dataclass(frozen=True)
class ImageEvidence:
    payload_id: str
    filename: str
    image_hash: str
    barcode: str | None = None
    barcode_is_valid: bool = False
    barcode_type: str | None = None
    item_name: str | None = None
    brand: str | None = None
    weight: str | None = None
    packaging_type: str | None = None
    type: str | None = None
    confidence: float = 0.0
    source: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class GroupingDecision:
    group_id: str
    confidence: float
    reason: str
    needs_review: bool = False


@dataclass(frozen=True)
class ProductImageCluster:
    group_id: str
    images: list[ImagePayload]
    evidence: list[ImageEvidence] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    needs_review: bool = False

    @property
    def filenames(self) -> list[str]:
        return [image.filename for image in self.images]

    def to_image_group(self) -> ImageGroup:
        return ImageGroup(group_id=self.group_id, images=self.images)


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def group_key(filename: str) -> str:
    """Return the product-group key used by the hackathon sample filenames."""

    name = Path(filename).name
    if "_" not in name:
        return Path(name).stem
    match = PREFIX_PATTERN.match(name)
    return match.group("prefix") if match else Path(name).stem


def group_images_by_filename_prefix(images: Iterable[ImagePayload]) -> list[ImageGroup]:
    grouped: dict[str, list[ImagePayload]] = {}
    for image in images:
        grouped.setdefault(group_key(image.filename), []).append(image)

    return [
        ImageGroup(group_id=group_id, images=sorted(items, key=lambda item: item.filename))
        for group_id, items in sorted(grouped.items())
    ]


def group_images(images: Iterable[ImagePayload]) -> list[ImageGroup]:
    """Backward-compatible alias for development sample grouping."""

    return group_images_by_filename_prefix(images)


def _norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().upper()


def _similarity(left: str | None, right: str | None) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _pair_score(left: ImageEvidence, right: ImageEvidence) -> tuple[float, list[str]]:
    if left.barcode_is_valid and right.barcode_is_valid:
        if left.barcode == right.barcode:
            return 1.0, ["same valid barcode"]
        return 0.0, ["conflicting valid barcodes"]

    score = 0.0
    reasons: list[str] = []
    if _norm(left.brand) and _norm(left.brand) == _norm(right.brand):
        score += 0.35
        reasons.append("matching brand")
    if _norm(left.weight) and _norm(left.weight) == _norm(right.weight):
        score += 0.20
        reasons.append("matching weight")
    name_score = _similarity(left.item_name, right.item_name)
    if name_score >= 0.82:
        score += 0.30
        reasons.append("similar item name")
    if _norm(left.type) and _norm(left.type) == _norm(right.type):
        score += 0.10
        reasons.append("matching type")
    if _norm(left.packaging_type) and _norm(left.packaging_type) == _norm(right.packaging_type):
        score += 0.05
        reasons.append("matching packaging")
    return round(score, 2), reasons


def _cluster_score(evidence: ImageEvidence, cluster: ProductImageCluster) -> tuple[float, list[str]]:
    scores = [_pair_score(evidence, candidate) for candidate in cluster.evidence]
    if not scores:
        return 0.0, []
    return max(scores, key=lambda item: item[0])


def _cluster_id(index: int, evidence: list[ImageEvidence], confidence: float) -> str:
    valid_barcodes = sorted({item.barcode for item in evidence if item.barcode_is_valid and item.barcode})
    if len(valid_barcodes) == 1:
        return f"barcode-{valid_barcodes[0]}"
    prefix = "auto" if confidence >= HIGH_CONFIDENCE_THRESHOLD and len(evidence) > 1 else "review"
    return f"{prefix}-{index:03d}"


def infer_product_groups(
    images: Iterable[ImagePayload],
    evidence_by_payload_id: dict[str, ImageEvidence],
    thresholds: dict[str, float] | None = None,
) -> list[ProductImageCluster]:
    threshold = (thresholds or {}).get("high_confidence", HIGH_CONFIDENCE_THRESHOLD)
    image_by_payload_id = {image.payload_id: image for image in images}
    evidences = sorted(evidence_by_payload_id.values(), key=lambda item: (item.filename, item.payload_id))

    barcode_groups: dict[str, list[ImageEvidence]] = {}
    unassigned: list[ImageEvidence] = []
    for evidence in evidences:
        if evidence.barcode_is_valid and evidence.barcode:
            barcode_groups.setdefault(evidence.barcode, []).append(evidence)
        else:
            unassigned.append(evidence)

    clusters: list[ProductImageCluster] = []
    for barcode, items in sorted(barcode_groups.items()):
        group_images_for_barcode = [
            image_by_payload_id[item.payload_id]
            for item in items
            if item.payload_id in image_by_payload_id
        ]
        clusters.append(
            ProductImageCluster(
                group_id=f"barcode-{barcode}",
                images=group_images_for_barcode,
                evidence=items,
                confidence=1.0,
                reason="same valid barcode",
                needs_review=False,
            )
        )

    non_barcode_clusters: list[ProductImageCluster] = []
    for evidence in unassigned:
        image = image_by_payload_id.get(evidence.payload_id)
        if image is None:
            continue

        best_index = None
        best_score = 0.0
        best_reasons: list[str] = []
        for index, cluster in enumerate(non_barcode_clusters):
            score, reasons = _cluster_score(evidence, cluster)
            if score > best_score:
                best_index = index
                best_score = score
                best_reasons = reasons

        if best_index is not None and best_score >= threshold:
            cluster = non_barcode_clusters[best_index]
            non_barcode_clusters[best_index] = ProductImageCluster(
                group_id=cluster.group_id,
                images=sorted([*cluster.images, image], key=lambda item: item.filename),
                evidence=sorted([*cluster.evidence, evidence], key=lambda item: item.filename),
                confidence=max(cluster.confidence or 0.0, best_score),
                reason=", ".join(best_reasons) or "matching product evidence",
                needs_review=False,
            )
        else:
            non_barcode_clusters.append(
                ProductImageCluster(
                    group_id="",
                    images=[image],
                    evidence=[evidence],
                    confidence=evidence.confidence,
                    reason="insufficient matching evidence; kept separate",
                    needs_review=True,
                )
            )

    for index, cluster in enumerate(non_barcode_clusters, start=1):
        confidence = cluster.confidence if len(cluster.evidence) > 1 else min(cluster.confidence, 0.5)
        clusters.append(
            ProductImageCluster(
                group_id=_cluster_id(index, cluster.evidence, confidence),
                images=cluster.images,
                evidence=cluster.evidence,
                confidence=confidence,
                reason=cluster.reason,
                needs_review=cluster.needs_review or confidence < threshold,
            )
        )

    return sorted(clusters, key=lambda cluster: cluster.group_id)
