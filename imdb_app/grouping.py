"""Utilities for grouping product images into candidate catalog rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


PREFIX_PATTERN = re.compile(r"^(?P<prefix>[^_]+)")


class NamedImage(Protocol):
    name: str


@dataclass(frozen=True)
class ImagePayload:
    filename: str
    image_bytes: bytes


@dataclass(frozen=True)
class ImageGroup:
    group_id: str
    images: list[ImagePayload]

    @property
    def filenames(self) -> list[str]:
        return [image.filename for image in self.images]


def group_key(filename: str) -> str:
    """Return the product-group key used by the hackathon sample filenames."""

    name = Path(filename).name
    if "_" not in name:
        return Path(name).stem
    match = PREFIX_PATTERN.match(name)
    return match.group("prefix") if match else Path(name).stem


def group_images(images: Iterable[ImagePayload]) -> list[ImageGroup]:
    grouped: dict[str, list[ImagePayload]] = {}
    for image in images:
        grouped.setdefault(group_key(image.filename), []).append(image)

    return [
        ImageGroup(group_id=group_id, images=sorted(items, key=lambda item: item.filename))
        for group_id, items in sorted(grouped.items())
    ]
