from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import app
from imdb_app.grouping import ImagePayload
from imdb_app.models import Attribute, ProductRecord
from imdb_app.store import ProductStore

from app import available_sample_groups, load_all_sample_payloads, load_sample_group


def test_available_sample_groups_reads_hackathon_style_names(tmp_path: Path):
    (tmp_path / "S1_1.jpg").write_bytes(b"1")
    (tmp_path / "S1_2.jpg").write_bytes(b"2")
    (tmp_path / "S2_1.jpg").write_bytes(b"3")
    (tmp_path / "ignore.txt").write_bytes(b"x")

    assert available_sample_groups(tmp_path) == ["S1", "S2"]


def test_load_sample_group_returns_sorted_payloads(tmp_path: Path):
    (tmp_path / "S1_2.jpg").write_bytes(b"2")
    (tmp_path / "S1_1.jpg").write_bytes(b"1")

    payloads = load_sample_group("S1", tmp_path)

    assert [payload.filename for payload in payloads] == ["S1_1.jpg", "S1_2.jpg"]
    assert [payload.image_bytes for payload in payloads] == [b"1", b"2"]


def test_load_all_sample_payloads_returns_sorted_jpgs(tmp_path: Path):
    (tmp_path / "S2_1.jpg").write_bytes(b"2")
    (tmp_path / "S1_2.jpg").write_bytes(b"3")
    (tmp_path / "S1_1.jpg").write_bytes(b"1")
    (tmp_path / "ignore.txt").write_bytes(b"x")

    payloads = load_all_sample_payloads(tmp_path)

    assert [payload.filename for payload in payloads] == ["S1_1.jpg", "S1_2.jpg", "S2_1.jpg"]
    assert [payload.image_bytes for payload in payloads] == [b"1", b"3", b"2"]


@dataclass
class _DummyStatus:
    writes: list[str]
    updated: bool = False

    def write(self, message: str) -> None:
        self.writes.append(message)

    def update(self, **_kwargs) -> None:
        self.updated = True


class _DummyStatusFactory:
    def __init__(self) -> None:
        self.instances: list[_DummyStatus] = []

    def __call__(self, *_args, **_kwargs) -> _DummyStatus:
        instance = _DummyStatus(writes=[])
        self.instances.append(instance)
        return instance


class _PipelineStub:
    async def process_group(self, group) -> ProductRecord:
        return ProductRecord(
            id=group.group_id,
            filename=group.group_id,
            filenames=[image.filename for image in group.images],
            item_name=Attribute(value=group.group_id, confidence=1.0, source="stub"),
        )


def test_process_image_payloads_processes_all_groups(monkeypatch):
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)
    monkeypatch.setattr(app.settings, "group_processing_concurrency", 2)

    payloads = [
        ImagePayload(filename="S1_1.jpg", image_bytes=b"1"),
        ImagePayload(filename="S1_2.jpg", image_bytes=b"2"),
        ImagePayload(filename="S2_1.jpg", image_bytes=b"3"),
    ]

    processed, errors = app.process_image_payloads(payloads, _PipelineStub(), ProductStore())

    assert errors == []
    assert [record.id for record in processed] == ["S1", "S2"]
    assert status_factory.instances[0].updated is True
