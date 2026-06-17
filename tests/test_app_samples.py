from __future__ import annotations

from dataclasses import dataclass
import app
from imdb_app.exporter import Exporter
from imdb_app.grouping import ImageEvidence, ImagePayload, ProductImageCluster
from imdb_app.models import Attribute, ProductRecord
from imdb_app.store import ProductStore


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


def test_process_reviewed_clusters_uses_reviewed_group_ids(monkeypatch):
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)
    monkeypatch.setattr(app.settings, "group_processing_concurrency", 2)

    cluster = ProductImageCluster(
        group_id="auto-001",
        images=[ImagePayload(filename="random-a.jpg", image_bytes=b"1"), ImagePayload(filename="random-b.jpg", image_bytes=b"2")],
        evidence=[
            ImageEvidence(payload_id="p1", filename="random-a.jpg", image_hash="h1", brand="Fizz"),
            ImageEvidence(payload_id="p2", filename="random-b.jpg", image_hash="h2", brand="Fizz"),
        ],
        confidence=0.9,
        reason="test",
    )

    processed, errors = app.process_reviewed_clusters([cluster], _PipelineStub(), ProductStore())

    assert errors == []
    assert [record.id for record in processed] == ["auto-001"]
    assert status_factory.instances[0].updated is True


def test_render_workflow_empty_state_does_not_claim_duplicate_found(monkeypatch):
    captions: list[str] = []
    successes: list[str] = []

    monkeypatch.setattr(app.st, "markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "container", lambda **_kwargs: _DummyContainer())
    monkeypatch.setattr(app.st, "caption", lambda message: captions.append(message))
    monkeypatch.setattr(app.st, "success", lambda message: successes.append(message))
    monkeypatch.setattr(app.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "render_scorecard", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "render_merge_suggestions", app.render_merge_suggestions)
    monkeypatch.setattr(app, "render_export_controls", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "get_suggestions", lambda: [])

    app.render_workflow([], 0.55, ProductStore(), Exporter())

    assert "Duplicate checks will appear after extraction." in captions
    assert "No duplicate found" not in successes


class _DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
