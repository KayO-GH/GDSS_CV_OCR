from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import app
from imdb_app.exporter import Exporter
from imdb_app.grouping import ImageEvidence, ImagePayload, ProductImageCluster
from imdb_app.model_catalog import get_model_profile
from imdb_app.models import Attribute, ProductRecord
from imdb_app.store import ProductStore


@dataclass
class _DummyStatus:
    writes: list[str]
    updated: bool = False
    label: str | None = None
    state: str | None = None

    def write(self, message: str) -> None:
        self.writes.append(message)

    def update(self, **kwargs) -> None:
        self.updated = True
        self.label = kwargs.get("label")
        self.state = kwargs.get("state")


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


class _GroupingPipelineStub:
    def __init__(self, evidence: list[ImageEvidence]) -> None:
        self.evidence = evidence
        self.called = False

    async def analyze_images_for_grouping(self, payloads, cache=None) -> list[ImageEvidence]:
        del payloads, cache
        self.called = True
        return self.evidence


class _RetryPipelineStub:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def process_group(self, group) -> ProductRecord:
        self.calls[group.group_id] = self.calls.get(group.group_id, 0) + 1
        if group.group_id == "bad" and self.calls[group.group_id] == 1:
            raise RuntimeError("temporary failure")
        return ProductRecord(
            id=f"{group.group_id}-record",
            filename=group.group_id,
            filenames=[image.filename for image in group.images],
            item_name=Attribute(value=group.group_id, confidence=1.0, source="stub"),
        )


class _DummyMetric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def metric(self, label: str, value, delta=None) -> None:
        self.calls.append((label, str(value), None if delta is None else str(delta)))


def _clear_batch_state() -> None:
    for key in [
        "batch_runs",
        "current_batch_id",
        "batch_failed_clusters_by_id",
        "inferred_image_clusters",
        "uploaded_image_payloads",
    ]:
        app.st.session_state.pop(key, None)


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


def test_identify_product_groups_records_batch_history(monkeypatch):
    _clear_batch_state()
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)

    class _NoEvidencePipeline:
        async def analyze_images_for_grouping(self, payloads, cache=None):
            raise AssertionError("evidence grouping should not run when usable prefixes exist")

    payloads = [
        ImagePayload(filename="S1_1.jpg", image_bytes=b"1"),
        ImagePayload(filename="S1_2.jpg", image_bytes=b"2"),
        ImagePayload(filename="S2_1.jpg", image_bytes=b"3"),
        ImagePayload(filename="S2_2.jpg", image_bytes=b"4"),
    ]

    clusters, errors = app.identify_product_groups(payloads, _NoEvidencePipeline(), consider_filename_prefixes=True)

    assert errors == []
    assert [cluster.group_id for cluster in clusters] == ["S1", "S2"]
    assert app.current_batch_id() == "batch-001"
    assert len(app.batch_history()) == 1
    batch = app.batch_history()[0]
    assert batch.uploaded_filenames == ["S1_1.jpg", "S1_2.jpg", "S2_1.jpg", "S2_2.jpg"]
    assert batch.image_count == 4
    assert batch.inferred_groups == ["S1", "S2"]
    assert batch.status == "identified"


def test_create_batch_run_keeps_multiple_batches_isolated():
    _clear_batch_state()
    first_payload = [ImagePayload(filename="A_1.jpg", image_bytes=b"1")]
    second_payload = [ImagePayload(filename="B_1.jpg", image_bytes=b"2")]

    first = app.create_batch_run(first_payload, [ProductImageCluster(group_id="A", images=first_payload)])
    second = app.create_batch_run(second_payload, [ProductImageCluster(group_id="B", images=second_payload)])

    assert first.id == "batch-001"
    assert second.id == "batch-002"
    assert app.current_batch_id() == "batch-002"
    assert first.uploaded_filenames == ["A_1.jpg"]
    assert second.uploaded_filenames == ["B_1.jpg"]


def test_process_reviewed_clusters_tracks_failed_groups_and_retries_only_failures(monkeypatch):
    _clear_batch_state()
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)
    monkeypatch.setattr(app.settings, "group_processing_concurrency", 1)
    store = ProductStore()
    pipeline = _RetryPipelineStub()
    good_payload = ImagePayload(filename="good.jpg", image_bytes=b"1")
    bad_payload = ImagePayload(filename="bad.jpg", image_bytes=b"2")
    clusters = [
        ProductImageCluster(group_id="good", images=[good_payload]),
        ProductImageCluster(group_id="bad", images=[bad_payload]),
    ]
    batch = app.create_batch_run([good_payload, bad_payload], clusters)

    processed, errors = app.process_reviewed_clusters(clusters, pipeline, store, batch_id=batch.id)

    assert [record.id for record in processed] == ["good-record"]
    assert errors == ["bad: temporary failure"]
    assert batch.status == "partial"
    assert batch.processed_group_ids == ["good"]
    assert batch.failed_group_ids == ["bad"]
    assert [cluster.group_id for cluster in app.failed_clusters_for_batch(batch.id)] == ["bad"]

    retry_processed, retry_errors = app.process_reviewed_clusters(app.failed_clusters_for_batch(batch.id), pipeline, store, batch_id=batch.id)

    assert retry_errors == []
    assert [record.id for record in retry_processed] == ["bad-record"]
    assert pipeline.calls == {"good": 1, "bad": 2}
    assert batch.status == "processed"
    assert batch.failed_group_ids == []
    assert app.failed_clusters_for_batch(batch.id) == []


def test_render_workflow_empty_state_does_not_claim_duplicate_found(monkeypatch):
    _clear_batch_state()
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

    app.render_workflow([], 0.55, ProductStore(), Exporter(), False)

    assert "Duplicate checks will appear after extraction." in captions
    assert "No duplicate found" not in successes


def test_render_scorecard_skips_benchmark_when_disabled(monkeypatch):
    captions: list[str] = []
    markdowns: list[str] = []
    metric_sets: list[_DummyMetric] = []

    def fake_columns(count: int):
        columns = [_DummyMetric() for _ in range(count)]
        metric_sets.extend(columns)
        return columns

    monkeypatch.setattr(app.st, "columns", fake_columns)
    monkeypatch.setattr(app.st, "dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "caption", lambda message: captions.append(message))
    monkeypatch.setattr(app.st, "markdown", lambda message, **_kwargs: markdowns.append(message))
    monkeypatch.setattr(app, "evaluate_aligned_records", lambda records: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(app, "evaluate_records", lambda records: (_ for _ in ()).throw(AssertionError("should not run")))

    records = [
        ProductRecord(
            id="one",
            item_name=Attribute(value="ITEM ONE"),
            barcode=Attribute(value="6034000482027"),
            manufacturer=Attribute(value="ACME"),
            brand=Attribute(value="BRAND"),
            weight=Attribute(value="200G"),
            packaging_type=Attribute(value="BOX"),
            country=Attribute(value="GHANA"),
            type=Attribute(value="SNACK"),
        )
    ]

    app.render_scorecard(records, enable_hackathon_benchmark=False)

    labels = [label for metric in metric_sets for label, *_ in metric.calls]
    assert "Field completion" in labels
    assert "Ground-truth aligned rows" not in labels
    assert "Hackathon workbook comparison" not in markdowns
    assert not any("Ground-truth match is shown only for aligned rows" in caption for caption in captions)


def test_render_scorecard_renders_benchmark_when_enabled(monkeypatch):
    captions: list[str] = []
    markdowns: list[str] = []
    metric_sets: list[_DummyMetric] = []

    def fake_columns(count: int):
        columns = [_DummyMetric() for _ in range(count)]
        metric_sets.extend(columns)
        return columns

    monkeypatch.setattr(app.st, "columns", fake_columns)
    monkeypatch.setattr(app.st, "dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "caption", lambda message: captions.append(message))
    monkeypatch.setattr(app.st, "markdown", lambda message, **_kwargs: markdowns.append(message))
    monkeypatch.setattr(app, "GROUND_TRUTH_PATH", SimpleNamespace(exists=lambda: True))

    class _Eval:
        aligned_count = 1
        row_count = 2
        normalized_accuracy = 0.31

    class _RowOrder:
        normalized_accuracy = 0.15

    calls = {"aligned": 0, "row_order": 0}

    def fake_aligned(records):
        calls["aligned"] += 1
        return _Eval()

    def fake_row_order(records):
        calls["row_order"] += 1
        return _RowOrder()

    monkeypatch.setattr(app, "evaluate_aligned_records", fake_aligned)
    monkeypatch.setattr(app, "evaluate_records", fake_row_order)

    records = [
        ProductRecord(
            id="one",
            item_name=Attribute(value="ITEM ONE"),
            barcode=Attribute(value="6034000482027"),
            manufacturer=Attribute(value="ACME"),
            brand=Attribute(value="BRAND"),
            weight=Attribute(value="200G"),
            packaging_type=Attribute(value="BOX"),
            country=Attribute(value="GHANA"),
            type=Attribute(value="SNACK"),
        )
    ]

    app.render_scorecard(records, enable_hackathon_benchmark=True)

    labels = [label for metric in metric_sets for label, *_ in metric.calls]
    assert calls == {"aligned": 1, "row_order": 1}
    assert "Ground-truth aligned rows" in labels
    assert any("Hackathon workbook comparison" in item for item in markdowns)
    assert any("Ground-truth match is shown only for aligned rows" in caption for caption in captions)


def test_render_model_cost_summary_shows_usage_and_unknown_pricing(monkeypatch):
    captions: list[str] = []
    markdowns: list[str] = []
    metric_sets: list[_DummyMetric] = []

    def fake_columns(count: int):
        columns = [_DummyMetric() for _ in range(count)]
        metric_sets.extend(columns)
        return columns

    monkeypatch.setattr(app.st, "columns", fake_columns)
    monkeypatch.setattr(app.st, "caption", lambda message: captions.append(message))
    monkeypatch.setattr(app.st, "markdown", lambda message, **_kwargs: markdowns.append(message))

    record = ProductRecord(
        id="one",
        metadata={
            "model_usage": {
                "request_count": 1,
                "image_count": 2,
                "model_id": "command-a-vision-07-2025",
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                "cost_usd": None,
                "cost_available": False,
                "cost_is_estimated": True,
                "pricing": {"note": "Pricing unavailable for this model."},
            }
        },
    )

    app.render_model_cost_summary([record])

    labels = [label for metric in metric_sets for label, *_ in metric.calls]
    values = [value for metric in metric_sets for _, value, *_ in metric.calls]
    assert "##### Model usage and cost" in markdowns
    assert "Model requests" in labels
    assert "Tokens" in labels
    assert "n/a" in values
    assert any("Pricing unavailable for this model." in caption for caption in captions)


def test_visible_model_profiles_only_show_usable_entries(monkeypatch):
    monkeypatch.setattr(app.settings, "cohere_api_key", "cohere-key")
    monkeypatch.setattr(app.settings, "hf_token", "hf-key")
    monkeypatch.setattr(app.settings, "OPENAI_KEY", None)

    profiles = app.visible_model_profiles()

    assert [profile.key for profile in profiles] == [
        "hf-glm-4-6v-flash",
        "hf-qwen3-vl-235b-a22b-instruct",
        "cohere-command-a-vision-07-2025",
    ]


def test_default_visible_model_profile_prefers_first_visible_when_no_explicit_default(monkeypatch):
    monkeypatch.setattr(app.settings, "default_model_key", None)

    profiles = [
        get_model_profile("hf-glm-4-6v-flash"),
        get_model_profile("hf-qwen3-vl-235b-a22b-instruct"),
        get_model_profile("cohere-command-a-vision-07-2025"),
    ]

    selected = app.default_visible_model_profile(profiles)

    assert selected.key == "hf-glm-4-6v-flash"


def test_default_visible_model_profile_respects_explicit_default(monkeypatch):
    monkeypatch.setattr(app.settings, "default_model_key", "cohere-command-a-vision-07-2025")

    profiles = [
        get_model_profile("hf-glm-4-6v-flash"),
        get_model_profile("hf-qwen3-vl-235b-a22b-instruct"),
        get_model_profile("cohere-command-a-vision-07-2025"),
    ]

    selected = app.default_visible_model_profile(profiles)

    assert selected.key == "cohere-command-a-vision-07-2025"


def test_identify_product_groups_uses_filename_prefixes_when_toggle_enabled(monkeypatch):
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)

    class _NoEvidencePipeline:
        async def analyze_images_for_grouping(self, payloads, cache=None):
            raise AssertionError("evidence grouping should not run when usable prefixes exist")

    payloads = [
        ImagePayload(filename="S227094844_568727218.jpg", image_bytes=b"4"),
        ImagePayload(filename="S227303151_569242991.jpg", image_bytes=b"8"),
        ImagePayload(filename="S227303151_569242988.jpg", image_bytes=b"5"),
        ImagePayload(filename="S227094844_568727215.jpg", image_bytes=b"1"),
        ImagePayload(filename="S227303151_569242989.jpg", image_bytes=b"6"),
        ImagePayload(filename="S227094844_568727217.jpg", image_bytes=b"3"),
        ImagePayload(filename="S227094844_568727216.jpg", image_bytes=b"2"),
        ImagePayload(filename="S227303151_569242990.jpg", image_bytes=b"7"),
    ]

    clusters, errors = app.identify_product_groups(payloads, _NoEvidencePipeline(), consider_filename_prefixes=True)

    assert errors == []
    assert [cluster.group_id for cluster in clusters] == ["S227094844", "S227303151"]
    assert clusters[0].filenames == [
        "S227094844_568727215.jpg",
        "S227094844_568727216.jpg",
        "S227094844_568727217.jpg",
        "S227094844_568727218.jpg",
    ]
    assert status_factory.instances[0].label == "Grouped by filename prefix"


def test_identify_product_groups_ignores_prefixes_when_toggle_disabled(monkeypatch):
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)
    monkeypatch.setattr(app, "prefix_group_clusters", lambda payloads: (_ for _ in ()).throw(AssertionError("should not be called")))

    payload = ImagePayload(filename="S227094844_568727215.jpg", image_bytes=b"1")
    evidence = [ImageEvidence(payload_id=payload.payload_id, filename=payload.filename, image_hash="h1", brand="SIYA")]
    pipeline = _GroupingPipelineStub(evidence)
    expected = [
        ProductImageCluster(
            group_id="auto-001",
            images=[payload],
            evidence=evidence,
            confidence=0.5,
            reason="test",
            needs_review=True,
        )
    ]
    monkeypatch.setattr(app, "infer_product_groups", lambda payloads, evidence_by_payload_id: expected)

    clusters, errors = app.identify_product_groups([payload], pipeline, consider_filename_prefixes=False)

    assert errors == []
    assert pipeline.called is True
    assert clusters == expected
    assert status_factory.instances[0].label == "Grouped by image evidence"


def test_identify_product_groups_falls_back_to_evidence_when_prefixes_are_not_usable(monkeypatch):
    status_factory = _DummyStatusFactory()
    monkeypatch.setattr(app.st, "status", status_factory)

    first = ImagePayload(filename="front.jpg", image_bytes=b"1")
    second = ImagePayload(filename="back.jpg", image_bytes=b"2")
    evidence = [
        ImageEvidence(payload_id=first.payload_id, filename=first.filename, image_hash="h1", brand="BAMA"),
        ImageEvidence(payload_id=second.payload_id, filename=second.filename, image_hash="h2", brand="BAMA"),
    ]
    pipeline = _GroupingPipelineStub(evidence)
    expected = [
        ProductImageCluster(
            group_id="auto-001",
            images=[first, second],
            evidence=evidence,
            confidence=0.9,
            reason="matching product evidence",
            needs_review=False,
        )
    ]
    monkeypatch.setattr(app, "infer_product_groups", lambda payloads, evidence_by_payload_id: expected)

    clusters, errors = app.identify_product_groups([first, second], pipeline, consider_filename_prefixes=True)

    assert errors == []
    assert pipeline.called is True
    assert clusters == expected
    assert status_factory.instances[0].label == "Grouped by image evidence"


def test_identify_product_groups_single_image_still_returns_candidate_group(monkeypatch):
    payload = ImagePayload(filename="single.jpg", image_bytes=b"1")
    evidence = [ImageEvidence(payload_id=payload.payload_id, filename=payload.filename, image_hash="h1", brand="ONE")]
    pipeline = _GroupingPipelineStub(evidence)
    expected = [
        ProductImageCluster(
            group_id="review-001",
            images=[payload],
            evidence=evidence,
            confidence=0.5,
            reason="insufficient matching evidence; kept separate",
            needs_review=True,
        )
    ]
    monkeypatch.setattr(app.st, "status", _DummyStatusFactory())
    monkeypatch.setattr(app, "infer_product_groups", lambda payloads, evidence_by_payload_id: expected)

    clusters, errors = app.identify_product_groups([payload], pipeline, consider_filename_prefixes=True)

    assert errors == []
    assert pipeline.called is True
    assert clusters == expected


def test_next_manual_group_id_skips_existing_ids():
    clusters = [
        ProductImageCluster(group_id="auto-merged-001", images=[]),
        ProductImageCluster(group_id="auto-merged-002", images=[]),
        ProductImageCluster(group_id="review-split-001", images=[]),
    ]

    assert app._next_manual_group_id(clusters, "auto-merged") == "auto-merged-003"
    assert app._next_manual_group_id(clusters, "review-split") == "review-split-002"


class _DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
