from __future__ import annotations

import pytest

from imdb_app.grouping import ImageGroup, ImagePayload
from imdb_app.grouping_benchmark import evaluate_group_predictions


def test_evaluate_group_predictions_returns_perfect_score_for_exact_match():
    payloads = [
        ImagePayload(filename="A_1.jpg", image_bytes=b"1"),
        ImagePayload(filename="A_2.jpg", image_bytes=b"2"),
        ImagePayload(filename="B_1.jpg", image_bytes=b"3"),
    ]
    truth = [
        ImageGroup(group_id="A", images=payloads[:2]),
        ImageGroup(group_id="B", images=payloads[2:]),
    ]
    predicted = [
        ImageGroup(group_id="X", images=payloads[:2]),
        ImageGroup(group_id="Y", images=payloads[2:]),
    ]

    report = evaluate_group_predictions(truth, predicted)

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.exact_group_matches == 2


def test_evaluate_group_predictions_penalizes_singletons():
    payloads = [
        ImagePayload(filename="A_1.jpg", image_bytes=b"1"),
        ImagePayload(filename="A_2.jpg", image_bytes=b"2"),
        ImagePayload(filename="B_1.jpg", image_bytes=b"3"),
    ]
    truth = [
        ImageGroup(group_id="A", images=payloads[:2]),
        ImageGroup(group_id="B", images=payloads[2:]),
    ]
    predicted = [ImageGroup(group_id=image.filename, images=[image]) for image in payloads]

    report = evaluate_group_predictions(truth, predicted)

    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.false_negative_pairs == 1


def test_evaluate_group_predictions_requires_same_file_set():
    truth = [ImageGroup(group_id="A", images=[ImagePayload(filename="A_1.jpg", image_bytes=b"1")])]
    predicted = [ImageGroup(group_id="A", images=[ImagePayload(filename="other.jpg", image_bytes=b"1")])]

    with pytest.raises(ValueError, match="Predicted groups must cover exactly same filenames as truth groups"):
        evaluate_group_predictions(truth, predicted)
