from __future__ import annotations

import pandas as pd

from imdb_app.evaluator import evaluate_aligned_predictions, evaluate_predictions
from imdb_app.models import EXPORT_COLUMNS


def test_evaluator_reports_exact_and_normalized_accuracy():
    ground_truth = pd.DataFrame(
        [
            {
                "ITEM_NAME": "FIZZ COLA 330ML",
                "BARCODE": "6034000482027",
                "WEIGHT": "330ML",
                "PACKAGING  TYPE": "PLASTIC BOTTLE",
            }
        ],
        columns=EXPORT_COLUMNS,
    ).fillna("")
    predictions = pd.DataFrame(
        [
            {
                "ITEM_NAME": "FIZZ COLA 330ML",
                "BARCODE": "6034-0004-82027",
                "WEIGHT": "330 ML",
                "PACKAGING  TYPE": "PLASTIC BOTTLE",
            }
        ],
        columns=EXPORT_COLUMNS,
    ).fillna("")

    report = evaluate_predictions(predictions, ground_truth)

    assert report.row_count == 1
    assert report.expected_row_count == 1
    assert report.exact_accuracy < report.normalized_accuracy
    assert report.normalized_accuracy == 1.0


def test_aligned_evaluator_handles_reordered_predictions():
    ground_truth = pd.DataFrame(
        [
            {"ITEM_NAME": "BAMA MAYONNAISE", "BARCODE": "8410300363439", "BRAND": "BAMA", "WEIGHT": "909ML"},
            {"ITEM_NAME": "TAPOK PREMIUM BLACK TEA", "BARCODE": "8901035064345", "BRAND": "TAPOK", "WEIGHT": "2G"},
        ],
        columns=EXPORT_COLUMNS,
    ).fillna("")
    predictions = ground_truth.iloc[[1, 0]].reset_index(drop=True)

    row_order = evaluate_predictions(predictions, ground_truth)
    aligned = evaluate_aligned_predictions(predictions, ground_truth)

    assert row_order.normalized_accuracy < 1.0
    assert aligned.aligned_count == 2
    assert aligned.normalized_accuracy == 1.0
