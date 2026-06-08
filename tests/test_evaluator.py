from __future__ import annotations

import pandas as pd

from imdb_app.evaluator import evaluate_predictions
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
