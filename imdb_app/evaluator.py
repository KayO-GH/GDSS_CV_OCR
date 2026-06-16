"""Evaluation helpers for comparing predictions with hackathon ground truth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import EXPORT_COLUMNS, ProductRecord
from .validators import validate_barcode


GROUND_TRUTH_PATH = Path("hackathon_material/Hackathon Materials/output_results.xlsx")


@dataclass(frozen=True)
class ColumnScore:
    column: str
    exact_matches: int
    normalized_matches: int
    compared: int

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.compared if self.compared else 0.0

    @property
    def normalized_accuracy(self) -> float:
        return self.normalized_matches / self.compared if self.compared else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    row_count: int
    expected_row_count: int
    column_scores: list[ColumnScore]
    mode: str = "row_order"
    aligned_count: int = 0
    unmatched_prediction_count: int = 0
    unmatched_truth_count: int = 0

    @property
    def exact_accuracy(self) -> float:
        total_matches = sum(score.exact_matches for score in self.column_scores)
        total_compared = sum(score.compared for score in self.column_scores)
        return total_matches / total_compared if total_compared else 0.0

    @property
    def normalized_accuracy(self) -> float:
        total_matches = sum(score.normalized_matches for score in self.column_scores)
        total_compared = sum(score.compared for score in self.column_scores)
        return total_matches / total_compared if total_compared else 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Column": score.column,
                    "Exact": score.exact_accuracy,
                    "Normalized": score.normalized_accuracy,
                    "Compared": score.compared,
                }
                for score in self.column_scores
            ]
        )


def records_to_frame(records: Iterable[ProductRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.values_for_export() for record in records], columns=EXPORT_COLUMNS).fillna("")


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=str).fillna("")
    return frame.reindex(columns=EXPORT_COLUMNS, fill_value="")


def normalize_for_match(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"[\d\s\-.]+", text):
        text = re.sub(r"[^0-9]", "", text)
    text = text.replace(" ", "") if re.search(r"\d", text) and re.search(r"[A-Z]", text) else text
    return text


def evaluate_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> EvaluationReport:
    predictions = predictions.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")
    ground_truth = ground_truth.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")

    compared_rows = min(len(predictions), len(ground_truth))
    column_scores: list[ColumnScore] = []

    for column in EXPORT_COLUMNS:
        exact_matches = 0
        normalized_matches = 0
        for index in range(compared_rows):
            predicted = str(predictions.iloc[index][column]).strip()
            expected = str(ground_truth.iloc[index][column]).strip()
            if predicted == expected:
                exact_matches += 1
            if normalize_for_match(predicted) == normalize_for_match(expected):
                normalized_matches += 1
        column_scores.append(
            ColumnScore(
                column=column,
                exact_matches=exact_matches,
                normalized_matches=normalized_matches,
                compared=compared_rows,
            )
        )

    return EvaluationReport(
        row_count=len(predictions),
        expected_row_count=len(ground_truth),
        column_scores=column_scores,
        mode="row_order",
        aligned_count=compared_rows,
        unmatched_prediction_count=max(len(predictions) - compared_rows, 0),
        unmatched_truth_count=max(len(ground_truth) - compared_rows, 0),
    )


def evaluate_records(records: Iterable[ProductRecord], ground_truth_path: Path = GROUND_TRUTH_PATH) -> EvaluationReport:
    return evaluate_predictions(records_to_frame(records), load_ground_truth(ground_truth_path))


def _row_text(row: pd.Series, column: str) -> str:
    return normalize_for_match(row.get(column, ""))


def _barcode_key(row: pd.Series) -> str | None:
    validation = validate_barcode(row.get("BARCODE", ""))
    return validation.value if validation.is_valid else None


def _fuzzy_score(predicted: pd.Series, expected: pd.Series) -> float:
    item_score = SequenceMatcher(None, _row_text(predicted, "ITEM_NAME"), _row_text(expected, "ITEM_NAME")).ratio()
    score = item_score * 0.55
    for column, weight in [
        ("BRAND", 0.15),
        ("WEIGHT", 0.12),
        ("PACKAGING  TYPE", 0.10),
        ("TYPE", 0.08),
    ]:
        if _row_text(predicted, column) and _row_text(predicted, column) == _row_text(expected, column):
            score += weight
    return score


def align_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> list[tuple[int, int]]:
    predictions = predictions.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")
    ground_truth = ground_truth.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")
    unused_truth = set(range(len(ground_truth)))
    pairs: list[tuple[int, int]] = []

    truth_by_barcode = {
        barcode: index
        for index, row in ground_truth.iterrows()
        if (barcode := _barcode_key(row))
    }

    for predicted_index, predicted in predictions.iterrows():
        barcode = _barcode_key(predicted)
        if barcode and barcode in truth_by_barcode and truth_by_barcode[barcode] in unused_truth:
            truth_index = truth_by_barcode[barcode]
            pairs.append((predicted_index, truth_index))
            unused_truth.remove(truth_index)
            continue

        best_index = None
        best_score = 0.0
        for truth_index in unused_truth:
            score = _fuzzy_score(predicted, ground_truth.iloc[truth_index])
            if score > best_score:
                best_index = truth_index
                best_score = score
        if best_index is not None and best_score >= 0.72:
            pairs.append((predicted_index, best_index))
            unused_truth.remove(best_index)

    return pairs


def evaluate_aligned_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> EvaluationReport:
    predictions = predictions.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")
    ground_truth = ground_truth.reindex(columns=EXPORT_COLUMNS, fill_value="").fillna("")
    pairs = align_predictions(predictions, ground_truth)

    aligned_predictions = pd.DataFrame(
        [predictions.iloc[predicted_index].to_dict() for predicted_index, _ in pairs],
        columns=EXPORT_COLUMNS,
    )
    aligned_truth = pd.DataFrame(
        [ground_truth.iloc[truth_index].to_dict() for _, truth_index in pairs],
        columns=EXPORT_COLUMNS,
    )
    report = evaluate_predictions(aligned_predictions, aligned_truth)
    return EvaluationReport(
        row_count=len(predictions),
        expected_row_count=len(ground_truth),
        column_scores=report.column_scores,
        mode="aligned",
        aligned_count=len(pairs),
        unmatched_prediction_count=len(predictions) - len(pairs),
        unmatched_truth_count=len(ground_truth) - len(pairs),
    )


def evaluate_aligned_records(records: Iterable[ProductRecord], ground_truth_path: Path = GROUND_TRUTH_PATH) -> EvaluationReport:
    return evaluate_aligned_predictions(records_to_frame(records), load_ground_truth(ground_truth_path))
