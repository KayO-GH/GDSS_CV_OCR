"""Evaluation helpers for comparing predictions with hackathon ground truth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import EXPORT_COLUMNS, ProductRecord


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
    )


def evaluate_records(records: Iterable[ProductRecord], ground_truth_path: Path = GROUND_TRUTH_PATH) -> EvaluationReport:
    return evaluate_predictions(records_to_frame(records), load_ground_truth(ground_truth_path))
