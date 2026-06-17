"""Benchmark helpers for evaluating product-image grouping quality."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from .grouping import ImageGroup, ImagePayload, ProductImageCluster, group_images_by_filename_prefix


HACKATHON_IMAGE_DIR = Path("hackathon_material/Hackathon Materials/product images")


@dataclass(frozen=True)
class PairwiseGroupingReport:
    image_count: int
    truth_group_count: int
    predicted_group_count: int
    true_positive_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    exact_group_matches: int

    @property
    def precision(self) -> float:
        total = self.true_positive_pairs + self.false_positive_pairs
        return self.true_positive_pairs / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positive_pairs + self.false_negative_pairs
        return self.true_positive_pairs / total if total else 0.0

    @property
    def f1(self) -> float:
        if not self.precision or not self.recall:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def load_hackathon_image_payloads(image_dir: Path = HACKATHON_IMAGE_DIR) -> list[ImagePayload]:
    """Load hackathon sample images as payloads."""

    return [ImagePayload(filename=path.name, image_bytes=path.read_bytes()) for path in sorted(image_dir.glob("*.jpg"))]


def truth_groups_from_filenames(payloads: Iterable[ImagePayload]) -> list[ImageGroup]:
    """Ground truth grouping for hackathon sample set based on source bundle ids."""

    return group_images_by_filename_prefix(payloads)


def groups_from_clusters(clusters: Sequence[ProductImageCluster]) -> list[ImageGroup]:
    """Convert inferred clusters into evaluation-ready groups."""

    return [cluster.to_image_group() for cluster in clusters]


def _group_sets(groups: Sequence[ImageGroup]) -> list[frozenset[str]]:
    return [frozenset(group.filenames) for group in groups]


def evaluate_group_predictions(
    truth_groups: Sequence[ImageGroup],
    predicted_groups: Sequence[ImageGroup],
) -> PairwiseGroupingReport:
    """Compare predicted grouping against truth using pairwise cluster metrics."""

    truth_sets = _group_sets(truth_groups)
    predicted_sets = _group_sets(predicted_groups)

    truth_lookup = {filename: index for index, group in enumerate(truth_groups) for filename in group.filenames}
    predicted_lookup = {filename: index for index, group in enumerate(predicted_groups) for filename in group.filenames}

    all_truth_files = set(truth_lookup)
    all_pred_files = set(predicted_lookup)
    if all_truth_files != all_pred_files:
        missing_truth = sorted(all_truth_files - all_pred_files)
        missing_pred = sorted(all_pred_files - all_truth_files)
        msg = (
            "Predicted groups must cover exactly same filenames as truth groups. "
            f"Missing from predictions: {missing_truth[:5]}. "
            f"Unexpected in predictions: {missing_pred[:5]}."
        )
        raise ValueError(msg)

    true_positive_pairs = 0
    false_positive_pairs = 0
    false_negative_pairs = 0
    all_files = sorted(all_truth_files)
    for left, right in combinations(all_files, 2):
        same_truth = truth_lookup[left] == truth_lookup[right]
        same_predicted = predicted_lookup[left] == predicted_lookup[right]
        if same_truth and same_predicted:
            true_positive_pairs += 1
        elif same_predicted and not same_truth:
            false_positive_pairs += 1
        elif same_truth and not same_predicted:
            false_negative_pairs += 1

    exact_group_matches = sum(1 for group in predicted_sets if group in truth_sets)
    return PairwiseGroupingReport(
        image_count=len(all_files),
        truth_group_count=len(truth_groups),
        predicted_group_count=len(predicted_groups),
        true_positive_pairs=true_positive_pairs,
        false_positive_pairs=false_positive_pairs,
        false_negative_pairs=false_negative_pairs,
        exact_group_matches=exact_group_matches,
    )
