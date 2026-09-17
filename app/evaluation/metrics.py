"""Metric helpers used by the retained offline COCO evaluator."""
from __future__ import annotations

from collections import defaultdict


def precision_recall_curve(outcomes: list[dict], gt_count: int) -> tuple[list[float], list[float]]:
    if gt_count <= 0:
        return [], []
    tp = fp = 0
    precision, recall = [], []
    for outcome in sorted(outcomes, key=lambda item: (-item["score"], item["prediction_id"])):
        tp += int(outcome["matched"])
        fp += int(not outcome["matched"])
        precision.append(tp / (tp + fp))
        recall.append(tp / gt_count)
    return precision, recall


def interpolated_ap(precision: list[float], recall: list[float]) -> float:
    return sum(max((p for p, r in zip(precision, recall) if r >= threshold), default=0.0) for threshold in (step / 100 for step in range(101))) / 101 if precision else 0.0


def summarize_coco_threshold(outcomes: list[dict], gt_count_by_category: dict[int, int]) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[int(outcome["category_id"])].append(outcome)
    per_category = {str(category): interpolated_ap(*precision_recall_curve(grouped.get(category, []), count)) for category, count in gt_count_by_category.items()}
    return {"per_category_ap": per_category, "mean_ap": sum(per_category.values()) / len(per_category) if per_category else 0.0}
