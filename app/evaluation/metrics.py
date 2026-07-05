"""*** COCO 与 HICO-DET 评估指标聚合辅助逻辑。 ***"""

from __future__ import annotations

from collections import defaultdict


def precision_recall_curve(match_outcomes: list[dict], gt_count: int) -> tuple[list[float], list[float]]:
    if gt_count <= 0:
        return [], []

    sorted_outcomes = sorted(match_outcomes, key=lambda item: (-item["score"], item["prediction_id"]))
    true_positives = 0
    false_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for outcome in sorted_outcomes:
        if outcome["matched"]:
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / gt_count
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def interpolated_ap(precisions: list[float], recalls: list[float]) -> float:
    if not precisions or not recalls:
        return 0.0

    ap = 0.0
    for recall_threshold in [step / 100 for step in range(101)]:
        precision_candidates = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= recall_threshold
        ]
        ap += max(precision_candidates, default=0.0)
    return ap / 101


def summarize_coco_threshold(
    match_outcomes: list[dict],
    gt_count_by_category: dict[int, int],
) -> dict:
    outcomes_by_category: dict[int, list[dict]] = defaultdict(list)
    for outcome in match_outcomes:
        outcomes_by_category[int(outcome["category_id"])].append(outcome)

    per_category_ap: dict[str, float] = {}
    for category_id, gt_count in gt_count_by_category.items():
        precisions, recalls = precision_recall_curve(outcomes_by_category.get(category_id, []), gt_count)
        per_category_ap[str(category_id)] = interpolated_ap(precisions, recalls)

    mean_ap = (
        sum(per_category_ap.values()) / len(per_category_ap)
        if per_category_ap
        else 0.0
    )
    return {
        "per_category_ap": per_category_ap,
        "mean_ap": mean_ap,
    }


def summarize_hicodet_threshold(
    match_outcomes: list[dict],
    gt_count_by_triplet: dict[str, int],
) -> dict:
    outcomes_by_triplet: dict[str, list[dict]] = defaultdict(list)
    for outcome in match_outcomes:
        outcomes_by_triplet[outcome["triplet_key"]].append(outcome)

    per_triplet: dict[str, dict[str, float | int]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for triplet_key, gt_count in gt_count_by_triplet.items():
        triplet_outcomes = outcomes_by_triplet.get(triplet_key, [])
        tp = sum(1 for item in triplet_outcomes if item["matched"])
        fp = sum(1 for item in triplet_outcomes if not item["matched"])
        fn = max(gt_count - tp, 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / gt_count if gt_count > 0 else 0.0
        per_triplet[triplet_key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    return {
        "per_triplet": per_triplet,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
        },
    }
