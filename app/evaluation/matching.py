"""IoU matching used by the retained offline COCO evaluator."""
from __future__ import annotations

from collections import defaultdict

from app.evaluation.types import BBox, CocoGtInstance, CocoPredictionInstance, MatchDebugRecord


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def coco_match_by_threshold(predictions: list[CocoPredictionInstance], ground_truth: list[CocoGtInstance], iou_threshold: float) -> tuple[list[dict], list[MatchDebugRecord]]:
    grouped: dict[tuple[str, int], list[CocoGtInstance]] = defaultdict(list)
    for truth in ground_truth:
        grouped[(truth.image_id, truth.category_id)].append(truth)
    for truths in grouped.values():
        truths.sort(key=lambda truth: truth.annotation_id)
    used, outcomes, debug = set(), [], []
    for prediction in sorted(predictions, key=lambda item: (-item.score, item.image_id, item.annotation_id)):
        best, best_iou = None, 0.0
        for truth in grouped.get((prediction.image_id, prediction.category_id), []):
            if truth.annotation_id not in used:
                value = bbox_iou(prediction.bbox_xyxy_abs, truth.bbox_xyxy_abs)
                if value > best_iou:
                    best, best_iou = truth, value
        matched = best is not None and best_iou >= iou_threshold
        if matched:
            used.add(best.annotation_id)
        outcomes.append({"image_id": prediction.image_id, "prediction_id": prediction.annotation_id, "category_id": prediction.category_id, "score": prediction.score, "matched": matched, "gt_id": best.annotation_id if matched else None, "iou": best_iou})
        debug.append(MatchDebugRecord(image_id=prediction.image_id, prediction_id=prediction.annotation_id, gt_id=best.annotation_id if matched else None, score=prediction.score, iou_threshold=iou_threshold, matched=matched, iou=best_iou, details={"category_id": prediction.category_id, "raw_label": prediction.raw_label}))
    return outcomes, debug
