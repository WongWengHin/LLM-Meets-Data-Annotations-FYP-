"""*** 离线评估中的匹配与 IoU 分配计算逻辑。 ***"""

from __future__ import annotations

from collections import defaultdict

from app.evaluation.types import (
    BBox,
    CocoGtInstance,
    CocoPredictionInstance,
    HicoGtInstance,
    HicoPredictionInstance,
    MatchDebugRecord,
)

def hicodet_metric_key(instance: HicoGtInstance | HicoPredictionInstance) -> str:
    if instance.predicate_id is not None and instance.object_id is not None:
        return f"{instance.subject_name.casefold()}|verb:{instance.predicate_id}|object:{instance.object_id}"
    return instance.triplet_key.casefold()


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def coco_match_by_threshold(
    predictions: list[CocoPredictionInstance],
    ground_truth: list[CocoGtInstance],
    iou_threshold: float,
) -> tuple[list[dict], list[MatchDebugRecord]]:
    gt_by_image_category: dict[tuple[str, int], list[CocoGtInstance]] = defaultdict(list)
    for gt_instance in ground_truth:
        gt_by_image_category[(gt_instance.image_id, gt_instance.category_id)].append(gt_instance)

    for gt_instances in gt_by_image_category.values():
        gt_instances.sort(key=lambda item: item.annotation_id)

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (-item.score, item.image_id, item.annotation_id),
    )
    used_gt_ids: set[str] = set()
    outcomes: list[dict] = []
    debug_records: list[MatchDebugRecord] = []

    for prediction in sorted_predictions:
        candidates = gt_by_image_category.get((prediction.image_id, prediction.category_id), [])
        best_gt: CocoGtInstance | None = None
        best_iou = 0.0
        for gt_instance in candidates:
            if gt_instance.annotation_id in used_gt_ids:
                continue
            iou = bbox_iou(prediction.bbox_xyxy_abs, gt_instance.bbox_xyxy_abs)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_instance

        matched = best_gt is not None and best_iou >= iou_threshold
        if matched and best_gt is not None:
            used_gt_ids.add(best_gt.annotation_id)

        outcomes.append(
            {
                "image_id": prediction.image_id,
                "prediction_id": prediction.annotation_id,
                "category_id": prediction.category_id,
                "score": prediction.score,
                "matched": matched,
                "gt_id": best_gt.annotation_id if matched and best_gt is not None else None,
                "iou": best_iou,
            }
        )
        debug_records.append(
            MatchDebugRecord(
                image_id=prediction.image_id,
                prediction_id=prediction.annotation_id,
                gt_id=best_gt.annotation_id if matched and best_gt is not None else None,
                score=prediction.score,
                iou_threshold=iou_threshold,
                matched=matched,
                iou=best_iou,
                details={"category_id": prediction.category_id, "raw_label": prediction.raw_label},
            )
        )

    return outcomes, debug_records


def hicodet_match_by_threshold(
    predictions: list[HicoPredictionInstance],
    ground_truth: list[HicoGtInstance],
    iou_threshold: float,
) -> tuple[list[dict], list[MatchDebugRecord]]:
    gt_by_image: dict[str, list[HicoGtInstance]] = defaultdict(list)
    for gt_instance in ground_truth:
        gt_by_image[gt_instance.image_id].append(gt_instance)

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (-item.score, item.image_id, item.annotation_id),
    )
    used_gt_ids: set[str] = set()
    outcomes: list[dict] = []
    debug_records: list[MatchDebugRecord] = []

    for prediction in sorted_predictions:
        candidates = [
            gt_instance
            for gt_instance in gt_by_image.get(prediction.image_id, [])
            if _hicodet_labels_match(prediction, gt_instance)
        ]
        best_gt: HicoGtInstance | None = None
        best_subject_iou = 0.0
        best_object_iou = 0.0
        best_pair_score = -1.0

        for gt_instance in candidates:
            if gt_instance.annotation_id in used_gt_ids:
                continue
            subject_iou = bbox_iou(prediction.subject_bbox_xyxy_abs, gt_instance.subject_bbox_xyxy_abs)
            object_iou = bbox_iou(prediction.object_bbox_xyxy_abs, gt_instance.object_bbox_xyxy_abs)
            pair_score = min(subject_iou, object_iou)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_gt = gt_instance
                best_subject_iou = subject_iou
                best_object_iou = object_iou

        matched = (
            best_gt is not None
            and best_subject_iou >= iou_threshold
            and best_object_iou >= iou_threshold
        )
        if matched and best_gt is not None:
            used_gt_ids.add(best_gt.annotation_id)

        outcomes.append(
            {
                "image_id": prediction.image_id,
                "prediction_id": prediction.annotation_id,
                "triplet_key": hicodet_metric_key(prediction),
                "score": prediction.score,
                "matched": matched,
                "gt_id": best_gt.annotation_id if matched and best_gt is not None else None,
                "subject_iou": best_subject_iou,
                "object_iou": best_object_iou,
            }
        )
        debug_records.append(
            MatchDebugRecord(
                image_id=prediction.image_id,
                prediction_id=prediction.annotation_id,
                gt_id=best_gt.annotation_id if matched and best_gt is not None else None,
                score=prediction.score,
                iou_threshold=iou_threshold,
                matched=matched,
                iou=min(best_subject_iou, best_object_iou),
                details={
                    "triplet_key": prediction.triplet_key,
                    "subject_iou": best_subject_iou,
                    "object_iou": best_object_iou,
                },
            )
        )

    return outcomes, debug_records


def _hicodet_labels_match(prediction: HicoPredictionInstance, ground_truth: HicoGtInstance) -> bool:
    if (
        prediction.predicate_id is not None
        and ground_truth.predicate_id is not None
        and prediction.object_id is not None
        and ground_truth.object_id is not None
    ):
        return (
            prediction.predicate_id == ground_truth.predicate_id
            and prediction.object_id == ground_truth.object_id
            and prediction.subject_name.casefold() == ground_truth.subject_name.casefold()
        )

    return prediction.triplet_key.casefold() == ground_truth.triplet_key.casefold()
