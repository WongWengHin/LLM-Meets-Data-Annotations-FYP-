"""*** 离线 COCO 与 HICO-DET 评估流程的回归测试。 ***"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.evaluation.adapters.coco_adapter import adapt_predictions_to_coco, gt_count_by_category, load_coco_ground_truth
from app.evaluation.adapters.hicodet_adapter import adapt_predictions_to_hicodet, gt_count_by_triplet, load_hicodet_ground_truth
from app.evaluation.label_mapping import load_label_mapping
from app.evaluation.matching import bbox_iou, coco_match_by_threshold, hicodet_match_by_threshold
from app.evaluation.metrics import summarize_coco_threshold, summarize_hicodet_threshold

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class EvaluationTests(unittest.TestCase):
    def test_bbox_iou_overlap_cases(self) -> None:
        self.assertEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(bbox_iou((0, 0, 10, 10), (11, 11, 20, 20)), 0.0)
        self.assertEqual(bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_coco_matching_summary(self) -> None:
        annotation_path = FIXTURE_DIR / "coco_annotations.json"
        prediction_path = FIXTURE_DIR / "coco_prediction.json"
        label_mapping = load_label_mapping(FIXTURE_DIR / "label_mapping.json")

        gt_instances, image_info_by_id, name_to_category_id = load_coco_ground_truth(annotation_path)
        prediction_exports = [json.loads(prediction_path.read_text(encoding="utf-8"))]
        predictions, unmapped = adapt_predictions_to_coco(
            prediction_exports,
            image_info_by_id,
            name_to_category_id,
            label_mapping=label_mapping["coco"]["object_to_category_id"],
        )
        self.assertEqual(unmapped, [])
        outcomes, _ = coco_match_by_threshold(predictions, gt_instances, 0.5)
        summary = summarize_coco_threshold(outcomes, gt_count_by_category(gt_instances))
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["matched"])
        self.assertGreaterEqual(summary["mean_ap"], 0.99)

    def test_hicodet_matching_summary(self) -> None:
        annotation_path = FIXTURE_DIR / "hicodet_annotations.json"
        prediction_path = FIXTURE_DIR / "hicodet_prediction.json"
        label_mapping = load_label_mapping(FIXTURE_DIR / "label_mapping.json")

        gt_instances, image_info_by_id = load_hicodet_ground_truth(annotation_path)
        prediction_exports = [json.loads(prediction_path.read_text(encoding="utf-8"))]
        predictions, issues = adapt_predictions_to_hicodet(
            prediction_exports,
            image_info_by_id,
            object_mapping=label_mapping["hicodet"]["object_to_id"],
            predicate_mapping=label_mapping["hicodet"]["predicate_to_id"],
        )
        self.assertEqual(issues, [])
        outcomes, _ = hicodet_match_by_threshold(predictions, gt_instances, 0.5)
        summary = summarize_hicodet_threshold(outcomes, gt_count_by_triplet(gt_instances))
        self.assertTrue(outcomes[0]["matched"])
        self.assertEqual(summary["overall"]["tp"], 1)
        self.assertAlmostEqual(summary["overall"]["precision"], 1.0)
        self.assertAlmostEqual(summary["overall"]["recall"], 1.0)
        self.assertEqual(predictions[0].object_id, 2)
        self.assertEqual(predictions[0].predicate_id, 76)


if __name__ == "__main__":
    unittest.main()
