"""*** 离线 COCO 与 HICO-DET 评估流程的回归测试。 ***"""

from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

from app.evaluation.adapters.coco_adapter import adapt_predictions_to_coco, gt_count_by_category, load_coco_ground_truth
from app.evaluation.adapters.hicodet_adapter import adapt_predictions_to_hicodet, gt_count_by_triplet, load_hicodet_ground_truth
from app.evaluation.label_mapping import load_label_mapping
from app.evaluation.matching import bbox_iou, coco_match_by_threshold, hicodet_match_by_threshold
from app.evaluation.metrics import summarize_coco_threshold, summarize_hicodet_threshold
from app.evaluation.run_eval import run_hicodet

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
        self.assertAlmostEqual(summary["overall"]["f1"], 1.0)
        self.assertAlmostEqual(summary["triplet_mean_ap"], 1.0)
        self.assertAlmostEqual(summary["mean_ap"], 1.0)
        self.assertAlmostEqual(next(iter(summary["per_triplet"].values()))["ap"], 1.0)
        self.assertEqual(predictions[0].object_id, 2)
        self.assertEqual(predictions[0].predicate_id, 76)

    def test_hicodet_summary_includes_semantic_block(self) -> None:
        annotation_path = FIXTURE_DIR / "hicodet_annotations.json"
        prediction_path = FIXTURE_DIR / "hicodet_prediction.json"
        label_mapping = load_label_mapping(FIXTURE_DIR / "label_mapping.json")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            predictions_dir = tmp_path / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_dir.joinpath("img1-prediction.json").write_text(
                json.dumps(
                    {
                        "image_id": "img1",
                        "image": {"image_id": "img1", "file_name": "img1.jpg", "width": 100, "height": 100},
                        "predictions": [
                            {
                                "annotation_id": "pred1",
                                "subject": "person",
                                "object": "bike",
                                "predicate": "ride",
                                "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                                "object_bbox": [0.4, 0.4, 0.8, 0.9],
                                "sentences": ["A person is riding a bicycle."],
                                "score": 0.8,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            semantic_reference_path = tmp_path / "semantic.json"
            semantic_reference_path.write_text(
                json.dumps(
                    {
                        "dataset_metadata": {"name": "HICO-DET dataset"},
                        "records": [
                        {
                            "image_id": "img1",
                            "file_name": "img1.jpg",
                            "annotation_id": "img1_001_ride",
                            "subject": "person",
                            "predicate": "ride",
                            "object": "bicycle",
                            "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                            "object_bbox": [0.4, 0.4, 0.8, 0.9],
                            "references": ["A person is riding a bicycle."],
                            "source_predicates": ["ride"],
                            "triplet_key": "person|ride|bicycle",
                        }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_dir = tmp_path / "output"
            run_hicodet(
                predictions_dir,
                annotation_path,
                output_dir,
                label_mapping=label_mapping,
                semantic_reference_path=semantic_reference_path,
            )

            summary_payload = json.loads((output_dir / "hicodet-summary.json").read_text(encoding="utf-8"))
            self.assertIn("semantic", summary_payload)
            self.assertEqual(summary_payload["semantic"]["matched_pair_count"], 1)
            self.assertAlmostEqual(summary_payload["semantic"]["bleu"], 1.0)
            self.assertAlmostEqual(summary_payload["semantic"]["meteor"], 1.0)
            self.assertTrue((output_dir / "hicodet-semantic-pairs.csv").exists())


if __name__ == "__main__":
    unittest.main()
