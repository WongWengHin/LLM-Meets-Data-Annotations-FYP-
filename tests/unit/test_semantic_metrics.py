from __future__ import annotations

import unittest

from app.evaluation.semantic_metrics import evaluate_hico_semantic_pairs, summarize_semantic_matches
from app.evaluation.semantic_reference import build_hico_semantic_records


class SemanticMetricsTests(unittest.TestCase):
    def test_evaluate_hico_semantic_pairs_scores_matched_pair(self) -> None:
        reference_payload = {
            "dataset_metadata": {"name": "HICO-DET dataset"},
            "images": [
                {
                    "id": "img1",
                    "file_name": "img1.jpg",
                    "annotations": [
                        {
                            "subject": "person",
                            "object": "bicycle",
                            "predicate": ["ride"],
                            "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                            "object_bbox": [0.4, 0.4, 0.8, 0.9],
                            "sentences": ["A person is riding a bicycle."],
                        }
                    ],
                }
            ],
        }
        semantic_records = build_hico_semantic_records(reference_payload)
        prediction_exports = [
            {
                "image_id": "img1",
                "image": {"image_id": "img1", "file_name": "img1.jpg"},
                "predictions": [
                    {
                        "annotation_id": "pred1",
                        "subject": "person",
                        "object": "bicycle",
                        "predicate": "ride",
                        "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                        "object_bbox": [0.4, 0.4, 0.8, 0.9],
                        "sentences": ["A person is riding a bicycle."],
                        "score": 0.95,
                    }
                ],
            }
        ]

        matches, issues = evaluate_hico_semantic_pairs(prediction_exports, semantic_records, iou_threshold=0.5)
        summary = summarize_semantic_matches(matches)

        self.assertEqual(issues, [])
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["matched"])
        self.assertAlmostEqual(matches[0]["bleu"], 1.0)
        self.assertAlmostEqual(matches[0]["meteor"], 1.0)
        self.assertAlmostEqual(summary["bleu"], 1.0)
        self.assertAlmostEqual(summary["meteor"], 1.0)
        self.assertEqual(summary["matched_pair_count"], 1)

    def test_evaluate_hico_semantic_pairs_matches_on_file_name_alias(self) -> None:
        semantic_records = [
            {
                "image_id": "2149",
                "file_name": "HICO_train2015_00000001.jpg",
                "annotation_id": "2149_001_ride",
                "subject": "person",
                "predicate": "ride",
                "object": "motorcycle",
                "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                "object_bbox": [0.4, 0.4, 0.8, 0.9],
                "references": ["A person is riding a motorcycle."],
                "source_predicates": ["ride"],
                "triplet_key": "person|ride|motorcycle",
            }
        ]
        prediction_exports = [
            {
                "image_id": "HICO_train2015_00000001",
                "image": {"image_id": "HICO_train2015_00000001", "file_name": "HICO_train2015_00000001.jpg"},
                "predictions": [
                    {
                        "annotation_id": "pred1",
                        "subject": "person",
                        "object": "motorcycle",
                        "predicate": "ride",
                        "subject_bbox": [0.1, 0.1, 0.4, 0.8],
                        "object_bbox": [0.4, 0.4, 0.8, 0.9],
                        "sentences": ["A person is riding a motorcycle."],
                        "score": 0.95,
                    }
                ],
            }
        ]

        matches, issues = evaluate_hico_semantic_pairs(prediction_exports, semantic_records, iou_threshold=0.5)

        self.assertEqual(issues, [])
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["matched"])
        self.assertEqual(matches[0]["reference_id"], "2149_001_ride")


if __name__ == "__main__":
    unittest.main()
