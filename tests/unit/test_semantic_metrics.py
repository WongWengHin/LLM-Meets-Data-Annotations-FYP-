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


if __name__ == "__main__":
    unittest.main()
