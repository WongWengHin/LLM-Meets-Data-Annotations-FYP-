from __future__ import annotations

import unittest

from app.evaluation.semantic_reference import build_hico_semantic_records


class SemanticReferenceTests(unittest.TestCase):
    def test_build_hico_semantic_records_expands_multi_predicates(self) -> None:
        payload = {
            "dataset_metadata": {"name": "HICO-DET dataset"},
            "images": [
                {
                    "id": 1,
                    "file_name": "img1.jpg",
                    "annotations": [
                        {
                            "subject": "person",
                            "object": "bicycle",
                            "predicate": ["ride", "sit_on"],
                            "subject_bbox": [0.1, 0.2, 0.3, 0.4],
                            "object_bbox": [0.2, 0.3, 0.4, 0.5],
                            "sentences": [
                                "A person is riding a bicycle.",
                                "The person is seated on the bicycle.",
                            ],
                        }
                    ],
                }
            ],
        }

        records = build_hico_semantic_records(payload)

        self.assertEqual(len(records), 2)
        self.assertEqual({record["predicate"] for record in records}, {"ride", "sit_on"})
        self.assertTrue(all(record["references"] for record in records))
        self.assertEqual(records[0]["triplet_key"], "person|ride|bicycle")


if __name__ == "__main__":
    unittest.main()
