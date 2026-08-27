from __future__ import annotations

import unittest

from app.evaluation.semantic_reference import build_hico_semantic_records, load_hico_semantic_fixture
from pathlib import Path
import json
import tempfile


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

    def test_load_hico_semantic_fixture_converts_raw_payload(self) -> None:
        raw_payload = {
            "dataset_metadata": {"name": "HICO-DET dataset"},
            "images": [
                {
                    "id": 2149,
                    "file_name": "HICO_train2015_00000001.jpg",
                    "annotations": [
                        {
                            "subject": "person",
                            "object": "motorcycle",
                            "predicate": ["ride"],
                            "subject_bbox": [0.1, 0.2, 0.3, 0.4],
                            "object_bbox": [0.2, 0.3, 0.4, 0.5],
                            "sentences": ["A person is riding a motorcycle."],
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.json"
            path.write_text(json.dumps(raw_payload), encoding="utf-8")

            loaded = load_hico_semantic_fixture(path)

        self.assertIn("records", loaded)
        self.assertEqual(len(loaded["records"]), 1)
        self.assertEqual(loaded["records"][0]["triplet_key"], "person|ride|motorcycle")

    def test_build_hico_semantic_records_accepts_references_field(self) -> None:
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
                            "predicate": ["ride"],
                            "subject_bbox": [0.1, 0.2, 0.3, 0.4],
                            "object_bbox": [0.2, 0.3, 0.4, 0.5],
                            "references": ["A person is riding a bicycle."],
                        }
                    ],
                }
            ],
        }

        records = build_hico_semantic_records(payload)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["triplet_key"], "person|ride|bicycle")
        self.assertEqual(records[0]["references"], ["A person is riding a bicycle."])


if __name__ == "__main__":
    unittest.main()
