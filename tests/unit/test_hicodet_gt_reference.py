from __future__ import annotations

import unittest

from app.evaluation.hicodet_gt_reference import build_hicodet_gt_fixture


class HicoGtReferenceTests(unittest.TestCase):
    def test_build_hicodet_gt_fixture_expands_predicates_and_converts_boxes(self) -> None:
        payload = {
            "dataset_metadata": {"name": "HICO-DET dataset"},
            "images": [
                {
                    "id": 2149,
                    "file_name": "HICO_train2015_00000001.jpg",
                    "annotations": [
                        {
                            "subject": "person",
                            "object": "motorcycle",
                            "subject_bbox": [0.325, 0.06875, 0.3421875, 0.55625],
                            "object_bbox": [0.0953125, 0.20833333333333334, 0.796875, 0.6270833333333333],
                            "predicate": ["race", "ride"],
                            "sentences": ["A person is riding a motorcycle."],
                        }
                    ],
                }
            ],
        }

        fixture = build_hicodet_gt_fixture(payload)

        self.assertEqual(len(fixture["images"]), 1)
        self.assertEqual(len(fixture["annotations"]), 2)
        self.assertEqual(fixture["images"][0]["image_id"], "2149")
        self.assertEqual(fixture["images"][0]["width"], 1)
        self.assertEqual(fixture["images"][0]["height"], 1)
        self.assertEqual(fixture["annotations"][0]["subject_bbox"], [0.325, 0.06875, 0.6671875, 0.625])
        self.assertEqual(fixture["annotations"][0]["object_bbox"], [0.0953125, 0.20833333333333334, 0.8921875, 0.8354166666666667])
        self.assertEqual(fixture["annotations"][0]["triplet_key"], "person|race|motorcycle")
        self.assertEqual(fixture["annotations"][1]["triplet_key"], "person|ride|motorcycle")


if __name__ == "__main__":
    unittest.main()
