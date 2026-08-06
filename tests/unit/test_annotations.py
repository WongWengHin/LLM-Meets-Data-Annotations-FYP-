from __future__ import annotations

import unittest
from pathlib import Path

from app.domain.annotations.exporters import export_frontend_response, export_prediction_file
from app.domain.annotations.normalizer import parse_and_normalize_model_output
from app.domain.annotations.validator import validate_annotation


class AnnotationFormatTests(unittest.TestCase):
    def test_object_detection_payload_without_caption_is_normalized_and_valid(self) -> None:
        raw_output = '''
        {
          "annotations": [
            {
              "annotation_id": "000001_001",
              "image_id": "000001",
              "class": "dog",
              "bbox": [0.245, 0.312, 0.624, 0.856],
              "sentences": [
                "A dog is visible in the image.",
                "The dog has brown fur and a short coat.",
                "The dog is standing on the ground."
              ]
            }
          ]
        }
        '''

        normalized = parse_and_normalize_model_output(raw_output, Path("sample.jpg"))

        self.assertEqual(normalized["annotation_type"], "object_detection")
        self.assertEqual(len(normalized["annotations"]), 1)
        self.assertEqual(normalized["annotations"][0]["class"], "dog")
        self.assertEqual(normalized["annotations"][0]["bbox"], [0.245, 0.312, 0.624, 0.856])
        self.assertEqual(normalized["annotations"][0]["annotation_id"], "000001_001")

        validation = validate_annotation(normalized)
        self.assertTrue(validation["is_valid"], validation)
        self.assertEqual(validation["errors"], [])

    def test_object_detection_prediction_export_omits_caption_and_duplicate_bbox_fields(self) -> None:
        master_annotation = {
            "schema_version": "3.0.0",
            "image": {
                "image_id": "000001",
                "file_name": "000001.jpg",
                "width": 640,
                "height": 427,
            },
            "caption": "",
            "annotations": [
                {
                    "annotation_type": "object_detection",
                    "annotation_id": "000001_001",
                    "image_id": "000001",
                    "class": "dog",
                    "bbox": [0.245, 0.312, 0.624, 0.856],
                    "category": "dog",
                    "bounding_box": [0.245, 0.312, 0.624, 0.856],
                    "sentences": ["A dog is visible in the image."],
                    "score": None,
                    "dataset_labels": None,
                }
            ],
            "generation": {
                "provider": "openai",
                "model": "gpt-5.5",
                "prompt_mode": "standard",
            },
        }

        exported = export_prediction_file(master_annotation)
        prediction = exported["predictions"][0]

        self.assertNotIn("caption", exported)
        self.assertEqual(prediction["class"], "dog")
        self.assertEqual(prediction["bbox"], [0.245, 0.312, 0.624, 0.856])
        self.assertNotIn("category", prediction)
        self.assertNotIn("bounding_box", prediction)

    def test_frontend_response_for_object_detection_uses_compact_annotation_shape(self) -> None:
        master_annotation = {
            "schema_version": "3.0.0",
            "image": {
                "image_id": "000001",
                "file_name": "000001.jpg",
                "width": 640,
                "height": 427,
            },
            "caption": "",
            "annotations": [
                {
                    "annotation_type": "object_detection",
                    "annotation_id": "000001_001",
                    "image_id": "000001",
                    "class": "dog",
                    "bbox": [0.245, 0.312, 0.624, 0.856],
                    "category": "dog",
                    "bounding_box": [0.245, 0.312, 0.624, 0.856],
                    "sentences": ["A dog is visible in the image."],
                    "score": None,
                    "dataset_labels": None,
                }
            ],
            "task": "object_detection",
            "validation": {
                "is_valid": True,
                "errors": [],
                "warnings": [],
            },
            "source": {
                "type": "model_generated",
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-5.5",
                "prompt_mode": "standard",
            },
        }

        frontend_response = export_frontend_response(master_annotation, "/uploads/000001.jpg", 123)
        frontend_annotation = frontend_response["annotations"][0]

        self.assertEqual(frontend_annotation["annotation_id"], "000001_001")
        self.assertEqual(frontend_annotation["image_id"], "000001")
        self.assertEqual(frontend_annotation["class"], "dog")
        self.assertEqual(frontend_annotation["bbox"], [0.245, 0.312, 0.624, 0.856])
        self.assertEqual(frontend_annotation["sentences"], ["A dog is visible in the image."])
        self.assertNotIn("subject", frontend_annotation)
        self.assertNotIn("object", frontend_annotation)
        self.assertNotIn("predicate", frontend_annotation)
        self.assertNotIn("subject_bbox", frontend_annotation)
        self.assertNotIn("object_bbox", frontend_annotation)
        self.assertNotIn("category", frontend_annotation)
        self.assertNotIn("bounding_box", frontend_annotation)

    def test_validator_rejects_object_detection_legacy_hoi_keys(self) -> None:
        normalized = {
            "annotation_type": "object_detection",
            "caption": "",
            "annotations": [
                {
                    "annotation_type": "object_detection",
                    "annotation_id": "000001_001",
                    "image_id": "000001",
                    "class": "dog",
                    "bbox": [0.245, 0.312, 0.624, 0.856],
                    "subject": None,
                    "object": None,
                    "predicate": None,
                    "subject_bbox": None,
                    "object_bbox": None,
                    "category": "apple",
                    "bounding_box": [0.245, 0.312, 0.624, 0.856],
                    "sentences": ["A dog is visible in the image."],
                }
            ],
        }

        validation = validate_annotation(normalized)
        self.assertFalse(validation["is_valid"])
        self.assertTrue(any("subject" in error or "object" in error or "predicate" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
