from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.routes import _is_completed_annotation, _unique_destination
from app.core.config import AppConfig, AppPaths, AuthConfig, ImagePreprocessConfig, LLMConfig
from app.core.user_database import get_batch_run


class ApiRoutesTests(unittest.TestCase):
    def test_only_valid_completed_annotations_are_saved(self) -> None:
        self.assertTrue(_is_completed_annotation({
            "agent": {"task_status": "completed"},
            "validation": {"is_valid": True},
        }))
        self.assertFalse(_is_completed_annotation({
            "agent": {"task_status": "review_required"},
            "validation": {"is_valid": False},
        }))

    def test_unique_destination_keeps_original_filename(self) -> None:
        upload_dir = Path(".tmp_tests") / f"api_routes_uploads_{uuid.uuid4().hex}"
        upload_dir.mkdir(parents=True)
        try:
            existing = upload_dir / "HICO_train2015_00000001.jpg"
            existing.write_bytes(b"old-bytes")

            destination = _unique_destination(upload_dir, "HICO_train2015_00000001.jpg")

            self.assertEqual(destination, existing)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    def test_backend_batch_upload_processes_multiple_images_under_one_batch(self) -> None:
        base_dir = Path(".tmp_tests") / f"api_routes_batch_{uuid.uuid4().hex}"
        base_dir.mkdir(parents=True)
        try:
            config = _test_config(base_dir)
            _create_directories(config.paths)
            (config.paths.frontend_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            app = create_app(config)

            with (
                TestClient(app) as client,
                patch("app.api.routes.AnnotationAgent", return_value=_FakeAnnotationAgent()),
            ):
                response = client.post(
                    "/api/batches/upload",
                    files=[
                        ("images", ("one.jpg", b"one-bytes", "image/jpeg")),
                        ("images", ("two.jpg", b"two-bytes", "image/jpeg")),
                    ],
                    data={"prompt_mode": "hoi"},
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            batch = payload["batch"]
            self.assertEqual(batch["total_images"], 2)
            self.assertEqual(batch["processed_images"], 2)
            self.assertEqual(batch["successful_images"], 2)
            self.assertEqual(batch["failed_images"], 0)
            self.assertEqual(batch["prompt_mode"], "hoi")
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["failures"], [])
            self.assertEqual({item["batch_id"] for item in payload["results"]}, {batch["batch_id"]})

            stored_batch = get_batch_run(config.paths.user_db_path, batch["batch_id"])
            self.assertIsNotNone(stored_batch)
            self.assertEqual(stored_batch["status"], "completed")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


class _FakeAnnotationAgent:
    def annotate_image(self, image_path: Path, *, prompt_mode: str) -> dict:
        return {
            "message": "Image uploaded and annotated successfully.",
            "filename": image_path.name,
            "uploaded_image_url": f"/uploads/{image_path.name}",
            "size_bytes": image_path.stat().st_size,
            "schema_version": "3.0.0",
            "image_id": image_path.stem,
            "caption": "A person is holding a cup.",
            "annotation_count": 1,
            "validation": {"is_valid": True, "errors": [], "warnings": []},
            "source": {"type": "model_generated"},
            "annotations": [
                {
                    "annotation_id": f"{image_path.stem}_001",
                    "image_id": image_path.stem,
                    "subject": "person",
                    "object": "cup",
                    "predicate": "hold",
                    "subject_bbox": [0.1, 0.1, 0.5, 0.9],
                    "object_bbox": [0.5, 0.3, 0.7, 0.6],
                    "category": "person hold cup",
                    "bounding_box": [0.1, 0.1, 0.7, 0.9],
                    "sentences": ["A person is holding a cup."],
                }
            ],
            "agent": {
                "task_id": f"task_{image_path.stem}",
                "task_status": "completed",
                "stage": "export",
                "strategy": "llm_only",
                "attempt_count": 1,
                "review_required": False,
                "review_reason": None,
            },
            "artifacts": {
                "raw_output_file": None,
                "master_annotation_file": None,
                "prediction_file": None,
                "evaluation_file": None,
                "inference_image_file": None,
                "normalized_output_file": None,
                "validation_report_file": None,
                "orchestration_trace_file": None,
                "task_state_file": None,
            },
        }


def _test_config(base_dir: Path) -> AppConfig:
    paths = AppPaths(
        base_dir=base_dir,
        frontend_dir=base_dir / "frontend",
        data_dir=base_dir / "data",
        user_db_path=base_dir / "data" / "user_results.db",
        uploads_dir=base_dir / "data" / "uploads",
        inference_images_dir=base_dir / "data" / "inference_images",
        raw_outputs_dir=base_dir / "data" / "raw_outputs",
        master_annotations_dir=base_dir / "data" / "master_annotations",
        predictions_dir=base_dir / "data" / "predictions",
        ground_truth_dir=base_dir / "data" / "ground_truth",
        evaluation_exports_dir=base_dir / "data" / "evaluation_exports",
        normalized_outputs_dir=base_dir / "data" / "normalized_outputs",
        validation_reports_dir=base_dir / "data" / "validation_reports",
        orchestration_traces_dir=base_dir / "data" / "orchestration_traces",
        task_states_dir=base_dir / "data" / "task_states",
    )
    return AppConfig(
        llm=LLMConfig(
            provider="annotation",
            api_key="test-key",
            base_url="https://example.invalid",
            model="gpt-5.2",
            timeout_seconds=30,
        ),
        image_preprocess=ImagePreprocessConfig(enable_resize=False, max_side=768, jpeg_quality=85),
        paths=paths,
        auth=AuthConfig(
            enabled=False,
            require_login=False,
            session_secret_key="unit-test-secret",
            google_client_id="",
            google_client_secret="",
            google_redirect_uri="",
        ),
        host="127.0.0.1",
        port=8000,
    )


def _create_directories(paths: AppPaths) -> None:
    for path in (
        paths.frontend_dir,
        paths.data_dir,
        paths.uploads_dir,
        paths.inference_images_dir,
        paths.raw_outputs_dir,
        paths.master_annotations_dir,
        paths.predictions_dir,
        paths.ground_truth_dir,
        paths.evaluation_exports_dir,
        paths.normalized_outputs_dir,
        paths.validation_reports_dir,
        paths.orchestration_traces_dir,
        paths.task_states_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
