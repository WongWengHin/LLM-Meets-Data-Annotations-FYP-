"""*** 单编排 agent 的图像标注与导出工作流实现。 ***"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.agent.policies import advance_strategy_after_failure, should_retry
from app.agent.state import AnnotationTaskState, create_initial_task_state
from app.agent.tools import (
    build_artifact_paths,
    build_export_payloads,
    build_frontend_payload,
    run_llm_tool,
    run_normalize_tool,
    run_preprocess_tool,
    run_schema_tool,
    run_validate_tool,
)
from app.agent.trace import append_trace_event, write_json, write_task_state, write_trace
from app.core.config import AppConfig


class AnnotationAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def annotate_image(self, image_path: Path) -> dict:
        state = create_initial_task_state(image_path)
        state.status = "running"
        state.artifacts = self._initial_artifact_map(build_artifact_paths(self.config, image_path.stem))
        append_trace_event(state, stage="created", status="success", details={"image": image_path.name})
        self._persist_state(state)

        try:
            inference_image = self._preprocess(state)
            self._run_attempts(state, inference_image.image_path)
        except Exception as exc:
            state.status = "failed"
            state.last_error = str(exc)
            append_trace_event(state, stage=state.stage, status="error", details={"error": str(exc)})
            self._finalize(state)
            raise

        self._finalize(state)
        return self._build_response(state)

    def _preprocess(self, state: AnnotationTaskState):
        state.stage = "preprocess"
        inference_image = run_preprocess_tool(state.image_path, self.config)
        state.inference_image = {
            "file_name": inference_image.image_path.name,
            "resized": inference_image.resized,
            "original_size": {
                "width": inference_image.original_size[0],
                "height": inference_image.original_size[1],
            },
            "inference_size": {
                "width": inference_image.inference_size[0],
                "height": inference_image.inference_size[1],
            },
        }
        append_trace_event(
            state,
            stage="preprocess",
            status="success",
            details=state.inference_image,
        )
        if inference_image.resized:
            state.artifacts["inference_image_file"] = str(
                inference_image.image_path.relative_to(self.config.paths.base_dir)
            )
        self._persist_state(state)
        return inference_image

    def _run_attempts(self, state: AnnotationTaskState, inference_image_path: Path) -> None:
        last_exception: Exception | None = None

        while state.attempt_count < state.max_attempts:
            state.attempt_count += 1
            state.last_error = None
            self._persist_state(state)

            try:
                self._annotate(state, inference_image_path)
                self._normalize(state)
                self._validate(state)

                if state.validation and state.validation.get("is_valid"):
                    self._export(state)
                    state.status = "completed"
                    return

                last_exception = ValueError("Validation failed.")
                state.last_error = "Validation failed."
                append_trace_event(
                    state,
                    stage="validate",
                    status="failed",
                    details={"errors": state.validation.get("errors", []) if state.validation else []},
                )
            except Exception as exc:
                last_exception = exc
                state.last_error = str(exc)
                append_trace_event(state, stage=state.stage, status="error", details={"error": str(exc)})

            self._persist_state(state)
            if not should_retry(state, last_exception):
                break
            advance_strategy_after_failure(state)
            append_trace_event(
                state,
                stage="annotate",
                status="retrying",
                details={"next_strategy": state.strategy},
            )

        state.status = "review_required"
        state.review_required = True
        state.review_reason = self._build_review_reason(state, last_exception)
        if state.master_annotation is not None:
            state.master_annotation.setdefault("review", {})
            state.master_annotation["review"]["status"] = "review_required"
            state.master_annotation["review"]["reviewed_by"] = None
            state.master_annotation["review"]["reviewed_at"] = None
            state.master_annotation["review"]["reason"] = state.review_reason

    def _annotate(self, state: AnnotationTaskState, inference_image_path: Path) -> None:
        state.stage = "annotate"
        prompt_mode, raw_model_output = run_llm_tool(inference_image_path, self.config)
        state.prompt_mode = prompt_mode
        state.raw_model_output = raw_model_output
        append_trace_event(
            state,
            stage="annotate",
            status="success",
            details={"prompt_mode": prompt_mode},
        )
        self._write_optional_artifact(
            "raw_output_file",
            state,
            {
                "raw_model_output": raw_model_output,
                "prompt_mode": prompt_mode,
                "attempt": state.attempt_count,
                "inference_image": state.inference_image,
            },
        )
        self._persist_state(state)

    def _normalize(self, state: AnnotationTaskState) -> None:
        state.stage = "normalize"
        state.normalized_annotation = run_normalize_tool(state.raw_model_output or "", state.image_path)
        append_trace_event(
            state,
            stage="normalize",
            status="success",
            details={"annotation_count": len(state.normalized_annotation.get("annotations", []))},
        )
        self._write_optional_artifact("normalized_output_file", state, state.normalized_annotation)
        self._persist_state(state)

    def _validate(self, state: AnnotationTaskState) -> None:
        state.stage = "validate"
        state.validation = run_validate_tool(state.normalized_annotation or {})
        append_trace_event(
            state,
            stage="validate",
            status="success" if state.validation.get("is_valid") else "failed",
            details=state.validation,
        )
        self._write_optional_artifact("validation_report_file", state, state.validation)
        if state.raw_model_output and state.normalized_annotation and state.validation:
            state.master_annotation = run_schema_tool(
                image_path=state.image_path,
                raw_model_output=state.raw_model_output,
                normalized_annotation=state.normalized_annotation,
                validation=state.validation,
                prompt_mode=state.prompt_mode or "standard",
                config=self.config,
            )
        self._persist_state(state)

    def _export(self, state: AnnotationTaskState) -> None:
        state.stage = "export"
        if state.master_annotation is None:
            raise RuntimeError("Master annotation is not available for export.")
        prediction_payload, evaluation_payload = build_export_payloads(state.master_annotation)
        self._write_optional_artifact("master_annotation_file", state, state.master_annotation)
        self._write_optional_artifact("prediction_file", state, prediction_payload)
        self._write_optional_artifact("evaluation_file", state, evaluation_payload)
        append_trace_event(state, stage="export", status="success")
        self._persist_state(state)

    def _build_response(self, state: AnnotationTaskState) -> dict:
        if state.master_annotation is not None:
            response = build_frontend_payload(state.master_annotation, state.image_path)
        else:
            response = {
                "message": "Image uploaded but annotation could not be completed automatically.",
                "filename": state.image_path.name,
                "uploaded_image_url": f"/uploads/{state.image_path.name}",
                "size_bytes": state.image_path.stat().st_size,
                "schema_version": "3.0.0",
                "image_id": state.image_path.stem,
                "caption": "No caption returned by the model.",
                "annotation_count": 0,
                "validation": {
                    "is_valid": False,
                    "errors": [state.last_error or "Unknown annotation error."],
                    "warnings": [],
                },
                "source": {"type": "agent_failed"},
                "annotations": [],
            }

        if state.status == "review_required":
            response["message"] = "Image annotated with review required."
        elif state.status == "completed":
            response["message"] = "Image uploaded and annotated successfully."

        response["agent"] = {
            "task_id": state.task_id,
            "task_status": state.status,
            "stage": state.stage,
            "strategy": state.strategy,
            "attempt_count": state.attempt_count,
            "review_required": state.review_required,
            "review_reason": state.review_reason,
        }
        response["artifacts"] = state.artifacts
        return response

    def _persist_state(self, state: AnnotationTaskState) -> None:
        artifact_paths = build_artifact_paths(self.config, state.image_path.stem)
        write_task_state(artifact_paths["task_state_file"], state)
        state.artifacts["task_state_file"] = str(
            artifact_paths["task_state_file"].relative_to(self.config.paths.base_dir)
        )

    def _finalize(self, state: AnnotationTaskState) -> None:
        state.completed_at = datetime.now(timezone.utc).isoformat()
        artifact_paths = build_artifact_paths(self.config, state.image_path.stem)
        if state.master_annotation is not None:
            self._write_optional_artifact("master_annotation_file", state, state.master_annotation)
        write_trace(artifact_paths["orchestration_trace_file"], state)
        state.artifacts["orchestration_trace_file"] = str(
            artifact_paths["orchestration_trace_file"].relative_to(self.config.paths.base_dir)
        )
        write_task_state(artifact_paths["task_state_file"], state)
        state.artifacts["task_state_file"] = str(
            artifact_paths["task_state_file"].relative_to(self.config.paths.base_dir)
        )

    def _write_optional_artifact(self, key: str, state: AnnotationTaskState, payload: dict) -> None:
        artifact_paths = build_artifact_paths(self.config, state.image_path.stem)
        path = artifact_paths[key]
        write_json(path, payload)
        state.artifacts[key] = str(path.relative_to(self.config.paths.base_dir))

    def _build_review_reason(self, state: AnnotationTaskState, exc: Exception | None) -> str:
        if exc is not None:
            return f"Automatic annotation requires review after {state.attempt_count} attempt(s): {exc}"
        return f"Automatic annotation requires review after {state.attempt_count} attempt(s)."

    def _initial_artifact_map(self, artifact_paths: dict[str, Path]) -> dict[str, str | None]:
        relative: dict[str, str | None] = {key: None for key in artifact_paths}
        relative["inference_image_file"] = None
        return relative
