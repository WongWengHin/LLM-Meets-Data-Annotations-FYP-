"""*** 标注 agent 调用的确定性工具封装层。 ***"""

from __future__ import annotations

from pathlib import Path

from app.domain.annotations.exporters import (
    build_json_artifact_path,
    export_evaluation_record,
    export_frontend_response,
    export_prediction_file,
)
from app.domain.annotations.normalizer import parse_and_normalize_model_output
from app.domain.annotations.schema import build_master_annotation
from app.domain.annotations.validator import validate_annotation
from app.services.llm.client import request_image_annotation
from app.core.config import AppConfig
from app.domain.images.preprocess import InferenceImageResult, prepare_inference_image


def run_preprocess_tool(image_path: Path, config: AppConfig) -> InferenceImageResult:
    return prepare_inference_image(image_path, config.paths.inference_images_dir, config.image_preprocess)


def run_llm_tool(inference_image_path: Path, config: AppConfig) -> tuple[str, str]:
    return request_image_annotation(inference_image_path, config.llm)


def run_normalize_tool(raw_model_output: str, image_path: Path) -> dict:
    return parse_and_normalize_model_output(raw_model_output, image_path)


def run_validate_tool(normalized_annotation: dict) -> dict:
    return validate_annotation(normalized_annotation)


def run_schema_tool(
    *,
    image_path: Path,
    raw_model_output: str,
    normalized_annotation: dict,
    validation: dict,
    prompt_mode: str,
    config: AppConfig,
) -> dict:
    return build_master_annotation(
        image_path=image_path,
        raw_model_output=raw_model_output,
        normalized_annotation=normalized_annotation,
        validation=validation,
        prompt_mode=prompt_mode,
        provider=config.llm.provider,
        model=config.llm.model,
    )


def build_frontend_payload(master_annotation: dict, image_path: Path) -> dict:
    return export_frontend_response(
        master_annotation,
        uploaded_image_url=f"/uploads/{image_path.name}",
        size_bytes=image_path.stat().st_size,
    )


def build_artifact_paths(config: AppConfig, image_stem: str) -> dict[str, Path]:
    return {
        "raw_output_file": build_json_artifact_path(config.paths.raw_outputs_dir, image_stem, "raw-output"),
        "normalized_output_file": build_json_artifact_path(config.paths.normalized_outputs_dir, image_stem, "normalized"),
        "validation_report_file": build_json_artifact_path(config.paths.validation_reports_dir, image_stem, "validation"),
        "master_annotation_file": build_json_artifact_path(config.paths.master_annotations_dir, image_stem, "master-annotation"),
        "prediction_file": build_json_artifact_path(config.paths.predictions_dir, image_stem, "prediction"),
        "evaluation_file": build_json_artifact_path(config.paths.evaluation_exports_dir, image_stem, "evaluation"),
        "orchestration_trace_file": build_json_artifact_path(config.paths.orchestration_traces_dir, image_stem, "trace"),
        "task_state_file": build_json_artifact_path(config.paths.task_states_dir, image_stem, "task"),
    }


def build_export_payloads(master_annotation: dict) -> tuple[dict, dict]:
    return export_prediction_file(master_annotation), export_evaluation_record(master_annotation)
