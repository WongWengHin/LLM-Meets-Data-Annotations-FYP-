"""*** 旧版确定性标注流水线的辅助逻辑。 ***"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.annotations.exporters import (
    build_json_artifact_path,
    export_evaluation_record,
    export_frontend_response,
    export_ground_truth_record,
    export_prediction_file,
)
from app.domain.annotations.normalizer import parse_and_normalize_model_output
from app.domain.annotations.schema import build_master_annotation
from app.domain.annotations.validator import validate_annotation
from app.services.llm.client import request_image_annotation
from app.core.config import AppConfig
from app.core.errors import AnnotationValidationError
from app.domain.images.preprocess import prepare_inference_image


def run_annotation_pipeline(image_path: Path, config: AppConfig) -> dict:
    print(f"[pipeline] Starting annotation pipeline for {image_path.name}")
    print("[pipeline] Preparing inference image...")
    inference_image = prepare_inference_image(
        image_path,
        config.paths.inference_images_dir,
        config.image_preprocess,
    )
    if inference_image.resized:
        print(
            "[pipeline] Inference image ready: "
            f"{inference_image.image_path.name} "
            f"({inference_image.original_size[0]}x{inference_image.original_size[1]} -> "
            f"{inference_image.inference_size[0]}x{inference_image.inference_size[1]})"
        )
    else:
        print(
            "[pipeline] Reusing original image for inference: "
            f"{inference_image.image_path.name} "
            f"({inference_image.inference_size[0]}x{inference_image.inference_size[1]})"
        )

    print("[pipeline] Requesting annotation from LLM...")
    active_prompt_mode, raw_model_output = request_image_annotation(inference_image.image_path, config.llm)
    print("[pipeline] LLM response received.")

    print("[pipeline] Normalizing model output...")
    normalized_annotation = parse_and_normalize_model_output(raw_model_output, image_path)
    print("[pipeline] Validating normalized annotation...")
    validation = validate_annotation(normalized_annotation)

    if not validation["is_valid"]:
        print(f"[pipeline] Validation failed with {len(validation['errors'])} error(s).")
        raise AnnotationValidationError(validation["errors"])

    print(
        "[pipeline] Validation passed: "
        f"{len(normalized_annotation.get('annotations', []))} annotation(s), "
        f"{len(validation.get('warnings', []))} warning(s)."
    )

    print("[pipeline] Building master annotation and exports...")
    master_annotation = build_master_annotation(
        image_path=image_path,
        raw_model_output=raw_model_output,
        normalized_annotation=normalized_annotation,
        validation=validation,
        prompt_mode=active_prompt_mode,
        provider=config.llm.provider,
        model=config.llm.model,
    )

    raw_output_path = build_json_artifact_path(config.paths.raw_outputs_dir, image_path.stem, "raw-output")
    master_annotation_path = build_json_artifact_path(config.paths.master_annotations_dir, image_path.stem, "master-annotation")
    prediction_path = build_json_artifact_path(config.paths.predictions_dir, image_path.stem, "prediction")
    evaluation_path = build_json_artifact_path(config.paths.evaluation_exports_dir, image_path.stem, "evaluation")

    _write_json(
        raw_output_path,
        {
            "raw_model_output": raw_model_output,
            "prompt_mode": active_prompt_mode,
            "inference_image": {
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
            },
        },
    )
    _write_json(master_annotation_path, master_annotation)
    _write_json(prediction_path, export_prediction_file(master_annotation))
    _write_json(evaluation_path, export_evaluation_record(master_annotation))

    frontend_response = export_frontend_response(
        master_annotation,
        uploaded_image_url=f"/uploads/{image_path.name}",
        size_bytes=image_path.stat().st_size,
    )
    frontend_response["artifacts"] = {
        "raw_output_file": str(raw_output_path.relative_to(config.paths.base_dir)),
        "master_annotation_file": str(master_annotation_path.relative_to(config.paths.base_dir)),
        "prediction_file": str(prediction_path.relative_to(config.paths.base_dir)),
        "evaluation_file": str(evaluation_path.relative_to(config.paths.base_dir)),
        "inference_image_file": (
            str(inference_image.image_path.relative_to(config.paths.base_dir))
            if inference_image.resized
            else None
        ),
    }
    print(f"[pipeline] Completed annotation pipeline for {image_path.name}")
    return frontend_response


def export_ground_truth_from_master(master_annotation: dict, output_path: Path) -> None:
    _write_json(output_path, export_ground_truth_record(master_annotation))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
