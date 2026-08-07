"""*** 用于生成前端、预测与评估产物的导出构建逻辑。 ***"""

from __future__ import annotations

from pathlib import Path


def export_frontend_response(master_annotation: dict, uploaded_image_url: str, size_bytes: int) -> dict:
    image = master_annotation["image"]
    return {
        "message": "Image uploaded and annotated successfully.",
        "filename": image["file_name"],
        "uploaded_image_url": uploaded_image_url,
        "size_bytes": size_bytes,
        "schema_version": master_annotation["schema_version"],
        "image_id": image["image_id"],
        "caption": master_annotation["caption"],
        "annotation_count": len(master_annotation["annotations"]),
        "validation": {
            "is_valid": master_annotation["validation"]["is_valid"],
            "errors": master_annotation["validation"]["errors"],
            "warnings": master_annotation["validation"]["warnings"],
        },
        "source": master_annotation["source"],
        "annotations": master_annotation["annotations"],
    }


def export_prediction_file(master_annotation: dict) -> dict:
    image = master_annotation["image"]
    generation = master_annotation["generation"]
    return {
        "schema_version": master_annotation["schema_version"],
        "image_id": image["image_id"],
        "image": {
            "image_id": image["image_id"],
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
        },
        "caption": master_annotation["caption"],
        "predictions": [
            {
                # "annotation_id": annotation["annotation_id"],
                "image_id": image["image_id"],
                "subject": annotation["subject"],
                "object": annotation["object"],
                "predicate": annotation["predicate"],
                "subject_bbox": annotation["subject_bbox"],
                "object_bbox": annotation["object_bbox"],
                "category": annotation["category"],
                "bounding_box": annotation["bounding_box"],
                "sentences": annotation["sentences"],
                "score": annotation.get("score"),
                "dataset_labels": annotation.get("dataset_labels"),
                "generation": {
                    "provider": generation["provider"],
                    "model": generation["model"],
                    "prompt_mode": generation["prompt_mode"],
                },
            }
            for annotation in master_annotation["annotations"]
        ],
    }


def export_ground_truth_record(master_annotation: dict) -> dict:
    review = master_annotation["review"]
    if review["status"] != "approved":
        raise ValueError("Ground truth export requires review.status to be 'approved'.")

    image = master_annotation["image"]
    return {
        "schema_version": master_annotation["schema_version"],
        "images": [
            {
                "id": image["image_id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
        ],
        "caption": master_annotation["caption"],
        "annotations": [
            {
                # "id": annotation["annotation_id"],
                "image_id": image["image_id"],
                "subject": annotation["subject"],
                "object": annotation["object"],
                "predicate": annotation["predicate"],
                "subject_bbox": annotation["subject_bbox"],
                "object_bbox": annotation["object_bbox"],
                "category": annotation["category"],
                "bounding_box": annotation["bounding_box"],
                "sentences": annotation["sentences"],
                "score": annotation.get("score"),
                "review": {
                    "status": review["status"],
                    "reviewed_by": review["reviewed_by"],
                    "reviewed_at": review["reviewed_at"],
                },
            }
            for annotation in master_annotation["annotations"]
        ],
        "categories": [],
    }


def export_evaluation_record(master_annotation: dict) -> dict:
    review = master_annotation["review"]
    if review["status"] == "rejected":
        raise ValueError("Rejected records are excluded from evaluation exports.")

    return {
        "schema_version": master_annotation["schema_version"],
        "image_id": master_annotation["image"]["image_id"],
        "caption": master_annotation["caption"],
        "bbox_export": [
            {
                "image_id": master_annotation["image"]["image_id"],
                # "annotation_id": annotation["annotation_id"],
                "subject": annotation["subject"],
                "object": annotation["object"],
                "predicate": annotation["predicate"],
                "subject_bbox": annotation["subject_bbox"],
                "object_bbox": annotation["object_bbox"],
                "category": annotation["category"],
                "bounding_box": annotation["bounding_box"],
                "score": annotation.get("score"),
                "dataset_labels": annotation.get("dataset_labels"),
            }
            for annotation in master_annotation["annotations"]
        ],
        "text_export": [
            {
                "image_id": master_annotation["image"]["image_id"],
                # "annotation_id": annotation["annotation_id"],
                "subject": annotation["subject"],
                "object": annotation["object"],
                "predicate": annotation["predicate"],
                "category": annotation["category"],
                "sentences": annotation["sentences"],
            }
            for annotation in master_annotation["annotations"]
        ],
    }


def build_json_artifact_path(directory: Path, image_stem: str, suffix: str) -> Path:
    return directory / f"{image_stem}-{suffix}.json"
