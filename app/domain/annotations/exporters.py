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
            _build_prediction_payload(annotation, image["image_id"], generation)
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
            _build_ground_truth_payload(annotation, image["image_id"], review)
            for annotation in master_annotation["annotations"]
        ],
        "categories": [],
    }


def export_evaluation_record(master_annotation: dict) -> dict:
    review = master_annotation["review"]
    if review["status"] == "rejected":
        raise ValueError("Rejected records are excluded from evaluation exports.")

    image_id = master_annotation["image"]["image_id"]
    return {
        "schema_version": master_annotation["schema_version"],
        "image_id": image_id,
        "caption": master_annotation["caption"],
        "bbox_export": [
            _build_evaluation_bbox_payload(annotation, image_id)
            for annotation in master_annotation["annotations"]
        ],
        "text_export": [
            _build_evaluation_text_payload(annotation, image_id)
            for annotation in master_annotation["annotations"]
        ],
    }


def _build_prediction_payload(annotation: dict, image_id: str, generation: dict) -> dict:
    if annotation.get("annotation_type") == "object_detection":
        return {
            "annotation_id": annotation["annotation_id"],
            "image_id": image_id,
            "class": annotation.get("class"),
            "bbox": annotation.get("bbox"),
            "category": annotation.get("category"),
            "bounding_box": annotation.get("bounding_box"),
            "sentences": annotation.get("sentences"),
            "score": annotation.get("score"),
            "dataset_labels": annotation.get("dataset_labels"),
            "generation": {
                "provider": generation["provider"],
                "model": generation["model"],
                "prompt_mode": generation["prompt_mode"],
            },
        }

    return {
        "annotation_id": annotation["annotation_id"],
        "image_id": image_id,
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


def _build_ground_truth_payload(annotation: dict, image_id: str, review: dict) -> dict:
    if annotation.get("annotation_type") == "object_detection":
        return {
            "id": annotation["annotation_id"],
            "image_id": image_id,
            "class": annotation.get("class"),
            "bbox": annotation.get("bbox"),
            "category": annotation.get("category"),
            "bounding_box": annotation.get("bounding_box"),
            "sentences": annotation.get("sentences"),
            "score": annotation.get("score"),
            "review": {
                "status": review["status"],
                "reviewed_by": review["reviewed_by"],
                "reviewed_at": review["reviewed_at"],
            },
        }

    return {
        "id": annotation["annotation_id"],
        "image_id": image_id,
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


def _build_evaluation_bbox_payload(annotation: dict, image_id: str) -> dict:
    if annotation.get("annotation_type") == "object_detection":
        return {
            "image_id": image_id,
            "annotation_id": annotation["annotation_id"],
            "class": annotation.get("class"),
            "bbox": annotation.get("bbox"),
            "category": annotation.get("category"),
            "bounding_box": annotation.get("bounding_box"),
            "score": annotation.get("score"),
            "dataset_labels": annotation.get("dataset_labels"),
        }

    return {
        "image_id": image_id,
        "annotation_id": annotation["annotation_id"],
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


def _build_evaluation_text_payload(annotation: dict, image_id: str) -> dict:
    if annotation.get("annotation_type") == "object_detection":
        return {
            "image_id": image_id,
            "annotation_id": annotation["annotation_id"],
            "class": annotation.get("class"),
            "category": annotation.get("category"),
            "sentences": annotation.get("sentences"),
        }

    return {
        "image_id": image_id,
        "annotation_id": annotation["annotation_id"],
        "subject": annotation["subject"],
        "object": annotation["object"],
        "predicate": annotation["predicate"],
        "category": annotation["category"],
        "sentences": annotation["sentences"],
    }


def build_json_artifact_path(directory: Path, image_stem: str, suffix: str) -> Path:
    return directory / f"{image_stem}-{suffix}.json"
