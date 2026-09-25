"""*** 标准化标注结果的校验规则定义。 ***"""

from __future__ import annotations

from app.evaluation.hicodet_registry import (
    allowed_predicates_for_object,
    is_hico_object,
    is_hico_verb,
    is_valid_hico_interaction,
    valid_hico_interactions,
)


def validate_annotation(normalized_annotation: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    annotation_type = normalized_annotation.get("annotation_type") or "hoi"
    no_humans_detected = normalized_annotation.get("no_humans_detected") is True
    if annotation_type == "hoi" and not no_humans_detected:
        caption = normalized_annotation.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            errors.append("caption must be a non-empty string.")

    annotations = normalized_annotation.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        if no_humans_detected:
            errors.append("No humans detected. Please try object detection instead.")
        else:
            errors.append("No annotations returned. The model was unable to determine any humans.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            errors.append(f"annotations[{index}] must be an object.")
            continue

        if annotation_type == "object_detection":
            for forbidden_key in (
                "subject",
                "object",
                "predicate",
                "subject_bbox",
                "object_bbox",
                "category",
                "bounding_box",
            ):
                if forbidden_key in annotation:
                    errors.append(
                        f"annotations[{index}].{forbidden_key} is not part of the object-detection schema. "
                        "Use only annotation_id, image_id, class, bbox, sentences, and optional score."
                    )

            annotation_id = annotation.get("annotation_id")
            if not isinstance(annotation_id, str) or not annotation_id.strip():
                errors.append(f"annotations[{index}].annotation_id must be a non-empty string.")

            image_id = annotation.get("image_id")
            if image_id is not None and (not isinstance(image_id, str) or not image_id.strip()):
                errors.append(f"annotations[{index}].image_id must be a non-empty string when provided.")

            class_name = annotation.get("class")
            if not isinstance(class_name, str) or not class_name.strip():
                errors.append(f"annotations[{index}].class must be a non-empty string.")

            bbox = annotation.get("bbox")
            if not _is_valid_bbox(bbox):
                errors.append(f"annotations[{index}].bbox must be a normalized [x1, y1, x2, y2] box.")

            score = annotation.get("score")
            if score is not None and not _is_valid_score(score):
                errors.append(f"annotations[{index}].score must be a number between 0 and 1 when provided.")

            sentences = annotation.get("sentences")
            if not isinstance(sentences, list) or not sentences:
                errors.append(f"annotations[{index}].sentences must be a non-empty list.")
                continue

            cleaned_sentences = [sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip()]
            if len(cleaned_sentences) != len(sentences) or not cleaned_sentences:
                errors.append(f"annotations[{index}].sentences must contain only non-empty strings.")
            elif len({sentence.casefold() for sentence in cleaned_sentences}) != len(cleaned_sentences):
                warnings.append(f"annotations[{index}].sentences contains duplicated sentence content.")
            continue

        subject = annotation.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            errors.append(f"annotations[{index}].subject must be a non-empty string.")
        elif subject != "person":
            errors.append(f"annotations[{index}].subject must be 'person' for HICO-DET HOI annotations.")

        object_name = annotation.get("object")
        if not isinstance(object_name, str) or not object_name.strip():
            errors.append(f"annotations[{index}].object must be a non-empty string.")
        elif object_name == "person":
            errors.append(f"annotations[{index}].object must not be 'person'; HICO-DET uses person as the subject.")
        elif not is_hico_object(object_name):
            errors.append(f"annotations[{index}].object must be a canonical HICO-DET object label.")

        predicate = annotation.get("predicate")
        if not isinstance(predicate, str) or not predicate.strip():
            errors.append(f"annotations[{index}].predicate must be a non-empty string.")
        elif not is_hico_verb(predicate):
            errors.append(f"annotations[{index}].predicate must be a canonical HICO-DET verb label.")

        if (
            isinstance(object_name, str)
            and isinstance(predicate, str)
            and object_name.strip()
            and predicate.strip()
            and object_name != "person"
            and is_hico_object(object_name)
            and is_hico_verb(predicate)
            and valid_hico_interactions()
            and not is_valid_hico_interaction(predicate, object_name)
        ):
            allowed = allowed_predicates_for_object(object_name)
            hint = f" Allowed predicates for {object_name}: {', '.join(allowed)}." if allowed else ""
            errors.append(
                f"annotations[{index}] predicate-object pair is not in the local HICO-DET interaction registry."
                f"{hint}"
            )

        category = annotation.get("category")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"annotations[{index}].category must be a non-empty string.")

        subject_bbox = annotation.get("subject_bbox")
        if not _is_valid_bbox(subject_bbox):
            errors.append(f"annotations[{index}].subject_bbox must be a normalized [x1, y1, x2, y2] box.")

        object_bbox = annotation.get("object_bbox")
        if not _is_valid_bbox(object_bbox):
            errors.append(f"annotations[{index}].object_bbox must be a normalized [x1, y1, x2, y2] box.")

        bbox = annotation.get("bounding_box")
        if not _is_valid_bbox(bbox):
            errors.append(f"annotations[{index}].bounding_box must be a normalized [x1, y1, x2, y2] box.")

        score = annotation.get("score")
        if score is not None and not _is_valid_score(score):
            errors.append(f"annotations[{index}].score must be a number between 0 and 1 when provided.")

        sentences = annotation.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            errors.append(f"annotations[{index}].sentences must be a non-empty list.")
            continue

        cleaned_sentences = [sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip()]
        if len(cleaned_sentences) != len(sentences) or not cleaned_sentences:
            errors.append(f"annotations[{index}].sentences must contain only non-empty strings.")
        elif len({sentence.casefold() for sentence in cleaned_sentences}) != len(cleaned_sentences):
            warnings.append(f"annotations[{index}].sentences contains duplicated sentence content.")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _is_valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False

    if not all(0 <= value <= 1 for value in (x1, y1, x2, y2)):
        return False

    return x1 < x2 and y1 < y2


def _is_valid_score(score: object) -> bool:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return False

    return 0 <= numeric_score <= 1
