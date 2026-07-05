from __future__ import annotations


def validate_annotation(normalized_annotation: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    caption = normalized_annotation.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        errors.append("caption must be a non-empty string.")

    annotations = normalized_annotation.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        errors.append("annotations must be a non-empty list.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            errors.append(f"annotations[{index}] must be an object.")
            continue

        subject = annotation.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            errors.append(f"annotations[{index}].subject must be a non-empty string.")

        object_name = annotation.get("object")
        if not isinstance(object_name, str) or not object_name.strip():
            errors.append(f"annotations[{index}].object must be a non-empty string.")

        predicate = annotation.get("predicate")
        if not isinstance(predicate, str) or not predicate.strip():
            errors.append(f"annotations[{index}].predicate must be a non-empty string.")

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
