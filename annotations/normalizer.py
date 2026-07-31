from __future__ import annotations

import json
import re
from pathlib import Path

from images.bbox_utils import combine_bounding_boxes, normalize_optional_bbox


def clean_json_text(content: str) -> str:
    cleaned = (content or "").strip().lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def extract_json_candidates(content: str) -> list[str]:
    cleaned = clean_json_text(content)
    candidates: list[str] = []

    if cleaned:
        candidates.append(cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1].strip())

    stack = 0
    object_start: int | None = None
    for index, char in enumerate(cleaned):
        if char == "{":
            if stack == 0:
                object_start = index
            stack += 1
        elif char == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and object_start is not None:
                    candidates.append(cleaned[object_start : index + 1].strip())
                    object_start = None

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    unique_candidates.sort(key=len, reverse=True)
    return unique_candidates


def loosen_common_json_issues(candidate: str) -> str:
    relaxed = candidate.strip()
    relaxed = re.sub(r",(\s*[}\]])", r"\1", relaxed)
    return relaxed


def parse_json_object_with_fallback(raw_model_output: str) -> tuple[dict, str]:
    candidates = extract_json_candidates(raw_model_output)
    if not candidates:
        raise ValueError("Model returned empty content.")

    last_error: json.JSONDecodeError | None = None
    best_candidate = candidates[0]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, candidate
        except json.JSONDecodeError as exc:
            last_error = exc

        relaxed_candidate = loosen_common_json_issues(candidate)
        if relaxed_candidate != candidate:
            try:
                parsed = json.loads(relaxed_candidate)
                if isinstance(parsed, dict):
                    return parsed, relaxed_candidate
            except json.JSONDecodeError as exc:
                last_error = exc

    if last_error is not None:
        raise ValueError(f"Model did not return valid JSON. Raw content: {best_candidate}") from last_error

    raise ValueError(f"Model did not return a JSON object. Raw content: {best_candidate}")


def pick_first_value(data: dict, *keys: str) -> object:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def parse_and_normalize_model_output(raw_model_output: str, image_path: Path | None = None) -> dict:
    data, cleaned = parse_json_object_with_fallback(raw_model_output)

    raw_annotations = data.get("annotations")
    if not isinstance(raw_annotations, list):
        nested_annotation = data.get("annotation")
        raw_annotations = [nested_annotation] if isinstance(nested_annotation, dict) else [data]

    normalized_annotations = [
        _normalize_annotation_item(item, image_path, cleaned, index + 1)
        for index, item in enumerate(raw_annotations)
    ]

    caption = _normalize_caption(data, normalized_annotations)
    return {
        "caption": caption,
        "annotations": normalized_annotations,
    }


def _normalize_annotation_item(
    item: object,
    image_path: Path | None,
    raw_content: str,
    index: int,
) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"Annotation item at index {index} is not a JSON object. Raw content: {raw_content}")

    subject = _normalize_label(
        pick_first_value(item, "subject", "person", "person_class", "human", "human_class", "subject_class")
    )
    object_name = _normalize_label(
        pick_first_value(item, "object", "object_name", "object_class", "target", "target_class", "item_class")
    )
    predicate = _normalize_label(
        pick_first_value(item, "predicate", "interaction", "interaction_verb", "verb", "action", "relation")
    )

    legacy_category = _normalize_label(
        pick_first_value(item, "category", "class", "label", "name", "interaction_label")
    )
    parsed_subject, parsed_predicate, parsed_object = _parse_legacy_category(legacy_category)
    subject = subject or parsed_subject or "person"
    object_name = object_name or parsed_object or (legacy_category if legacy_category and legacy_category != subject else "unknown")
    predicate = predicate or parsed_predicate or "interact_with"

    subject_bbox = normalize_optional_bbox(
        pick_first_value(
            item,
            "subject_bbox",
            "subject_bounding_box",
            "person_bbox",
            "person_bounding_box",
            "human_bbox",
            "human_bounding_box",
        ),
        image_path,
        raw_content,
        f"annotations[{index}].subject_bbox",
    )
    object_bbox = normalize_optional_bbox(
        pick_first_value(
            item,
            "object_bbox",
            "object_bounding_box",
            "object_box",
            "object_bbox_normalized",
            "target_bbox",
            "target_bounding_box",
        ),
        image_path,
        raw_content,
        f"annotations[{index}].object_bbox",
    )
    legacy_bbox = normalize_optional_bbox(
        pick_first_value(item, "bounding_box", "bbox", "box", "coordinates", "union_bbox", "interaction_bounding_box"),
        image_path,
        raw_content,
        f"annotations[{index}].bounding_box",
    )

    if subject_bbox is None and legacy_bbox is not None:
        subject_bbox = legacy_bbox
    if object_bbox is None and legacy_bbox is not None:
        object_bbox = legacy_bbox

    if subject_bbox is None:
        raise ValueError(f"Annotation item at index {index} is missing subject_bbox. Raw content: {raw_content}")
    if object_bbox is None:
        raise ValueError(f"Annotation item at index {index} is missing object_bbox. Raw content: {raw_content}")

    sentences = _normalize_sentences(
        pick_first_value(item, "sentences", "data_annotation_sentences", "descriptions", "captions", "annotation_sentences"),
        fallback=[f"A {subject} is {predicate} a {object_name}."],
        raw_content=raw_content,
        field_name=f"annotations[{index}].sentences",
    )

    return _build_annotation_record(
        subject=subject,
        object_name=object_name,
        predicate=predicate,
        subject_bbox=subject_bbox,
        object_bbox=object_bbox,
        sentences=sentences,
        score=item.get("score"),
    )


def _normalize_caption(data: dict, annotations: list[dict]) -> str:
    caption = pick_first_value(data, "caption", "image_caption", "description", "scene_description", "summary")
    if isinstance(caption, str) and caption.strip():
        return caption.strip()

    return _derive_caption_from_annotations(annotations)


def _derive_caption_from_annotations(annotations: list[dict]) -> str:
    if annotations:
        first_sentence = annotations[0].get("sentences", [])
        if first_sentence and isinstance(first_sentence[0], str) and first_sentence[0].strip():
            return first_sentence[0].strip()
    return "No caption returned by the model."


def _normalize_sentences(
    sentences: object,
    *,
    fallback: list[str],
    raw_content: str,
    field_name: str,
) -> list[str]:
    if not isinstance(sentences, list):
        if isinstance(sentences, str) and sentences.strip():
            sentences = [sentences.strip()]
        else:
            sentences = fallback

    normalized_sentences = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    if not normalized_sentences:
        normalized_sentences = fallback
    if not normalized_sentences:
        raise ValueError(f"{field_name} is empty. Raw content: {raw_content}")
    return normalized_sentences


def _normalize_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _parse_legacy_category(category: str) -> tuple[str, str, str]:
    if not category:
        return "", "", ""

    tokens = category.split()
    if len(tokens) >= 3 and tokens[0].casefold() == "person":
        return "person", tokens[1], " ".join(tokens[2:])
    if len(tokens) == 2:
        return tokens[0], "interact_with", tokens[1]
    return "", "", ""


def _build_annotation_record(
    *,
    subject: str,
    object_name: str,
    predicate: str,
    subject_bbox: list[float] | None,
    object_bbox: list[float] | None,
    sentences: list[str],
    score: object = None,
) -> dict:
    category = " ".join(part for part in (subject, predicate, object_name) if part).strip()
    bounding_box = (
        combine_bounding_boxes(subject_bbox, object_bbox)
        if subject_bbox is not None and object_bbox is not None
        else subject_bbox or object_bbox
    )

    record = {
        "subject": subject,
        "object": object_name,
        "predicate": predicate,
        "subject_bbox": subject_bbox,
        "object_bbox": object_bbox,
        "category": category or "unknown",
        "bounding_box": bounding_box,
        "sentences": sentences,
    }
    if score is not None:
        record["score"] = score
    return record
