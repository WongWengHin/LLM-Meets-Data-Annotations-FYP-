"""Semantic evaluation helpers for HICO-style matched HOI pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from statistics import mean

from app.evaluation.matching import bbox_iou


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize_text(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.casefold())


def bleu_score(candidate: str, references: list[str], max_n: int = 4) -> float:
    candidate_tokens = tokenize_text(candidate)
    reference_tokens = [tokenize_text(reference) for reference in references if reference.strip()]
    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_length = len(candidate_tokens)
    reference_lengths = [len(tokens) for tokens in reference_tokens if tokens]
    if not reference_lengths:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        candidate_counts = _ngram_counts(candidate_tokens, n)
        if not candidate_counts:
            precisions.append(0.0)
            continue

        max_reference_counts: Counter[tuple[str, ...]] = Counter()
        for ref_tokens in reference_tokens:
            ref_counts = _ngram_counts(ref_tokens, n)
            for ngram, count in ref_counts.items():
                if count > max_reference_counts.get(ngram, 0):
                    max_reference_counts[ngram] = count

        clipped = sum(min(count, max_reference_counts.get(ngram, 0)) for ngram, count in candidate_counts.items())
        total = sum(candidate_counts.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    geometric_mean = math.exp(sum(math.log(max(precision, 1e-9)) for precision in precisions) / max_n)
    best_reference_length = min(reference_lengths, key=lambda length: (abs(length - candidate_length), length))
    brevity_penalty = 1.0 if candidate_length > best_reference_length else math.exp(1 - best_reference_length / max(candidate_length, 1))
    return brevity_penalty * geometric_mean


def meteor_score(candidate: str, references: list[str]) -> float:
    candidate_tokens = tokenize_text(candidate)
    if not candidate_tokens:
        return 0.0

    best_score = 0.0
    for reference in references:
        reference_tokens = tokenize_text(reference)
        if not reference_tokens:
            continue
        if candidate_tokens == reference_tokens:
            return 1.0

        matches, chunks = _align_tokens(candidate_tokens, reference_tokens)
        if matches == 0:
            continue

        precision = matches / len(candidate_tokens)
        recall = matches / len(reference_tokens)
        f_mean = (10 * precision * recall) / (recall + 9 * precision) if precision + recall > 0 else 0.0
        penalty = 0.5 * ((chunks / matches) ** 3)
        score = f_mean * (1 - penalty)
        best_score = max(best_score, score)

    return best_score


def score_semantic_pair(candidate_sentences: list[str], reference_sentences: list[str]) -> dict[str, float | int]:
    candidates = [sentence.strip() for sentence in candidate_sentences if sentence and sentence.strip()]
    references = [sentence.strip() for sentence in reference_sentences if sentence and sentence.strip()]
    if not candidates or not references:
        return {
            "bleu": 0.0,
            "meteor": 0.0,
            "candidate_count": len(candidates),
            "reference_count": len(references),
        }

    bleu_values: list[float] = []
    meteor_values: list[float] = []
    for candidate in candidates:
        bleu_values.append(bleu_score(candidate, references))
        meteor_values.append(meteor_score(candidate, references))

    return {
        "bleu": mean(bleu_values) if bleu_values else 0.0,
        "meteor": mean(meteor_values) if meteor_values else 0.0,
        "candidate_count": len(candidates),
        "reference_count": len(references),
    }


def evaluate_hico_semantic_pairs(
    prediction_exports: list[dict],
    semantic_records: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    records_by_image: dict[str, list[dict]] = defaultdict(list)
    for record in semantic_records:
        for alias in _record_image_aliases(record):
            records_by_image[alias].append(record)

    used_record_ids: set[str] = set()
    matches: list[dict] = []
    issues: list[dict] = []

    for export_payload in prediction_exports:
        image_id = _extract_image_id(export_payload)
        if not image_id:
            issues.append({"reason": "missing_image_id", "prediction_file": export_payload.get("image_id")})
            continue

        image_aliases = _export_image_aliases(export_payload)
        predictions = export_payload.get("predictions", [])
        if not isinstance(predictions, list):
            issues.append({"image_id": image_id, "reason": "predictions_not_list"})
            continue

        sorted_predictions = sorted(
            [prediction for prediction in predictions if isinstance(prediction, dict)],
            key=lambda item: (-float(item.get("score") or 0.0), str(item.get("annotation_id") or "")),
        )

        for prediction in sorted_predictions:
            triplet_key = _triplet_key_from_prediction(prediction)
            candidate_sentences = _extract_prediction_sentences(prediction)
            image_records: list[dict] = []
            seen_records: set[str] = set()
            for alias in image_aliases:
                for record in records_by_image.get(alias, []):
                    record_id = str(record.get("annotation_id") or "")
                    if record_id in seen_records:
                        continue
                    seen_records.add(record_id)
                    image_records.append(record)
            exact_candidates = [
                record
                for record in image_records
                if str(record.get("triplet_key") or "").casefold() == triplet_key.casefold()
                and str(record.get("annotation_id") or "") not in used_record_ids
            ]
            candidates = exact_candidates or [
                record
                for record in image_records
                if str(record.get("annotation_id") or "") not in used_record_ids
            ]

            best_record: dict | None = None
            best_subject_iou = 0.0
            best_object_iou = 0.0
            best_pair_score = -1.0

            for record in candidates:
                subject_bbox = record.get("subject_bbox")
                object_bbox = record.get("object_bbox")
                if not (_is_bbox(subject_bbox) and _is_bbox(object_bbox)):
                    continue

                subject_pred_bbox = prediction.get("subject_bbox")
                object_pred_bbox = prediction.get("object_bbox")
                if not (_is_bbox(subject_pred_bbox) and _is_bbox(object_pred_bbox)):
                    continue

                subject_iou = bbox_iou(tuple(float(value) for value in subject_pred_bbox), tuple(float(value) for value in subject_bbox))
                object_iou = bbox_iou(tuple(float(value) for value in object_pred_bbox), tuple(float(value) for value in object_bbox))
                pair_score = min(subject_iou, object_iou)
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_record = record
                    best_subject_iou = subject_iou
                    best_object_iou = object_iou

            matched = (
                best_record is not None
                and best_subject_iou >= iou_threshold
                and best_object_iou >= iou_threshold
            )
            if matched and best_record is not None:
                used_record_ids.add(str(best_record.get("annotation_id") or ""))

            reference_sentences = _extract_reference_sentences(best_record) if best_record is not None else []
            semantic_scores = score_semantic_pair(candidate_sentences, reference_sentences) if matched else {
                "bleu": 0.0,
                "meteor": 0.0,
                "candidate_count": len(candidate_sentences),
                "reference_count": len(reference_sentences),
            }

            matches.append(
                {
                    "image_id": image_id,
                    "prediction_id": str(prediction.get("annotation_id") or ""),
                    "reference_id": str(best_record.get("annotation_id") or "") if best_record else None,
                    "triplet_key": triplet_key,
                    "matched": matched,
                    "subject_iou": best_subject_iou,
                    "object_iou": best_object_iou,
                    "bleu": semantic_scores["bleu"],
                    "meteor": semantic_scores["meteor"],
                    "candidate_count": semantic_scores["candidate_count"],
                    "reference_count": semantic_scores["reference_count"],
                    "prediction_sentences": candidate_sentences,
                    "reference_sentences": reference_sentences,
                }
            )

    return matches, issues


def summarize_semantic_matches(matches: list[dict], *, matched_only: bool = True) -> dict:
    relevant_matches = [match for match in matches if match["matched"]] if matched_only else list(matches)
    per_triplet: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for match in relevant_matches:
        grouped[match["triplet_key"]].append(match)

    for triplet_key, items in grouped.items():
        per_triplet[triplet_key] = {
            "count": len(items),
            "bleu": mean(item["bleu"] for item in items) if items else 0.0,
            "meteor": mean(item["meteor"] for item in items) if items else 0.0,
        }

    return {
        "matched_pair_count": len(relevant_matches),
        "pair_count": len(matches),
        "match_rate": len(relevant_matches) / len(matches) if matches else 0.0,
        "bleu": mean(item["bleu"] for item in relevant_matches) if relevant_matches else 0.0,
        "meteor": mean(item["meteor"] for item in relevant_matches) if relevant_matches else 0.0,
        "per_triplet": per_triplet,
    }


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _align_tokens(candidate_tokens: list[str], reference_tokens: list[str]) -> tuple[int, int]:
    reference_positions: list[int] = []
    used_indices: set[int] = set()
    for candidate_index, token in enumerate(candidate_tokens):
        for reference_index, reference_token in enumerate(reference_tokens):
            if reference_index in used_indices:
                continue
            if token == reference_token:
                reference_positions.append(reference_index)
                used_indices.add(reference_index)
                break

    if not reference_positions:
        return 0, 0

    chunks = 1
    for previous, current in zip(reference_positions, reference_positions[1:]):
        if current != previous + 1:
            chunks += 1

    return len(reference_positions), chunks


def _extract_image_id(export_payload: dict) -> str:
    image_block = export_payload.get("image", {})
    if isinstance(image_block, dict):
        for key in ("image_id", "id"):
            value = image_block.get(key)
            if value is not None:
                return str(value)

    for key in ("image_id", "id"):
        value = export_payload.get(key)
        if value is not None:
            return str(value)

    return ""


def _export_image_aliases(export_payload: dict) -> list[str]:
    aliases: list[str] = []
    image_block = export_payload.get("image", {})
    if isinstance(image_block, dict):
        for key in ("image_id", "id", "file_name"):
            value = image_block.get(key)
            if value is None:
                continue
            aliases.extend(_expand_aliases(str(value)))

    for key in ("image_id", "id", "file_name"):
        value = export_payload.get(key)
        if value is not None:
            aliases.extend(_expand_aliases(str(value)))

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = alias.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _record_image_aliases(record: dict) -> list[str]:
    aliases: list[str] = []
    for key in ("image_id", "id", "file_name"):
        value = record.get(key)
        if value is not None:
            aliases.extend(_expand_aliases(str(value)))

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = alias.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _triplet_key_from_prediction(prediction: dict) -> str:
    subject = str(prediction.get("subject") or "").strip()
    predicate = str(prediction.get("predicate") or "").strip()
    object_name = str(prediction.get("object") or "").strip()
    return f"{subject}|{predicate}|{object_name}"


def _extract_prediction_sentences(prediction: dict) -> list[str]:
    sentences = prediction.get("sentences")
    if isinstance(sentences, list):
        cleaned = [str(sentence).strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip()]
        if cleaned:
            return cleaned

    caption = prediction.get("caption")
    if isinstance(caption, str) and caption.strip():
        return [caption.strip()]

    return []


def _extract_reference_sentences(record: dict | None) -> list[str]:
    if not record:
        return []
    references = record.get("references")
    if not isinstance(references, list):
        return []
    return [str(sentence).strip() for sentence in references if isinstance(sentence, str) and sentence.strip()]


def _is_bbox(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value)


def _expand_aliases(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    aliases = [text]
    path_text = text.replace("\\", "/")
    stem = path_text.rsplit("/", 1)[-1]
    if stem not in aliases:
        aliases.append(stem)
    if "." in stem:
        stem_no_ext = stem.rsplit(".", 1)[0]
        if stem_no_ext not in aliases:
            aliases.append(stem_no_ext)
    return aliases
