"""Caption metrics used by the COCO evaluation workflow."""

from __future__ import annotations

import math
import re
from collections import Counter


def sentence_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(sentence).strip() for sentence in value if str(sentence).strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _bleu(candidate: list[str], references: list[list[str]]) -> float:
    if not candidate or not references:
        return 0.0
    precisions = []
    for order in range(1, 5):
        candidate_ngrams = Counter(
            tuple(candidate[index:index + order])
            for index in range(len(candidate) - order + 1)
        )
        if not candidate_ngrams:
            precisions.append(1.0)
            continue
        reference_max = Counter()
        for reference in references:
            reference_ngrams = Counter(
                tuple(reference[index:index + order])
                for index in range(len(reference) - order + 1)
            )
            for ngram, count in reference_ngrams.items():
                reference_max[ngram] = max(reference_max[ngram], count)
        clipped = sum(min(count, reference_max[ngram]) for ngram, count in candidate_ngrams.items())
        total = sum(candidate_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))
    reference_length = min(len(reference) for reference in references)
    brevity_penalty = 1.0 if len(candidate) > reference_length else math.exp(1.0 - reference_length / len(candidate))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4.0)


def _meteor(candidate: list[str], reference: list[str]) -> float:
    if not candidate or not reference:
        return 0.0
    reference_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        reference_positions.setdefault(token, []).append(index)
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for candidate_index, token in enumerate(candidate):
        available = [index for index in reference_positions.get(token, []) if index not in used]
        if available:
            reference_index = available[0]
            used.add(reference_index)
            matches.append((candidate_index, reference_index))
    if not matches:
        return 0.0
    precision = len(matches) / len(candidate)
    recall = len(matches) / len(reference)
    harmonic = 10 * precision * recall / (recall + 9 * precision)
    chunks = 1
    for previous, current in zip(matches, matches[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    return harmonic * (1 - 0.5 * (chunks / len(matches)) ** 3)


class SentenceMetricScorer:
    def score_text(self, predicted: str, references: list[str]) -> tuple[float, float]:
        candidate = _tokens(predicted)
        reference_tokens = [_tokens(reference) for reference in references]
        bleu = _bleu(candidate, reference_tokens)
        meteor = max((_meteor(candidate, reference) for reference in reference_tokens), default=0.0)
        return bleu, meteor