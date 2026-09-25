from __future__ import annotations

import runpy
from functools import lru_cache
from pathlib import Path
from typing import Any


HICO_OBJECTS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


HICO_VERBS = (
    "adjust",
    "assemble",
    "block",
    "blow",
    "board",
    "break",
    "brush_with",
    "buy",
    "carry",
    "catch",
    "chase",
    "check",
    "clean",
    "control",
    "cook",
    "cut",
    "cut_with",
    "direct",
    "drag",
    "dribble",
    "drink_with",
    "drive",
    "dry",
    "eat",
    "eat_at",
    "exit",
    "feed",
    "fill",
    "flip",
    "flush",
    "fly",
    "greet",
    "grind",
    "groom",
    "herd",
    "hit",
    "hold",
    "hop_on",
    "hose",
    "hug",
    "hunt",
    "inspect",
    "install",
    "jump",
    "kick",
    "kiss",
    "lasso",
    "launch",
    "lick",
    "lie_on",
    "lift",
    "light",
    "load",
    "lose",
    "make",
    "milk",
    "move",
    "no_interaction",
    "open",
    "operate",
    "pack",
    "paint",
    "park",
    "pay",
    "peel",
    "pet",
    "pick",
    "pick_up",
    "point",
    "pour",
    "pull",
    "push",
    "race",
    "read",
    "release",
    "repair",
    "ride",
    "row",
    "run",
    "sail",
    "scratch",
    "serve",
    "set",
    "shear",
    "sign",
    "sip",
    "sit_at",
    "sit_on",
    "slide",
    "smell",
    "spin",
    "squeeze",
    "stab",
    "stand_on",
    "stand_under",
    "stick",
    "stir",
    "stop_at",
    "straddle",
    "swing",
    "tag",
    "talk_on",
    "teach",
    "text_on",
    "throw",
    "tie",
    "toast",
    "train",
    "turn",
    "type_on",
    "walk",
    "wash",
    "watch",
    "wave",
    "wear",
    "wield",
    "zip",
)


_OBJECTS = set(HICO_OBJECTS)
_VERBS = set(HICO_VERBS)


def is_hico_object(label: str) -> bool:
    return label in _OBJECTS


def is_hico_verb(label: str) -> bool:
    return label in _VERBS


def is_valid_hico_interaction(predicate: str, object_name: str) -> bool:
    return (predicate, object_name) in valid_hico_interactions()


def allowed_predicates_for_object(object_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(predicate for predicate, candidate_object in valid_hico_interactions() if candidate_object == object_name)
    )


@lru_cache(maxsize=1)
def valid_hico_interactions() -> frozenset[tuple[str, str]]:
    """Return predicate-object pairs covered by the local HICO ground truth file."""
    ground_truth_path = Path(__file__).parents[2] / "evaluation_new" / "HICO-DET" / "hico_gt.py"
    if not ground_truth_path.is_file():
        return frozenset()

    namespace = runpy.run_path(str(ground_truth_path))
    payload = next(
        (value for name, value in namespace.items() if name.startswith("HICO_SENTENCES_") and isinstance(value, dict)),
        None,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        return frozenset()

    pairs: set[tuple[str, str]] = set()
    for image in payload["images"]:
        if not isinstance(image, dict) or not isinstance(image.get("annotations"), list):
            continue
        for annotation in image["annotations"]:
            if not isinstance(annotation, dict):
                continue
            object_name = _text(annotation.get("object"))
            predicates = annotation.get("predicate")
            predicates = predicates if isinstance(predicates, list) else [predicates]
            for predicate in predicates:
                predicate_name = _text(predicate)
                if predicate_name and object_name:
                    pairs.add((predicate_name, object_name))
    return frozenset(pairs)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""
