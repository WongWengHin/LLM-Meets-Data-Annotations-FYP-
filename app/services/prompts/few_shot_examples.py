from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

COCO_EXAMPLES = [
    (
        BASE_DIR / "data/few_shots/coco/example_01.jpg",
        BASE_DIR / "data/few_shots/coco/example_01.json",
    ),
    (
        BASE_DIR / "data/few_shots/coco/example_02.jpg",
        BASE_DIR / "data/few_shots/coco/example_02.json",
    ),
    (
        BASE_DIR / "data/few_shots/coco/example_03.jpg",
        BASE_DIR / "data/few_shots/coco/example_03.json",
    ),
    (
        BASE_DIR / "data/few_shots/coco/example_04.jpg",
        BASE_DIR / "data/few_shots/coco/example_04.json",
    ),
    (
        BASE_DIR / "data/few_shots/coco/example_05.jpg",
        BASE_DIR / "data/few_shots/coco/example_05.json",
    )
]

HICODET_EXAMPLES = [
    (
        BASE_DIR / "data/few_shots/hicodet/example_01.jpg",
        BASE_DIR / "data/few_shots/hicodet/example_01.json",
    ),
    (
        BASE_DIR / "data/few_shots/hicodet/example_02.jpg",
        BASE_DIR / "data/few_shots/hicodet/example_02.json",
    ),
    (
        BASE_DIR / "data/few_shots/hicodet/example_03.jpg",
        BASE_DIR / "data/few_shots/hicodet/example_03.json",
    ),
    (
        BASE_DIR / "data/few_shots/hicodet/example_04.jpg",
        BASE_DIR / "data/few_shots/hicodet/example_04.json",
    ),
    (
        BASE_DIR / "data/few_shots/hicodet/example_05.jpg",
        BASE_DIR / "data/few_shots/hicodet/example_05.json",
    )
]


def get_few_shot_examples(prompt_mode: str):
    if prompt_mode == "hoi":
        return HICODET_EXAMPLES

    return COCO_EXAMPLES