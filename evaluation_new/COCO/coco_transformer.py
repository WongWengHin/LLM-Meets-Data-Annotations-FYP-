from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class COCOTransformerError(ValueError):
    pass


def _read_json(path: Path) -> dict:
    """Read a JSON file."""

    if not path.exists():
        raise COCOTransformerError(
            f"Input file does not exist: {path}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise COCOTransformerError(
            f"Failed to parse JSON from {path}: {exc}"
        )


def validate_coco_input(payload: dict) -> list[str]:
    """
    Validate the entire COCO input and return ALL errors.

    The function does not stop after finding the first problem.
    """

    errors: list[str] = []

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")

    # ======================================================
    # Top-level validation
    # ======================================================

    if not isinstance(images, list):
        errors.append(
            "Input JSON missing or invalid 'images' array"
        )

    if not isinstance(annotations, list):
        errors.append(
            "Input JSON missing or invalid 'annotations' array"
        )

    if not isinstance(categories, list):
        errors.append(
            "Input JSON missing or invalid 'categories' array"
        )

    # Cannot safely continue if these are missing
    if errors:
        return errors

    # ======================================================
    # Validate images
    # ======================================================

    image_lookup: dict[str, dict] = {}

    for index, img in enumerate(images):

        if not isinstance(img, dict):
            errors.append(
                f"Image record #{index} is not an object"
            )
            continue

        if "id" not in img:
            errors.append(
                f"Image record #{index} missing 'id'"
            )
            continue

        image_id = str(img["id"])

        if image_id in image_lookup:
            errors.append(
                f"Duplicate image id: {image_id}"
            )
            continue

        try:
            width = int(img.get("width"))
        except Exception:
            errors.append(
                f"Image {image_id} has invalid width: "
                f"{img.get('width')!r}"
            )
            continue

        try:
            height = int(img.get("height"))
        except Exception:
            errors.append(
                f"Image {image_id} has invalid height: "
                f"{img.get('height')!r}"
            )
            continue

        if width <= 0:
            errors.append(
                f"Image {image_id} width must be > 0, "
                f"got {width}"
            )

        if height <= 0:
            errors.append(
                f"Image {image_id} height must be > 0, "
                f"got {height}"
            )

        image_lookup[image_id] = {
            "width": width,
            "height": height,
        }

    # ======================================================
    # Validate categories
    # ======================================================

    category_lookup: dict[int, str] = {}

    for index, cat in enumerate(categories):

        if not isinstance(cat, dict):
            errors.append(
                f"Category record #{index} is not an object"
            )
            continue

        if "id" not in cat:
            errors.append(
                f"Category record #{index} missing 'id'"
            )
            continue

        if "name" not in cat:
            errors.append(
                f"Category {cat.get('id')} missing 'name'"
            )
            continue

        try:
            category_id = int(cat["id"])
        except Exception:
            errors.append(
                f"Category record #{index} has invalid id: "
                f"{cat.get('id')!r}"
            )
            continue

        if category_id in category_lookup:
            errors.append(
                f"Duplicate category id: {category_id}"
            )
            continue

        category_lookup[category_id] = str(
            cat["name"]
        )

    # ======================================================
    # Validate annotations
    # ======================================================

    annotation_ids: set[str] = set()

    for index, ann in enumerate(annotations):

        if not isinstance(ann, dict):
            errors.append(
                f"Annotation record #{index} is not an object"
            )
            continue

        if "id" not in ann:
            errors.append(
                f"Annotation record #{index} missing 'id'"
            )
            ann_id = f"index_{index}"
        else:
            ann_id = str(ann["id"])

            if ann_id in annotation_ids:
                errors.append(
                    f"Duplicate annotation id: {ann_id}"
                )

            annotation_ids.add(ann_id)

        # --------------------------------------------------
        # image_id
        # --------------------------------------------------

        if "image_id" not in ann:
            errors.append(
                f"Annotation {ann_id} missing 'image_id'"
            )
            continue

        image_id = str(ann["image_id"])

        image_info = image_lookup.get(image_id)

        if image_info is None:
            errors.append(
                f"Annotation {ann_id} references unknown "
                f"image_id {image_id}"
            )
            continue

        # --------------------------------------------------
        # category_id
        # --------------------------------------------------

        if "category_id" not in ann:
            errors.append(
                f"Annotation {ann_id} missing 'category_id'"
            )
        else:
            try:
                category_id = int(
                    ann["category_id"]
                )

                if category_id not in category_lookup:
                    errors.append(
                        f"Annotation {ann_id} references "
                        f"unknown category_id {category_id}"
                    )

            except Exception:
                errors.append(
                    f"Annotation {ann_id} has invalid "
                    f"category_id: "
                    f"{ann.get('category_id')!r}"
                )

        # --------------------------------------------------
        # bbox
        # --------------------------------------------------

        bbox = ann.get("bbox")

        if not isinstance(bbox, list):
            errors.append(
                f"Annotation {ann_id} bbox must be a list: "
                f"{bbox!r}"
            )
            continue

        if len(bbox) != 4:
            errors.append(
                f"Annotation {ann_id} bbox must contain "
                f"4 values: {bbox}"
            )
            continue

        try:
            x, y, w, h = (
                float(value)
                for value in bbox
            )
        except Exception:
            errors.append(
                f"Annotation {ann_id} bbox contains "
                f"non-numeric values: {bbox}"
            )
            continue

        if w <= 0:
            errors.append(
                f"Annotation {ann_id} bbox width must be "
                f"> 0, got {w}"
            )

        if h <= 0:
            errors.append(
                f"Annotation {ann_id} bbox height must be "
                f"> 0, got {h}"
            )

        if w <= 0 or h <= 0:
            continue

        # Convert COCO [x, y, width, height]
        # to [x1, y1, x2, y2]

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        image_width = image_info["width"]
        image_height = image_info["height"]

        nx1 = x1 / image_width
        ny1 = y1 / image_height
        nx2 = x2 / image_width
        ny2 = y2 / image_height

        normalized_coordinates = {
            "nx1": nx1,
            "ny1": ny1,
            "nx2": nx2,
            "ny2": ny2,
        }

        for name, value in normalized_coordinates.items():

            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"Annotation {ann_id} normalized "
                    f"{name} out of range 0..1: "
                    f"{value:.6f} "
                    f"(image {image_id}, "
                    f"size {image_width}x{image_height}, "
                    f"bbox={bbox})"
                )

        # --------------------------------------------------
        # sentences
        # --------------------------------------------------

        sentences = ann.get(
            "data_annotation_sentences"
        )

        if (
            sentences is not None
            and not isinstance(sentences, list)
        ):
            errors.append(
                f"Annotation {ann_id} has invalid "
                f"data_annotation_sentences: "
                f"{sentences!r}"
            )

    return errors


def transform_coco(
    input_json: Path | str,
    output_json: Path | str,
) -> dict:
    """
    Transform COCO-format JSON into the internal
    normalized ground-truth format.

    The entire input is validated first. If any errors
    are found, all errors are printed and no output file
    is written.
    """

    input_path = Path(input_json)
    output_path = Path(output_json)

    payload = _read_json(input_path)

    # ======================================================
    # Validate everything first
    # ======================================================

    errors = validate_coco_input(payload)

    if errors:

        print()
        print("=" * 70)
        print(
            f"COCO VALIDATION FAILED: "
            f"{len(errors)} ERROR(S) FOUND"
        )
        print("=" * 70)

        for index, error in enumerate(
            errors,
            start=1,
        ):
            print(f"{index}. {error}")

        print("=" * 70)
        print(
            "Ground-truth JSON was NOT generated."
        )
        print()

        raise COCOTransformerError(
            f"Input validation failed with "
            f"{len(errors)} error(s)"
        )

    # ======================================================
    # Build lookups
    # ======================================================

    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]

    image_lookup: dict[str, dict] = {}

    for img in images:

        image_id = str(img["id"])

        entry = {
            "image_id": image_id,
            "file_name": img.get("file_name"),
            "width": int(img["width"]),
            "height": int(img["height"]),
        }

        if "difficulty" in img:
            entry["difficulty"] = img.get(
                "difficulty"
            )

        image_lookup[image_id] = entry

    category_lookup = {
        int(category["id"]): str(
            category["name"]
        )
        for category in categories
    }

    # ======================================================
    # Transform annotations
    # ======================================================

    out_annotations = []

    for ann in annotations:

        annotation_id = str(ann["id"])
        image_id = str(ann["image_id"])

        image_info = image_lookup[image_id]

        category_id = int(
            ann["category_id"]
        )

        class_name = category_lookup[
            category_id
        ]

        x, y, width, height = (
            float(value)
            for value in ann["bbox"]
        )

        x1 = x
        y1 = y

        x2 = x + width
        y2 = y + height

        image_width = image_info["width"]
        image_height = image_info["height"]

        normalized_bbox = [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        ]

        sentences = ann.get(
            "data_annotation_sentences"
        )

        if sentences is None:
            sentences = []

        out_annotations.append(
            {
                "annotation_id":
                    annotation_id,

                "image_id":
                    image_id,

                "class":
                    class_name,

                "bbox":
                    normalized_bbox,

                "sentences":
                    [
                        str(sentence)
                        for sentence in sentences
                    ],
            }
        )

    output = {
        "images": list(
            image_lookup.values()
        ),
        "annotations": out_annotations,
    }

    # ======================================================
    # Write transformed output
    # ======================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("COCO TRANSFORMATION SUCCESSFUL")
    print("=" * 70)
    print(
        f"Images transformed: "
        f"{len(output['images'])}"
    )
    print(
        f"Annotations transformed: "
        f"{len(output['annotations'])}"
    )
    print(
        f"Output: {output_path}"
    )
    print("=" * 70)

    return output


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Transform COCO JSON into internal "
            "normalized ground-truth format"
        )
    )

    parser.add_argument(
        "input_json",
        help="Path to input COCO JSON",
    )

    parser.add_argument(
        "output_json",
        help="Path to output ground-truth JSON",
    )

    args = parser.parse_args()

    transform_coco(
        Path(args.input_json),
        Path(args.output_json),
    )


if __name__ == "__main__":
    _cli()