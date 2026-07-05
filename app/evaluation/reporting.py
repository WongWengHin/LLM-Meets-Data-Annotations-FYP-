"""*** 评估摘要、CSV 结果与调试文件的输出辅助逻辑。 ***"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.evaluation.types import MatchDebugRecord


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", ""] + lines
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def write_debug_reports(debug_dir: Path, records: list[MatchDebugRecord]) -> None:
    by_image: dict[str, list[dict]] = {}
    for record in records:
        by_image.setdefault(record.image_id, []).append(
            {
                "prediction_id": record.prediction_id,
                "gt_id": record.gt_id,
                "score": record.score,
                "iou_threshold": record.iou_threshold,
                "matched": record.matched,
                "iou": record.iou,
                "details": record.details,
            }
        )

    for image_id, image_records in by_image.items():
        write_json(debug_dir / f"{image_id}.json", {"image_id": image_id, "matches": image_records})
