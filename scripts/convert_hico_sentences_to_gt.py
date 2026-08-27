"""Convert a HICO_sentences.py-style source file into a HICO-DET ground-truth JSON fixture."""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.hicodet_gt_reference import build_hicodet_gt_fixture, save_hicodet_gt_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HICO_sentences.py-style payload into JSON for HICO-DET evaluation."
    )
    parser.add_argument(
        "source",
        help="Path to HICO_sentences.py or a git ref like remotes/upstream/branch:HICO_sentences.py.",
    )
    parser.add_argument("output", help="Path to write the converted GT JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.source)
    fixture = build_hicodet_gt_fixture(payload)
    output_path = Path(args.output)
    save_hicodet_gt_fixture(fixture, output_path)
    print(f"Converted {args.source} -> {output_path}")
    print(f"Images written: {len(fixture['images'])}")
    print(f"Annotations written: {len(fixture['annotations'])}")


def _load_payload(source: str) -> dict:
    source_path = Path(source)
    if source_path.exists():
        return _load_payload_from_path(source_path)

    git_text = _git_show(source)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        handle.write(git_text)

    try:
        return _load_payload_from_path(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_payload_from_path(path: Path) -> dict:
    namespace = runpy.run_path(str(path))
    for key, value in namespace.items():
        if key.startswith("HICO_SENTENCES_") and isinstance(value, dict):
            return value
    raise RuntimeError("No HICO_SENTENCES_* dictionary was found in the source file.")


def _git_show(ref: str) -> str:
    completed = subprocess.run(
        ["git", "show", ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


if __name__ == "__main__":
    main()
