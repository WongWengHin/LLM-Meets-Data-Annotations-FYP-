"""Convert a HICO_sentences.py-style source file into a semantic reference JSON fixture."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.semantic_reference import build_hico_semantic_fixture, save_hico_semantic_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HICO_sentences.py-style Python source file into JSON semantic references."
    )
    parser.add_argument("input", help="Path to the HICO_sentences.py source file.")
    parser.add_argument("output", help="Path to write the converted JSON fixture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    namespace = runpy.run_path(str(input_path))
    payload = _extract_hico_payload(namespace)
    fixture = build_hico_semantic_fixture(payload)
    save_hico_semantic_fixture(fixture, output_path)
    print(f"Converted {input_path} -> {output_path}")
    print(f"Records written: {len(fixture['records'])}")


def _extract_hico_payload(namespace: dict[str, object]) -> dict:
    for key, value in namespace.items():
        if key.startswith("HICO_SENTENCES_") and isinstance(value, dict):
            return value
    raise RuntimeError("No HICO_SENTENCES_* dictionary was found in the input file.")


if __name__ == "__main__":
    main()
