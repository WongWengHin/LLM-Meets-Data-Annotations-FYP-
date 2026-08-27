from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.api.routes import _unique_destination


class ApiRoutesTests(unittest.TestCase):
    def test_unique_destination_keeps_original_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_dir = Path(tmp_dir)
            existing = upload_dir / "HICO_train2015_00000001.jpg"
            existing.write_bytes(b"old-bytes")

            destination = _unique_destination(upload_dir, "HICO_train2015_00000001.jpg")

            self.assertEqual(destination, existing)


if __name__ == "__main__":
    unittest.main()
