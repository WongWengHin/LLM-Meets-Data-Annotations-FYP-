from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.user_database import (
    get_user_image_by_id,
    get_user_results,
    init_user_database,
    save_user_annotation_result,
    save_user_image,
)


class UserResultsDatabaseTests(unittest.TestCase):
    def test_numeric_image_ids_are_stored_without_leading_zeroes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "user_results.db"
            init_user_database(database_path)
            save_user_image(
                database_path, image_id="000000076731", owner_id="user",
                filename="000000076731.jpg", file_data=b"image", size_bytes=5,
            )
            save_user_annotation_result(
                database_path, image_id="000000076731", task_id="task",
                task_status="completed", annotation_result={"annotations": []},
            )
            result = get_user_results(database_path, "user")
            self.assertEqual(result[0]["image_id"], "76731")
            self.assertIsNotNone(get_user_image_by_id(database_path, "000000076731", "user"))

    def test_results_are_isolated_by_google_user_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database_path = Path(directory) / "user_results.db"
            init_user_database(database_path)
            save_user_image(
                database_path,
                image_id="first-image",
                owner_id="google:user-one",
                filename="first.jpg",
                file_data=b"first-image-data",
                size_bytes=16,
            )
            save_user_annotation_result(
                database_path,
                image_id="first-image",
                task_id="task-1",
                task_status="completed",
                annotation_result={"annotations": [{"annotation_id": "first-image_001"}]},
            )
            save_user_image(
                database_path,
                image_id="second-image",
                owner_id="google:user-two",
                filename="second.jpg",
                file_data=b"second-image-data",
                size_bytes=17,
            )

            first_user_results = get_user_results(database_path, "google:user-one")
            second_user_results = get_user_results(database_path, "google:user-two")

            self.assertEqual([row["image_id"] for row in first_user_results], ["first-image"])
            self.assertEqual([row["image_id"] for row in second_user_results], ["second-image"])
            self.assertIsNone(get_user_image_by_id(database_path, "first-image", "google:user-two"))
            self.assertEqual(
                get_user_image_by_id(database_path, "first-image", "google:user-one")["file_data"],
                b"first-image-data",
            )

if __name__ == "__main__":
    unittest.main()
