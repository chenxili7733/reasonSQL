from __future__ import annotations

import unittest


class CompatImportTests(unittest.TestCase):
    def test_common_prompt_compat(self) -> None:
        from common.prompt import build_messages, build_prompt_text, load_schema

        self.assertTrue(callable(build_messages))
        self.assertTrue(callable(build_prompt_text))
        self.assertTrue(callable(load_schema))

    def test_common_data_compat(self) -> None:
        from common.data import discover_split, normalize_bird_row

        self.assertTrue(callable(discover_split))
        row = normalize_bird_row({"db_id": "a", "question": "q", "SQL": "SELECT 1"})
        self.assertEqual(row["db_id"], "a")


if __name__ == "__main__":
    unittest.main()

