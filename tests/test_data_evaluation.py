from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reasonsql.data import (
    PrepareConfig,
    deterministic_subset,
    discover_split,
    load_json_rows,
    normalize_bird_row,
    prepare_bird,
)
from reasonsql.evaluation import (
    align_predictions,
    evaluate_predictions,
    load_predictions,
    write_bird_sql_file,
)


def _create_split(root: Path, split: str, rows: list[dict]) -> Path:
    split_root = root / split
    db_root = split_root / f"{split}_databases"
    demo_root = db_root / "demo"
    demo_root.mkdir(parents=True)
    conn = sqlite3.connect(demo_root / "demo.sqlite")
    conn.executescript(
        """
        CREATE TABLE items (id INTEGER, name TEXT);
        INSERT INTO items VALUES (1, 'alpha'), (2, 'beta');
        """
    )
    conn.close()
    (split_root / f"{split}.json").write_text(
        json.dumps(rows, ensure_ascii=False),
        encoding="utf-8",
    )
    return db_root


class DataTests(unittest.TestCase):
    def test_normalize_and_subset(self) -> None:
        row = normalize_bird_row(
            {"database_id": "demo", "Question": "q", "query": "SELECT 1", "level": "Simple"},
            7,
        )
        self.assertEqual(row["question_id"], 7)
        self.assertEqual(row["difficulty"], "simple")
        rows = [{"question_id": index} for index in range(20)]
        self.assertEqual(deterministic_subset(rows, 5, 42), deterministic_subset(rows, 5, 42))
        self.assertNotEqual(deterministic_subset(rows, 5, 42), deterministic_subset(rows, 5, 43))

    def test_discover_and_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bird"
            train_rows = [
                {
                    "question_id": index,
                    "db_id": "demo",
                    "question": f"question {index}",
                    "evidence": "",
                    "SQL": "SELECT name FROM items ORDER BY id",
                    "difficulty": "simple",
                }
                for index in range(3)
            ]
            dev_rows = [dict(train_rows[0])]
            _create_split(root, "train", train_rows)
            _create_split(root, "dev", dev_rows)
            split = discover_split(root, "train")
            self.assertEqual(split.json_path.name, "train.json")
            self.assertEqual(len(load_json_rows(split.json_path)), 3)

            out = Path(temp) / "processed"
            captured: list[dict] = []

            def fake_parquet(_path, rows):
                captured.extend(rows)

            with patch("reasonsql.data.write_parquet", side_effect=fake_parquet):
                manifest = prepare_bird(
                    PrepareConfig(
                        bird_dir=root,
                        output_dir=out,
                        rl_size=2,
                        workers=1,
                    )
                )
            self.assertEqual(manifest["train_examples"], 3)
            self.assertEqual(manifest["rl_examples"], 2)
            self.assertEqual(len(captured), 2)
            self.assertTrue((out / "bird_sft.jsonl").is_file())
            self.assertTrue((out / "bird_dev.jsonl").is_file())


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.rows = [
            {
                "question_id": 10,
                "db_id": "demo",
                "question": "List names",
                "evidence": "",
                "SQL": "SELECT name FROM items ORDER BY id",
                "difficulty": "simple",
            },
            {
                "question_id": 11,
                "db_id": "demo",
                "question": "Count items",
                "evidence": "",
                "SQL": "SELECT COUNT(*) FROM items",
                "difficulty": "moderate",
            },
        ]
        self.db_root = _create_split(self.root, "dev", self.rows)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_alignment_and_execution_evaluation(self) -> None:
        predictions = [
            {"question_id": 11, "pred_sql": "SELECT 0"},
            {"question_id": 10, "response": "```sql\nSELECT name FROM items ORDER BY id\n```"},
        ]
        aligned = align_predictions(self.rows, predictions)
        self.assertEqual(aligned[0]["question_id"], 10)
        records, summary = evaluate_predictions(self.rows, predictions, self.db_root, workers=1)
        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.by_status["correct"], 1)
        self.assertEqual(records[1]["status"], "mismatch")

    def test_prediction_io(self) -> None:
        predictions = [
            {"question_id": 10, "db_id": "demo", "pred_sql": "SELECT name FROM items"},
            {"question_id": 11, "db_id": "demo", "pred_sql": "SELECT COUNT(*) FROM items"},
        ]
        jsonl = self.root / "predictions.jsonl"
        jsonl.write_text(
            "".join(json.dumps(item) + "\n" for item in predictions),
            encoding="utf-8",
        )
        loaded = load_predictions(jsonl)
        self.assertEqual(len(loaded), 2)
        sql_file = self.root / "predict_dev.sql"
        write_bird_sql_file(sql_file, loaded)
        self.assertIn("----- bird -----", sql_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

