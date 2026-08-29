#!/usr/bin/env python3
"""CPU-only end-to-end smoke test for prompt, reward and evaluation paths."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasonsql.evaluation import evaluate_predictions  # noqa: E402
from reasonsql.prompts import build_prompt_text  # noqa: E402
from reasonsql.rewards.bird_reward import score_sql  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        db_root = Path(temp) / "databases"
        database = db_root / "toy"
        database.mkdir(parents=True)
        conn = sqlite3.connect(database / "toy.sqlite")
        conn.executescript(
            """
            CREATE TABLE products (id INTEGER, name TEXT, price REAL);
            INSERT INTO products VALUES
              (1, 'keyboard', 80.0),
              (2, 'monitor', 220.0),
              (3, 'mouse', 25.0);
            """
        )
        conn.close()

        prompt = build_prompt_text(
            "toy",
            "Which product is the most expensive?",
            "Return the product name.",
            db_root,
        )
        gold_sql = "SELECT name FROM products ORDER BY price DESC LIMIT 1"
        response = "```sql\nSELECT name FROM products ORDER BY price DESC LIMIT 1;\n```"
        reward = score_sql(response, gold_sql, "toy", str(db_root))
        rows = [
            {
                "question_id": 0,
                "db_id": "toy",
                "question": "Which product is the most expensive?",
                "evidence": "Return the product name.",
                "SQL": gold_sql,
                "difficulty": "simple",
            }
        ]
        records, summary = evaluate_predictions(
            rows,
            [{"question_id": 0, "response": response}],
            db_root,
            workers=1,
        )
        result = {
            "prompt_chars": len(prompt),
            "reward": reward.to_dict(),
            "evaluation": summary.to_dict(),
            "status": records[0]["status"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if reward.total != 1.0 or summary.correct != 1:
            raise SystemExit("smoke test failed")


if __name__ == "__main__":
    main()

