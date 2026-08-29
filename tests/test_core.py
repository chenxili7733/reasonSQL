from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from reasonsql.prompts import PromptConfig, build_prompt_text, format_target, serialize_schema
from reasonsql.rewards.bird_reward import extract_sql_block, score_sql, selection_scores
from reasonsql.rewards.sql_executor import (
    ExecError,
    ExecutionConfig,
    QueryResult,
    SQLiteExecutor,
    results_equal,
    validate_read_only_query,
)


class DatabaseFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_root = Path(self.tempdir.name)
        self.db_id = "demo"
        database_dir = self.db_root / self.db_id
        database_dir.mkdir()
        db_path = database_dir / f"{self.db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                salary REAL,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );
            INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales');
            INSERT INTO employees VALUES
              (1, 'Alice', 1, 100.0),
              (2, 'Bob', 1, 90.0),
              (3, 'Bob', 1, 90.0),
              (4, 'Cara', 2, 80.0);
            """
        )
        conn.close()

        description_dir = database_dir / "database_description"
        description_dir.mkdir()
        (description_dir / "employees.csv").write_text(
            "table,column,column_description,value_description\n"
            "employees,salary,Annual salary,US dollars\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class PromptTests(DatabaseFixture):
    def test_schema_and_prompt_are_stable(self) -> None:
        cfg = PromptConfig(max_sample_rows=1)
        schema = serialize_schema(self.db_root, self.db_id, cfg)
        self.assertIn("CREATE TABLE employees", schema)
        self.assertIn("employees.salary: Annual salary | US dollars", schema)
        self.assertIn("foreign key: employees.department_id -> departments.id", schema)
        prompt = build_prompt_text(
            self.db_id,
            "Which employee has the highest salary?",
            "salary means annual salary",
            self.db_root,
            cfg,
        )
        self.assertIn("## Schema", prompt)
        self.assertIn("## Evidence", prompt)
        self.assertIn("```sql code block", prompt)

    def test_target_format(self) -> None:
        self.assertEqual(format_target("SELECT 1"), "```sql\nSELECT 1;\n```")
        self.assertEqual(format_target("SELECT 1;", "sql_tags"), "<sql>SELECT 1;</sql>")


class ExecutorTests(DatabaseFixture):
    def test_read_only_and_multiple_statement_guards(self) -> None:
        executor = SQLiteExecutor(self.db_root)
        with self.assertRaises(ExecError) as update_error:
            executor.execute(self.db_id, "UPDATE employees SET salary = 0")
        self.assertEqual(update_error.exception.kind, "unsafe_statement")

        with self.assertRaises(ExecError) as multi_error:
            executor.execute(self.db_id, "SELECT 1; SELECT 2")
        self.assertEqual(multi_error.exception.kind, "multiple_statements")

        # A semicolon inside a literal is not a second statement.
        result = executor.execute(self.db_id, "SELECT 'a;b'")
        self.assertEqual(result.rows, (("a;b",),))

    def test_duplicate_rows_are_preserved(self) -> None:
        executor = SQLiteExecutor(self.db_root)
        duplicated = executor.execute(
            self.db_id,
            "SELECT name FROM employees WHERE name = 'Bob'",
        )
        single = QueryResult(rows=(("Bob",),), column_names=("name",), ordered=False)
        self.assertFalse(results_equal(duplicated, single))

    def test_column_permutation_and_order_semantics(self) -> None:
        cfg = ExecutionConfig(allow_column_permutation=True)
        predicted = QueryResult(
            rows=(("Alice", 100), ("Bob", 90)),
            column_names=("name", "salary"),
            ordered=False,
        )
        gold = QueryResult(
            rows=((100, "Alice"), (90, "Bob")),
            column_names=("salary", "name"),
            ordered=False,
        )
        self.assertTrue(results_equal(predicted, gold, cfg))
        ordered_gold = QueryResult(gold.rows, gold.column_names, ordered=True)
        self.assertFalse(results_equal(predicted, ordered_gold, cfg))

    def test_numeric_normalization(self) -> None:
        predicted = QueryResult(rows=((3,), ("3.000000",)), column_names=("x",), ordered=False)
        gold = QueryResult(rows=((3.0,), (3,)), column_names=("x",), ordered=False)
        self.assertTrue(results_equal(predicted, gold))

    def test_validation(self) -> None:
        self.assertEqual(validate_read_only_query("WITH x AS (SELECT 1) SELECT * FROM x").split()[0], "WITH")
        with self.assertRaises(ExecError):
            validate_read_only_query("PRAGMA table_info(employees)")


class RewardTests(DatabaseFixture):
    def test_extract_sql_contract(self) -> None:
        sql, format_ok = extract_sql_block("```sql\nSELECT 1;\n```")
        self.assertEqual(sql, "SELECT 1;")
        self.assertTrue(format_ok)
        sql, format_ok = extract_sql_block("Explanation\n```sql\nSELECT 1;\n```")
        self.assertEqual(sql, "SELECT 1;")
        self.assertFalse(format_ok)

    def test_execution_correct_is_dominant(self) -> None:
        gold = "SELECT name FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)"
        correct = score_sql(
            "```sql\nSELECT name FROM employees ORDER BY salary DESC LIMIT 1\n```",
            gold,
            self.db_id,
            str(self.db_root),
        )
        wrong = score_sql(
            "```sql\nSELECT name FROM employees WHERE department_id = 1\n```",
            gold,
            self.db_id,
            str(self.db_root),
        )
        self.assertEqual(correct.execution, 1.0)
        self.assertEqual(correct.total, 1.0)
        self.assertLess(wrong.total, correct.total)
        self.assertLessEqual(wrong.total, 0.49)

    def test_selection_scores(self) -> None:
        table_score, column_score = selection_scores(
            "SELECT e.name FROM employees AS e",
            "SELECT name FROM employees WHERE salary > 100",
        )
        self.assertEqual(table_score, 1.0)
        self.assertGreater(column_score, 0.0)
        _, star_score = selection_scores(
            "SELECT * FROM employees",
            "SELECT name FROM employees",
        )
        self.assertEqual(star_score, 0.0)

    def test_breakdown_is_serializable(self) -> None:
        result = score_sql(
            "not sql",
            "SELECT 1",
            self.db_id,
            str(self.db_root),
        )
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()

