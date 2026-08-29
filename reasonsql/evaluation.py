"""Prediction loading, execution evaluation and BIRD-format export."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import normalize_bird_row
from .rewards.bird_reward import extract_sql_block
from .rewards.sql_executor import ExecError, ExecutionConfig, SQLiteExecutor


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    correct: int
    execution_accuracy: float
    by_difficulty: dict[str, dict[str, float | int]]
    by_status: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_predictions(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            payload = [json.loads(line) for line in handle if line.strip()]
    elif source.suffix == ".sql":
        payload = []
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                sql, separator, db_id = line.rstrip("\n").partition("\t----- bird -----\t")
                payload.append({"question_id": index, "pred_sql": sql, "db_id": db_id if separator else ""})
    else:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = [
                {"question_id": key, **(value if isinstance(value, dict) else {"pred_sql": value})}
                for key, value in payload.items()
            ]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported prediction structure in {source}")
    return [dict(item) for item in payload]


def _prediction_sql(item: dict[str, Any]) -> tuple[str | None, str]:
    for key in ("pred_sql", "predicted_sql", "SQL", "sql", "query"):
        if item.get(key):
            return str(item[key]).strip(), str(item.get("response") or item[key])
    response = str(item.get("response") or item.get("output") or "")
    return extract_sql_block(response)[0], response


def align_predictions(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(item["question_id"]): item for item in predictions if item.get("question_id") is not None}
    aligned: list[dict[str, Any]] = []
    positional = len(by_id) == 0
    for index, row in enumerate(gold_rows):
        if positional:
            prediction = predictions[index] if index < len(predictions) else {}
        else:
            prediction = by_id.get(str(row["question_id"]), {})
        sql, response = _prediction_sql(prediction)
        aligned.append(
            {
                "question_id": row["question_id"],
                "db_id": row["db_id"],
                "difficulty": row["difficulty"],
                "question": row["question"],
                "gold_sql": row["SQL"],
                "pred_sql": sql or "",
                "response": response,
            }
        )
    return aligned


def _evaluate_one(args: tuple[dict[str, Any], str, ExecutionConfig]) -> dict[str, Any]:
    item, database_dir, execution_config = args
    output = dict(item)
    if not item["pred_sql"]:
        output["status"] = "no_sql"
        output["correct"] = False
        return output
    executor = SQLiteExecutor(database_dir, execution_config)
    try:
        correct = executor.is_execution_correct(item["db_id"], item["pred_sql"], item["gold_sql"])
        output["status"] = "correct" if correct else "mismatch"
        output["correct"] = correct
    except ExecError as exc:
        output["status"] = exc.kind
        output["correct"] = False
        output["error"] = exc.message
    return output


def evaluate_predictions(
    gold_rows: Iterable[dict[str, Any]],
    predictions: list[dict[str, Any]],
    database_dir: str | os.PathLike[str],
    workers: int = 8,
    execution_config: ExecutionConfig | None = None,
) -> tuple[list[dict[str, Any]], EvaluationSummary]:
    normalized_gold = [normalize_bird_row(row, index) for index, row in enumerate(gold_rows)]
    aligned = align_predictions(normalized_gold, predictions)
    config = execution_config or ExecutionConfig()
    jobs = [(item, str(database_dir), config) for item in aligned]
    if workers <= 1:
        records = [_evaluate_one(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_evaluate_one, jobs))
    summary = summarize_records(records)
    return records, summary


def summarize_records(records: list[dict[str, Any]]) -> EvaluationSummary:
    total = len(records)
    correct = sum(bool(item.get("correct")) for item in records)
    status = Counter(str(item.get("status", "unknown")) for item in records)
    difficulty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        difficulty_groups[str(item.get("difficulty", "unknown"))].append(item)
    by_difficulty: dict[str, dict[str, float | int]] = {}
    for difficulty, group in sorted(difficulty_groups.items()):
        group_correct = sum(bool(item.get("correct")) for item in group)
        by_difficulty[difficulty] = {
            "total": len(group),
            "correct": group_correct,
            "execution_accuracy": group_correct / len(group) if group else 0.0,
        }
    return EvaluationSummary(
        total=total,
        correct=correct,
        execution_accuracy=correct / total if total else 0.0,
        by_difficulty=by_difficulty,
        by_status=dict(status),
    )


def write_evaluation(
    path: str | os.PathLike[str],
    records: list[dict[str, Any]],
    summary: EvaluationSummary,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"summary": summary.to_dict(), "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_bird_sql_file(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> None:
    """Write the line-oriented format consumed by the classic BIRD evaluator."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in records:
            sql = str(item.get("pred_sql") or "").replace("\n", " ").strip()
            handle.write(f"{sql}\t----- bird -----\t{item['db_id']}\n")

