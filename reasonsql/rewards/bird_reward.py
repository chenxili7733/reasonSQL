"""Composite reward for BIRD Text-to-SQL trajectories.

Execution correctness is deliberately dominant.  Dense shaping rewards are
only used for incorrect trajectories, so a syntactically pretty but wrong SQL
query can never outrank an execution-correct query.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .sql_executor import ExecError, get_executor, validate_read_only_query

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None
    exp = None


_FENCE_RE = re.compile(r"```(?:\s*sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_TAG_RE = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_QUERY_START_RE = re.compile(r"^\s*(?:--[^\n]*\n\s*)*(SELECT|WITH|VALUES)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardConfig:
    execution_reward: float = 1.0
    syntax_weight: float = 0.10
    table_weight: float = 0.10
    column_weight: float = 0.20
    format_weight: float = 0.05
    dense_reward_cap: float = 0.49

    @classmethod
    def from_env(cls) -> "RewardConfig":
        mapping = {
            "execution_reward": "REASONSQL_EXECUTION_REWARD",
            "syntax_weight": "REASONSQL_SYNTAX_WEIGHT",
            "table_weight": "REASONSQL_TABLE_WEIGHT",
            "column_weight": "REASONSQL_COLUMN_WEIGHT",
            "format_weight": "REASONSQL_FORMAT_WEIGHT",
            "dense_reward_cap": "REASONSQL_DENSE_REWARD_CAP",
        }
        values: dict[str, float] = {}
        for field, variable in mapping.items():
            if variable in os.environ:
                values[field] = float(os.environ[variable])
        return cls(**values)

    def validate(self) -> None:
        if self.execution_reward <= 0:
            raise ValueError("execution_reward must be positive")
        if self.dense_reward_cap >= self.execution_reward:
            raise ValueError("dense_reward_cap must be smaller than execution_reward")
        if min(self.syntax_weight, self.table_weight, self.column_weight, self.format_weight) < 0:
            raise ValueError("reward weights must be non-negative")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    execution: float
    syntax: float
    table_selection: float
    column_selection: float
    format: float
    error_kind: str | None = None
    sql: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_sql_block(text: str | None) -> tuple[str | None, bool]:
    """Extract one query and report whether the response exactly follows a contract."""

    if not isinstance(text, str) or not text.strip():
        return None, False
    stripped = text.strip()
    matches: list[tuple[str, re.Pattern[str]]] = []
    for pattern in (_FENCE_RE, _SQL_TAG_RE, _ANSWER_RE):
        found = pattern.findall(stripped)
        matches.extend((candidate, pattern) for candidate in found)
    if len(matches) == 1:
        sql, pattern = matches[0]
        outside = pattern.sub("", stripped).strip()
        return sql.strip(), outside == ""
    if _QUERY_START_RE.match(stripped):
        return stripped, False
    return None, False


def _fallback_references(sql: str) -> tuple[set[str], set[str], bool]:
    tables = {
        match.casefold()
        for match in re.findall(r"\b(?:FROM|JOIN)\s+[`\"\[]?([\w\u0080-\uffff]+)", sql, re.IGNORECASE)
    }
    columns: set[str] = set()
    for match in re.findall(r"(?:\b\w+\.)?[`\"\[]?([\w\u0080-\uffff]+)[`\"\]]?", sql):
        lowered = match.casefold()
        if lowered not in {
            "select",
            "from",
            "where",
            "join",
            "left",
            "right",
            "inner",
            "outer",
            "on",
            "and",
            "or",
            "as",
            "group",
            "order",
            "by",
            "limit",
            "offset",
            "asc",
            "desc",
            "distinct",
            "with",
        } and lowered not in tables:
            columns.add(lowered)
    has_star = bool(re.search(r"(?<!\w)(?:\w+\.)?\*(?!\w)", sql))
    return tables, columns, has_star


def sql_references(sql: str) -> tuple[set[str], set[str], bool]:
    """Return table names, column names and whether ``*`` is used."""

    if not sqlglot:
        return _fallback_references(sql)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return _fallback_references(sql)
    tables = {node.name.casefold() for node in tree.find_all(exp.Table) if node.name}
    columns = {node.name.casefold() for node in tree.find_all(exp.Column) if node.name and node.name != "*"}
    has_star = any(True for _ in tree.find_all(exp.Star))
    return tables, columns, has_star


def _f1(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    intersection = len(predicted & gold)
    precision = intersection / len(predicted)
    recall = intersection / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def selection_scores(predicted_sql: str, gold_sql: str) -> tuple[float, float]:
    predicted_tables, predicted_columns, has_star = sql_references(predicted_sql)
    gold_tables, gold_columns, _ = sql_references(gold_sql)
    table_score = _f1(predicted_tables, gold_tables)
    column_score = 0.0 if has_star and gold_columns else _f1(predicted_columns, gold_columns)
    return table_score, column_score


def score_sql(
    solution_str: str,
    gold_sql: str,
    db_id: str,
    db_dir: str,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    cfg = config or RewardConfig.from_env()
    cfg.validate()
    sql, format_ok = extract_sql_block(solution_str)
    format_score = 1.0 if format_ok else 0.0
    if sql is None:
        dense = min(cfg.dense_reward_cap, cfg.format_weight * format_score)
        return RewardBreakdown(dense, 0.0, 0.0, 0.0, 0.0, format_score, "no_sql", None)

    syntax_score = 0.0
    error_kind: str | None = None
    try:
        validate_read_only_query(sql)
        syntax_score = 1.0
    except ExecError as exc:
        error_kind = exc.kind

    table_score, column_score = selection_scores(sql, gold_sql)
    execution_score = 0.0
    if syntax_score:
        try:
            execution_score = float(get_executor(db_dir).is_execution_correct(db_id, sql, gold_sql))
        except ExecError as exc:
            error_kind = exc.kind

    if execution_score:
        total = cfg.execution_reward
    else:
        total = min(
            cfg.dense_reward_cap,
            cfg.syntax_weight * syntax_score
            + cfg.table_weight * table_score
            + cfg.column_weight * column_score
            + cfg.format_weight * format_score,
        )
    return RewardBreakdown(
        total=float(total),
        execution=execution_score,
        syntax=syntax_score,
        table_selection=table_score,
        column_selection=column_score,
        format=format_score,
        error_kind=error_kind,
        sql=sql,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """veRL custom reward entry point."""

    del data_source
    info = extra_info or {}
    db_id = info.get("db_id")
    db_dir = info.get("db_dir") or os.environ.get("BIRD_TRAIN_DB_DIR")
    if not db_id or not db_dir or not ground_truth:
        return 0.0
    return score_sql(solution_str, ground_truth, str(db_id), str(db_dir)).total

