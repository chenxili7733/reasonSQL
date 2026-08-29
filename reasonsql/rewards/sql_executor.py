"""Safe SQLite execution and result comparison for BIRD-style Text-to-SQL.

Important properties:

* opens database files in immutable, read-only mode;
* rejects multiple statements and non-query statements before execution;
* installs a SQLite authorizer as a second write-protection layer;
* enforces both wall-clock and virtual-machine-step limits;
* preserves duplicate rows instead of collapsing results into a ``set``;
* optionally accepts semantically equivalent output-column permutations.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - exercised on minimal bootstrap environments
    sqlglot = None
    exp = None


class ExecError(RuntimeError):
    """A categorized SQL validation or execution error."""

    def __init__(self, kind: str, message: str = ""):
        self.kind = kind
        self.message = message
        super().__init__(f"[{kind}] {message}".rstrip())


@dataclass(frozen=True)
class ExecutionConfig:
    timeout_seconds: float = 5.0
    progress_steps: int = 10_000
    max_rows: int = 100_000
    numeric_decimals: int = 6
    allow_column_permutation: bool = True
    max_permutation_columns: int = 8
    gold_cache_size: int = 50_000

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.progress_steps < 100:
            raise ValueError("progress_steps must be >= 100")
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")


@dataclass(frozen=True)
class QueryResult:
    rows: tuple[tuple[Any, ...], ...]
    column_names: tuple[str, ...]
    ordered: bool

    @property
    def width(self) -> int:
        if self.rows:
            return len(self.rows[0])
        return len(self.column_names)


_LEADING_KEYWORD_RE = re.compile(
    r"^\s*(?:(?:--[^\n]*(?:\n|$))|(?:/\*.*?\*/\s*))*([a-zA-Z_]+)",
    re.DOTALL,
)
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _split_sql_statements(sql: str) -> list[str]:
    """Split on semicolons while respecting strings, quoted identifiers and comments."""

    statements: list[str] = []
    start = 0
    index = 0
    quote: str | None = None
    line_comment = False
    block_comment = False
    while index < len(sql):
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < len(sql) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if char == quote:
                if nxt == quote and quote in {"'", '"'}:
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if char == "-" and nxt == "-":
            line_comment = True
            index += 2
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "[":
            quote = "]"
            index += 1
            continue
        if char == ";":
            part = sql[start:index].strip()
            if part:
                statements.append(part)
            start = index + 1
        index += 1

    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _fallback_validate_query(sql: str) -> None:
    statements = _split_sql_statements(sql)
    if len(statements) != 1:
        raise ExecError("multiple_statements", "exactly one SQL statement is required")
    match = _LEADING_KEYWORD_RE.match(statements[0])
    keyword = match.group(1).casefold() if match else ""
    if keyword not in {"select", "with", "values"}:
        raise ExecError("unsafe_statement", f"only SELECT/WITH/VALUES queries are allowed, got {keyword!r}")


def validate_read_only_query(sql: str) -> str:
    """Return the single cleaned statement or raise :class:`ExecError`."""

    if not isinstance(sql, str) or not sql.strip():
        raise ExecError("empty_sql", "SQL is empty")
    statements = _split_sql_statements(sql)
    if len(statements) != 1:
        raise ExecError("multiple_statements", "exactly one SQL statement is required")
    statement = statements[0].strip()

    if sqlglot is None:
        _fallback_validate_query(statement)
        return statement

    try:
        parsed = sqlglot.parse(statement, read="sqlite")
    except Exception as exc:
        raise ExecError("syntax", str(exc)[:300]) from exc
    if len(parsed) != 1 or parsed[0] is None:
        raise ExecError("multiple_statements", "exactly one SQL statement is required")

    tree = parsed[0]
    forbidden_names = {
        "Alter",
        "Analyze",
        "Attach",
        "Command",
        "Copy",
        "Create",
        "Delete",
        "Detach",
        "Drop",
        "Insert",
        "LoadData",
        "Merge",
        "Pragma",
        "Transaction",
        "TruncateTable",
        "Update",
        "Use",
    }
    for node in tree.walk():
        if node.__class__.__name__ in forbidden_names:
            raise ExecError("unsafe_statement", f"forbidden SQL node: {node.__class__.__name__}")

    # sqlglot query expressions expose a Query base class in supported versions.
    query_type = getattr(exp, "Query", ()) if exp is not None else ()
    if query_type and not isinstance(tree, query_type):
        # VALUES may be represented separately depending on sqlglot version.
        values_type = getattr(exp, "Values", ())
        if not values_type or not isinstance(tree, values_type):
            raise ExecError("unsafe_statement", f"not a read-only query: {tree.__class__.__name__}")
    return statement


def _canonical_decimal(value: Any, decimals: int) -> Decimal | None:
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return Decimal(str(round(value, decimals))).normalize()
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped):
            return None
        try:
            return Decimal(stripped).quantize(Decimal(1).scaleb(-decimals)).normalize()
        except (InvalidOperation, ValueError):
            return None
    return None


def normalize_value(value: Any, decimals: int = 6) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    number = _canonical_decimal(value, decimals)
    if number is not None:
        return ("number", number)
    if isinstance(value, bytes):
        return ("bytes", bytes(value))
    return ("text", str(value).strip())


def normalize_row(row: Sequence[Any], decimals: int = 6) -> tuple[tuple[str, Any], ...]:
    return tuple(normalize_value(value, decimals) for value in row)


def _normalize_rows(
    rows: Iterable[Sequence[Any]],
    decimals: int,
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    return tuple(normalize_row(row, decimals) for row in rows)


def _permute_rows(
    rows: tuple[tuple[tuple[str, Any], ...], ...],
    permutation: tuple[int, ...],
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    return tuple(tuple(row[index] for index in permutation) for row in rows)


def _rows_equal(
    predicted: tuple[tuple[tuple[str, Any], ...], ...],
    gold: tuple[tuple[tuple[str, Any], ...], ...],
    ordered: bool,
) -> bool:
    return predicted == gold if ordered else Counter(predicted) == Counter(gold)


def results_equal(
    predicted: QueryResult,
    gold: QueryResult,
    config: ExecutionConfig | None = None,
) -> bool:
    """Compare results while preserving duplicates and respecting gold ``ORDER BY``."""

    cfg = config or ExecutionConfig()
    pred_rows = _normalize_rows(predicted.rows, cfg.numeric_decimals)
    gold_rows = _normalize_rows(gold.rows, cfg.numeric_decimals)
    if len(pred_rows) != len(gold_rows) or predicted.width != gold.width:
        return False

    ordered = gold.ordered
    if _rows_equal(pred_rows, gold_rows, ordered):
        return True
    width = gold.width
    if (
        not cfg.allow_column_permutation
        or width <= 1
        or width > cfg.max_permutation_columns
        or ordered
    ):
        return False
    for permutation in itertools.permutations(range(width)):
        if permutation == tuple(range(width)):
            continue
        if _rows_equal(_permute_rows(pred_rows, permutation), gold_rows, ordered=False):
            return True
    return False


def result_set(rows: Iterable[Sequence[Any]]) -> set[tuple[tuple[str, Any], ...]]:
    """Legacy helper. New code should use :func:`results_equal` to preserve duplicates."""

    return set(_normalize_rows(rows, ExecutionConfig().numeric_decimals))


def _authorizer(action_code: int, _arg1: str, _arg2: str, _db: str, _trigger: str) -> int:
    denied_names = (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
    )
    denied = {getattr(sqlite3, name) for name in denied_names if hasattr(sqlite3, name)}
    return sqlite3.SQLITE_DENY if action_code in denied else sqlite3.SQLITE_OK


class SQLiteExecutor:
    """Thread-safe executor; each call uses its own read-only connection."""

    def __init__(self, db_dir: str | os.PathLike[str], config: ExecutionConfig | None = None):
        self.db_dir = Path(db_dir).expanduser().resolve()
        self.config = config or ExecutionConfig()
        self.config.validate()
        self._gold_cache: OrderedDict[tuple[str, str], QueryResult] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _db_path(self, db_id: str) -> Path:
        candidate = (self.db_dir / db_id / f"{db_id}.sqlite").resolve()
        if self.db_dir not in candidate.parents:
            raise ExecError("invalid_db_id", f"database path escapes root: {db_id!r}")
        if not candidate.is_file():
            raise ExecError("missing_database", str(candidate))
        return candidate

    @staticmethod
    def validate_sql(sql: str) -> str:
        return validate_read_only_query(sql)

    def execute(self, db_id: str, sql: str) -> QueryResult:
        statement = validate_read_only_query(sql)
        db_path = self._db_path(db_id)
        deadline = time.monotonic() + self.config.timeout_seconds
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1",
            uri=True,
            timeout=self.config.timeout_seconds,
        )
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(_authorizer)
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() >= deadline else 0,
                self.config.progress_steps,
            )
            cursor = conn.execute(statement)
            rows = cursor.fetchmany(self.config.max_rows + 1)
            if len(rows) > self.config.max_rows:
                raise ExecError("too_many_rows", f"query returned more than {self.config.max_rows} rows")
            names = tuple(column[0] for column in (cursor.description or ()))
            return QueryResult(tuple(tuple(row) for row in rows), names, bool(_ORDER_BY_RE.search(statement)))
        except ExecError:
            raise
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            lowered = message.casefold()
            if "interrupted" in lowered:
                kind = "timeout"
            elif "not authorized" in lowered or "readonly" in lowered:
                kind = "unsafe_statement"
            elif "syntax" in lowered or "incomplete input" in lowered:
                kind = "syntax"
            else:
                kind = "execution"
            raise ExecError(kind, message[:300]) from exc
        finally:
            conn.close()

    def gold_result(self, db_id: str, gold_sql: str) -> QueryResult:
        key = (db_id, gold_sql)
        with self._cache_lock:
            cached = self._gold_cache.get(key)
            if cached is not None:
                self._gold_cache.move_to_end(key)
                return cached
        result = self.execute(db_id, gold_sql)
        with self._cache_lock:
            self._gold_cache[key] = result
            self._gold_cache.move_to_end(key)
            while len(self._gold_cache) > self.config.gold_cache_size:
                self._gold_cache.popitem(last=False)
        return result

    def is_execution_correct(self, db_id: str, predicted_sql: str, gold_sql: str) -> bool:
        predicted = self.execute(db_id, predicted_sql)
        gold = self.gold_result(db_id, gold_sql)
        return results_equal(predicted, gold, self.config)


_EXECUTORS: dict[tuple[str, ExecutionConfig], SQLiteExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def get_executor(
    db_dir: str | os.PathLike[str],
    config: ExecutionConfig | None = None,
) -> SQLiteExecutor:
    cfg = config or ExecutionConfig()
    key = (str(Path(db_dir).expanduser().resolve()), cfg)
    with _EXECUTORS_LOCK:
        if key not in _EXECUTORS:
            _EXECUTORS[key] = SQLiteExecutor(key[0], cfg)
        return _EXECUTORS[key]

