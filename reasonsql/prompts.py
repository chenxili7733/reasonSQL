"""BIRD prompt and schema serialization utilities.

The same builder is used by SFT, RL and evaluation so that prompt drift cannot
silently invalidate an ablation.  The implementation deliberately keeps all
database access read-only.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier without accepting SQL fragments."""

    return '"' + str(name).replace('"', '""') + '"'


@dataclass(frozen=True)
class PromptConfig:
    """Controls schema serialization and output format."""

    max_sample_rows: int = 2
    max_cell_chars: int = 48
    max_description_chars: int = 160
    max_schema_chars: int = 24_000
    include_column_descriptions: bool = True
    include_sample_rows: bool = True
    include_foreign_keys: bool = True
    output_format: str = "fenced_sql"

    def validate(self) -> None:
        if self.max_sample_rows < 0:
            raise ValueError("max_sample_rows must be >= 0")
        if self.max_schema_chars < 1_000:
            raise ValueError("max_schema_chars must be >= 1000")
        if self.output_format not in {"fenced_sql", "sql_tags", "raw_sql"}:
            raise ValueError(f"unsupported output_format: {self.output_format}")


def _description_dir(database_root: Path, db_id: str) -> Path:
    return database_root / db_id / "database_description"


def database_path(database_root: str | os.PathLike[str], db_id: str) -> Path:
    root = Path(database_root).expanduser().resolve()
    candidate = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in candidate.parents:
        raise ValueError(f"invalid db_id outside database root: {db_id!r}")
    return candidate


def _read_descriptions(database_root: Path, db_id: str, cfg: PromptConfig) -> dict[tuple[str, str], str]:
    descriptions: dict[tuple[str, str], str] = {}
    root = _description_dir(database_root, db_id)
    if not root.is_dir() or not cfg.include_column_descriptions:
        return descriptions

    for csv_path in sorted(root.glob("*.csv")):
        rows: list[dict[str, Any]] = []
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with csv_path.open(encoding=encoding, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                break
            except (UnicodeDecodeError, csv.Error):
                continue
        for row in rows:
            table = str(row.get("table") or row.get("table_name") or csv_path.stem).strip()
            column = str(row.get("column") or row.get("column_name") or "").strip()
            description = str(
                row.get("column_description")
                or row.get("colum_description")  # typo in some BIRD files
                or row.get("description")
                or ""
            ).strip()
            value_description = str(row.get("value_description") or "").strip()
            text = " | ".join(part for part in (description, value_description) if part)
            if table and column and text:
                descriptions[(table.casefold(), column.casefold())] = text[: cfg.max_description_chars]
    return descriptions


def _sample_rows(conn: sqlite3.Connection, table: str, cfg: PromptConfig) -> list[tuple[Any, ...]]:
    if not cfg.include_sample_rows or cfg.max_sample_rows == 0:
        return []
    sql = f"SELECT * FROM {quote_identifier(table)} LIMIT ?"
    try:
        return conn.execute(sql, (cfg.max_sample_rows,)).fetchall()
    except sqlite3.Error:
        return []


def _foreign_keys(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})").fetchall()
    except sqlite3.Error:
        return []
    return [f"{table}.{row[3]} -> {row[2]}.{row[4]}" for row in rows]


def _format_sample_value(value: Any, max_chars: int) -> str:
    if value is None:
        return "NULL"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max(0, max_chars - 1)] + "…"
    return text


@lru_cache(maxsize=256)
def _serialize_schema_cached(
    database_root: str,
    db_id: str,
    cfg: PromptConfig,
    database_mtime_ns: int,
) -> str:
    del database_mtime_ns  # part of the cache key
    cfg.validate()
    root = Path(database_root)
    path = database_path(root, db_id)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    descriptions = _read_descriptions(root, db_id, cfg)
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        objects = conn.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY name
            """
        ).fetchall()

        blocks: list[str] = []
        for object_type, table, ddl in objects:
            lines = [str(ddl).strip().rstrip(";") + ";"]
            if object_type == "table":
                columns = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
                for column in columns:
                    description = descriptions.get((table.casefold(), str(column[1]).casefold()))
                    if description:
                        lines.append(f"-- {table}.{column[1]}: {description}")
                if cfg.include_foreign_keys:
                    lines.extend(f"-- foreign key: {fk}" for fk in _foreign_keys(conn, table))
                samples = _sample_rows(conn, table, cfg)
                if samples:
                    rendered = [
                        " | ".join(_format_sample_value(value, cfg.max_cell_chars) for value in row)
                        for row in samples
                    ]
                    lines.append(f"-- sample rows: {' ; '.join(rendered)}")
            blocks.append("\n".join(lines))
    finally:
        conn.close()

    schema = "\n\n".join(blocks)
    if len(schema) > cfg.max_schema_chars:
        schema = schema[: cfg.max_schema_chars].rstrip() + "\n-- schema truncated by ReasonSQL"
    return schema


def serialize_schema(
    database_root: str | os.PathLike[str],
    db_id: str,
    config: PromptConfig | None = None,
) -> str:
    cfg = config or PromptConfig()
    path = database_path(database_root, db_id)
    mtime_ns = path.stat().st_mtime_ns if path.exists() else 0
    return _serialize_schema_cached(str(Path(database_root).expanduser().resolve()), db_id, cfg, mtime_ns)


def _format_instruction(output_format: str) -> str:
    if output_format == "fenced_sql":
        return "Return exactly one executable SQLite query inside a single ```sql code block. No explanation."
    if output_format == "sql_tags":
        return "Return exactly one executable SQLite query between <sql> and </sql>. No explanation."
    return "Return exactly one executable SQLite query and no explanation."


def build_prompt_text(
    db_id: str,
    question: str,
    evidence: str | None,
    db_dir: str | os.PathLike[str],
    config: PromptConfig | None = None,
) -> str:
    cfg = config or PromptConfig()
    schema = serialize_schema(db_dir, db_id, cfg)
    evidence_text = str(evidence or "").strip() or "(none)"
    return (
        "You are an expert SQLite developer. Use only the supplied schema.\n\n"
        f"## Database\n{db_id}\n\n"
        f"## Schema\n{schema}\n\n"
        f"## Evidence\n{evidence_text}\n\n"
        f"## Question\n{str(question).strip()}\n\n"
        f"## Output contract\n{_format_instruction(cfg.output_format)}"
    )


def build_messages(
    db_id: str,
    question: str,
    evidence: str | None,
    db_dir: str | os.PathLike[str],
    config: PromptConfig | None = None,
) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_prompt_text(db_id, question, evidence, db_dir, config)}]


def format_target(sql: str, output_format: str = "fenced_sql") -> str:
    clean = str(sql).strip().rstrip(";") + ";"
    if output_format == "fenced_sql":
        return f"```sql\n{clean}\n```"
    if output_format == "sql_tags":
        return f"<sql>{clean}</sql>"
    return clean

