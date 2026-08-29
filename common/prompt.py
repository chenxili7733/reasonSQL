"""Compatibility wrapper for the legacy ``common.prompt`` import path.

New code should import directly from ``reasonsql.prompts``.
"""

from reasonsql.prompts import (  # noqa: F401
    PromptConfig,
    build_messages,
    build_prompt_text,
    database_path,
    format_target,
    quote_identifier,
    serialize_schema,
)


def load_schema(db_dir: str, db_id: str) -> str:
    """Return a serialized schema string for callers of the old helper."""

    return serialize_schema(db_dir, db_id)


__all__ = [
    "PromptConfig",
    "build_messages",
    "build_prompt_text",
    "database_path",
    "format_target",
    "load_schema",
    "quote_identifier",
    "serialize_schema",
]
