"""Compatibility facade for the old ``common`` namespace.

Historically data helpers lived under ``common``; they now live under
``reasonsql.data``.  This module is kept as an import-compatibility shim.
"""

from reasonsql.data import (  # noqa: F401
    BirdSplit,
    PrepareConfig,
    deterministic_subset,
    discover_split,
    load_json_rows,
    normalize_bird_row,
    prepare_bird,
    sha256_file,
    write_jsonl,
    write_parquet,
)

__all__ = [
    "BirdSplit",
    "PrepareConfig",
    "deterministic_subset",
    "discover_split",
    "load_json_rows",
    "normalize_bird_row",
    "prepare_bird",
    "sha256_file",
    "write_jsonl",
    "write_parquet",
]

