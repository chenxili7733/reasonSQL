#!/usr/bin/env python3
"""Export ReasonSQL JSON predictions to the classic BIRD .sql line format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasonsql.evaluation import load_predictions, write_bird_sql_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_predictions(args.predictions)
    missing_db = [item for item in records if not item.get("db_id")]
    if missing_db:
        raise ValueError("all prediction records need db_id for BIRD export")
    write_bird_sql_file(args.output, records)
    print(args.output)


if __name__ == "__main__":
    main()

