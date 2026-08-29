#!/usr/bin/env python3
"""Prepare BIRD inputs with the Arctic-Text2SQL-R1 preprocessing pipeline.

The filtered Hugging Face train set contains examples only.  Its SQLite files
and schema definitions are shared with the original BIRD train release.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESS_DIR = ROOT / "reasonsql" / "process_data"
BUILD_INDEX = PROCESS_DIR / "build_contents_index.py"
PROCESS_DATASET = PROCESS_DIR / "process_dataset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "train-filtered", "all"),
        default="all",
        help="Dataset split(s) to prepare.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--value-limit-num", type=int, default=2)
    parser.add_argument(
        "--base-train-dir",
        type=Path,
        default=ROOT / "data" / "bird" / "train",
        help="Original BIRD train release, used for filtered train databases and tables.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def build_index(db_root: Path, index_root: Path, threads: int, overwrite: bool) -> None:
    if index_root.exists():
        if not overwrite:
            raise FileExistsError(f"{index_root} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(index_root)
    run(
        [
            sys.executable,
            str(BUILD_INDEX),
            "--db-root",
            str(db_root),
            "--index-root",
            str(index_root),
            "--temp-dir",
            str(index_root.parent / "tmp_contents"),
            "--threads",
            str(threads),
        ]
    )


def process(
    input_json: Path,
    output_json: Path,
    db_root: Path,
    tables_json: Path,
    mode: str,
    index_root: Path,
    value_limit_num: int,
) -> None:
    run(
        [
            sys.executable,
            str(PROCESS_DATASET),
            "--input_data_file",
            str(input_json),
            "--output_data_file",
            str(output_json),
            "--db_path",
            str(db_root),
            "--tables",
            str(tables_json),
            "--source",
            "bird",
            "--mode",
            mode,
            "--value_limit_num",
            str(value_limit_num),
            "--db_content_index_path",
            str(index_root),
        ]
    )


def prepare_dev(data_dir: Path, threads: int, value_limit_num: int, overwrite: bool) -> None:
    dev_dir = data_dir / "bird" / "dev_20240627"
    input_json = dev_dir / "dev.json"
    tables_json = dev_dir / "dev_tables.json"
    db_root = ensure_databases(dev_dir, "dev_databases")
    output_json = data_dir / "bird" / "dev_bird.json"
    for path in (input_json, tables_json, db_root):
        if not path.exists():
            raise FileNotFoundError(f"Missing BIRD dev input: {path}")
    ensure_output_path(output_json, overwrite)
    index_root = dev_dir / "db_contents_index"
    build_index(db_root, index_root, threads, overwrite)
    process(input_json, output_json, db_root, tables_json, "dev", index_root, value_limit_num)


def write_filtered_train(data_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Download the requested dataset and emit its train records as BIRD JSON."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install datasets first: python -m pip install 'datasets>=3.0'") from error

    train_dir = data_dir / "bird" / "train_filtered"
    train_json = train_dir / "train.json"
    ensure_output_path(train_json, overwrite)
    dataset = load_dataset("birdsql/bird23-train-filtered", cache_dir=str(train_dir))
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    rows: list[dict[str, Any]] = [dict(row) for row in split]
    required = {"db_id", "question", "evidence", "SQL"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Filtered dataset is missing required BIRD columns: {sorted(missing)}")
    with train_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return train_dir, train_json


def ensure_databases(dataset_dir: Path, directory_name: str) -> Path:
    """Return an extracted BIRD database directory, extracting its zip if needed."""
    db_root = dataset_dir / directory_name
    if db_root.is_dir():
        return db_root
    archive = dataset_dir / f"{directory_name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing {db_root} and {archive}. Download the matching original BIRD release first."
        )
    print(f"Extracting {archive} to {dataset_dir}", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(dataset_dir)
    if not db_root.is_dir():
        raise FileNotFoundError(f"{archive} did not contain {db_root.name}/")
    return db_root


def filter_tables(source_tables: Path, db_ids: set[str], destination: Path, overwrite: bool) -> None:
    ensure_output_path(destination, overwrite)
    with source_tables.open(encoding="utf-8") as handle:
        tables = json.load(handle)
    selected = [table for table in tables if table["db_id"] in db_ids]
    found = {table["db_id"] for table in selected}
    missing = db_ids.difference(found)
    if missing:
        raise ValueError(f"Schema entries missing for {len(missing)} databases, e.g. {sorted(missing)[:5]}")
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_filtered_train(args: argparse.Namespace) -> None:
    train_dir, train_json = write_filtered_train(args.data_dir, args.overwrite)
    with train_json.open(encoding="utf-8") as handle:
        db_ids = {row["db_id"] for row in json.load(handle)}
    source_tables = args.base_train_dir / "train_tables.json"
    if not source_tables.is_file():
        raise FileNotFoundError(f"Missing original BIRD schema file: {source_tables}")
    db_root = ensure_databases(args.base_train_dir, "train_databases")
    missing_dbs = [db_id for db_id in db_ids if not (db_root / db_id / f"{db_id}.sqlite").is_file()]
    if missing_dbs:
        raise FileNotFoundError(f"Missing {len(missing_dbs)} train databases, e.g. {missing_dbs[:5]}")
    tables_json = train_dir / "train_tables.json"
    filter_tables(source_tables, db_ids, tables_json, args.overwrite)
    output_json = train_dir / "train_bird.json"
    ensure_output_path(output_json, args.overwrite)
    index_root = train_dir / "db_contents_index"
    build_index(db_root, index_root, args.threads, args.overwrite)
    process(train_json, output_json, db_root, tables_json, "train", index_root, args.value_limit_num)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.split in ("dev", "all"):
        prepare_dev(args.data_dir, args.threads, args.value_limit_num, args.overwrite)
    if args.split in ("train-filtered", "all"):
        prepare_filtered_train(args)


if __name__ == "__main__":
    main()
