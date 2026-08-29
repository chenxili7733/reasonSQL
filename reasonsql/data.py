"""BIRD dataset discovery and deterministic conversion utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .prompts import PromptConfig, build_messages, build_prompt_text, format_target


@dataclass(frozen=True)
class BirdSplit:
    name: str
    json_path: Path
    database_dir: Path


@dataclass(frozen=True)
class PrepareConfig:
    bird_dir: Path
    output_dir: Path
    rl_size: int = 2_000
    seed: int = 42
    workers: int = 8
    overwrite: bool = False
    prompt: PromptConfig = PromptConfig()

    def validate(self) -> None:
        if self.rl_size < 1:
            raise ValueError("rl_size must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.prompt.validate()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_split_roots(bird_dir: Path, split: str) -> Iterator[Path]:
    direct = bird_dir / split
    if direct.exists():
        yield direct
    for child in sorted(bird_dir.iterdir() if bird_dir.exists() else []):
        if child.is_dir() and split.casefold() in child.name.casefold() and child != direct:
            yield child


def discover_split(bird_dir: str | os.PathLike[str], split: str) -> BirdSplit:
    """Discover common official and mirrored BIRD directory layouts."""

    root = Path(bird_dir).expanduser().resolve()
    json_names = [f"{split}.json", f"{split}_20240627.json", "train.json" if split == "train" else "dev.json"]
    db_names = [
        f"{split}_databases",
        "train_databases" if split == "train" else "dev_databases",
        "databases",
    ]
    for split_root in _candidate_split_roots(root, split):
        json_path = next((split_root / name for name in json_names if (split_root / name).is_file()), None)
        database_dir = next((split_root / name for name in db_names if (split_root / name).is_dir()), None)
        if json_path and database_dir:
            return BirdSplit(split, json_path.resolve(), database_dir.resolve())

    # Also support flat layouts.
    json_path = next((root / name for name in json_names if (root / name).is_file()), None)
    database_dir = next((root / name for name in db_names if (root / name).is_dir()), None)
    if json_path and database_dir:
        return BirdSplit(split, json_path.resolve(), database_dir.resolve())
    raise FileNotFoundError(
        f"could not discover BIRD {split!r} under {root}; expected {split}.json and {split}_databases/"
    )


def load_json_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "examples", "rows"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"expected a list of rows in {source}")
        rows = payload
    return [normalize_bird_row(row, index) for index, row in enumerate(rows)]


def normalize_bird_row(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize field-name differences among BIRD mirrors."""

    db_id = row.get("db_id") or row.get("database_id")
    question = row.get("question") or row.get("Question")
    sql = row.get("SQL") or row.get("sql") or row.get("query")
    if not db_id or not question or not sql:
        missing = [name for name, value in (("db_id", db_id), ("question", question), ("SQL", sql)) if not value]
        raise ValueError(f"BIRD row {index} is missing required fields: {missing}")
    question_id = row.get("question_id", row.get("id", index))
    return {
        "question_id": question_id,
        "db_id": str(db_id),
        "question": str(question),
        "evidence": str(row.get("evidence") or row.get("external_knowledge") or ""),
        "SQL": str(sql),
        "difficulty": str(row.get("difficulty") or row.get("level") or "unknown").casefold(),
    }


def _sft_item(args: tuple[dict[str, Any], str, PromptConfig]) -> dict[str, Any]:
    row, database_dir, prompt_cfg = args
    prompt = build_prompt_text(row["db_id"], row["question"], row["evidence"], database_dir, prompt_cfg)
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": format_target(row["SQL"], prompt_cfg.output_format)},
        ],
        "metadata": {
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "difficulty": row["difficulty"],
        },
    }


def _rl_item(args: tuple[dict[str, Any], str, PromptConfig]) -> dict[str, Any]:
    row, database_dir, prompt_cfg = args
    prompt = build_messages(row["db_id"], row["question"], row["evidence"], database_dir, prompt_cfg)
    return {
        "data_source": "bird",
        "ability": "text2sql",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": row["SQL"]},
        "extra_info": {
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "difficulty": row["difficulty"],
            "split": "train",
        },
    }


def deterministic_subset(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Sample deterministically while preserving a stable output order."""

    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(rows)), size))
    return [rows[index] for index in chosen]


def _ordered_map(function, jobs: list[Any], workers: int) -> list[Any]:
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_parquet(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("writing RL parquet requires `pip install -e '.[data]'`") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)


def prepare_bird(config: PrepareConfig) -> dict[str, Any]:
    config.validate()
    train = discover_split(config.bird_dir, "train")
    dev = discover_split(config.bird_dir, "dev")
    train_rows = load_json_rows(train.json_path)
    dev_rows = load_json_rows(dev.json_path)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "sft": output_dir / "bird_sft.jsonl",
        "rl_jsonl": output_dir / f"bird_rl_{min(config.rl_size, len(train_rows))}.jsonl",
        "rl_parquet": output_dir / f"bird_rl_{min(config.rl_size, len(train_rows))}.parquet",
        "dev": output_dir / "bird_dev.jsonl",
        "manifest": output_dir / "bird_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("output files already exist; pass --overwrite:\n" + "\n".join(existing))

    train_jobs = [(row, str(train.database_dir), config.prompt) for row in train_rows]
    sft_rows = _ordered_map(_sft_item, train_jobs, config.workers)
    write_jsonl(targets["sft"], sft_rows)

    rl_source_rows = deterministic_subset(train_rows, config.rl_size, config.seed)
    rl_jobs = [(row, str(train.database_dir), config.prompt) for row in rl_source_rows]
    rl_rows = _ordered_map(_rl_item, rl_jobs, config.workers)
    write_jsonl(targets["rl_jsonl"], rl_rows)
    write_parquet(targets["rl_parquet"], rl_rows)
    write_jsonl(targets["dev"], dev_rows)

    difficulty_counts: dict[str, int] = {}
    for row in train_rows:
        difficulty_counts[row["difficulty"]] = difficulty_counts.get(row["difficulty"], 0) + 1
    manifest = {
        "schema_version": 1,
        "train_json": str(train.json_path),
        "train_sha256": sha256_file(train.json_path),
        "train_database_dir": str(train.database_dir),
        "dev_json": str(dev.json_path),
        "dev_sha256": sha256_file(dev.json_path),
        "dev_database_dir": str(dev.database_dir),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "rl_examples": len(rl_rows),
        "seed": config.seed,
        "difficulty_counts": difficulty_counts,
        "prompt_config": asdict(config.prompt),
        "outputs": {name: str(path) for name, path in targets.items() if name != "manifest"},
    }
    targets["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

