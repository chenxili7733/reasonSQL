#!/usr/bin/env python3
"""Convert filtered BIRD ``input_seq``/``output_seq`` rows to OpenAI messages."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", payload.get("examples")))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {path}")
    rows = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not row.get("input_seq") or not row.get("output_seq"):
            raise ValueError(f"row {index} must contain non-empty input_seq and output_seq")
        rows.append(row)
    return rows


def to_openai(row: dict[str, Any]) -> dict[str, Any]:
    """Keep source fields for traceability while adding the standard messages field."""
    result = dict(row)
    result.setdefault("question_id", None)
    result["messages"] = [
        {"role": "user", "content": str(row["input_seq"])},
        {"role": "assistant", "content": str(row["output_seq"])},
    ]
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rl(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("writing RL parquet requires pandas and pyarrow") from exc
    records = []
    for row in rows:
        records.append(
            {
                "data_source": "bird",
                "ability": "text2sql",
                "prompt": [row["messages"][0]],
                "reward_model": {"style": "rule", "ground_truth": row["output_seq"]},
                "extra_info": {
                    "question_id": row.get("question_id"),
                    "db_id": row.get("db_id"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "split": "train",
                },
            }
        )
    write_jsonl(path.with_suffix(".jsonl"), records)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/bird/train_filtered/train_bird.json")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional source rows containing db_id; defaults to train.json beside --input.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--rl-size", type=int, default=2000)
    parser.add_argument("--smoke-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rl_size < 1 or args.smoke_size < 1:
        raise SystemExit("--rl-size and --smoke-size must be positive")
    source = load_rows(args.input)
    metadata_path = args.metadata or args.input.with_name("train.json")
    if metadata_path.is_file():
        if metadata_path.name == "train.json":
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = load_rows(metadata_path)
        if isinstance(metadata, list) and len(metadata) == len(source):
            for index, row in enumerate(source):
                meta = metadata[index]
                for key in ("db_id", "question_id", "evidence", "difficulty"):
                    if key in meta:
                        row[key] = meta[key]
    for index, row in enumerate(source):
        row.setdefault("question_id", index)
    converted = [to_openai(row) for row in source]
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(converted)), min(args.rl_size, len(converted))))
    rl_rows = [converted[index] for index in indices]
    smoke_rows = converted[: min(args.smoke_size, len(converted))]
    out = args.out_dir
    write_jsonl(out / "bird_sft_openai.jsonl", converted)
    write_jsonl(out / "bird_sft_smoke_openai.jsonl", smoke_rows)
    write_rl(out / f"bird_rl_openai_{len(rl_rows)}.parquet", rl_rows)
    manifest = {
        "source": str(args.input.resolve()),
        "sft_examples": len(converted),
        "rl_examples": len(rl_rows),
        "smoke_examples": len(smoke_rows),
        "seed": args.seed,
        "outputs": {
            "sft": str((out / "bird_sft_openai.jsonl").resolve()),
            "smoke": str((out / "bird_sft_smoke_openai.jsonl").resolve()),
            "rl_parquet": str((out / f"bird_rl_openai_{len(rl_rows)}.parquet").resolve()),
            "rl_jsonl": str((out / f"bird_rl_openai_{len(rl_rows)}.jsonl").resolve()),
        },
    }
    (out / "openai_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
