#!/usr/bin/env python3
"""Generate and/or execution-evaluate BIRD development predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasonsql.data import discover_split, load_json_rows  # noqa: E402
from reasonsql.evaluation import (  # noqa: E402
    evaluate_predictions,
    load_predictions,
    write_bird_sql_file,
    write_evaluation,
)
from reasonsql.generation import GenerationConfig, generate  # noqa: E402
from reasonsql.prompts import PromptConfig, build_messages  # noqa: E402
from reasonsql.rewards.bird_reward import extract_sql_block  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="HF/vLLM model directory used to generate predictions")
    source.add_argument("--predictions", type=Path, help="existing JSON/JSONL/.sql predictions")
    parser.add_argument("--bird-dir", type=Path, default=ROOT / "data" / "bird")
    parser.add_argument("--dev-file", type=Path)
    parser.add_argument("--dev-db-dir", type=Path)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "eval" / "bird_eval.json")
    parser.add_argument("--raw-predictions", type=Path)
    parser.add_argument("--bird-sql-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovered = discover_split(args.bird_dir, "dev")
    dev_file = args.dev_file or discovered.json_path
    dev_db_dir = args.dev_db_dir or discovered.database_dir
    rows = load_json_rows(dev_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    if args.model:
        prompt_cfg = PromptConfig()
        messages = [
            build_messages(row["db_id"], row["question"], row["evidence"], dev_db_dir, prompt_cfg)
            for row in rows
        ]
        generation_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
        )
        responses = generate(args.model, messages, generation_cfg, args.backend)
        predictions = []
        for row, response in zip(rows, responses):
            sql, _ = extract_sql_block(response)
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "db_id": row["db_id"],
                    "pred_sql": sql or "",
                    "response": response,
                }
            )
        raw_path = args.raw_predictions or args.output.with_name(args.output.stem + "_predictions.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        predictions = load_predictions(args.predictions)

    records, summary = evaluate_predictions(rows, predictions, dev_db_dir, workers=args.workers)
    write_evaluation(args.output, records, summary)
    if args.bird_sql_output:
        write_bird_sql_file(args.bird_sql_output, records)

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    print(f"details: {args.output}")
    if args.bird_sql_output:
        print(f"classic BIRD SQL file: {args.bird_sql_output}")


if __name__ == "__main__":
    main()

