#!/usr/bin/env python3
"""Analyze ReasonSQL evaluation output by status, difficulty and database."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("evaluation file must contain a records list")

    total = len(records)
    by_status = Counter(str(item.get("status", "unknown")) for item in records)
    by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    by_database: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in records:
        difficulty = str(item.get("difficulty", "unknown"))
        status = str(item.get("status", "unknown"))
        by_difficulty[difficulty][status] += 1
        db_stats = by_database[str(item.get("db_id", "unknown"))]
        db_stats[0] += 1
        db_stats[1] += int(bool(item.get("correct")))

    report = {
        "total": total,
        "status": dict(by_status),
        "difficulty": {key: dict(value) for key, value in sorted(by_difficulty.items())},
        "worst_databases": [
            {"db_id": db_id, "total": counts[0], "correct": counts[1], "accuracy": counts[1] / counts[0]}
            for db_id, counts in sorted(
                by_database.items(),
                key=lambda item: (item[1][1] / item[1][0], -item[1][0], item[0]),
            )[:20]
        ],
    }

    print(f"total: {total}")
    print("\nstatus:")
    for status, count in by_status.most_common():
        ratio = count / total if total else 0.0
        print(f"  {status:24s} {count:5d} ({ratio:7.2%})")

    print("\ndifficulty:")
    for difficulty, counts in sorted(by_difficulty.items()):
        group_total = sum(counts.values())
        correct = counts.get("correct", 0)
        print(f"  {difficulty:16s} EX={correct / group_total if group_total else 0:7.2%} n={group_total}")

    print("\nlowest-accuracy databases:")
    for item in report["worst_databases"][:10]:
        print(
            f"  {item['db_id']:32s} {item['correct']:4d}/{item['total']:4d} "
            f"({item['accuracy']:7.2%})"
        )

    print("\nrepresentative failures:")
    shown = 0
    for item in records:
        if item.get("correct") or shown >= args.show:
            continue
        print(
            f"\n[{item.get('status')}] qid={item.get('question_id')} "
            f"db={item.get('db_id')} difficulty={item.get('difficulty')}"
        )
        print(f"  question: {str(item.get('question', ''))[:300]}")
        print(f"  predicted: {str(item.get('pred_sql', ''))[:500]}")
        print(f"  gold:      {str(item.get('gold_sql', ''))[:500]}")
        shown += 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nanalysis JSON: {args.output_json}")


if __name__ == "__main__":
    main()

