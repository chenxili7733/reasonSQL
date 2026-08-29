#!/usr/bin/env python3
"""Compare evaluation JSON files and print a compact ablation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="evaluation JSON files")
    parser.add_argument("--json", type=Path, help="write the table as JSON")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", payload)
        rows.append(
            {
                "name": path.stem,
                "path": str(path),
                "execution_accuracy": summary.get("execution_accuracy"),
                "correct": summary.get("correct"),
                "total": summary.get("total"),
                "difficulty": summary.get("by_difficulty", {}),
                "status": summary.get("by_status", {}),
            }
        )

    ordered = sorted(rows, key=lambda row: (row["execution_accuracy"] is None, -(row["execution_accuracy"] or 0.0)))
    print(f"{'experiment':40s} {'EX':>8s}  {'correct':>7s}  {'total':>6s}")
    for row in ordered:
        ex = f"{row['execution_accuracy']:.2%}" if row["execution_accuracy"] is not None else "ERR"
        print(
            f"{row['name'][:40]:40s} {ex:>8s}  "
            f"{str(row['correct'] or '-'):>7s}  {str(row['total'] or '-'):>6s}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\ntable written to {args.json}")


if __name__ == "__main__":
    main()

