#!/usr/bin/env python3
"""Prepare SFT, RL and evaluation artifacts from a local BIRD checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasonsql.data import PrepareConfig, prepare_bird  # noqa: E402
from reasonsql.prompts import PromptConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird-dir", type=Path, default=ROOT / "data" / "bird")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--rl-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-sample-rows", type=int, default=2)
    parser.add_argument("--max-schema-chars", type=int, default=24_000)
    parser.add_argument("--no-sample-rows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = PromptConfig(
        max_sample_rows=args.max_sample_rows,
        max_schema_chars=args.max_schema_chars,
        include_sample_rows=not args.no_sample_rows,
    )
    manifest = prepare_bird(
        PrepareConfig(
            bird_dir=args.bird_dir,
            output_dir=args.out_dir,
            rl_size=args.rl_size,
            seed=args.seed,
            workers=args.workers,
            overwrite=args.overwrite,
            prompt=prompt,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

