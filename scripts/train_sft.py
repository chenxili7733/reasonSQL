"""Python launcher for LLaMA-Factory SFT."""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/sft.yaml"))
    p.add_argument("--gpus", default=os.environ.get("STUDENT_GPUS", "0,1,2,3"))
    ns, extra = p.parse_known_args()
    data = os.environ.get("DATA_FILE", str(ROOT / "data/processed/bird_sft_openai.jsonl"))
    if not Path(data).is_file():
        raise SystemExit(
            f"Missing SFT data: {data}. Run scripts/prepare_openai_data.py first."
        )
    run_env = os.environ.copy()
    run_env.update(
        CUDA_VISIBLE_DEVICES=ns.gpus, FORCE_TORCHRUN="1", TOKENIZERS_PARALLELISM="false"
    )
    subprocess.run(
        ["llamafactory-cli", "train", ns.config, *extra],
        cwd=ROOT,
        env=run_env,
        check=True,
    )


if __name__ == "__main__":
    main()
