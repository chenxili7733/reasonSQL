"""Python launcher for pure GRPO; extra arguments are passed to veRL."""
from __future__ import annotations
import argparse
from _verl_common import (
    ROOT,
    check_grpo_inputs,
    hydra_args,
    load_config,
    launch,
    normalize_grpo_paths,
    setup_environment,
)


def main() -> None:
    setup_environment()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/verl/grpo.yaml"))
    ns, extra = p.parse_known_args()
    config = load_config(ns.config)
    normalize_grpo_paths(config)
    check_grpo_inputs(config)
    launch(hydra_args(config), extra)


if __name__ == "__main__":
    main()
