"""Python launcher for pure GRPO; extra arguments are passed to veRL."""
from __future__ import annotations
import argparse
from _verl_common import ROOT, check_inputs, common_args, env, launch, setup_environment


def main() -> None:
    setup_environment()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--student-model",
        default=env(
            "STUDENT_MODEL", str(ROOT / "checkpoints/student_sft_qwen2.5_coder_7b")
        ),
    )
    p.add_argument(
        "--rl-data",
        default=env("RL_DATA", str(ROOT / "data/processed/bird_rl_2000.parquet")),
    )
    p.add_argument(
        "--output-dir", default=env("OUTPUT_DIR", str(ROOT / "checkpoints/grpo"))
    )
    p.add_argument("--gpus-per-node", default=env("STUDENT_GPUS_PER_NODE", "4"))
    p.add_argument("--nodes", default=env("STUDENT_NNODES", "1"))
    p.add_argument("--rollout-n", default=env("ROLLOUT_N", "8"))
    p.add_argument("--train-batch-size", default=env("TRAIN_BATCH_SIZE", "32"))
    p.add_argument("--max-prompt-length", default=env("MAX_PROMPT_LENGTH", "4096"))
    p.add_argument("--max-response-length", default=env("MAX_RESPONSE_LENGTH", "512"))
    p.add_argument("--epochs", default=env("TOTAL_EPOCHS", "3"))
    p.add_argument("--param-offload", default=env("FSDP_PARAM_OFFLOAD", "True"))
    p.add_argument("--optimizer-offload", default=env("FSDP_OPTIMIZER_OFFLOAD", "True"))
    ns, extra = p.parse_known_args()
    check_inputs(ns.student_model, ns.rl_data)
    launch(
        common_args(
            ns.student_model,
            ns.rl_data,
            ns.output_dir,
            experiment="grpo",
            student_gpus=ns.gpus_per_node,
            student_nodes=ns.nodes,
            rollout_n=ns.rollout_n,
            train_batch=ns.train_batch_size,
            prompt_len=ns.max_prompt_length,
            response_len=ns.max_response_length,
            epochs=ns.epochs,
            param_offload=ns.param_offload,
            optimizer_offload=ns.optimizer_offload,
        ),
        extra,
    )


if __name__ == "__main__":
    main()
