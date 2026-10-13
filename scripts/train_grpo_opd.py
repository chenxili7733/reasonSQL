"""Python launcher for GRPO + veRL native OPD."""
from __future__ import annotations
import argparse
from _verl_common import (
    ROOT,
    check_inputs,
    env,
    hydra_args,
    load_config,
    launch,
    normalize_grpo_paths,
    resolve_path,
    setup_environment,
)


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
        "--teacher-model",
        default=env("TEACHER_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    )
    p.add_argument(
        "--rl-data",
        default=env("RL_DATA", str(ROOT / "data/processed/bird_rl_openai_2000.parquet")),
    )
    p.add_argument(
        "--output-dir", default=env("OUTPUT_DIR", str(ROOT / "checkpoints/grpo_opd"))
    )
    p.add_argument("--gpus-per-node", default=env("STUDENT_GPUS_PER_NODE", "4"))
    p.add_argument("--nodes", default=env("STUDENT_NNODES", "1"))
    p.add_argument("--teacher-gpus-per-node", default=env("TEACHER_GPUS_PER_NODE", "4"))
    p.add_argument("--teacher-nodes", default=env("TEACHER_NNODES", "1"))
    p.add_argument("--teacher-tp", default=env("TEACHER_TP", "4"))
    p.add_argument("--teacher-replicas", default=env("TEACHER_REPLICAS", "1"))
    p.add_argument("--rollout-n", default=env("ROLLOUT_N", "8"))
    p.add_argument("--train-batch-size", default=env("TRAIN_BATCH_SIZE", "32"))
    p.add_argument("--max-prompt-length", default=env("MAX_PROMPT_LENGTH", "4096"))
    p.add_argument("--max-response-length", default=env("MAX_RESPONSE_LENGTH", "512"))
    p.add_argument("--epochs", default=env("TOTAL_EPOCHS", "3"))
    p.add_argument("--opd-loss-mode", default=env("OPD_LOSS_MODE", "forward_kl_topk"))
    p.add_argument("--opd-topk", default=env("OPD_TOPK", "32"))
    p.add_argument("--param-offload", default=env("FSDP_PARAM_OFFLOAD", "True"))
    p.add_argument("--optimizer-offload", default=env("FSDP_OPTIMIZER_OFFLOAD", "True"))
    ns, extra = p.parse_known_args()
    check_inputs(ns.student_model, ns.rl_data)
    try:
        tp, replicas, teacher_gpus, teacher_nodes = map(
            int,
            (
                ns.teacher_tp,
                ns.teacher_replicas,
                ns.teacher_gpus_per_node,
                ns.teacher_nodes,
            ),
        )
    except ValueError as exc:
        raise SystemExit("Teacher topology values must be integers") from exc
    if (
        min(tp, replicas, teacher_gpus, teacher_nodes) < 1
        or tp * replicas != teacher_gpus * teacher_nodes
    ):
        raise SystemExit(
            "Teacher topology must satisfy TP * replicas == teacher GPUs/node * teacher nodes"
        )
    config = load_config(str(ROOT / "configs/verl/grpo.yaml"))
    normalize_grpo_paths(config)
    # Keep this legacy CLI interface as native veRL/Hydra overrides.
    args = hydra_args(config)
    args += [
        f"actor_rollout_ref.model.path={resolve_path(ns.student_model)}",
        f"data.train_files={resolve_path(ns.rl_data)}",
        f"data.val_files={resolve_path(ns.rl_data)}",
        f"trainer.default_local_dir={resolve_path(ns.output_dir)}",
        f"trainer.n_gpus_per_node={ns.gpus_per_node}",
        f"trainer.nnodes={ns.nodes}",
        f"trainer.total_epochs={ns.epochs}",
        "trainer.experiment_name=grpo-opd",
        f"data.train_batch_size={ns.train_batch_size}",
        f"data.max_prompt_length={ns.max_prompt_length}",
        f"data.max_response_length={ns.max_response_length}",
        f"actor_rollout_ref.rollout.n={ns.rollout_n}",
        f"actor_rollout_ref.rollout.max_model_len={int(ns.max_prompt_length) + int(ns.max_response_length)}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={ns.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={ns.optimizer_offload}",
        "distillation.enabled=True",
        f"distillation.teacher_models.teacher_model.model_path={ns.teacher_model}",
        f"distillation.teacher_models.teacher_model.num_replicas={replicas}",
        "distillation.teacher_models.teacher_model.inference.dtype=bfloat16",
        f"distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size={tp}",
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85",
        f"distillation.teacher_models.teacher_model.inference.max_model_len={int(ns.max_prompt_length) + int(ns.max_response_length) + 1}",
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=16384",
        "distillation.teacher_models.teacher_model.inference.max_num_seqs=128",
        f"distillation.distillation_loss.loss_mode={ns.opd_loss_mode}",
        f"distillation.distillation_loss.topk={ns.opd_topk}",
        "distillation.distillation_loss.use_policy_gradient=False",
        "distillation.distillation_loss.use_task_rewards=True",
        f"distillation.distillation_loss.distillation_loss_coef={env('REASONSQL_OPD_LAMBDA_START', '1.0')}",
        "distillation.distillation_loss.target_probs_ratio=1.0",
        f"distillation.nnodes={teacher_nodes}",
        f"distillation.n_gpus_per_node={teacher_gpus}",
    ]
    launch(args, extra)


if __name__ == "__main__":
    main()
