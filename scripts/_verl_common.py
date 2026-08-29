"""Shared launch helpers for ReasonSQL veRL experiments."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def setup_environment() -> None:
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
    os.environ.setdefault(
        "BIRD_TRAIN_DB_DIR", str(ROOT / "data/bird/train/train_databases")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("REASONSQL_OPD_SCHEDULE", "cosine")
    os.environ.setdefault("REASONSQL_OPD_LAMBDA_START", "1.0")
    os.environ.setdefault("REASONSQL_OPD_LAMBDA_END", "0.2")
    os.environ.setdefault("REASONSQL_OPD_WARMUP_STEPS", "0")


def check_inputs(student_model: str, rl_data: str) -> None:
    checks = (
        (student_model, "student model", Path(student_model).exists()),
        (rl_data, "RL data", Path(rl_data).is_file()),
        (
            os.environ["BIRD_TRAIN_DB_DIR"],
            "BIRD database directory",
            Path(os.environ["BIRD_TRAIN_DB_DIR"]).is_dir(),
        ),
    )
    for path, label, exists in checks:
        if not exists:
            raise SystemExit(f"Missing {label}: {path}")


def common_args(
    student_model: str,
    rl_data: str,
    output_dir: str,
    *,
    experiment: str,
    student_gpus: str,
    student_nodes: str,
    rollout_n: str,
    train_batch: str,
    prompt_len: str,
    response_len: str,
    epochs: str,
    param_offload: str,
    optimizer_offload: str,
) -> list[str]:
    max_len = int(prompt_len) + int(response_len)
    return [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"data.train_files={rl_data}",
        f"data.val_files={rl_data}",
        f"data.train_batch_size={train_batch}",
        f"data.max_prompt_length={prompt_len}",
        f"data.max_response_length={response_len}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={student_model}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=128",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optimizer_offload}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=0.95",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        f"actor_rollout_ref.rollout.max_model_len={max_len}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=16384",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"custom_reward_function.path={ROOT / 'reasonsql/verl_reward.py'}",
        "custom_reward_function.name=compute_score",
        "trainer.project_name=reasonsql",
        f"trainer.experiment_name={experiment}",
        "trainer.logger=[console,wandb]",
        f"trainer.nnodes={student_nodes}",
        f"trainer.n_gpus_per_node={student_gpus}",
        f"trainer.total_epochs={epochs}",
        "trainer.save_freq=50",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={output_dir}",
        "trainer.device=cuda",
    ]


def launch(args: list[str], extra: list[str]) -> None:
    subprocess.run(
        ["python", "-m", "verl.trainer.main_ppo", *args, *extra], cwd=ROOT, check=True
    )
