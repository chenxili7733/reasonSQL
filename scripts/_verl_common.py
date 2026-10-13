"""Shared launch helpers for ReasonSQL veRL experiments."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

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


def load_config(path: str) -> dict[str, Any]:
    """Load a veRL/Hydra override configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"Missing configuration file: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit("Configuration root must be a YAML mapping")
    return config


def resolve_path(value: str) -> str:
    """Resolve project-relative paths while preserving absolute paths."""
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def hydra_args(values: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested mapping into Hydra ``key=value`` overrides."""
    args: list[str] = []
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(hydra_args(value, name))
        elif isinstance(value, bool):
            args.append(f"{name}={value}")
        elif isinstance(value, list):
            args.append(f"{name}=[{','.join(map(str, value))}]")
        elif value is not None:
            args.append(f"{name}={value}")
    return args


def config_value(config: dict[str, Any], dotted_key: str) -> Any:
    """Return a nested config value, failing with a useful launcher error."""
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise SystemExit(f"Configuration is missing '{dotted_key}'")
        value = value[key]
    return value


def normalize_grpo_paths(config: dict[str, Any]) -> None:
    """Resolve the project-relative paths used by the GRPO launcher in place."""
    for dotted_key in (
        "actor_rollout_ref.model.path",
        "data.train_files",
        "data.val_files",
        "trainer.default_local_dir",
    ):
        parent: Any = config
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                raise SystemExit(f"Configuration is missing '{dotted_key}'")
            parent = parent[key]
        value = parent.get(keys[-1]) if isinstance(parent, dict) else None
        if not isinstance(value, str):
            raise SystemExit(f"Configuration field '{dotted_key}' must be a string path")
        parent[keys[-1]] = resolve_path(value)


def check_grpo_inputs(config: dict[str, Any]) -> None:
    """Validate the model and training data named by a native veRL config."""
    check_inputs(
        str(config_value(config, "actor_rollout_ref.model.path")),
        str(config_value(config, "data.train_files")),
    )


def launch(args: list[str], extra: list[str]) -> None:
    subprocess.run(
        ["python", "-m", "verl.trainer.main_ppo", *args, *extra], cwd=ROOT, check=True
    )
