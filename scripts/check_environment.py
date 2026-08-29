#!/usr/bin/env python3
"""Preflight checks for ReasonSQL training nodes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module_version(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, str(exc)
    return True, str(getattr(module, "__version__", "unknown"))


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", default=str(ROOT / "checkpoints" / "student_sft_qwen2.5_coder_7b"))
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--rl-data", default=str(ROOT / "data" / "processed" / "bird_rl_2000.parquet"))
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--check-tokenizers", action="store_true")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="Fail if optional FlashAttention 2 is not installed.",
    )
    args = parser.parse_args()

    report: dict[str, object] = {"python": sys.version, "modules": {}, "files": {}, "gpu": {}}
    failed = False
    for module in ("torch", "transformers", "vllm", "verl", "ray", "sqlglot", "pandas", "pyarrow"):
        ok, version = module_version(module)
        report["modules"][module] = {"ok": ok, "version_or_error": version}
        failed |= not ok

    flash_ok, flash_version = module_version("flash_attn")
    report["modules"]["flash_attn"] = {
        "ok": flash_ok,
        "required": args.require_flash_attn,
        "version_or_error": flash_version,
    }
    failed |= args.require_flash_attn and not flash_ok

    verl_version = distribution_version("verl")
    report["verl_compatibility"] = {
        "expected": "0.8.0",
        "installed_distribution_version": verl_version,
        "ok": verl_version == "0.8.0",
    }
    failed |= verl_version != "0.8.0"

    for name, path in {
        "student_model": args.student_model,
        "rl_data": args.rl_data,
        "reward_file": str(ROOT / "reasonsql" / "verl_reward.py"),
    }.items():
        exists = Path(path).expanduser().exists()
        report["files"][name] = {"path": path, "exists": exists}
        failed |= not exists

    try:
        import torch

        count = torch.cuda.device_count()
        devices = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 1024**3, 2),
                    "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else None,
                }
            )
        report["gpu"] = {"count": count, "devices": devices}
        if count < args.expected_gpus:
            failed = True
            report["gpu"]["error"] = f"expected at least {args.expected_gpus} visible GPUs"
    except Exception as exc:
        failed = True
        report["gpu"] = {"error": str(exc)}

    if args.check_tokenizers:
        try:
            from transformers import AutoTokenizer

            student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
            teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
            probes = [
                "SELECT department_id, COUNT(*) FROM employees GROUP BY department_id;",
                "```sql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```",
                "Unicode: 北京, café, naïve",
            ]
            same = student.get_vocab() == teacher.get_vocab()
            encoded_same = all(student.encode(probe) == teacher.encode(probe) for probe in probes)
            report["tokenizers"] = {
                "vocab_equal": same,
                "probe_token_ids_equal": encoded_same,
                "student_vocab_size": len(student),
                "teacher_vocab_size": len(teacher),
            }
            failed |= not (same and encoded_same)
        except Exception as exc:
            failed = True
            report["tokenizers"] = {"error": str(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
