from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LaunchConfigTests(unittest.TestCase):
    def test_opd_uses_verl_v080_teacher_config_tree(self) -> None:
        script = (ROOT / "scripts" / "11_run_grpo_opd.sh").read_text(encoding="utf-8")
        required = (
            "distillation.teacher_models.teacher_model.model_path=",
            "distillation.teacher_models.teacher_model.num_replicas=",
            "distillation.teacher_models.teacher_model.inference.dtype=",
            "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=",
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=",
            "distillation.teacher_models.teacher_model.inference.max_model_len=",
            "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=",
            "distillation.teacher_models.teacher_model.inference.max_num_seqs=",
        )
        for key in required:
            self.assertIn(key, script)

        self.assertNotIn('"distillation.teacher_model.', script)

    def test_sft_does_not_require_optional_flash_attention(self) -> None:
        for config_name in ("sft.yaml", "teacher_lora_sft.yaml"):
            config = (ROOT / "configs" / config_name).read_text(encoding="utf-8")
            self.assertIn("flash_attn: auto", config)
            self.assertNotIn("flash_attn: fa2", config)


if __name__ == "__main__":
    unittest.main()
