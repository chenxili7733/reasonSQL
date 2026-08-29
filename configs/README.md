# Training configuration

- `sft.yaml`: full-parameter SFT for the 7B student using DeepSpeed ZeRO-3.
- `teacher_lora_sft.yaml`: optional LoRA domain adaptation for the 32B teacher.
- `teacher_lora_merge.yaml`: merges the optional teacher adapter.
- `ds_zero3.json`: shared DeepSpeed configuration.

GRPO and native OPD options are passed explicitly by:

- `scripts/10_run_grpo.sh`
- `scripts/11_run_grpo_opd.sh`

Keeping the veRL overrides in executable arrays avoids copying veRL's entire
Hydra configuration tree into this repository. The launch scripts are pinned
to veRL `v0.8.0`.

