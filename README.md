# ReasonSQL

基于复合执行奖励、GRPO 与在线策略蒸馏（OPD）的 Text-to-SQL 后训练工程。

训练目标：

```text
Qwen2.5-Coder-7B-Instruct
        │
        ├── BIRD SFT
        │
        ├── GRPO + SQL composite reward
        │
        └── GRPO + native veRL OPD
                  └── Qwen2.5-Coder-32B-Instruct teacher
```

工程默认面向：

- 学生：Qwen2.5-Coder-7B-Instruct，全参数训练；
- 教师：Qwen2.5-Coder-32B-Instruct，冻结推理；
- 学生资源池：4×A100-80GB；
- 教师资源池：4×A100-40GB/80GB；
- RL 框架：veRL `v0.8.0`；
- Rollout/教师推理：vLLM；
- 学生分片：FSDP；
- SFT：LLaMA-Factory + DeepSpeed ZeRO-3。

> 当前仓库的 CPU 代码路径、配置生成和 Shell 语法已经通过本地测试。真正的
> A100 端到端训练必须在 CUDA 集群上执行 Smoke Run 后再扩大数据量。不要在没有
> 实验日志的情况下直接把 `68.7%` 当作仓库已经验证的结果。

## 1. 设计原则

### 1.1 统一 Prompt

SFT、RL 与评估共用 `reasonsql/prompts.py`：

- CREATE TABLE；
- 外键；
- BIRD column description；
- 少量样例行；
- evidence；
- 严格 SQL 输出协议。

所有数据预处理参数会写入 `data/processed/bird_manifest.json`，防止不同实验的
Prompt 悄然发生变化。

### 1.2 安全 SQL 执行

`reasonsql/rewards/sql_executor.py` 提供：

- SQLite immutable/read-only URI；
- SQL 解析和单语句校验；
- SQLite authorizer 二次写保护；
- 墙钟超时；
- 最大结果行数；
- Gold 执行结果 LRU 缓存；
- 重复行保留；
- 数值规范化；
- 非 ORDER BY 查询的结果行无序比较；
- 可选输出列置换匹配。

### 1.3 执行正确性优先的复合奖励

默认奖励：

```text
execution correct                     -> 1.00
execution incorrect:
    syntax                            -> 0.10
    table selection F1                -> 0.10
    column selection F1               -> 0.20
    exact output format               -> 0.05
    dense reward cap                  -> 0.49
```

因此，格式正确但语义错误的 SQL 永远不会超过执行正确的 SQL。

### 1.4 原生 OPD

GRPO+OPD 使用 veRL v0.8.0 原生能力：

1. 学生 vLLM 生成 on-policy token IDs；
2. veRL TeacherServerActor 将完整序列 token IDs 直接发送给教师；
3. 教师返回 sampled-token 或 Top-K 概率；
4. 学生同时计算任务 GRPO Loss 与 token-level distillation loss；
5. 教师冻结，不传递梯度。

仓库不再使用“decode 文本后再让教师重新 tokenize”的旧方案，避免 Token 对齐错误。

## 2. 目录结构

```text
.
├── configs/
│   ├── sft.yaml
│   ├── teacher_lora_sft.yaml
│   ├── teacher_lora_merge.yaml
│   ├── ds_zero3.json
│   └── dataset_info.json
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── eval/
│   ├── eval_bird.py
│   ├── analyze_errors.py
│   ├── eval_series.sh
│   ├── export_bird_submission.py
│   └── compare_results.py
├── patches/
│   └── apply_patch.py
├── reasonsql/
│   ├── data.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── prompts.py
│   ├── verl_reward.py
│   ├── opd/
│   │   └── schedule.py
│   └── rewards/
│       ├── bird_reward.py
│       └── sql_executor.py
├── common/
│   ├── prompt.py
│   └── data.py
├── scripts/
│   ├── 00_download_bird.sh
│   ├── 01_prepare_data.py
│   ├── 02_train_sft.sh
│   ├── 03_teacher_server.sh
│   ├── 04_train_teacher_lora.sh
│   ├── 05_merge_teacher_lora.sh
│   ├── 06_install_verl.sh
│   ├── 10_run_grpo.sh
│   ├── 11_run_grpo_opd.sh
│   ├── 12_run_ablation.sh
│   ├── 20_merge_verl_checkpoint.sh
│   ├── check_environment.py
│   └── smoke_test.py
└── tests/
```

## 3. 安装

### 3.1 CPU 开发环境

```bash
cd /path/to/reasonsql
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

make check
make test
make smoke
```

### 3.2 GPU 环境

先安装与机器驱动匹配的 CUDA PyTorch，再安装项目与 GPU 依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-gpu.txt
bash scripts/06_install_verl.sh
```

`scripts/06_install_verl.sh` 默认将 veRL v0.8.0 安装到
`third_party/verl`。

SFT 配置使用 `flash_attn: auto`：环境中安装了兼容的 FlashAttention 2
时可使用该后端，否则由 LLaMA-Factory/Transformers 回退到当前环境可用的
attention 实现。若希望强制检查 FlashAttention，可执行：

```bash
python3 scripts/check_environment.py --require-flash-attn
```

如果要启用 OPD 系数逐步退火，再执行：

```bash
python3 patches/apply_patch.py \
  --verl-src third_party/verl \
  --dry-run

python3 patches/apply_patch.py \
  --verl-src third_party/verl
```

该补丁只增加动态 `distillation_loss_coef`，不会替换 veRL 的教师服务、
Token 对齐或原生 distillation loss。

### 3.3 Docker

```bash
docker compose build
docker compose run --rm reasonsql make smoke
```

训练时挂载：

- `./data`；
- `./checkpoints`；
- Hugging Face 缓存；
- 全部 NVIDIA GPU。

不同集群的驱动与 CUDA 约束不同，如基础镜像不兼容，请通过
`CUDA_IMAGE` build arg 更换镜像。

## 4. BIRD 数据准备

### 4.1 下载

```bash
bash scripts/00_download_bird.sh
```

默认目录：

```text
data/bird/train/train.json
data/bird/train/train_databases/
data/bird/dev/dev.json
data/bird/dev/dev_databases/
```

BIRD 版本会更新。正式报告结果时必须记录：

- train/dev 文件 SHA-256；
- 数据集发布日期或压缩包版本；
- 是否使用 evidence；
- Prompt schema/value 配置；
- 最终官方 evaluator 版本。

下载源可通过环境变量覆盖：

```bash
BIRD_TRAIN_URL=... \
BIRD_DEV_REPO=... \
BIRD_DEV_FILE=... \
bash scripts/00_download_bird.sh
```

### 4.2 转换

```bash
python3 scripts/01_prepare_data.py \
  --bird-dir data/bird \
  --out-dir data/processed \
  --rl-size 2000 \
  --seed 42 \
  --workers 16
```

输出：

```text
data/processed/bird_sft.jsonl
data/processed/bird_rl_2000.jsonl
data/processed/bird_rl_2000.parquet
data/processed/bird_dev.jsonl
data/processed/bird_manifest.json
```

重新生成时需要显式传入 `--overwrite`，避免无意覆盖已有实验数据。

## 5. SFT

### 5.1 7B 学生全参数 SFT

训练入口现在由 Python 启动器实现，Shell 仅保留兼容性包装。可直接运行：

```bash
python scripts/train_sft.py --gpus 0,1,2,3
```

GRPO 与 GRPO+OPD 分别使用 `scripts/train_grpo.py` 和
`scripts/train_grpo_opd.py`；命令行参数会转换为 veRL 参数，未识别参数仍会
原样传给 veRL。原有 `.sh` 命令仍然兼容。

```bash
STUDENT_GPUS=0,1,2,3 \
bash scripts/02_train_sft.sh
```

默认关键参数：

| 参数 | 值 |
|---|---:|
| cutoff length | 4096 |
| epochs | 2 |
| learning rate | 2e-5 |
| per-device micro batch | 1 |
| gradient accumulation | 8 |
| precision | BF16 |
| optimizer sharding | DeepSpeed ZeRO-3 |
| gradient checkpointing | enabled |

输出：

```text
checkpoints/student_sft_qwen2.5_coder_7b
```

### 5.2 可选：教师 LoRA

直接使用原始 32B 教师前，应先评估其 Text-to-SQL 能力是否强于 SFT 后的学生。
如果教师不够强，可使用 train split 做 LoRA：

```bash
TEACHER_GPUS=0,1,2,3 \
bash scripts/04_train_teacher_lora.sh

bash scripts/05_merge_teacher_lora.sh
```

合并输出：

```text
checkpoints/teacher_qwen2.5_coder_32b_merged
```

不要使用 BIRD dev 对教师做训练。

## 6. 环境检查

八卡 OPD 节点：

```bash
python3 scripts/check_environment.py \
  --expected-gpus 8 \
  --student-model checkpoints/student_sft_qwen2.5_coder_7b \
  --teacher-model Qwen/Qwen2.5-Coder-32B-Instruct \
  --rl-data data/processed/bird_rl_2000.parquet \
  --check-tokenizers
```

`--check-tokenizers` 会检查：

- vocab 是否一致；
- SQL/Unicode probe 的 token IDs 是否一致。

OPD 不允许学生和教师拥有不一致的 Token ID 映射。

## 7. 纯 GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
STUDENT_GPUS_PER_NODE=4 \
STUDENT_MODEL=checkpoints/student_sft_qwen2.5_coder_7b \
RL_DATA=data/processed/bird_rl_2000.parquet \
bash scripts/10_run_grpo.sh
```

常用覆盖：

```bash
TRAIN_BATCH_SIZE=16 \
ROLLOUT_N=4 \
MAX_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=512 \
bash scripts/10_run_grpo.sh \
  trainer.total_training_steps=20 \
  trainer.logger='[console]'
```

第一次运行建议只执行 5～20 steps，确认：

- rollout 不 OOM；
- SQL reward 进程能访问数据库；
- 各组 reward 存在方差；
- checkpoint 可恢复；
- checkpoint 可成功合并。

## 8. GRPO + OPD

默认一台 8 GPU 机器：

```text
4 GPU: student actor + rollout
4 GPU: frozen 32B teacher
```

启动脚本使用 veRL v0.8.0 的
`distillation.teacher_models.teacher_model.inference.*` 配置树。教师的
`TEACHER_TP × TEACHER_REPLICAS` 必须等于教师资源池的 GPU 总数
`TEACHER_GPUS_PER_NODE × TEACHER_NNODES`，脚本会在启动前检查这一条件。

启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
STUDENT_GPUS_PER_NODE=4 \
TEACHER_GPUS_PER_NODE=4 \
TEACHER_TP=4 \
TEACHER_REPLICAS=1 \
TEACHER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
bash scripts/11_run_grpo_opd.sh
```

4×A100-80GB 教师池可尝试两套 TP2 副本：

```bash
TEACHER_TP=2 \
TEACHER_REPLICAS=2 \
bash scripts/11_run_grpo_opd.sh
```

4×A100-40GB 教师池建议：

```bash
TEACHER_TP=4
TEACHER_REPLICAS=1
```

默认 OPD Loss：

```text
loss_mode                 = forward_kl_topk
topk                      = 32
use_policy_gradient       = false
use_task_rewards          = true
teacher coefficient       = 1.0
```

启用退火补丁后，默认：

```text
schedule = cosine
lambda   = 1.0 -> 0.2
```

可覆盖：

```bash
REASONSQL_OPD_SCHEDULE=linear \
REASONSQL_OPD_LAMBDA_START=0.8 \
REASONSQL_OPD_LAMBDA_END=0.1 \
REASONSQL_OPD_WARMUP_STEPS=20 \
bash scripts/11_run_grpo_opd.sh
```

如果不打退火补丁，veRL 使用启动参数中的固定系数，训练仍然可以运行。

## 9. Ablation

```bash
SEED=42 bash scripts/12_run_ablation.sh
SEED=43 bash scripts/12_run_ablation.sh
SEED=44 bash scripts/12_run_ablation.sh
```

至少报告：

1. SFT；
2. SFT + GRPO；
3. SFT + GRPO + OPD；
4. execution-only reward；
5. composite reward；
6. fixed OPD coefficient；
7. annealed OPD coefficient；
8. 三个随机种子的 mean/std。

## 10. Checkpoint 合并

veRL FSDP actor checkpoint：

```bash
bash scripts/20_merge_verl_checkpoint.sh \
  checkpoints/grpo_opd/global_step_100/actor \
  checkpoints/grpo_opd/global_step_100/actor_hf
```

后续评估使用 `actor_hf`。

## 11. 评估

### 11.1 生成并执行评估

vLLM：

```bash
python3 eval/eval_bird.py \
  --model checkpoints/grpo_opd/global_step_100/actor_hf \
  --backend vllm \
  --tensor-parallel-size 1 \
  --bird-dir data/bird \
  --output data/eval/grpo_opd.json \
  --bird-sql-output data/eval/predict_dev.sql
```

Transformers：

```bash
python3 eval/eval_bird.py \
  --model checkpoints/student_sft_qwen2.5_coder_7b \
  --backend transformers \
  --batch-size 8 \
  --bird-dir data/bird \
  --output data/eval/sft.json
```

评估已有预测：

```bash
python3 eval/eval_bird.py \
  --predictions data/eval/predictions.jsonl \
  --bird-dir data/bird \
  --output data/eval/scored.json
```

### 11.2 错误分析

```bash
python3 eval/analyze_errors.py \
  --predictions data/eval/grpo_opd.json \
  --output-json data/eval/grpo_opd_analysis.json
```

### 11.3 官方评测

本仓库的执行比较器用于训练奖励、回归测试和快速本地评估。最终对外报告的
BIRD Dev/Test 数值必须再使用对应数据版本的 BIRD 官方 evaluator。

导出经典 `.sql` 格式：

```bash
python3 eval/export_bird_submission.py \
  --predictions data/eval/grpo_opd_predictions.jsonl \
  --output data/eval/predict_dev.sql
```

### 11.4 多 checkpoint 同口径对比

```bash
bash eval/eval_series.sh \
  checkpoints/student_sft_qwen2.5_coder_7b \
  checkpoints/grpo/global_step_100/actor_hf \
  checkpoints/grpo_opd/global_step_100/actor_hf
```

脚本会对每个 checkpoint 复用完全一致的 prompt、采样和评测参数，并把
预测与评分分别保存到 `data/eval/series/`。

已有多个评分 JSON 时直接汇总：

```bash
python3 eval/compare_results.py \
  data/eval/series/*.json \
  --json data/eval/series_summary.json
```

## 12. 硬件建议

| 阶段 | 4×A100-80GB | 4×A100-40GB |
|---|---|---|
| 7B 全参 SFT | 推荐 | 需要 offload，较紧 |
| 7B 全参 GRPO | 可行，先小 batch | 不推荐 |
| 7B GRPO+OPD 学生池 | 可行 | 不推荐 |
| 32B BF16 教师推理 | 可行 | TP4 可行 |
| 32B 全参数训练 | 不可行 | 不可行 |

完整推荐配置是 8 GPU：

```text
4×A100-80GB student
4×A100-40GB/80GB teacher
```

如果只有 4 张 GPU，优先完成：

1. SFT；
2. 纯 GRPO；
3. 使用 LoRA/量化教师或离线 teacher logprob 的简化实验。

不要把 7B 全参学生和 32B BF16 教师同时硬塞进 4×A100-40GB。

## 13. 常见故障

### Rollout OOM

依次降低：

```text
TRAIN_BATCH_SIZE
ROLLOUT_N
actor_rollout_ref.rollout.gpu_memory_utilization
actor_rollout_ref.actor.ppo_max_token_len_per_gpu
MAX_PROMPT_LENGTH
MAX_RESPONSE_LENGTH
```

并保持 FSDP parameter/optimizer offload。

### 教师吞吐过低

- A100-80GB：尝试 `TEACHER_TP=2, TEACHER_REPLICAS=2`；
- A100-40GB：保持 `TEACHER_TP=4, TEACHER_REPLICAS=1`；
- 降低 `OPD_TOPK`；
- 确保没有返回 full-vocabulary logits；
- 检查 Prompt 是否被无必要扩展。

### Reward 全为零或组内无方差

- 检查 SQL 输出协议；
- 检查 `BIRD_TRAIN_DB_DIR`；
- 检查数据库路径和权限；
- 减小 rollout temperature 之前先确认模型能生成可解析 SQL；
- 检查 composite reward 各子项；
- 增大 rollout group size。

### OPD 没有产生收益

- 先单独评估教师；
- 确认教师强于 SFT 学生；
- 检查 tokenizer/token IDs；
- 降低 distillation coefficient；
- 比较 `forward_kl_topk` 与 sampled-token loss；
- 检查任务 reward 是否仍被启用；
- 不要同时添加过强的 reference KL。

### veRL 补丁锚点失败

退火补丁只支持 veRL v0.8.0。先执行：

```bash
git -C third_party/verl checkout v0.8.0
python3 patches/apply_patch.py --verl-src third_party/verl --dry-run
```

如果使用其他 veRL 版本，不要强制应用补丁；先对照该版本的
`ray_trainer.py` 与 `distillation/losses.py` 手工审核。

## 14. 测试

```bash
make check
make test
make smoke
```

测试覆盖：

- SQL 单语句与只读限制；
- 字符串内分号；
- 重复结果行；
- ORDER BY 语义；
- 输出列置换；
- 数字规范化；
- Prompt/schema 构造；
- 格式提取；
- 复合奖励；
- BIRD 数据发现与转换；
- 预测对齐；
- 执行评估；
- OPD 系数 schedule；
- veRL 补丁幂等性。

兼容导入：

- `common.prompt` → `reasonsql.prompts`；
- `common.data` → `reasonsql.data`。

## 15. 结果复现清单

对外声称准确率前，请保存：

- Git commit；
- veRL tag/commit；
- Docker image digest；
- BIRD manifest；
- student/teacher model revision；
- tokenizer hash；
- 所有 Hydra overrides；
- W&B run；
- 随机种子；
- GPU 类型与数量；
- 推理 temperature、candidate 数、是否 self-consistency；
- 本地评估结果；
- 官方 evaluator 输出。

只有这些信息齐全，SFT、GRPO 与 GRPO+OPD 的增益才具有可复现性。
