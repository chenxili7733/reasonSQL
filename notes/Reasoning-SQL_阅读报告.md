# Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL —— 阅读报告

> **论文链接**: https://arxiv.org/abs/2503.23157
> **论文题目**: Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL
> **作者**: Mohammadreza Pourreza(Google Cloud)、Shayan Talaei(Stanford)、Ruoxi Sun(Google DeepMind)、Xingchen Wan、Hailong Li、Azalia Mirhoseini、Amin Saberi、Sercan Ö. Arik(Google Cloud/Stanford)
> **状态**: arXiv:2503.23157v2(2025-04),Preprint,Under review

---

## 一、主要解决的问题

Text-to-SQL 涉及多个推理密集的子任务:自然语言理解、数据库 schema 理解、精确 SQL 查询构建。论文指出现有方法存在以下问题:

1. **推理能力受限**:基于 SFT 或 few-shot 提示的方法在模糊、多步查询等需要深度推理的场景下表现不足,其性能被底层 LLM 的推理能力所限制。
2. **奖励稀疏(reward sparsity)**:Text-to-SQL 最直观的奖励是执行准确率(execution accuracy),但它是**二值且稀疏**的——许多候选 SQL 虽包含部分正确的逻辑却得不到任何反馈,难以区分"接近正确"与"完全错误",严重阻碍策略优化。
3. **依赖人工推理设计**:现有方法往往使用手工设计的推理路径(带归纳偏置),效果受限;且此前缺乏显式用 RL 自动优化 LLM 推理过程以提升 Text-to-SQL 的方法。

论文借鉴 DeepSeek-R1、OpenAI o1 等"奖励驱动自探索"的成功经验,提出**用 RL 自动优化 Text-to-SQL 推理过程**,并通过专门设计的**部分奖励(partial rewards)** 缓解奖励稀疏问题。

---

## 二、创新点

1. **首个 RL 驱动的 Text-to-SQL 推理优化框架(Reasoning-SQL)**:自动优化 LLM 的推理过程,而非人工设计推理路径,使模型逐步涌现出结构化、有效的 SQL 推理风格。
2. **专为 Text-to-SQL 设计的部分奖励套件(Partial Rewards Suite)**:提出 **LLM-as-a-Judge(AI 反馈)、语法检查、Schema Linking、N-Gram 相似度** 四类部分奖励 + 格式奖励 + 执行准确率,加权组合形成稠密、信息量充足的反馈,系统解决奖励稀疏问题。
3. **涌现式推理优于人工设计**:RL 训练不强制推理格式,模型自然涌现出结构化推理;将该推理风格作为 few-shot 提示迁移到其他基础模型,效果优于 ACT-SQL、Divide-and-Conquer、Query Plan 等人工设计策略。
4. **小模型 + 低成本的 SOTA 表现**:14B 开源模型在 BIRD 上超越 o3-mini(+4%)、Gemini-1.5-Pro-002(+3%);接入标准 pipeline 后 BIRD test 达 **72.78%**,推理成本比 Gemini-1.5-Pro 低 **93%**。

---

## 三、使用的技术和算法

### 3.1 强化学习协议:GRPO
- 采用 **GRPO(Group Relative Policy Optimization)**,相比 PPO 无需独立价值模型、内存开销更小,与 DeepSeek-R1 同源。
- 对每个输入(自然语言问题 q + 数据库 schema)由策略模型 πθ 生成 **G 个候选 SQL**,按组内相对表现计算优势 A_i,再通过裁剪 + KL 散度正则更新策略:

```
J_GRPO(θ) = E[ 1/G Σ_i min( r_i^ratio·A_i, clip(r_i^ratio, 1±ε)·A_i ) - β·D_KL(πθ‖πref) ]
```

- 超参:lr=1e-6(常数调度,0.1% warmup),effective batch size=32,3 epochs,group size G=6,8×NVIDIA H100。

### 3.2 训练数据
- 主训练集:BIRD 训练集 9,428 对(70 个数据库,覆盖航空、电影、销售等领域);用 Gemini-2.0-flash 与 GPT-4o 双重标记过滤噪声/歧义样本后保留 **8,026 对**。
- **Schema linking 前置处理**:用 Gemini-1.5-Pro 过滤数据库 schema(解耦 schema linking 与 SQL 生成);训练时与 ground-truth schema 合并以保证信息完整,推理时仅用过滤后的 schema。

### 3.3 基础模型与对比基线
- 基础模型:**Qwen2.5-Coder 3B / 7B / 14B**。
- SFT 基线:
  - 直接 SFT(direct SQL prediction,极大似然学习 gold SQL,lr=1e-5、cosine 调度);
  - **STaR-SFT**:基于 Self-Taught Reasoner 框架,用 CHASE-SQL 的 Divide-and-Conquer 提示 + Gemini-1.5-Pro-002 生成 CoT 轨迹,正确样本入训练集、错误样本给 ground-truth 提示。

### 3.4 集成到 CHASE-SQL 多阶段 Pipeline
- 将训练好的模型接入 CHASE-SQL 流水线(schema linking → SQL 生成 → 自纠正 → 查询选择),替换其中的 Gemini-1.5-Pro;
- 用 Qwen2.5-Coder-14B 训练二元成对验证器(pairwise verifier)做候选查询选择。

---

## 四、Reward 设计

最终复合奖励为各评估函数的加权和:

```
r = Σ_i w_i · r_i    (权重: w_exec=3, w_judge=2, 其余=1)
```

权重精心设计,保证**任何错误 SQL 的总奖励都不会超过正确 SQL**。

### 1. 执行准确率奖励(Execution Accuracy Reward, RLEF)
将候选 SQL 与 ground-truth SQL 分别在数据库上执行并比较结果,完全匹配得满分:

```
r_exec = 1[exec(o_i) == exec(gold)]
```

与下游目标完全对齐,但二值、稀疏——多数部分正确的查询得 0 分。

### 2. LLM-as-a-Judge 奖励(AI 反馈, RLAIF)
**仅对执行准确率为 0 的查询**调用 LLM 裁判(使用带专门评分准则的提示,基于逻辑一致性、结构相似性、语义正确性等评估候选与 gold 的接近程度),区分不同程度的错误答案,提供细微的部分奖励;执行正确的查询直接得满分 1:

```
r_judge = 1                (若 r_exec = 1)
r_judge = LLM-judge(o_i, gold) ∈ [0,1]   (若 r_exec = 0)
```

### 3. 语法检查奖励(Syntax Check Reward)
候选 SQL 语法有效且能无错执行即给正分(不论结果是否正确),引导模型优先生成语法规范、可执行的查询:

```
r_syntax = 1[syntactically valid and executable]
```

### 4. Schema Linking 奖励
计算候选查询与 gold 查询所用 schema 元素(表/列)集合的 **Jaccard 相似度**,直接针对"自然语言实体 → 数据库 schema"映射这一关键环节:

```
r_schema = Jaccard(schema_items(o_i), schema_items(gold))
```

### 5. N-Gram 相似度奖励
利用 SQL 的层次化结构,计算候选与 gold SQL 的 **n-gram(N=2,bigram)Jaccard 相似度**,促进词法与句法结构上的对齐:

```
r_ngram = Jaccard(ngrams(o_i), ngrams(gold))
```

### 6. 格式奖励(Format Reward)
鼓励输出符合预定结构(如 `<reasoning>` 与 `<answer>` 标签),符合格式即获奖励加成;该结构同时触发模型的零样本 CoT 推理,并随训练逐步优化:

```
r_format = 1[o_i matches required format]
```

### 完整伪代码逻辑

```
对每个候选 o_i:
  r_exec   ← 执行结果是否与 gold 一致
  r_judge  ← (r_exec=1 时为 1) 否则 LLM 裁判打分 ∈ [0,1]
  r_syntax ← 是否语法有效可执行
  r_schema ← schema 元素 Jaccard 相似度
  r_ngram  ← n-gram Jaccard 相似度
  r_format ← 是否符合输出格式
  r_i      ← 3·r_exec + 2·r_judge + r_syntax + r_schema + r_ngram + r_format
  A_i      ← r_i - 组内平均奖励
```

---

## 五、实验效果总结

评测指标为执行准确率(EX),基准为 BIRD(dev/test)、Spider、Spider-DK、Spider-Syn。

### 5.1 RQ1 训练目标消融(Qwen2.5-Coder-7B,BIRD-Dev)

| 后训练方法 | Thinking | EX(%) | FS EX(%) |
| --- | --- | --- | --- |
| 无后训练(Qwen2.5-Coder-7B) | ✗ | 58.73 | 64.01 |
| SFT(直接 SQL) | ✗ | 61.53 | 69.23 |
| STaR-SFT(带推理) | ✓ | 62.84 | 68.12 |
| GRPO no-reasoning(全部奖励,无推理) | ✗ | 62.25 | 68.12 |
| GRPO(exe) 仅执行奖励 | ✓ | 62.32 | 68.90 |
| GRPO(exe, syntax) | ✓ | 62.84 | 68.97 |
| GRPO(exe, syntax, schema) | ✓ | 63.10 | 69.23 |
| GRPO(exe, syntax, schema, ngram) | ✓ | 63.75 | 69.94 |
| GRPO(Only RLAIF) | ✓ | 63.75 | 69.16 |
| **GRPO(all rewards)** | ✓ | **64.01** | **70.66** |

- 每个部分奖励都有**针对性效果**:加语法奖励使 Syntax Check +1.37;加 schema 奖励使 Schema Items Jaccard +0.78;加 n-gram 奖励使 N-gram Jaccard +3.92,并同时带动整体 EX 提升。
- 全部奖励组合相比基线提升 **6.77%**,而传统 SFT 仅提升 4.11%。
- **带推理 vs 不带推理 GRPO 相差 2%**,证明逐步推理对 SQL 生成的重要性。

### 5.2 RQ2 基线对比(BIRD-Dev,全部奖励模型)

| 模型 | Thinking | EX(%) | FS EX(%) |
| --- | --- | --- | --- |
| GPT-4o-mini | ✗ | 60.82 | 66.29 |
| o3-mini | ✓ | 61.34 | 69.88 |
| Gemini-1.5-Pro-002 | ✗ | 62.25 | 67.47 |
| Gemini-2.0-Flash | ✗ | 66.49 | 71.90 |
| Qwen2.5-Coder-3B(all rewards) | ✓ | 58.67 | 65.31 |
| Qwen2.5-Coder-7B(all rewards) | ✓ | 64.01 | 70.66 |
| **Qwen2.5-Coder-14B(all rewards)** | ✓ | **65.31** | **72.03** |

- 三个规模(3B/7B/14B)上,RL 训练一致优于各自 SFT 版本,差距随模型规模增大而扩大(FS EX 提升 +1.69 / +2.54 / +3.20)。
- **14B 模型超过 o3-mini 4%、超过 Gemini-1.5-Pro-002 3%**,与 Gemini-2.0-Flash 持平,且可部署在单 GPU 上。

### 5.3 接入 CHASE-SQL Pipeline(BIRD)

| 方法 | 模型 | Dev EX(%) | Test EX(%) |
| --- | --- | --- | --- |
| CHASE-SQL | Gemini-1.5-Pro | 74.46 | 74.79 |
| XiYan-SQL | 模型集成 | 73.34 | 75.63 |
| OpenSearch-SQL v2 | GPT-4o | 69.3 | 72.28 |
| **Reasoning-SQL(Ours)** | Qwen2.5-Coder-14B(all rewards) | **72.29** | **72.78** |
| Reasoning-SQL(Ours) | Qwen2.5-Coder-7B(all rewards) | 68.05 | - |

- 达到开源小模型的 SOTA,超越多数 GPT-4 系流水线;与 CHASE-SQL 持平但**成本低 93%**。

### 5.4 RQ3 涌现式推理 vs 人工设计策略(BIRD-Dev)

| 提示策略 | Qwen-14B-code-instruct | CodeGemma-7B-it |
| --- | --- | --- |
| ACT-SQL | 62.84 | 48.37 |
| Divide-and-Conquer | 60.75 | 48.82 |
| Query Plan | 61.92 | 48.56 |
| **Reasoning-SQL 涌现式推理** | **64.21** | **50.19** |

- RL 训练自然涌现的推理风格,作为 few-shot 提示迁移到其他基础模型时,一致优于三种人工设计的推理策略,验证了其有效性与跨模型可迁移性。
- 训练动态:第 200 步时模型自发采用结构化推理并正确识别 join 条件(不强制任何推理格式)。

### 5.5 RQ4 泛化分析(Spider / Spider-DK / Spider-Syn)

| 模型 | Spider EX | Spider-DK EX | Spider-Syn EX |
| --- | --- | --- | --- |
| o3-mini | 78.82 | 71.77 | 73.01 |
| Gemini-2.0-Flash | 81.23 | 74.15 | 73.59 |
| Gemini-1.5-Pro-002 | 80.46 | 67.66 | 71.56 |
| Qwen2.5-Coder-7B SFT | 68.08 | 62.80 | 51.93 |
| Qwen2.5-Coder-7B(all rewards) | 78.72 | 73.27 | 69.34 |
| Qwen2.5-Coder-14B SFT | 75.04 | 69.15 | 61.02 |
| **Qwen2.5-Coder-14B(all rewards)** | **81.43** | **73.03** | **72.63** |

- RL 模型在三个 OOD 基准上全面超越 o3-mini、Gemini-1.5-Pro 等 SOTA 模型。
- 关键观察:**SFT 模型在 OOD 上甚至低于基础模型**(记忆化,SFT memorizes),而 **RL 模型稳定提升基础模型**(泛化,RL generalizes)。

### 5.6 补充分析(附录)
- **测试时计算(Test-time Compute)**:Pass@K 与 Average@K 对比显示,GRPO 模型候选池**平均质量稳定更高**(温度 0.5 下仅从 65.5% 降到 64.66%,SFT 从 63.75% 降到 60.82%),SFT 多样性略高但平均质量低。
- **难度分析(BIRD-Dev)**:7B(all)在 Simple/Moderate/Challenging 上分别 71.56 / 53.66 / 46.20,均优于 STaR-SFT 且**推理字符更少**(1180/1339/1421 vs 1550/2012/2346);推理长度随问题难度与所需 schema 连接数增长,呈正相关。
- **成本分析**:Gemini-1.5-Pro 约 $1.25/1M 输入 token、$5.00/1M 输出 token;本模型约 $0.08/1M 输入、$0.18/1M 输出,推理成本降低 93%,性能仅低约 1%。
- **训练动态**:前 200 步中准确率持续上升,平均推理字符数下降——模型从零样本 CoT 的冗长推理逐步收敛为简洁、一致的格式。

---

## 六、局限性与展望

- **依赖外部模型做 schema linking**:训练与推理均用 Gemini-1.5-Pro 过滤 schema,增加了对外部闭源服务的依赖。
- **论文为 preprint(under review)**:部分细节(如 LLM 裁判的具体提示)在附录中,完整评测有待正式发表。
- 未来方向:进一步探索更丰富的部分奖励组合、更大规模模型、以及将 RL 推理能力迁移到其他结构化输出任务。

## 七、一句话总结

Reasoning-SQL 是首个用 GRPO 自动优化 Text-to-SQL 推理过程的 RL 框架,通过"执行准确率 + LLM 裁判 + 语法检查 + schema linking + n-gram 相似度 + 格式"的加权部分奖励套件解决奖励稀疏问题,使 14B 开源模型在 BIRD 上超越 o3-mini/Gemini-1.5-Pro 等更大闭源模型(72.78%),并实现 93% 的成本降低,且验证了"RL 泛化、SFT 记忆化"与涌现式推理优于人工设计策略等重要结论。
