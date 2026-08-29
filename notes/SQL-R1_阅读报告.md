# SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning —— 阅读报告

> **论文链接**: https://arxiv.org/abs/2504.08600
> **论文题目**: SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning
> **作者**: Peixian Ma, Xialie Zhuang, Chengjin Xu, Xuhui Jiang, Ran Chen, Jian Guo(IDEA Research / 国际数字经济研究院等)
> **发表**: NeurIPS 2025(39th Conference on Neural Information Processing Systems)
> **代码**: https://github.com/IDEA-FinAI/SQL-R1

---

## 一、主要解决的问题

自然语言转 SQL(NL2SQL / Text2SQL)旨在把自然语言问题转换为结构化 SQL 语句,降低用户访问数据库的门槛。论文指出当前 NL2SQL 方法存在以下关键问题:

1. **复杂场景推理能力不足**:在多表连接(Multi-table Joins)、嵌套查询(Nested Queries)等复杂数据库场景下,SFT 训练的模型经常产生与用户意图不一致的 SQL,推理性能受限。
2. **SFT 的固有局限**:现有方法主要依赖监督微调(SFT),高度依赖数据库 schema 结构和训练数据规模,在新数据库环境(如金融、医疗)中领域适应与泛化能力不稳定。
3. **可解释性差**:缺乏对 NL2SQL 推理逻辑的解释,限制了模型在高风险领域(金融、医疗)的应用。

论文围绕三个核心问题展开:
- **Q1**:能否针对 NL2SQL 任务设计专门的强化学习(RL)算法,训练出具备显式推理能力的 NL2SQL 推理模型?
- **Q2**:对于基于 RL 的 NL2SQL 推理模型,是否需要特定的冷启动(cold start)方式(SFT)?
- **Q3**:能否构建可持续的数据工程,用合成数据支撑鲁棒、高效的 NL2SQL 推理模型训练?

---

## 二、创新点

1. **显式 NL2SQL 推理模型(SQL-R1)**:首次系统性将强化学习引入 NL2SQL 推理训练,模型在输出 SQL 的同时生成**显式、可读的推理过程**(`<think>...</think>` 与 `<answer>...</answer>` 格式),提升了可解释性。仅用**约 5K 条复杂合成数据**做 RL 增强训练即可达到领先精度(Spider-Test 88.6%、BIRD 66.6%~67.1%)。
2. **面向 NL2SQL 的专门奖励函数**:设计了包含"格式奖励 + 执行奖励 + 结果奖励 + 长度奖励"四部分的渐进式复合奖励机制,为推理训练提供分层、细粒度的反馈。
3. **冷启动训练策略探索**:系统研究了 SFT 冷启动对 RL 训练的影响(数据来源、数据量、是否带推理指令),并得出"冷启动并非普适必需,其效果取决于数据来源与规模"的重要结论。
4. **合成数据工程**:使用首个百万级合成 NL2SQL 数据集 SynSQL-2.5M,验证了"合成数据 + RL"的协同增益,并探索了 RL 训练阶段的数据工程(仅保留 Complex 难度样本、不带 CoT、不带数据库代表值)。

---

## 三、使用的技术和算法

### 3.1 数据工程
- **数据来源**:SynSQL-2.5M,首个百万级合成 NL2SQL 数据集(约 250 万样本),覆盖 16,000+ 个不同领域合成数据库,难度从单表查询到多表连接、函数、CTE 不等。每个样本为四元组(数据库、自然语言问题、SQL、CoT 解题过程)。
- **SFT 数据(SynSQL-200K)**:从 SynSQL-2.5M 采样 20 万条,四个难度等级各 5 万条,样本格式为 `<think>推理</think><answer>SQL</answer>`。
- **RL 数据(SynSQL-Complex-5K)**:随机采样 5K 条 **Complex 难度**的 NL-SQL 对(不包含原始 CoT),旨在针对性提升复杂 SQL 生成能力。

### 3.2 训练流程
- **基础模型**:Qwen2.5-Coder 系列(3B / 7B / 14B)。
- **Stage 1 SFT 冷启动**:在 SynSQL-200K 上全参数微调,使其具备指令跟随与推理生成能力;对比了两种冷启动策略:仅 SQL 生成的原始指令(类似 OmniSQL-7B)与带推理指令的完整微调。超参:lr=5e-5,batch size=1。
- **Stage 2 强化学习**:使用 **GRPO(Group Relative Policy Optimization)**,无需价值模型、内存开销小、奖励目标定义清晰。对每个问题采样 G=8 个 SQL 候选,通过组内相对优势计算奖励并更新策略;lr=3e-7,max response length=2048,使用 KL 散度正则(β)约束偏离参考策略。

GRPO 目标函数:

```
J_GRPO(θ) = E[ 1/G Σ_i min( r_i^ratio·A_i, clip(r_i^ratio, 1±ε)·A_i ) - β·D_KL(πθ‖πref) ]
```

其中 `r_i^ratio = πθ(o_i|v)/πold(o_i|v)` 为重要性采样比率,`A_i` 为组内相对优势。

### 3.3 推理/候选选择
- 自一致性投票(Self-Consistency):生成 8 个候选(温度 0.8),执行所有 SQL,按得分选出最优 SQL 作为最终答案;候选数取 8 是多样性与计算开销的最佳平衡点。

---

## 四、Reward 设计

SQL-R1 采用四类奖励组成的渐进式复合反馈机制:

### 1. 格式奖励(Format Reward, S_f)
要求模型将推理过程放在 `<think>...</think>`、最终答案放在 `<answer>...</answer>` 中,且 SQL 必须包裹在 ` ```sql...``` ` 中:

```
S_f =  1  (格式正确)
S_f = -1  (格式错误)
```

### 2. 执行奖励(Execution Reward, S_e)
评估 SQL 的语法可执行性,防止模型输出杂乱不可执行的内容;SQL 不可执行时不发放后续所有奖励,并设置执行时间限制避免生成过于复杂的 SQL:

```
S_e =  2  (SQL 可执行)
S_e =  0  (格式错误)
S_e = -2  (SQL 不可执行)
```

### 3. 结果奖励(Result Reward, S_r)
基于执行准确率(Execution Accuracy, EX)判断查询结果是否正确,是奖励机制中最重要的组成部分,结果错误时施加严厉惩罚以引导后续推理:

```
S_r =  3  (查询结果正确)
S_r =  0  (格式错误或不可执行)
S_r = -3  (查询结果错误)
```

### 4. 长度奖励(Length Reward, S_l)
激励模型生成更完整的推理过程,同时抑制多余解释。分为两部分:
- 第一半:回答总长度(lenthink + lenanswer)与最大响应长度的比例关系;
- 第二半:SQL 长度占整个 answer 的比例(抑制冗长废话);
- 超过最大长度时给予惩罚反馈。

```
S_l = 0.5×S_tl + S_al   (结果正确且 len_response <= MAX_LENGTH)
S_l = 0.5 + S_al         (结果正确且 len_response >  MAX_LENGTH)
S_l = 0                   (其他情况)
```

其中 `S_tl = (len_think + len_answer)/MAX_LENGTH`,`S_al = len_sql/len_answer`。

---

## 五、实验效果总结

评测指标为执行准确率(EX),基准为 Spider 与 BIRD(亦包含 Spider-DK / Spider-Syn / Spider-Realistic / Spider2.0 等)。

### 5.1 主基准结果

| 模型 | Spider(Dev) | Spider(Test) | BIRD(Dev) |
| --- | --- | --- | --- |
| SQL-R1 + Qwen2.5-Coder-3B | 78.1 | 78.9 | 54.6 |
| **SQL-R1 + Qwen2.5-Coder-7B** | **87.6** | **88.7** | **66.6** |
| SQL-R1 + Qwen2.5-Coder-14B | 86.7 | 88.1 | **67.1** |

- 摘要/结论中报告的最佳成绩为 **Spider-Test 88.6% 与 BIRD 67.1%**(论文各版本略有差异)。
- 与闭源模型对比:优于 GPT-4 系的 DAIL-SQL(86.2%)、MAC-SQL(82.8%)等,接近/优于 GPT-4o 系方法(MCTS-SQL 86.6%、OpenSearch-SQL 87.1%);7B 模型即可达到超过更大规模模型的精度,性价比突出。
- 优于同期开源 RL 推理方案:SQL-R1 在 Spider-Dev 与 BIRD-Dev 上超过 DeepRetrieval、Reasoning-SQL。

### 5.2 难度与规模分析(BIRD-Dev)

| 模型 | Simple | Moderate | Challenging | All |
| --- | --- | --- | --- | --- |
| SQL-R1-7B | 72.1 | 60.8 | 51.0 | 66.6 |
| SQL-R1-14B | 72.4 | 59.7 | **56.5**(+5.5) | 67.1 |

- 14B 相对 7B 的提升集中在 Challenging 难度(+5.5 个百分点),说明**模型容量主要解决复杂查询(多表连接、嵌套子查询、复杂聚合)的泛化难题**。

### 5.3 鲁棒性/泛化(Spider-DK / Spider-Syn / Spider-Realistic)

| 模型 | Spider-DK | Spider-Syn | Spider-Realistic |
| --- | --- | --- | --- |
| SQL-R1-7B | 78.1 | 76.7 | 83.3 |
| SQL-R1-14B | 79.3 | 78.5 | 86.2 |

均超过 SENSE、ROUTE、SQL-o1、OmniSQL、SQL-PaLM、PURPLE 等对比方法,验证了模型的泛化与鲁棒性。

### 5.4 冷启动(SFT)分析

| 模型 | SFT 数据 | 推理指令 | Spider(Dev) | Spider(Test) | BIRD(Dev) |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-7B(基线) | - | - | 77.4 | 79.4 | 58.2 |
| Qwen2.5-Coder-7B | SynSQL-200K | ✓ | 82.7 | 83.3 | 57.0 |
| SQL-R1-7B(无冷启动) | - | - | 84.5 | 86.1 | **63.1** |
| SQL-R1-7B | SynSQL-200K | ✓ | 84.7 | 86.4 | 59.2 |
| SQL-R1 + OmniSQL-7B | SynSQL-2.5M | - | 87.6 | 88.7 | 66.6 |
| SQL-R1-14B(无冷启动) | - | - | 86.7 | 88.1 | **67.1** |

**结论(Q2)**:SFT 冷启动并非普适必需——SynSQL-200K 冷启动在 BIRD 上反而使 RL 效果下降(63.1→59.2),而基于 SynSQL-2.5M 的 OmniSQL 冷启动显著更优;冷启动的效果取决于**数据来源与数据量**。

### 5.5 Reward 组件消融(BIRD-Dev)

| Reward 组合 | 准确率 |
| --- | --- |
| S_f + S_e + S_r + S_l(完整) | 63.1 |
| - 去掉格式奖励 S_f | 60.4(↓2.7) |
| - 去掉执行奖励 S_e | 60.7(↓2.4) |
| - 去掉结果奖励 S_r | 62.4(↓0.7) |
| - 去掉长度奖励 S_l | 61.0(↓2.1) |

任何单一奖励的缺失都会导致性能下降,验证了复合奖励机制的有效性。

### 5.6 其他补充实验
- **Spider2.0(SQLite 子集)**:SQL-R1-7B 达 20.0%,显著超过 GPT-4o(15.6%)、DeepSeek-V3(15.6%)、OmniSQL-7B(10.4%)。
- **效率对比(BIRD-Dev)**:SQL-R1 贪婪解码 0.4s/3.1K tokens(63.7%),自一致性(8 候选)1.1s/23.1K tokens(66.6%);相比 CHESS(251s/320.8K tokens、61.5%)在精度、延迟、token 开销上全面占优。
- **小模型收益更大**:3B/7B 经 RL 提升显著,14B 提升相对有限,说明 RL 对小模型性价比更高。
- **数据库代表值消融**:RL 训练加入数据库值(B.4)无显著提升且拉长输入,故最终训练阶段不加代表值,以增强模型的 schema linking 探索能力。
- **候选数消融**:8 个候选 + 温度 0.8 为多样性/开销最优配置。

---

## 六、局限性与展望

- **数据库方言受限**:目前仅在 SQLite 方言上训练与评测,尚需扩展到 Snowflake、DuckDB 等真实场景方言。
- **基础模型覆盖有限**:实验集中在 Qwen2.5-Coder 系列,未覆盖 Llama4 等更新模型。
- 未来方向:提升模型可解释性、扩展多表连接能力、探索更大规模合成数据生成以支撑可扩展训练。

## 七、一句话总结

SQL-R1 首次将 GRPO 强化学习与专门设计的四层奖励(格式/执行/结果/长度)系统性应用于 NL2SQL 推理训练,以极少量(5K)复杂合成数据实现接近/超过闭源大模型的表现,并系统回答了 RL 冷启动方式与合成数据工程这两个关键问题。
