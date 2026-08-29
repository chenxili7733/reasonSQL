# ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback —— 阅读报告

> **论文链接**: https://arxiv.org/abs/2503.19988
> **论文题目**: ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback
> **作者**: Bohan Zhai, Canwen Xu, Yuxiong He, Zhewei Yao(Snowflake Inc.)
> **代码**: 训练框架基于 Arctic Training;模型发布于 Huggingface(Qwen-2.5-coder-Arctic-ExCoT-32B、Llama-3.1-Arctic-ExCoT-70B)

---

## 一、主要解决的问题

论文研究如何让大语言模型(LLM)在 Text-to-SQL 任务上有效利用 Chain-of-Thought(CoT)推理。通过调查提出三个关键观察:

1. **Zero-shot CoT 几乎无增益**:不做专门训练、仅靠提示词让模型"一步步思考",在 Text-to-SQL 上相比基线没有任何提升——非结构化的推理步骤无法处理 schema 相关依赖与语法约束。
2. **不带 CoT 直接做 DPO 收效甚微**:直接对 SQL 输出做直接偏好优化(Direct Preference Optimization,DPO)而不引入中间推理步骤,性能提升有限——缺乏中间推理路径使偏好信号无法定位逻辑错误。
3. **CoT + 执行反馈效果显著**:一旦引入 CoT 推理,即使只用最简单的"执行正确性"反馈信号,配合 DPO 也能大幅提升模型推理能力——中间推理步骤暴露了逻辑/结构错误,让 DPO 能够更有效地纠正。

此外,中间逻辑错误(如 join 错配、聚合错误)往往不可逆地传播到最终 SQL 输出,这是 Text-to-SQL 推理的核心难点。

---

## 二、创新点

1. **ExCoT(Execution-Guided Chain-of-Thought Direct Preference Optimization)框架**:首次把 **CoT 推理 + off-policy/on-policy 迭代式 DPO** 结合,统一到"仅靠执行准确率(execution accuracy)作为反馈信号"的训练循环中。
2. **彻底去掉奖励模型与人工偏好标注**:不依赖专门的 reward model,也不需要人工标注的偏好对——用"SQL 在本地 SQLite 执行结果与 ground truth 是否一致"作为唯一的正确/错误标签,实现完全自指导的迭代改进。
3. **三种 CoT 生成范式对比**:No-CoT(直接 SQL)、Simple-CoT(简短推理)、Complex-CoT(分而治之,Divide-and-Conquer:先分解子问题+伪 SQL,再合并求解,最后优化),系统论证了结构化复杂推理对对齐的必要性。
4. **off-policy 与 on-policy 结合的双阶段对齐**:先用 GPT-4o 生成的高质量数据做 off-policy DPO(多样性,最大编辑距离采样),再用模型自身生成的数据做多轮 on-policy 迭代 DPO(近邻/最小编辑距离采样),逐步收敛到模型自身分布。

---

## 三、使用的技术和算法

### 3.1 CoT 数据生成
- 用 **GPT-4o** 作为合成基座模型,从 BIRD(9.2k)与 Spider(8.6k)训练集随机抽取示例,每提示附带 3 个 exemplar,采用 few-shot 提示生成候选解;**明确排除 ground-truth SQL**,鼓励多样且有创造性的推理。
- 每个问题生成最多 32 个候选解,覆盖 No-CoT / Simple-CoT / Complex-CoT(分而治之)三种推理风格。
- **Complex-CoT 提示**结构:主问题分析 → 伪 SQL 模板 → 子问题拆分(每个带分析与伪 SQL)→ 最终 SQL 组装 → 简化/优化阶段。

### 3.2 执行验证(Execution-Based Feedback)
- 把相关数据库 schema 装载进本地 **SQLite**;从模型输出中解析最后一个代码块作为最终 SQL(约定),执行并与 ground-truth 查询结果比对。
- 结果一致即判为正确(win),否则错误(lose)——**功能等价**判定,不要求语法一致。

### 3.3 分阶段训练流程
- **Stage 1(SFT)**:用执行验证过的正确 CoT 解做监督微调(LLaMA-3.1 70B 与 Qwen-2.5-Coder 32B 先在大规模自然语言数据集上预对齐)。得到 5.6k(BIRD)+ 6.1k(Spider)条验证过的样本。超参:lr=1e-5,2 epochs,64×H100,全局 batch 128,总 448 GPU 小时。
- **Stage 2(Off-Policy DPO)**:把 GPT-4o 生成解分为正确(win)/错误(lose)池,按**最大编辑距离**(furthest)挑选差异最大的配对做 off-policy DPO,强化正确解、抑制错误解。
- **Stage 3(On-Policy 迭代 DPO)**:用当前模型自己生成新候选解 → 执行验证 → 按**最小编辑距离**(nearest)构造 win/lose 对 → DPO 更新 → 重复多轮(默认 1 轮 off-policy + 2 轮 on-policy)。每轮 DPO:lr=1e-6,2 epochs,全局 batch 64,共 160 GPU 小时。

### 3.4 采样策略(课程式 curriculum)
- Off-policy 用 **furthest**(差异最大,放大模型最明显的错误);
- On-policy 用 **nearest**(差异最小,纠正模型自身分布上的"擦肩而过"式细微错误,快速迭代改进)。

---

## 四、Reward 设计

ExCoT **不训练任何显式奖励模型**,采用最简形式:

### 4.1 二元执行反馈(Execution Correctness)

```
R(SQL) = 1  (生成 SQL 与 ground-truth 查询结果一致)
R(SQL) = 0  (结果不一致或不可执行)
```

该信号是唯一的偏好来源:
- 正确解进入 win 池(用于 SFT 与 DPO 正样本);
- 错误解进入 lose 池(用于 DPO 负样本);
- 丢弃不可解析的中间输出。

### 4.2 DPO 的隐式奖励
DPO 把"执行正确性"转化为隐式奖励函数——最大化正确解与错误解的似然差,无需显式 reward model 或人工排序。论文核心论点:**CoT 是让这个二元信号充分发挥作用的关键**——中间推理步骤提供了"错误发生在哪一步"的内部线索,使执行反馈能够精确指导模型自我纠错。

---

## 五、实验效果总结

评测指标:执行准确率(EX%)与 SQL 有效性(Valid%),基准为 BIRD(dev/test)与 Spider(test)。

### 5.1 主结果

| 模型 / 阶段 | BIRD(Dev) EX | Spider(Test) EX |
| --- | --- | --- |
| LLaMA-3.1 70B(基线,No-CoT) | 57.37 | 78.81 |
| + SFT(GPT-4o 数据,Complex-CoT) | 58.14 | 81.42 |
| + Off-Policy DPO | 66.30 | 82.49 |
| **+ On-Policy 迭代 DPO(ExCoT 70B)** | **68.51** | **86.59** |
| Qwen-2.5-Coder 32B(ExCoT 32B) | 68.25 | 85.14 |
| GPT-4o | 54.69 | 76.53 |
| Claude 3.5-Sonnet | 50.13 | 69.91 |
| OpenAI o1-mini | 52.41 | 75.13 |
| OpenAI o3-mini | 53.72 | 72.80 |
| XiYan-SQL(QwenCoder-32B 底座) | 63.75 | 81.42 |

- LLaMA-3.1 70B:BIRD 从 57.37% 提升到 **68.51%**(+10.37),Spider 从 78.81% 提升到 **86.59%**。
- **BIRD test 68.53%(ExCoT-70B)**,为当时单模型(single-model)设置的 SOTA,超过 o1-mini、o3-mini、GPT-4o、Claude 等闭源模型。
- 有效性(Valid%)在 DPO 后普遍 >97%,说明精度提升不是靠"生成更合法的 SQL"这一单一因素,而是收敛到功能正确的查询。

### 5.2 关键消融

**DPO 轮次配置(BIRD-Dev)**:

| 配置 | BIRD EX | Spider EX |
| --- | --- | --- |
| 1×off-policy + 2×on-policy(默认) | **68.51** | **86.59** |
| 2×on-policy(去掉 off-policy) | 66.30 | 84.16 |
| 3×on-policy | 67.08 | 86.21 |

→ off-policy 轮必不可少,提供多样数据为 on-policy 打基础。

**采样策略(1 轮)**:

| 策略 | Off-Policy BIRD EX | On-Policy BIRD EX |
| --- | --- | --- |
| Random | 64.08 | 66.36 |
| Nearest | 61.99 | **67.21** |
| Furthest | **66.30** | 66.95 |

→ off-policy 用 furthest、on-policy 用 nearest 是最优课程配置。

**CoT 类型**:No-CoT 变体全程提升最小(DPO 后仅 +2.6/+2.2),印证"没有推理路径,偏好信号无法有效定位错误";Complex-CoT(分而治之)在最终阶段增益最大。

### 5.3 训练动态
- 有效偏好对数量随轮次递减(SFT 阶段 >2,000 对 → 第三轮 on-policy 约 560 对),说明对比对趋于耗尽,性能增益随之递减(边际收益递减/可能回落)。
- 平均 CoT 长度随训练从 560 tokens(SFT)增至 910 tokens(最终 on-policy 轮),说明迭代对齐促使模型生成更详尽的中间推理。
- 案例研究(BIRD 题号 10):GPT-4o 32 次生成 0/32 正确 → off-policy DPO 1/32 → on-policy R1 5/32 → R2 4/32,直观展示"自指导迭代"逐步发现正确 SQL 模式。

---

## 六、局限性与展望

- **高度复杂 schema 仍困难**:表关系复杂、依赖领域惯例的查询仍需更多领域知识或训练样本;探索"偏好优化 + 智能体"可缓解。
- **CoT 一致性无保证**:中间推理可能含部分真理、冗余步骤或矛盾,给错误诊断与用户信任带来挑战。
- **CoT 模板受约束**:当前使用结构化固定模板,自由风格 CoT 留给未来;仅尝试了 offline DPO,Online-DPO、PPO、GRPO 等更先进方法留待后续探索。

## 七、一句话总结

ExCoT 证明了"**CoT 推理 + 二元执行反馈 + 离策略/在策略迭代 DPO**"足以让开源 LLM 自指导式地大幅提升 Text-to-SQL 能力(LLaMA-3.1 70B 在 BIRD 上 57.37%→68.51%),完全不需要奖励模型或人工偏好标注,是"执行即奖励"范式的代表工作之一。
