# Reward-SQL: Boosting Text-to-SQL via Stepwise Execution-Aware Reasoning and Process-Supervised Rewards —— 阅读报告

> **论文链接**: https://arxiv.org/abs/2505.04671
> **论文题目**: Reward-SQL: Boosting Text-to-SQL via Stepwise Execution-Aware Reasoning and Process-Supervised Rewards
> **作者**: Yuxin Zhang, Meihao Fan, Ju Fan, Mingyang Yi, Yuyu Luo(中国人民大学、HKUST(GZ))、Guoliang Li(清华大学)、Bin Wu, Wenchao Zhou(阿里云)
> **发表**: Proc. ACM Manag. Data 4, 3(SIGMOD), Article 228, 2026
> **代码**: https://github.com/ruc-datalab/RewardSQL

---

## 一、主要解决的问题

论文针对 RL 后训练方法在复杂 Text-to-SQL 任务上的两个关键局限:

1. **缺乏逐步"执行感知"的推理(Stepwise Execution-Aware Reasoning)**:现有 RL 方法(如 Reasoning-SQL、ExCoT 变体)虽然引入中间推理步骤,但这些步骤不接触数据库——模型在生成过程中观察不到中间执行状态,只能等最终 SQL 执行完才看到不一致。这导致表选择、列选择、JOIN 键等错误级联放大(典型错误:多余 JOIN 导致聚合重复计数),并且逻辑上"看起来合理"的推理与实际数据分布脱节。
2. **缺乏过程级奖励(Process-Supervised Rewards)**:现有方法主要使用"结果奖励"(outcome reward)——最终 SQL 执行正确才给分。该奖励极度稀疏、不可分解,无法定位是哪个推理步骤出错,导致 credit assignment 困难、策略梯度方差大。

**实证证据**:作者用 Qwen3-8B + GRPO + 纯结果奖励在 BIRD 训练集上训练,仅得到 63.0% 的 BIRD-Dev 执行准确率;错误分布中最常见的是过滤条件错误(32.8%)、表选择错误(23.4%)、列选择错误(16.5%)——根因都是"一次性生成完整 SQL、无法观察中间数据库状态"。

---

## 二、创新点

1. **CoCTE(Chain of Common Table Expressions,CTE 链)推理框架**:借鉴数据库工程师"渐进式构建查询"的写法,把复杂 SQL 分解为**一串可独立执行、可验证的 CTE**(每条 CTE 生成后立即执行),用中间结果将推理"锚定"到真实数据库上,同时提高准确率与可解释性。相比纯语言 CoT,CoCTE 的每个中间步都是可执行、可验证的 SQL 片段。
2. **显式训练的过程奖励模型(PRM)**:首次为 Text-to-SQL 训练专门的 PRM,结合 **MCTS 生成的轨迹打分模型**(估计中间轨迹正确性)与 **逆熵加权**(inverse entropy weighting,突出信息量大的关键步骤、压制冗余步骤),把稀疏的结果信号转化为稠密、可解释的逐步监督。
3. **过程奖励贯穿训练与推理**:统一目标 `R_proc + R_out` 集成进 GRPO 训练(稳定优化、加速收敛、降低方差);推理阶段用过程奖励做 Best-of-N 选择,替代启发式的多数投票/自一致性。
4. **系统性验证泛化性**:在 5 个分布外(OOD)基准(Spider-DK、Spider-Syn、Spider-Realistic、ScienceBenchmark、EHRSQL)上做了零样本评测,证明跨域泛化能力强。

---

## 三、使用的技术和算法

Reward-SQL 分三阶段:

### Stage 1:模型初始化(Cold Start)
- 半自动流水线构造 CoCTE 训练语料:用强生成器 **DeepSeek-V3.1**(辅以 5 条人工标注 seed)把 BIRD/Spider 训练集中的 17,462 条 SQL 转换为 CoCTE 轨迹;每题在温度 1.0 下采样 3 次以增加多样性,再经 **AST 解析过滤**(剔除格式错误 CTE、未定义引用、命名冲突)。
- 语料规模:17,462 条 SQL → **51,996 条有效 CoCTE 轨迹**(接受率 90.7%);平均每条轨迹 **3.3 个 CTE 步**、平均 rationale 28.1 tokens、平均 CTE 11.6 tokens。
- 对策略模型做 SFT(下一 token 预测损失),使其学会"自然语言规划(rationale) + 可执行 CTE"交替的结构化推理输出。超参:lr=1e-5,batch=64,max context=4096,3 epochs。

### Stage 2:过程奖励设计(Process Reward Design)
- **轨迹打分模型 R_φ**:以 Qwen2.5-7B-Coder-Instruct 为底座独立训练(不与策略共享参数),输入是每个 CTE 片段 + 其 `SELECT * FROM` 执行结果 + `<step>` 标记,输出该中间轨迹的正确性置信度 s_i ∈ (0,1)。
  - **训练标签由 MCTS 自动生成**:对 BIRD 训练问题从冷启动策略采样 n=8 条 rollout(温度 1.0,平均每个问题得到 38 条不同根到叶轨迹),叶子节点以执行结果给出二元奖励 R_out,通过蒙特卡洛回传得到每步期望效用 σ_i = E[R_out | T≤i],再用**二元交叉熵**训练 R_φ。
- **逆熵权重 I_H**:基于策略模型 token 级熵定义轨迹熵并归一化,I_H(T≤i) = 1 - H̃(T≤i)。成功推理的熵随步骤单调下降;冗余轨迹平均熵更高(0.3142 vs 0.2966)、逆熵更低(0.47 vs 0.50),从而自动降低冗余步骤的权重。
- **过程奖励定义**:R_proc(T≤d) = Σ_{i=1..d} [ (Σ_{j≤i} I_H(T≤j)·R_φ(T≤j)) / (Σ_{j≤i} I_H(T≤j)) ](逆熵加权的逐步轨迹得分累积)。

### Stage 3:过程监督的 RL 训练与推理
- **RL 训练**:在 **GRPO** 框架下最大化 `E[ R_proc(T) + R_out(y) - λ·D_KL(πθ‖πref) ]`;每个 token 的组内优势 A 同时包含结果奖励与过程奖励,再做组内标准化;训练时 PRM 参数冻结,只更新策略。超参:G=8,clip=0.2,KL 系数 0.02,温度 1.0。
- **推理**:生成 N 条 CoCTE 轨迹,用过程奖励选最高分轨迹作为最终答案(**Best-of-N / PRM@N**);默认 N=8。

---

## 四、Reward 设计

### 4.1 结果奖励(Outcome Reward, R_out)
二元信号:预测 SQL 与标准 SQL 在数据库上执行结果一致则 1,否则 0。概念简单但极度稀疏、不可分解(单一最终子句错误会抹掉整条轨迹的反馈)。

### 4.2 过程奖励(Process Reward, R_proc)
对 CoCTE 轨迹中每个中间步骤提供稠密反馈,由两个互补组件构成:

1. **轨迹打分模型 R_φ**(正确性维度):估计中间轨迹 T≤i 最终成功的期望概率(MCTS 蒙特卡洛回传标签监督),值域 (0,1)。
2. **逆熵权重 I_H**(信息量/效率维度):低熵(高确定性)步骤权重更大,信息量小或冗余的步骤被自动降权。

最终形式(逆熵加权的逐步轨迹得分累积):

```
R_proc(T≤d) = Σ_{i=1..d} [ (Σ_{j≤i} I_H(T≤j) · R_φ(T≤j)) / (Σ_{j≤i} I_H(T≤j)) ]
```

### 4.3 统一 RL 目标

```
J(θ) = E_{(T,y)~πθ} [ R_proc(T) + R_out(y) - λ·D_KL(πθ ‖ πref) ]
```

设计要点:过程奖励提供稠密逐步监督以解决 credit assignment,结果奖励保持与最终目标对齐;二者结合并通过 KL 正则约束稳定性,避免 reward hacking(模型只优化过程得分而不产生正确最终答案)。

---

## 五、实验效果总结

基础模型 Qwen3-8B,评测指标为执行准确率(EX)。

### 5.1 主基准(BIRD-Dev / Spider-Test)

| 方法 | BIRD(Dev) | Spider(Test) |
| --- | --- | --- |
| Reasoning-SQL + Qwen2.5-7B | 64.0 | 78.7 |
| SQL-R1 + Qwen2.5-7B‡ | 65.1 | 81.4 |
| Arctic-SQL + Qwen2.5-7B‡ | 66.8 | 82.1 |
| **Reward-SQL + Greedy** | 65.9 | 81.2 |
| **Reward-SQL + PRM@8** | 68.5 | 83.1 |
| **Reward-SQL + PRM@32** | **70.3** | **85.4** |

- 在同等规模(7-8B)后训练方法中全面最优;PRM@32 在 BIRD 上达 **70.3%**,并超越部分 100 倍参数的闭源模型(GPT-4 系)。
- 基线验证:纯结果奖励的 GRPO 基线 BIRD-Dev 仅 63.0%,证明过程奖励是主要增益来源。

### 5.2 OOD 泛化(零样本,5 个基准)

| 模型 | 鲁棒性级 Avg(Spider-DK/Syn/Realistic) | 跨域级 Avg(ScienceBench/EHRSQL) | 总体 Avg |
| --- | --- | --- | --- |
| GPT-4o | 67.5 | 50.2 | 59.9 |
| SQL-R1 + Qwen2.5-7B | 75.5 | 45.3 | 63.4 |
| Arctic-SQL + Qwen2.5-7B | 76.5 | 46.4 | 64.4 |
| **Reward-SQL + Qwen3-8B** | **78.5** | **44.5** | **64.9** |

- 鲁棒性级 OOD(语言变体)上超过所有 7-8B 模型,甚至超过 GPT-4-Turbo/DeepSeek-V3 等大模型;总体平均超越所有对比方法。
- 跨域级 OOD(全新 schema 与术语)上所有后训练模型都会掉点(领域过拟合),Reward-SQL 与其他 RL 基线持平——作者将其列为未来挑战。

### 5.3 训练稳定性

| 方法 | 最终 Dev EX | 平均(±) | 方差(±) | 收敛步数 |
| --- | --- | --- | --- | --- |
| GRPO(R_out 纯结果) | 59.6 | 60.2 | 0.76 | 350 |
| **GRPO(R_proc + R_out)** | **61.8** | **61.5** | **0.58** | **250** |

过程奖励加速收敛(350→250 步)、降低方差(±0.76→±0.58),并显著提升最终精度。

### 5.4 消融实验(BIRD-Dev)

| 配置 | Greedy | PRM@8 |
| --- | --- | --- |
| **完整 Reward-SQL(SFT+GRPO+PRM)** | **66.0** | **68.7** |
| Direct SQL(无推理) | 63.6 | - |
| CoCTE(仅提示 DeepSeek-V3.1,5-shot) | 60.1 | - |
| NL-CoT(通用自然语言推理) | 64.2 | - |
| SFT only | 63.2 | 67.4 |
| GRPO(R_out only) | 62.6 | 67.0 |
| GRPO(R_proc + R_out) | 66.0 | 68.7 |
| 选择:Self-Consistency | - | 66.6 |
| 选择:Outcome Reward 模型 | - | 67.1 |
| **选择:Process Reward(PRM)** | - | **68.7** |

- CoCTE 优于直接 SQL(+2.4)与 NL-CoT(+1.8);对比纯提示 CoCTE(+5.9 差距)证明**训练+奖励设计才是关键,光有格式提示不够**。
- 训练加入过程奖励:greedy +3.4、PRM@8 +1.7;PRM 选择优于自一致性(+2.1)与 ORM(+1.6)。
- 冷启动语料来源敏感度:DeepSeek-only 68.4 / GPT-4-only 67.6 / Claude-only 68.9 / 混合 70.0——单源差异仅 ±0.6%,混合语料效果最好,说明模型不依赖特定 LLM 的分解风格。

### 5.5 错误分析与效率
- **错误率大幅下降**:基线 Qwen3-8B 总错误 45.57% → PRM@8 31.49% → PRM@32 29.66%(相对下降 34.9%)。最显著:幻觉错误 6.26%→0.26%,JOIN 键错误 -2.34%,表选择 -2.48%,列选择 -2.61%。
- **CTE 级错误分析**:200 个失败样本中 94.5% 与 CTE 分解无关(主要仍是过滤条件、schema 不匹配等 SQL 生成问题),说明 CoCTE 不引入系统性分解错误。
- **效率**:推理延迟 n=1 时 0.8s,n=32 时 12.31s(其中 SQL 执行与 PRM 打分为主要开销);PRM@16(69.6%,6.3s)为精度/延迟实际甜点;CoCTE 查询执行开销仅比专家扁平 SQL 高 <10%(230ms vs 210ms)。
- **训练成本**:MCTS 数据构造离线一次完成(16,248 实例,约 18 小时);PRM 训练 4×A800 共 28 GPU 小时;推理时无 MCTS 开销。

---

## 六、局限性与展望

- **跨域(全新 schema)泛化仍有限**:所有 RL 后训练模型在 ScienceBenchmark/EHRSQL 上显著落后于大型通用模型,存在领域过拟合。
- **测试时延计算**:PRM@N 精度随 N 对数增长,但 N>16 后边际收益递减、延迟翻倍;与理论 Pass@N 上限仍有差距,选择机制还有提升空间。
- **查询执行效率**:CoCTE 格式比扁平 SQL 有约 10% 执行开销,生产环境可探索"规则式 CTE 转扁平 SQL 编译器"与成本感知奖励。
- 未来方向:解决跨域泛化、设计 cost-aware 奖励优化执行效率。

## 七、一句话总结

Reward-SQL 用 **CoCTE(可执行的 CTE 链)推理** 替代纯语言 CoT,并训练了一个 **MCTS 标签 + 逆熵加权的过程奖励模型**,将稀疏结果奖励升级为稠密过程监督,同时作用于 GRPO 训练与 Best-of-N 推理,以 8B 模型在 BIRD 上达到 70.3% 并表现出较强的 OOD 泛化能力。
