# FlowCache 实验必要性审查：Gate 门控流程与 E1-E7 贡献分析

> **目标**：理清整个计划的完成流程，解释 Gate 与 E1-E7 的关系，并根据 IDEA 三条核心贡献评价每个正式实验是否应该存在。
> **日期**：2026-07-25
> **上游文档**：IDEA.rewritten.md、execution-plan.md、ccfa.yaml、experiments/experiment-designs.md

---

## 1. 总体完成流程

### 1.1 路线 A 的推进逻辑

FlowCache 采用 **Gate 门控 + 实验** 的分层架构。Gate 判定"这个方向是否还值得做"，Experiment 产出"论文需要展示的证据"。二者不是并列关系——**Gate 是决策节点，Experiment 是证据产出节点**。

整体流程如下：

```
Phase 0: 基础设施 (W1-W2)
  G0 Exactness & Loadability
    │
    ├── 通过 → 进入 Phase 1
    └── 失败 → 路线 B（characterization 论文）

Phase 1: 机会与画像 (W3-W7)
  W3-W5: Rollout 录制 trace
  W6: G1 Opportunity (oracle headroom ≥ 10%)
  W7: E1 Workload Characterization
    │
    ├── G1 通过 → 进入 Phase 2
    └── G1 失败 → 路线 B

Phase 2: 无损驻留 + 学习预测 (W7-W8, 并行)
  G3 Lossless Residency (p95 TTFT 改善 ≥ 15%)
  G5 Learning (可选，失败不触发路线切换)
    │
    ├── G3 通过 → 进入 Phase 3
    └── G3 失败 → 路线 B

Phase 3: 量化 + 双轴必要性 (W9-W10)
  G4 Quantization (量化无损 + 质量非劣)
  G2 Two-Axis Necessity (joint > 解耦组合)
    │
    ├── G2+G4 通过 → 进入 Phase 4
    └── 任一失败 → 路线 B

Phase 4: 主实验冻结 (W11)
  E3 Fidelity-Risk Estimation
  E4 End-to-End Main Results
  E5 Mechanism Ablation

Phase 5: 鲁棒性 + 写作 (W12-W14)
  E6 Generalization and Robustness (精简版)
  E7 Failure and Overhead (精简版)
  W13: 全量复跑 + 统计
  W14: 稿件撰写
```

### 1.2 关键约束

- **G0→G1→G3→G4→G2 是路线 A 的关键路径**（execution-plan.md §3.2）
- G5 不在关键路径上：失败只删除 GNN，不切换路线
- G0 和 G4 各允许**一次**模型/后端切换机会
- **不能删除量化后继续沿用 reuse–fidelity 主标题**（G4 失败动作）
- 路线 A 切换到 B 的条件：G0/G1/G2/G3/G4 任一关键门槛失败，或所有 closest baseline 均无法忠实比较

---

## 2. Gate 与 E1-E7 的关系

### 2.1 关系总览

| Gate/Experiment | 角色 | 产出什么 | 为谁服务 |
|---|---|---|---|
| **G0** | 正确性准入 | 引擎能跑、缓存恢复一致、codec 链路通 | 所有后续工作的基础 |
| **G1** | 可行性判定 | oracle headroom ≥ 10% ? closest baseline 可比 ? | 决定路线 A 是否有做的价值 |
| **E1** | 证据产出 | 工作负载画像（overlap, next-use, locality） | 为 G1 提供数据输入，论文 Section 4 的核心图表 |
| **G3** | 可行性判定 | 无损驻留是否有净收益 ? | 路线 A 地基——无损都无收益，加量化更无意义 |
| **E2** | 证据产出 | 复用价值预测器的准确率 vs 系统收益 | 为 G5 提供证据，论文 Section 5 的预测器分析 |
| **G5** | 可选判定 | GNN 预测器是否有净收益 ? | 决定论文是否用 GNN（非强制） |
| **G4** | 可行性判定 | 量化是否成立 ? 质量非劣是否可证 ? | 路线 A 必须通过——量化是不可或缺的维度 |
| **E3** | 证据产出 | 保真风险估计器的准确率、风险校准 | 为 G4 提供数据，论文 Section 5 的 fidelity 分析 |
| **G2** | 核心判定 | joint > 解耦组合 ? 双轴错位是否存在 ? | 论文核心 claim 是否成立 |
| **E4** | 证据产出 | 端到端主结果表（多 budget/并发/workload） | 论文的 Table 1 主表，Section 6 核心 |
| **E5** | 证据产出 | 消融实验（删模块性能下降 ? 单一分数错误分配 ?） | 论文 Section 6 的消融，解释"为什么 joint 更好" |
| **E6** | 证据产出 | 泛化与鲁棒性（workflow-family-out, 噪声, burst） | 论文 Section 7，证明方法不是过拟合 |
| **E7** | 证据产出 | 失败模式与开销分析 | 论文 Section 7/8，诚实报告边界 |

### 2.2 依赖链

```
G0 ──► G1 ──► G3 ──► G4 ──► G2 ──► 论文写作
 │      │      │      │      │
 │      E1     E2     E3     E4
 │             (G5)         E5
 │                          E6 (精简)
 │                          E7 (精简)
```

- **E1** 为 G1 提供输入数据（画像），但 E1 本身不判定路线
- **E2** 为 G5 提供证据，但 G5 不在关键路径上
- **E3** 为 G4 提供 fidelity 证据
- **E4** 是 G2 的端到端验证，也是论文的主表
- **E5** 是 G2 的机制解释（why joint works）
- **E6/E7** 是论文的鲁棒性/诚实性补充

---

## 3. 三条核心贡献与 E1-E7 的必要性评价

### 3.1 核心贡献回顾（IDEA §9）

| 贡献 | 描述 | 贡献类型 |
|---|---|---|
| **C1** | Cache-compatible agent workflow abstraction 与 trace protocol | 辅助评价（工具/协议） |
| **C2** | 复用价值–保真风险解耦的联合 residency controller | 主贡献（系统设计） |
| **C3** | 关于 reuse–fidelity 错位的系统性实证 | 主贡献（实证发现） |

### 3.2 逐实验评价

#### E1：缓存机会与工作负载画像

**直接支撑**：C1（trace protocol 的产物就是可重放 trace）、C3（locality 是实证的起点）

**必要性**：**必须存在**。E1 是三个贡献的共同前提。如果没有 exact-prefix locality，C2 和 C3 都无从谈起。E1 的画像图（overlap, next-use distance, oracle headroom）是论文 Section 4 的核心内容，不应该放在附录。

**但需要审视的是**：E1 目前覆盖了 5 个数据集（τ-bench, BFCL, StableToolBench, SWE, Toolathlon），加上 Mooncake 抽样。对于"证明 locality 存在"这个目的，τ-bench 495 + BFCL 800 已经足够。SWE、Toolathlon 和 Mooncake 的画像更多地是为 E5/E6/E7 提供结构多样性输入，而非 E1 自身必需。可以考虑将 SWE/Toolathlon/Mooncake 的画像降级为 E6 的附表。

---

#### E2：复用价值预测

**直接支撑**：C2（reuse-value estimator 是联合 controller 的一半）

**必要性**：**必须存在，但可以大幅简化**。E2 的核心问题是"预测器是否比简单启发式更好"，而不是"GNN 比 survival 模型好多少"。IDEA §4.3 明确说预测器选择顺序是：heuristic → survival → GNN（仅在前两者与 oracle 仍有明显差距时）。

当前 E2 设计包含 age/LRU、size/recompute-cost、survival/hazard、partial-DAG GNN 四个变体。实际上，如果 G3 已经用 heuristic/survival 通过了无损驻留门槛，G5 又判定 GNN 无净收益（或直接不启用），那么 E2 的 GNN 变体就没有存在的理由。**建议 E2 只保留 heuristic 和 survival/hazard 两个变体，GNN 变体仅在 G5 判定有必要时才启用。**

---

#### E3：保真风险估计

**直接支撑**：C2（fidelity-risk estimator 是联合 controller 的另一半）

**必要性**：**必须存在，但范围可压缩**。E3 的核心是证明"不同 block 的量化敏感度不同，且可以预测"。当前 E3 设计包含 4 个变体（uniform precision、static layer/position rule、norm/range proxy、FlowCache fidelity estimator），并要在 LongBench 1000 + GSM8K 300 + MuSiQue 300 + 2WikiMultihopQA 300 四个数据集上评估。

**问题**：E3 的量化质量评估（logit KL, QA EM/F1）与 E4 的端到端主结果（TTFT, SLO goodput）之间存在巨大的 gap。E3 证明"可以预测哪些 block 敏感"，但 E4 才是证明"这个预测能转化为系统收益"的场合。E3 的四个数据集可以精简：

- LongBench 1000（长上下文，KV 量化质量的主战场）— **保留**
- GSM8K 300（量化 accuracy sanity check）— **保留，但样本量可降至 100**
- MuSiQue 300 + 2WikiMultihopQA 300 — **可合并为一个"多跳 QA"类别，总样本 300，而非各 300**

---

#### E4：端到端主结果

**直接支撑**：C2（证明 joint controller 的端到端收益）、C3（错位的实证）

**必要性**：**必须存在，这是论文的 Table 1**。E4 是论文最核心的实验，在 3 个 KV budget × 3 个并发水平 × 2 类 workload 上比较 10+ 个 baseline/变体。这个实验没有商量余地。

**但可以审视**：E4 的主表是否需要所有 3 个主 workload（τ-bench 495 + BFCL 800 + StableToolBench 500）？从论文叙事角度，τ-bench + BFCL 作为两个不同 tool family 的主表已经足够。StableToolBench 可以作为 E6 的 workflow-family-out 泛化证据，而非 E4 的主表。

---

#### E5：机制消融

**直接支撑**：C2（解释"为什么 joint 比解耦好"）、C3（证明错位是 joint 收益的来源）

**必要性**：**必须存在，但问题是"消融什么"而不是"消融多少"**。IDEA §8 E5 的关键问题已经写得很清楚：

> 关键问题不是"删掉模块性能是否下降"，而是联合建模是否解决了单一分数的错误分配。

当前 E5 设计了 10 个消融轴（reuse-only, fidelity-only, 两者独立串联, joint utility, 无 partial DAG, 无成本校准, 无 parent-closure, 无 CPU tier, 静态阈值 vs 动态预算, 不同 controller 更新频率）。其中：

- **reuse-only / fidelity-only / 独立串联 / joint utility**：核心消融，直接回答"joint 是否必要"——**必须保留**
- **无 parent-closure / 无 CPU tier**：解释 controller 的哪些设计决策贡献了收益——**保留**
- **无 partial DAG / 无成本校准**：解释预测器特征和成本模型的作用——**保留，但可合并为一个"特征消融"表**
- **静态阈值 vs 动态预算 / 不同 controller 更新频率**：超参数敏感性——**可以降级为 E6 的一部分**

---

#### E6：泛化与鲁棒性

**直接支撑**：C3（实证的系统性——不只在单一 workload 上成立）

**必要性**：**需要存在，但可以大幅精简**。当前 E6 设计了 8 个轴（workflow-family-out, 不同上下文长度, DAG 边缺失/噪声, branch misprediction, burst arrival, GPU budget 突变, CPU 带宽竞争, predictor calibration drift）。对于一篇 ICWS 论文：

- **workflow-family-out**（τ-bench → BFCL 交叉）：必须保留——这是泛化的核心证据
- **branch noise / burst arrival / GPU budget 突变**：三个可以合并为一个"鲁棒性"表，用 2-3 种扰动类型
- **不同上下文长度**：可以在 E4 的主表中通过 context length 分层来体现，无需单独一节
- **CPU 带宽竞争 / predictor calibration drift**：属于 appendix 级别的内容，不建议进主文

**第二模型和额外 dataset-out 已明确标注为"仅资源允许时扩展"，正确。**

---

#### E7：失败与开销

**直接支撑**：C3（实证的系统性——诚实报告边界与失败条件）

**必要性**：**必须存在，但不需要独立成章**。E7 的内容（exact-prefix overlap 过低、量化误判、GPU↔CPU 抖动、parent block 缺失、大面积 invalidation、controller 开销超过 saved-prefill、graceful degradation）是论文诚实的体现。但其中许多内容实际上是 E4/E5 的"负面结果"附录，不需要单独作为一个实验章。

**建议**：将 E7 的内容拆分为两部分：
1. Controller 开销分析 → 合并入 E4 的开销表
2. 失败模式分析 → 合并入 E5/E6 的讨论或作为 appendix

---

## 4. 总结：必要性矩阵

| 实验 | 必要性 | 建议动作 | 对核心贡献的支撑 |
|---|---|---|---|
| **E1** | 必须存在 | 保留，但 SWE/Toolathlon/Mooncake 画像降级为 E6 附表 | C1+C3 前提 |
| **E2** | 必须存在 | 只保留 heuristic + survival 变体；GNN 变体仅在 G5 有必要时启用 | C2（reuse 一半） |
| **E3** | 必须存在 | 数据集压缩：LongBench 1000 + GSM8K 100 + 多跳 QA 合并 300 | C2（fidelity 一半） |
| **E4** | 必须存在 | 主表用 τ-bench + BFCL；StableToolBench 移至 E6 作为泛化证据 | C2+C3 核心 |
| **E5** | 必须存在 | 核心消融 4 轴保留；特征消融合并；超参数敏感度移至 E6 | C2+C3 机制解释 |
| **E6** | 需要存在 | 精简为 3 轴（workflow-family-out + 鲁棒性扰动 + branch noise） | C3 系统性 |
| **E7** | 不需要独立成章 | 拆分为 controller 开销（入 E4）+ 失败模式（入 E5/E6 appendix） | C3 诚实性 |

### 4.1 精简后的总实验量

| 实验 | 数据集 | 样本量 | 节省 |
|---|---|---|---|
| E1 | τ-bench 495 + BFCL 800 | 1,295 | 原 5 数据集 → 2 主数据集 |
| E2 | τ-bench 495 + BFCL 800（同 E1 trace） | 1,295 | 变体 4 → 2 |
| E3 | LongBench 1000 + GSM8K 100 + 多跳 QA 300 | 1,400 | 原 1,900 → 1,400 |
| E4 | τ-bench 495 + BFCL 800 | 1,295 | 原 3 主 workload → 2 |
| E5 | τ-bench 495 + BFCL 800（同 E4 trace） | 1,295 | 消融轴 10 → 6 |
| E6 | τ-bench/BFCL 交叉 + tool-wait 扰动 | ~1,500 | 轴 8 → 3 |
| E7 | 合并入 E4/E5/E6 | — | 独立章 → 分散 |

**总计**：从 ~8,800 样本、13 个数据集、7 个独立实验章，压缩为 ~6 个实验章、~10 个数据集、~6,000 核心样本。E6/E7 的精简版不丢失证据，但减少了独立章和重复运行。

### 4.2 不等价删除的了什么

- **第二模型泛化**：已明确后移，正确
- **GNN 预测器**：已在 G5 设计为可选，失败不触发路线切换，正确
- **StableToolBench 作为第三主 workload**：降级为 E6 泛化证据，不进入 E4 主表。这意味着论文的"two tool families"证据来自 τ-bench + BFCL，而非三个
- **合成 DAG**：已因用户禁令（v0.2）全部移除，结构多样性由 SWE 轨迹和 Toolathlon 的真实结构替代
- **HotpotQA / ShareGPT**：已排除，正确