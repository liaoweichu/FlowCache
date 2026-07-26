# FlowCache 14 周执行计划与 Stage Gate 管理

> 本文件基于 `IDEA.rewritten.md` 的 Section 7（可行性门槛 G0–G5）、Section 8（正式实验计划 E1–E7）、Section 11（风险、收缩与转向）和 Section 12（14 周执行计划）形式化而成，不引入新的研究内容或实验。
> 最后更新：2026-07-26（v0.5：BFCL 全面移除，单数据集 τ-bench 1,320 episodes）

---

## 1. 项目概述

| 项 | 内容 |
|---|---|
| 项目名称 | FlowCache: Decoupling Reuse Value and Fidelity Risk for Prefix Caching in Memory-Constrained Agent Workflows |
| 中文题目 | 面向内存受限 Agent 工作流的复用价值–保真风险解耦前缀缓存 |
| 目标投稿 | IEEE ICWS 2027（CFP TBD，截稿时间待官方发布，不得写成已确认的 2027-01） |
| 备选 venue | IEEE EDGE 2026/2027；具备真实双节点部署和长期服务证据后再考虑 IEEE TSC |
| 硬件 | 单卡 NVIDIA RTX 4090D 24GB |
| 结论边界 | memory-constrained GPU emulation，不直接等同于真实移动/边缘设备；不能单独支持真实手机/嵌入式能耗、云端排队、跨节点重叠、大规模集群调度或 TSC 级生产部署结论 |
| 项目阶段 | Conditional Go——只有通过 Section 7 可行性门槛 G0–G5 后才进入完整实现 |
| 当前路线 | 路线 A（推荐主线）尚未启动，G0–G5 均 `not_started` |
| 核心假设 | (1) 真实 Agent workload 存在非平凡 exact-prefix 再访问，且离线 oracle 明显优于 LRU/简单启发式；(2) 未来复用价值与量化敏感度并非高度一致；(3) 联合控制在扣除预测、迁移和量化开销后优于最强解耦基线；(4) 质量损失能被预先设定的非劣区间约束 |

### 1.1 路线总览

| 路线 | 定位 | 状态 | 触发条件 |
|---|---|---|---|
| A | Exact-prefix reuse value + fidelity risk + joint precision/residency（推荐主线） | 待启动 | 默认主线，必须通过 G0–G5 |
| B | When Does Workflow Structure Create Physical KV Reuse?（保守回退） | 备用 | G0/G1/G2/G3/G4 任一关键门槛失败 |
| C | Model-Scoped Shadow Frontiers for Heterogeneous Edge–Cloud Agents（高风险扩展） | 延后 | 具备真实双端点后，作为 TSC 扩展，不与路线 A 同时塞入首篇会议论文 |

---

## 2. W1–W14 每周计划表

| 周次 | 目标 | 对应 Gate / Experiment | 产物 | 失败动作 |
|---|---|---|---|---|
| W1 | 冻结一个主模型、后端和主机配置；完成 Q-storage codec/staging spike | G0（loadability/codec） | 模型/后端/主机配置冻结记录；Q-storage codec spike 报告；allocated/reserved 峰值测量 | G0 失败：允许切换一次受支持的模型/后端；仍失败则路线 A No-Go，转路线 B，不进入预测器开发 |
| W2 | 实现 block identity、precision lineage、父链、invalidation 和 exactness tests；冻结主工具 workload（τ-bench 1,320 episodes / StableToolBench）与真实轨迹子集 | G0（exactness） | block identity/lineage/父链实现；exactness test 通过证据；主工具 workload 与真实轨迹子集冻结 | 同 W1（G0 失败动作） |
| W3–W5 | Tier 1 主 workload（τ-bench 1,320 / STB 500）rollout 录制与 compiler/trace/replay；Tier 2 真实轨迹（SWE/Toolathlon/CATraces）整理；Tier 4 静态集整理 | —（可重放 trace） | 可重放 trace；cache-compatible 序列化规则；0.4.3 核验报告 | trace 不可重放则阻塞 G1，间接触发 G1 失败动作 |
| W6 | LRU/GDSF、同引擎 APC、offline oracle；至少一个 PBKV/KVFlow closest baseline | G1（opportunity/comparability） | LRU/GDSF/APC/oracle 实现；至少一个 closest baseline 可公平运行；oracle headroom 测量 | G1 失败：转向"何时工作流结构产生物理 KV 复用"的 benchmark/characterization 论文（即路线 B） |
| W7 | workload characterization 与无泄漏 split | E1 | E1 画像报告（workflow 长度/深度/宽度/分支率/工具等待、exact-prefix overlap、LCP tokens、next-use distance、working-set size、KV 占比、oracle vs LRU/heuristic headroom）；无泄漏 split | E1 画像若显示 exact-prefix overlap 过低或 oracle headroom 很小，反映 G1 未真正通过，需回溯 |
| W7–W8 | GPU BF16↔CPU BF16↔evict、heuristic/survival reuse estimator 与简单 controller | G3 / G5 | 无损 residency 控制器；heuristic/survival reuse estimator；简单 controller；G3/G5 评估 | G3 失败：路线 A No-Go；可保留实现作为工程基线，但不以无损 residency 单独投稿该主张。G5 失败：保留简单、可解释的 controller，不为论文形式强行加入 GNN（不触发路线切换） |
| W9 | 离线量化与组合干预、fidelity estimator、质量界功效分析 | G2 / G4 | 离线量化干预回放；fidelity estimator；质量非劣界/δ/样本量预注册；功效分析 | G4 失败：在 G0 已允许的一次模型/后端切换后仍失败，则路线 A No-Go 并转路线 B；不能删除量化后继续使用 reuse–fidelity 主标题投稿 |
| W10 | Q-storage 集成和 joint controller；与最强解耦组合直接比较 | G2 / G4 | Q-storage 集成；joint controller；joint vs 最强解耦组合对比 | G2 失败：若 joint policy 无净收益，reuse–fidelity 主线不成立，转路线 B；不把低相关分析单独包装成方法贡献 |
| W11 | 相同后端的主实验冻结 | E3 / E4 / E5 | E3 fidelity 评估；E4 端到端主结果（多 budget/并发/workload）；E5 机制消融 | 主实验无法冻结则阻塞 W13，需回溯 G2/G3/G4 |
| W12 | branch noise、burst、负对照、failure 和 overhead；第二模型/GNN 后移 | E6（精简版）/ E7（精简版） | E6 鲁棒性精简版；E7 失败与开销报告；负对照（LMSYS-Chat-1M 顺序式） | 第二模型和 GNN 仅在主结果已稳定且仍有时间时加入，否则后移不视为失败 |
| W13 | 全量复跑、统计、artifact 与复现说明 | —（冻结结果） | 全量复跑结果；workflow-level paired bootstrap 95% CI；artifact 包；复现说明 | 统计 CI 不足则不能提出质量非劣 claim，需回溯 G4 功效分析 |
| W14 | ICWS 稿件与图表 | —（不新增未验证 claim） | ICWS 稿件；图表；只使用已验证结果 | 不得新增未验证 claim 填充结论 |

### 2.1 计划弹性说明

- GNN 和第二模型只有在主结果已稳定且仍有时间时加入（W12 后移项）。
- 14 周是否为真实时间约束仍需确认（见 IDEA Section 14"仍需确认"）。
- 若 W3–W5 的 trace 不可重放，则 G1 缺少输入，间接触发 G1 失败动作。
- **v0.2（2026-07-25）**：数据集体系重构（详见 `experiments/experiment-designs.md` 0.4）——禁自建数据集、样本总量 ~880 → ~8,800；trace 录制窗由 W3–W4 延至 W3–W5（rollout ~33–40 GPU 小时），G1 顺延至 W6、E1 顺延至 W7；G3/E2/G5（W7–W8）及之后周次不变，顺延由 W9 后缓冲吸收。

---

## 3. Gate 依赖图与关键路径

### 3.1 依赖关系

```
                    G0  Exactness & Loadability  [W1–W2]
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       G1          G4          G3
   Opportunity  Quantization  Lossless
     [W6]       [W9–W10]     Residency
        │                       [W7–W8]
        │           │           │
        └─────►G3───┘           │
                │               │
                │       ┌───────┘
                ▼       ▼
                G2  Two-Axis Necessity  [W9–W10]
                │
                ▼
               G5  Learning  [W7–W8, 可选/并行]
```

### 3.2 关键路径

**关键路径（决定路线 A 生死）**：

```
G0 ──► G1 ──► G3 ──► G4 ──► G2
```

- G0 是所有后续 gate 的基础（exactness、loadability、codec/lineage spike）。
- G1 需要 G0 产出的可重放 trace 与 compiler。
- G3（无损 residency）需要 G0 的 loadability 与 G1 证明的 opportunity。
- G4（量化）需要 G0 的 Q-storage codec/lineage spike。
- G2（双轴必要性）需要 G1、G3、G4 全部就位，才能在相同质量约束下比较 reuse-only、fidelity-only、最强解耦组合与 joint policy。
- G5（学习）不在关键路径上：失败只删除 GNN，保留简单 controller，不触发路线切换。

### 3.3 依赖链形式化

| Gate | 周次 | 依赖前置 Gate | 关键路径上？ |
|---|---|---|---|
| G0 | W1–W2 | 无 | 是 |
| G1 | W6 | G0 | 是 |
| G2 | W9–W10 | G1, G3, G4 | 是 |
| G3 | W7–W8 | G0, G1 | 是 |
| G4 | W9–W10 | G0 | 是 |
| G5 | W7–W8 | G1, G3 | 否（可选，失败不触发路线切换） |

---

## 4. Experiment–Gate 映射表

| Experiment | 描述 | 周次 | 对应 Gate | 说明 |
|---|---|---|---|---|
| E1 | 缓存机会与工作负载画像 | W7 | G1 | E1 报告 exact-prefix overlap、next-use distance、oracle headroom，是 G1 opportunity 判定的中心证据，不应放在附录 |
| E2 | 复用价值预测 | W7–W8 | G5 | E2 的 GNN 变体仅在 G1/G5 有必要时启用；准确率提升若不能转换为系统收益，不构成贡献 |
| E3 | 保真风险估计 | W9, W11 | G4 | E3 需要量化干预回放，依赖 G4 的量化支持；指标包括 logit KL、QA EM/F1、工具调用正确率、风险校准 |
| E4 | 端到端主结果 | W11 | G2, G3, G4 | E4 在多 budget/并发/workload 上比较 joint policy 与最强解耦组合，是 G2 双轴必要性的端到端验证 |
| E5 | 机制消融 | W11 | G2 | E5 消融 joint utility、parent-closure、CPU tier 等，关键问题是联合建模是否解决单一分数的错误分配（即 G2） |
| E6 | 泛化与鲁棒性 | W12 | G5 | E6 在未见 workflow family 上测试（workflow-family-out），与 G5 的"未见 workflow family 上降低 policy regret"相关；第二模型/额外 dataset-out 仅作资源允许时的扩展 |
| E7 | 失败与开销 | W12 | —（独立） | E7 单独报告失败模式与控制器开销，不绑定特定 gate；包括 overlap 过低、量化误判、GPU↔CPU 抖动、controller 开销超 saved-prefill 等 |

### 4.1 写作就绪条件（IDEA Section 14）

进入论文写作前必须满足：

1. G0–G3 通过；
2. G1 证明真实决策空间存在；
3. 最强 close baseline 已能公平运行或清楚解释不兼容；
4. G2 和 G4 均通过，joint policy 在质量约束下胜过最强解耦组合；
5. 主 claim 有一张相同后端的 Pareto 主图和 workflow-level 置信区间；
6. 所有贡献均对应真实结果，不使用预期数字填充结论。

---

## 5. 失败回退决策表

| Gate | 通过条件（摘要） | 失败动作（原文） | 回退路线 | 是否触发 A→B 切换 |
|---|---|---|---|---|
| G0 | BF16 缓存恢复与完整重算一致；block identity/父链/invalidation 无错误；冻结模型/tokenizer/template/后端 revision；Q-storage codec/staging/precision-lineage 隔离跑通；后端能拦截恢复 KV；测量 allocated/reserved 峰值 | 允许切换一次受支持的模型/后端；仍失败则路线 A No-Go，转路线 B，不进入预测器开发 | B | 是（一次模型/后端切换机会后仍失败） |
| G1 | 统计 exact-prefix overlap/next-use distance/可节省 prefill time/KV 占比；比较 LRU/size-aware heuristic/离线 oracle；至少一个 PBKV/KVFlow closest baseline 可公平运行；oracle 相对最佳简单策略有约 10% miss-cost 或 p95 TTFT 改进空间 | 转向"何时工作流结构产生物理 KV 复用"的 benchmark/characterization 论文 | B | 是 |
| G2 | 测量 future-use/recompute value 与 D_{b,q} 的 rank correlation；分析四类块；在相同质量风险约束下比较 reuse-only/fidelity-only/最强解耦组合/joint policy；joint policy 计入自身开销后形成解耦组合达不到的延迟–容量改进 | 若 joint policy 无净收益，reuse–fidelity 主线不成立，转路线 B；不把低相关分析单独包装成方法贡献 | B | 是 |
| G3 | 实现 GPU BF16/CPU BF16/evict；恢复和迁移开销小于所节省 prefill；内部参考：固定质量下 p95 TTFT 改善约 15%，吞吐下降不超过约 5%；控制器优于 size-aware LRU/GDSF | 路线 A No-Go；可保留实现作为工程基线，但不以无损 residency 单独投稿该主张 | B | 是 |
| G4 | 真实后端支持目标模型 KV quantization 与恢复；active runtime 统一材料化为 BF16，staging 峰值与 tainted lineage 被追踪；量化/反量化不破坏 end-to-end latency；pilot 后预注册绝对质量非劣界/δ/样本量；95% CI 窄到足以检验该界；至少在一个真实工具 workload 上验证 | 在 G0 已允许的一次模型/后端切换后仍失败，则路线 A No-Go 并转路线 B；不能删除量化后继续使用 reuse–fidelity 主标题投稿 | B | 是（一次模型/后端切换机会后仍失败） |
| G5 | 学习模型在未见 workflow family 上相对最佳确定性启发式降低 policy regret；收益包含模型推理开销；内部参考：净端到端收益约 5%，或 regret 改善约 10% | 保留简单、可解释的 controller，不为论文形式强行加入 GNN | 不切换（仍走路线 A，仅删除 GNN） | 否 |

### 5.1 关键约束

- **不能删除量化后继续沿用 reuse–fidelity 主标题投稿**（G4 失败动作原文约束）。
- **不以无损 residency 单独投稿该主张**（G3 失败动作原文约束）。
- **不把低相关分析单独包装成方法贡献**（G2 失败动作原文约束）。
- G5 失败不触发路线切换，仅退化控制器。

---

## 6. 路线 A/B/C 切换条件汇总

### 6.1 路线 A→B 切换触发条件

依据 IDEA Section 12："如果 G0 的 codec/lineage spike、G1、G2、G3 或 G4 任一关键门槛失败，路线 A 停止并转路线 B。"

| 触发源 | 具体条件 | 来源 |
|---|---|---|
| G0 失败 | codec/lineage spike 失败，且一次模型/后端切换后仍无法通过 exactness/loadability | Section 7 G0 + Section 12 |
| G1 失败 | oracle headroom 很小（无约 10% miss-cost 或 p95 TTFT 改进空间），或所有 closest baseline 无法忠实比较 | Section 7 G1 + Section 11 风险表 |
| G2 失败 | joint policy 无净收益，或复用价值与保真风险高度一致（双轴不成立） | Section 7 G2 + Section 11 风险表 |
| G3 失败 | 无损 residency 恢复/迁移开销不小于所节省 prefill，或控制器不优于 size-aware LRU/GDSF | Section 7 G3 |
| G4 失败 | 后端不支持所需 KV quantization/offload，且一次模型/后端切换后仍失败；或量化收益被 codec/PCIe 抵消，端到端无正收益 | Section 7 G4 + Section 11 风险表 |
| 基线不可比 | 所有 closest baseline 均无法忠实比较，inspired variant 不能替代全部 close baselines | Section 11 风险表（likely-pivot） |

### 6.2 路线 B 内容

**When Does Workflow Structure Create Physical KV Reuse?**

- 构建 cache-compatible workflow compiler、trace benchmark、oracle 与简单 cost-aware policy。
- 方法新颖性较低，但语义扎实、单卡可完成。
- 更适合 ICWS/EDGE 的 workload/system characterization。
- 无损 GPU/CPU/evict 实现可保留为路线 B 的工程基线。

### 6.3 路线 A→C 切换

- **不存在直接的 A→C 切换**。路线 C 是高风险扩展，需真实双端点、网络 trace 和质量模型，作为 TSC 扩展。
- 路线 C 不得用单卡轮流加载两个模型替代真实端云证据。
- 路线 C 不与路线 A 同时塞入首篇会议论文。

### 6.4 路线 A 内部退化（非切换）

| 触发 | 退化动作 | 是否切换路线 |
|---|---|---|
| G5 失败（学习模型无净收益） | 删除 GNN，保留简单 cost-aware policy | 否（仍路线 A） |
| 简单 heuristic 接近 oracle | 删除 GNN，强调简单 cost-aware policy | 否（仍路线 A） |
| 多 block 量化误差非线性叠加 | 组合干预标定、风险上界和全任务质量约束 | 否（仍路线 A，修正方法） |
| 工作流 split 泄漏模板/底层问题 | group split、prefix dedup、冻结 test | 否（仍路线 A，修正证据） |

### 6.5 Venue 切换条件

| 条件 | 目标 venue | 来源 |
|---|---|---|
| 单卡模拟无法支撑 edge claim，且服务/QoS framing 较强 | ICWS（主投），使用 memory-constrained GPU 表述 | Section 5.4 + Section 13.1 |
| 贡献主要体现为缓存内核、显存层级和 edge resource scheduling，服务抽象较弱 | IEEE EDGE | Section 13.2 |
| 具备两个真实服务端点、长期多租户 workload、SLA/成本/能耗、网络指标、故障恢复、Shadow Frontiers 等新增机制 | IEEE TSC（延后） | Section 13.3 |

---

## 7. Stage Gate 状态追踪机制

### 7.1 状态枚举

Gate 与 Experiment 的 `status` 字段在 `ccfa.yaml` 中追踪以下四种状态：

| 状态 | 含义 |
|---|---|
| `not_started` | 尚未启动 |
| `in_progress` | 正在执行，尚未判定 |
| `passed` | 通过门槛（Gate）或完成（Experiment） |
| `failed` | 未通过门槛，触发对应失败动作 |

### 7.2 追踪字段

每个 Gate 在 `ccfa.yaml` 中记录：`id`、`description`、`status`、`week`、`depends_on`、`failure_action`、`fallback_route`、`triggers_route_switch`、`source`。

每个 Experiment 在 `ccfa.yaml` 中记录：`id`、`description`、`status`、`week`、`gates`、`source`。

### 7.3 更新规则

1. 进入某 gate/experiment 所在周次时，状态从 `not_started` → `in_progress`。
2. 收齐该 gate 的全部通过条件证据后，状态 → `passed`。
3. 任一通过条件无法满足且无法修复，状态 → `failed`，并执行 `failure_action`。
4. Gate 失败且 `triggers_route_switch: true` 时，路线 A 停止，转路线 B。
5. Gate 失败且 `triggers_route_switch: false`（仅 G5）时，执行退化动作，路线 A 继续。
6. Experiment 状态不直接触发路线切换，但其结果可能成为 gate 判定的证据（如 E1 → G1，E5 → G2）。

### 7.4 当前状态快照（2026-07-24）

| Gate | 状态 | 周次 |
|---|---|---|
| G0 | not_started | W1–W2 |
| G1 | not_started | W6 |
| G2 | not_started | W9–W10 |
| G3 | not_started | W7–W8 |
| G4 | not_started | W9–W10 |
| G5 | not_started | W7–W8 |

| Experiment | 状态 | 周次 |
|---|---|---|
| E1 | not_started | W7 |
| E2 | not_started | W7–W8 |
| E3 | not_started | W9, W11 |
| E4 | not_started | W11 |
| E5 | not_started | W11 |
| E6 | not_started | W12 |
| E7 | not_started | W12 |

---

## 8. 与 ccfa.yaml 的对应关系

本计划的 gate/experiment 定义、状态、依赖、失败动作、回退路线均同步至 `ccfa.yaml` 的 `gates`、`experiments`、`routes`、`status_schema` 字段。状态变更时应同时更新本文件第 7.4 节快照与 `ccfa.yaml` 对应字段。
