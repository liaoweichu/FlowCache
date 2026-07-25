# FlowCache 实验体系精简设计（v0.3）

> **目标**：在 14 周硬约束内完成实验，聚焦三条核心 claim（C1 trace 协议 / C2 联合控制器 / C3 reuse–fidelity 错位实证）的最小充分证据包。
> **上游文档**：IDEA.rewritten.md、experiments/experiment-designs.md（v0.2）、experiments/g2-pilot-design.md
> **取代范围**：本设计重构 experiment-designs.md 的章节体系；g2-pilot-design.md 保持有效；IDEA §7 的 Gate 判定逻辑不变，但运行方式改为复用正式实验数据。
> **创建日期**：2026-07-25
> **状态**：approved-pending-implementation

---

## 1. 背景与动机

### 1.1 v0.2 现状

experiment-designs.md v0.2 包含 6 个 Gate（G0–G5）+ 7 个正式实验（E1–E7），每个有 13 个子节的完整设计：

- 12+ 数据集，~8,800 个 workflow 级样本（v0.2 因"禁合成数据"从 ~880 膨胀到 ~8,800）
- 仅 E4 主表：18 cell（3 预算 × 3 并发 × 2 workload）× 13 个对照 × 3 seeds = ~702 replay
- E5 有 10 个消融轴，E6 有 8 个扰动轴，E7 有 8 种失败模式
- 仅 Tier-1 轨迹录制就要 33–40 GPU 小时

### 1.2 问题

1. **Gate 与 E 实验大量共享数据却按独立运行设计**——同一 trace 被 G1、E1、G3、E4 分别回放
2. **对照/消融轴数量远超 ICWS 审稿人验证核心 claim 所需**——13 个对照中多数功能重叠
3. **IDEA §6.1 自己写明"只保留两个主工具 workload"，但 v0.2 数据集体系已偏离**——主表混入 StableToolBench、SWE、Toolathlon 等
4. **14 周硬约束下无法排下**——按 v0.2 规模推算，W11 主实验冻结窗口至少需要 3–4 周而非 1 周

### 1.3 用户约束（brainstorming 阶段确认）

- 主要约束：审稿焦点/论文叙事 + 14 周日历时间硬约束
- 对既有 experiment-necessity-review.md 的态度：重新独立分析（不沿用其结论）
- 方案选择：A（claim 驱动重构）+ C（两阶段门控执行顺序）

---

## 2. 三条原则

1. **一次运行，多处消费**——Gate 判定不再独立运行，全部复用正式实验的 trace/数据。判定逻辑不变（阈值、统计检验照旧），但数据来源从"独立 Gate 实验"改为"正式实验的子集或同 trace"。
2. **样本量按功效反推**——E1 与 G2-Pilot 先行，用实测效应量（ρ、headroom）标定主实验规模。上限封顶为 workload 全量（τ-bench 495 + BFCL 800），下限为功效分析所需最小 N。
3. **核心 4 变体贯穿**——reuse-only / fidelity-only / decoupled-best / joint 既是主表行也是消融主体，不单设消融章节。其他消融轴作为主表的附加列或附录表。

---

## 3. 精简后的实验体系：13 章 → 5 章 + 2 个小判定

| 章 | 内容 | 支撑 claim | 数据来源 | 对应 v0.2 章节 |
|---|---|---|---|---|
| **Ch.1 工作负载画像** | overlap/LCP/next-use/working-set 画像 + oracle headroom；G1 判定复用此数据 | C1+C3 前提 | τ-bench 495 + BFCL 800 | E1 + G1 |
| **Ch.2 R–D 错位 Pilot** | 80 workflow 子集，Spearman ρ + 四象限，GO/NO-GO 判定路线 A | C3 存在性 | τ-bench 80 子集 | G2-Pilot（保持现有设计） |
| **Ch.3 估计器有效性** | reuse 侧 2 变体（heuristic vs survival）；fidelity 侧 2 变体（uniform vs norm/range proxy）；GNN 删除 | C2 的两半 | reuse 复用 Ch.1 trace；fidelity 用 LongBench 1000 + GSM8K 100 | E2 + E3 合并 |
| **Ch.4 端到端主结果** | 10 对照 × 6 cell；核心 4 变体 + 2 个设计消融（无 parent-closure、无 CPU tier）同表；开销透明账并入表列 | C2+C3 核心 | τ-bench 495 + BFCL 800 | E4 + E5 核心合并 |
| **Ch.5 鲁棒性与失败分析** | 3 轴：family-out、到达扰动、branch 噪声；失败模式从 Ch.4 负结果 cell 提取 | C3 系统性 | STB 500（family-out）+ SWE 200 + Toolathlon 200（压力面） | E6 + E7 合并 |

### 3.1 保留的 2 个小判定

- **G0（正确性准入，W1–W2）**：保持不变。这是基础设施判定，规模小且必须独立于正式实验。
- **G3 冒烟（W8）**：E4 主 cell × 4 个无损对照（No-Cache、APC-LRU、GDSF、Reuse-Only）× ~100 workflow 子集。本质是主表的 pilot run，防止无损驻留不成立时白做量化。G3 的完整阈值判定（p95 TTFT 改善 ~15%、吞吐 ≥ −5%、优于 size-aware LRU/GDSF）用 Ch.4 主表的无损对照行结果做最终确认。

### 3.2 删除的 Gate

- **G5（Learning）删除**：GNN 不启用是设计选择。论文主张"简单可解释 controller 足够"，这本身可写为发现（见 §7.2）。简单 controller 成为默认而非"gate 失败"。
- **G1、G2、G4 不再独立运行**：判定逻辑保留，数据来源改为复用正式实验。
  - G1：用 Ch.1 画像数据判定 oracle headroom ≥ 10% + closest baseline 可比性
  - G2：用 Ch.2 Pilot 数据判定 R–D 错位；最终的"joint > 解耦组合"用 Ch.4 主表判定
  - G4：用 Ch.3 fidelity 侧数据判定量化非劣 + 端到端不破坏延迟

---

## 4. Ch.4 端到端主表瘦身

### 4.1 对照 13 → 10

| # | 对照 | 说明 |
|---|---|---|
| 1 | No-Cache | cold recompute 下界 |
| 2 | APC-LRU | 同引擎实际 APC |
| 3 | GDSF | 强启发式代表（合并 LFU、LRU-K/2Q 的角色） |
| 4 | KVFlow† 或 PBKV† | ≥1 个可公平运行的 closest baseline |
| 5 | Uniform-Q8 | 统一 Q8（Q4 仅出现在 Ch.3 fidelity 侧，不进主表） |
| 6 | Reuse-Only | 核心变体 1：复用价值驱动驻留 + 统一精度 |
| 7 | Fidelity-Only | 核心变体 2：保真风险驱动精度 + 强启发式驻留 |
| 8 | **Decoupled-Best** | 核心变体 3：最强"reuse policy + uniform quantization"解耦组合 |
| 9 | **FlowCache-Joint** | 核心变体 4：待验联合 policy |
| 10 | Oracle-Cost | 离线上界 |

**删除**：LFU、LRU-K/2Q（GDSF 代表简单启发式）；Uniform-Q4（移入 Ch.3）。

### 4.2 cell 18 → 6

| cell | 预算 | 并发 | workload | seeds |
|---|---|---|---|---|
| 主-1 | 25% | 8 | τ-bench | 3 |
| 主-2 | 25% | 8 | BFCL | 3 |
| 主-3 | 50% | 8 | τ-bench | 1 |
| 主-4 | 50% | 8 | BFCL | 1 |
| 边界-1 | 10% | 16 | τ-bench | 1 |
| 边界-2 | 10% | 16 | BFCL | 1 |

**删除**：并发 4 档（与 8 重叠）、100% 预算档（上界参照，不进主表）。

### 4.3 运行量对比

- v0.2：18 cell × 13 对照 × 3 seeds = ~702 replay
- v0.3：10 对照 × 6 cell，其中主-1/主-2 各 3 seeds = 10×(4×1 + 2×3) = **100 replay**

### 4.4 设计消融并入主表

以下 2 个设计消融作为主表的附加列（同一 cell，仅切开关）：

- **无 parent-closure**：后继 block 在父 block 不可用时仍计为可复用
- **无 CPU tier**：仅 GPU + evict，禁用 CPU offload

v0.2 E5 的其余消融轴（无 partial DAG、无成本校准、静态阈值 vs 动态预算、不同 controller 更新频率）降级为附录表或移除。

---

## 5. 数据集体系：12+ → 7 核心 + 2 辅助

**7 个核心数据集**（计入样本总量）：

| 角色 | 数据集 | 样本 | 用途 |
|---|---|---|---|
| 主表 | τ-bench | 1,320（165 任务 × 8 seeds） | Ch.1/2/3/4 |
| 主表 | BFCL v3 multi-turn | 800 | Ch.1/3/4 |
| fidelity 质量面 | LongBench | 1,000 | Ch.3 |
| fidelity 质量面 | GSM8K | 100 | Ch.3（accuracy sanity） |
| family-out（仅 Ch.5） | StableToolBench | 500 | Ch.5 |
| 压力面（仅 Ch.5） | SWE 轨迹 | 200 | Ch.5 |
| 压力面（仅 Ch.5） | Toolathlon | 200 | Ch.5 |

**τ-bench seeds 数依据（2026-07-25 调研后冻结）**：τ-bench 原论文（arXiv 2406.12045, ICLR 2025）主表用 165 任务全量，pass^k 指标用 k∈{1,2,4,8}。3 seeds 只能算 pass^3，统计上不足以区分 consistency；8 seeds 与原论文 pass^8 完全对齐，最稳健。增量成本：165 × 5 = 825 episodes（相比 3 seeds 多 660 episodes），按 4090D ~30s/episode 估算约 5.5 GPU 小时。

**2 个辅助角色**（不计入样本总量）：

| 角色 | 数据集 | 用途 |
|---|---|---|
| 到达证据 | BurstGPT 窗口 | Ch.4 到达结构（replay 参数，不产生 workflow 样本） |
| 负对照 | LMSYS-Chat-1M | Ch.1 画像附注（一段话描述，500 会话仅做顺序式多轮的 exact-prefix overlap 对照） |

**删除**：CATraces（可得性 TBD）、Mooncake（窗口 TBD）、MuSiQue、2WikiMultihopQA（多跳 QA 质量 sanity 由 LongBench 的 QA 子任务覆盖）。

**核心样本总量**：~8,800 → **~4,120**（1320+800+1000+100+500+200+200）。相比 v0.2 降幅 53%。

---

## 6. 执行顺序（C 方案嵌入）

```
W1–W2   G0（不变）
W3–W5   τ-bench 495 + BFCL 800 轨迹录制
W6–W7   Ch.1 画像 + G1 判定（复用 trace）
W7–W8   Ch.2 Pilot + Ch.3 reuse 侧（并行）→ G2 存在性判定
W8      G3 冒烟（主 cell × 4 无损对照 × 100 子集）
W9      Ch.3 fidelity 侧 + G4 判定（复用其数据）
W9 末   用实测效应量标定 Ch.4 最终样本量（封顶 495/800）
W10–W11 Ch.4 主表（G2/G3 最终确认复用主表）
W12     Ch.5 鲁棒性（STB 500 录制在此窗口）
W13     复跑冻结 / W14 写作
```

### 6.1 样本量标定规则

W9 末根据 Ch.2 Pilot 实测的 Spearman ρ 和 Ch.1 画像的 oracle headroom，按功效分析反推 Ch.4 所需样本量：

- 若 ρ < 0.4（错位显著）：主 cell 用 τ-bench 200 + BFCL 300 即可达 80% 功效检测中等效应
- 若 0.4 ≤ ρ ≤ 0.7（灰区）：主 cell 用全量 495 + 800，并要求 joint vs Decoupled-Best 净收益 ≥ 5%
- 若 ρ > 0.7（NO-GO）：路线 A 停止，转路线 B（不进入 Ch.4）

封顶为 workload 全量，不超量运行。

---

## 7. 明确放弃的东西

### 7.1 实验范围

- **GNN 预测器**：论文主张"简单可解释 controller 足够"。G5 删除。
- **第三主 workload 进主表**：StableToolBench 仅作 family-out（Ch.5），不进入 Ch.4 主表。论文的"two tool families"证据来自 τ-bench + BFCL。
- **E7 独立章**：开销账并进 Ch.4 主表列，失败模式从 Ch.4 负结果 cell 提取。
- **多跳 QA 专用质量数据集**：MuSiQue、2WikiMultihopQA 删除，LongBench 的 QA 子任务覆盖。
- **Uniform-Q4 进主表**：Q4 仅在 Ch.3 fidelity 侧作为激进精度对照。

### 7.2 GNN 删除的论文叙事

GNN 删除不是"gate 失败"，而是设计选择。论文可写为发现：

> 在我们的 workload 上，校准的 survival/hazard 模型与 partial-DAG GNN 在 saved-prefill ms 和 policy regret 上的差距小于 GNN 自身的推理开销；因此 FlowCache 默认采用简单可解释的 controller。这一发现与 IDEA §4.3 的预测器选择顺序一致。

### 7.3 鲁棒性篇幅变薄的风险

Ch.5 仅 3 轴（family-out、到达扰动、branch 噪声），相比 v0.2 E6 的 8 轴大幅缩减。审稿人可能要求补实验。

**应对**：rebuttal 时用预留的 SWE/Toolathlon 余量（各 200 已录制，可扩展到 500）补 CPU 带宽竞争、predictor calibration drift 等轴。接受此风险。

---

## 8. 规模对比

| 维度 | v0.2 现状 | v0.3 精简后 | 降幅 |
|---|---|---|---|
| 章节 | 6 Gate + 7 E = 13 章 | 5 章 + 2 小判定 | 61% |
| 核心数据集 | 12+ | 7 | 42% |
| workflow 样本 | ~8,800 | ~4,120 | 53% |
| E4 replay 运行 | ~702 | ~100 | 86% |
| 独立 Gate 运行 | 6 次 | 0（全部复用） | 100% |
| Tier-1 轨迹录制 | 33–40 GPU 小时 | ~26 GPU 小时 | 35% |

---

## 9. 与 IDEA 的对应关系

| IDEA 要素 | v0.3 处理 |
|---|---|
| §6.1 "14 周版本只保留一个主模型和两个主工具 workload" | 主表严格 τ-bench + BFCL；STB/SWE/Toolathlon 仅 Ch.5 |
| §7 G0–G5 Gate | G0 保留；G1/G2/G4 复用正式实验数据判定；G3 用 Ch.4 无损行；G5 删除 |
| §8 E1–E7 | 合并为 5 章；E2+E3→Ch.3；E4+E5 核心→Ch.4；E6+E7→Ch.5 |
| §9 三条贡献 | C1←Ch.1；C2←Ch.3+Ch.4；C3←Ch.2+Ch.4+Ch.5 |
| §11 路线 A/B 切换 | 切换条件不变（G0/G1/G2/G3/G4 任一失败）；G5 不再触发切换 |
| §14 写作就绪条件 | 全部满足：G0–G3 通过、G1 证明决策空间、close baseline 可比、G2+G4 通过、Pareto 主图+CI、贡献对应真实结果 |

---

## 10. 后续动作

1. 本设计获用户批准后，调用 writing-plans skill 创建实施计划
2. 实施计划将重构 experiment-designs.md 为 v0.3（5 章结构），并更新 ccfa.yaml 的 Gate 状态字段
3. g2-pilot-design.md 保持有效，不重构
4. execution-plan.md 的周次安排按 §6 更新
