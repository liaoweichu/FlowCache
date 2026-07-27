# G3 实验设计：Lossless Residency（无损驻留）

> **项目**：FlowCache — 复用价值–保真风险解耦的前缀缓存
> **实验 ID**：G3（Lossless Residency）
> **关联 Gate**：G3（IDEA.rewritten.md §7）
> **前置条件**：G1′ PASSED（2026-07-27，headroom=45.80% @ 1 GiB c=4）
> **硬件**：本地 RTX 4090D 24GB + 云端 AutoDL Linux
> **创建日期**：2026-07-27
> **状态**：designed — 等待执行
> **依据**：`experiments/experiment-designs.md` §G3.1–G3.13

---

## 1. 实验目标与 Gate 关系

### 1.1 核心问题

G1′ 已验证物理前缀复用机会存在（Oracle-Cost 比 best simple heuristic 好 45.80%）。G3 要回答：

> **在只用无损动作（GPU BF16 ↔ CPU BF16 ↔ Evict，动作空间 A₀）时，一个简单的价值感知 controller 是否已经优于最强简单淘汰策略（GDSF / SizeCost-LRU）？**

G3 是联合 controller 的"地基"——如果连无损驻留都没有净收益，加入精度维度（G4/G2）也没有意义（experiment-designs.md §G3.1）。

### 1.2 G3 不证明的内容

- 不涉及任何量化精度决策（G4）；
- 不证明复用价值与保真风险错位（G2）；
- 不要求学习式预测器（G5 已删除；G3 使用 heuristic/survival 级别的 reuse 估计）。

### 1.3 G3 通过条件（IDEA §7 G3，对应 §G3.8）

| 条件 | 阈值 |
|---|---|
| 开销可行性 | 恢复 + 迁移开销 < 所节省 prefill（聚合层面成立；逐 block 违反比例如实报告） |
| 主收益 | 固定质量下 p95 TTFT 改善 ≥ ~15%（主 cell：1 GiB c=4 必须达标，其余 cell 报告趋势） |
| 吞吐非劣 | 吞吐下降 ≤ ~5% |
| 优于强启发式 | controller 显著优于 SizeCost-LRU / GDSF（bootstrap CI 不含 0） |

**全部满足 → G3 PASSED。** 任一关键条件失败 → 失败动作（§9）。

### 1.4 运行方式（IDEA §7 G3）

分两时点判定：

| 时点 | 内容 | 用途 |
|---|---|---|
| **W8 冒烟** | 主 cell × 4 无损对照（No-Cache、APC-LRU、GDSF、FlowCache-Lossless）× 100 workflow 子集 | pilot run，防止无损驻留不成立时白做量化 |
| **最终判定** | 9 cell 全量 + 6 对照 × 495 episodes × 3 replay 种子 | 阈值确认 |

最终阈值判定复用 Ch.4 主表的无损对照行（experiment-designs.md §Ch.4）。

---

## 2. 数据集与目标操作点

### 2.1 数据集

| 数据集 | 样本数 | 来源 |
|---|---|---|
| τ-bench | 495 episodes × 3 replay 种子 = 1,485 runs | 复用 G1′ 物理前缀访问流，不重新运行对话 |

> **v0.5 注**：原 BFCL v3 multi-turn 800 行已删除——BFCL 不再作为数据集。
> StableToolBench 500 与 Toolathlon 500 作为确认性 workload，若资源允许补充。

### 2.2 目标操作点（基于 G1′ 结果调整）

experiment-designs.md §G3.3 原用 % budget（10%/25%/50%），但 G1′ 已改为绝对 GiB（1/2/4/6 GiB），且发现 % budget 脱离 24GB GPU 现实（10% ≈ 41.4 GiB）。本设计统一用绝对 GiB：

| 容量 | c=1 | c=4 | c=8 | 备注 |
|---|---|---|---|---|
| 1 GiB | — | **主 cell** ✅ | 异常 ⚠️ | G1′ headroom 45.80% @ c=4 |
| 2 GiB | — | 对照 | 对照 | G1′ headroom 34.90% @ c=4，42.66% @ c=8 |
| 4 GiB | — | — | 对照 | G1′ headroom 16.63% @ c=8 |
| 6 GiB | — | — | — | G1′ headroom ≈ 0%，排除 |

**G3 网格**（9 cell）：

| cell | 容量 | 并发 | 角色 |
|---|---|---|---|
| 1 | 1 GiB | c=4 | **主判定 cell**（G1′ 最佳点） |
| 2 | 1 GiB | c=8 | 异常点复核（Oracle 退化是否消失） |
| 3 | 1 GiB | c=1 | 顺序基线对照 |
| 4 | 2 GiB | c=4 | 中等压力 |
| 5 | 2 GiB | c=8 | 高并发中等压力（G1′ 第二佳） |
| 6 | 2 GiB | c=1 | 顺序基线对照 |
| 7 | 4 GiB | c=4 | 低压力 |
| 8 | 4 GiB | c=8 | 低压力高并发 |
| 9 | 4 GiB | c=1 | 顺序基线对照 |

> **注**：原 G3 设计用并发 4/8/16，但 G1′ 仅测了 1/4/8。本设计沿用 G1′ 的 1/4/8，避免重新合成 trace；如需 c=16，后续在 simulate_concurrency.py 补充。

### 2.3 与 experiment-designs.md §G3.3 的差异说明

| 维度 | 原设计 | 本设计 | 理由 |
|---|---|---|---|
| 容量 | 10%/25%/50% budget | 1/2/4 GiB 绝对 | G1′ 已迁移到绝对 GiB，% 脱离 24GB GPU |
| 并发 | 4/8/16 | 1/4/8 | 沿用 G1′ 已合成的 trace，避免重做 |
| episodes | 495 × 3 种子 | 495 × 3 种子 | 一致 |

---

## 3. 动作空间与 Controller

### 3.1 动作空间 A₀（无损，IDEA §2.3）

| 动作 | 含义 | 触发条件 |
|---|---|---|
| `KEEP_GPU` | block 保留在 GPU BF16 | 默认 |
| `MIGRATE_TO_CPU` | block 迁移到 CPU pinned buffer（BF16） | 显存压力 + 估计会被复用 |
| `RESTORE_FROM_CPU` | block 从 CPU 恢复到 GPU | 即将被访问 |
| `EVICT` | block 直接淘汰 | 估计不会被复用或 CPU 满 |

**禁止动作**：任何量化（Q8/Q4）—— G3 只测无损。

### 3.2 Controller 架构（experiment-designs.md §G3.9 步骤 3.2–3.3）

| 组件 | 输入 | 输出 | 实现 |
|---|---|---|---|
| reuse 估计器 | block 特征（位置、role、age、share_count） | R_b 估计值 | heuristic 或 survival/hazard（不用 GNN） |
| 成本查表 | block 字节数、PCIe 状态 | 迁移/恢复 ms | 来自 §4 成本标定 |
| 决策器 | (R_b, 成本, 容量约束) | A₀ 动作 | 加权打分 + 安全水位 |

**决策规则**（heuristic 级别）：

$$\text{score}_b = \hat{R}_b - \lambda \cdot \text{hold\_cost}_b$$

- 按 score 降序保留 GPU 上的前 N 个 block（N = capacity_blocks）
- score > 0 且 CPU 有空间 → MIGRATE_TO_CPU
- score ≤ 0 → EVICT
- λ 在 W8 冒烟上用网格搜索优化

**安全水位**：保留 10% 容量作 buffer，避免 allocator reserved 导致临界 OOM。

**回退机制**：controller 内部异常时回退 SizeCost-LRU 并记录回退次数。

### 3.3 复用估计器（不用 GNN，IDEA §4.3 预测器选择顺序）

第一版用 heuristic：

| 信号 | 公式 | 来源 |
|---|---|---|
| 最近访问 | age = now - last_access_step | G1′ access_trace |
| 共享度 | share_count = distinct workflows accessing b in H | G1′ access_trace |
| 位置 | block_idx（前缀位置越靠前越可能复用） | G1′ access_trace |
| role-type | system/user/assistant/tool | G1′ access_trace |

$$\hat{R}_b = e^{-\beta \cdot \text{age}} \cdot (1 + \alpha \cdot \text{share\_count}) \cdot \text{position\_weight}(\text{block\_idx})$$

- β = 0.005 / step（~200 step 半衰期）
- α = 0.5
- position_weight: block_idx < 10 → 1.5；< 50 → 1.0；否则 0.7

若 heuristic 与 Oracle 仍有明显差距，W8 后升级为 survival/hazard 模型（仍非 GNN）。

---

## 4. 成本标定（experiment-designs.md §G3.6.1）

### 4.1 必须实测的成本项

| 成本项 | 符号 | 标定方法 | 拟合形式 |
|---|---|---|---|
| prefill 成本 | C^res_evict | block_size=16，父前缀长度 0.5K/1K/2K/4K/8K × 并发 1/4/8 实测 prefill ms | 分段线性或查表（记录 R²） |
| GPU→CPU 迁移 | C^place | pinned buffer 上不同字节数 × 并发负载的 D2H 时间 | 线性（截距 + 斜率/byte） |
| CPU→GPU 恢复 | C^res_CPU | 同上，H2D 方向 | 线性 |
| hold 机会成本 | C^hold | 同预算下被挤占 block 的期望 miss cost（oracle 辅助） | 标量/byte·step |
| controller 决策 | C^policy | 单次决策耗时 × 调用频率实测 | 标量/decision |

### 4.2 标定协议

1. **在云端 AutoDL 上运行**（RTX 4090D 24GB + PCIe 4.0 + pinned memory）
2. 每个数据点测 100 次取中位数，记录 P95
3. 拟合 R² ≥ 0.95 否则扩展采样点
4. **冻结**于 `experiments/g3/cost-model.json`，G3/G5/E4 共用，禁止各章各自标定

### 4.3 标定脚本

| 脚本 | 职责 | 运行环境 |
|---|---|---|
| experiments/g3/calibrate_costs.py | 实测 5 类成本，拟合模型 | 云端 |
| experiments/g3/test_calibrate_costs.py | 单元测试 | 本地 mock |

---

## 5. Baseline / 对照（experiment-designs.md §G3.4）

| 对照 | 说明 | 动作空间 |
|---|---|---|
| No-Cache | 下界参照 | 无缓存 |
| APC-LRU | 工程 baseline | GPU only evict |
| GDSF | 强启发式驱逐 | GPU only evict |
| SizeCost-LRU | size-aware LRU | GPU only evict |
| **FlowCache-Lossless** | 待验：value-aware 驻留 controller（A₀） | GPU ↔ CPU ↔ evict |
| Oracle-Cost | 上界参照（未来信息） | GPU only evict |

> **注**：No-Cache / APC-LRU / GDSF / SizeCost-LRU / Oracle-Cost 直接复用 `experiments/e1/compare_oracle.py` 的类。FlowCache-Lossless 是新增实现。

---

## 6. 测试指标（experiment-designs.md §G3.5）

| 类别 | 指标 |
|---|---|
| **主指标** | p95 TTFT（vs GDSF/SizeCost-LRU 的相对改善）、吞吐（req/s）相对变化 |
| 辅助 | TTFT/JCT p50/p99、token/block/byte hit rate、saved-prefill tokens/time |
| 开销 | 恢复时间（CPU→GPU）、迁移时间（GPU→CPU）、H2D/D2H 字节数、controller 单次决策耗时与总开销 |
| 约束 | 任务成功率 = BF16 基线（无损路径**零质量差异**，逐 workflow 核验）、GPU allocated/reserved、CPU pinned bytes |
| 可行性 | 恢复 + 迁移开销 < 所节省 prefill（逐 block 判定，报告违反比例） |

---

## 7. 运行协议（experiment-designs.md §G3.6）

### 7.1 open-loop replay（系统性能）

- 复用 G1′ 的 `access_trace_c{1,4,8}.jsonl`
- 所有策略同引擎、同模型、同 dtype、同预算、同请求顺序
- controller 触发时机：请求到达、暂停、恢复、完成、显存压力变化
- 保留安全水位（10% capacity buffer），避免临界 OOM
- 预测器失效回退：controller 异常时回退 SizeCost-LRU 并记录回退次数

### 7.2 closed-loop 质量抽检（无损零差异）

- 20 episodes closed-loop BF16 vs No-Cache
- 验证：任务成功率差 = 0；输出文本逐 token 一致率 = 100%；工具调用序列一致率 = 100%
- 若任一不一致 → 无损路径有 bug，修复后重跑

---

## 8. 统计检验（experiment-designs.md §G3.7）

- **主比较**：FlowCache-Lossless vs GDSF、vs SizeCost-LRU 的 per-workflow p95 TTFT 配对差
- **bootstrap**：paired workflow-level 1000 次，95% CI
- **多重比较**：9 cell × 2 主比较 = 18 检验，Bonferroni 校正（α = 0.05 / 18）
- **吞吐非劣**：CI 下界 > −5%（非劣式判定）

### 8.1 Go/No-Go 判定逻辑

```
G3 PASSED iff:
  (开销可行性) aggregate(restore_ms + migrate_ms) < aggregate(saved_prefill_ms)
  AND (主收益) p95_TTFT_improvement >= 15% at main cell (1 GiB, c=4)
  AND (吞吐非劣) throughput_drop <= 5% (CI lower > -5%)
  AND (优于强启发式) bootstrap_CI(FlowCache - GDSF) lower > 0
                     AND bootstrap_CI(FlowCache - SizeCost) lower > 0
```

---

## 9. 失败动作（experiment-designs.md §G3.12）

按 IDEA §7 G3：路线 A No-Go，转路线 B；实现保留为工程基线，但不以无损 residency 单独投稿该主张。更新 ccfa.yaml（G3 → failed，route → B）。

---

## 10. 代码结构

| 文件 | 职责 | 运行环境 |
|---|---|---|
| experiments/g3/calibrate_costs.py | 实测 5 类成本，拟合模型 | 云端 |
| experiments/g3/cache_manager.py | 无损三层动作 cache manager（A₀） | 本地 + 云端 |
| experiments/g3/reuse_estimator.py | heuristic/survival reuse 估计器 | 本地 |
| experiments/g3/controller.py | FlowCache-Lossless controller（含安全水位与回退） | 本地 |
| experiments/g3/run_g3_grid.py | 9 cell × 6 对照 × 495 episodes × 3 种子全网格 | 云端 |
| experiments/g3/g3_verdict.py | G3 判定报告（含 bootstrap CI） | 本地 |
| experiments/g3/plot_g3.py | G3 结果绘图（p95 TTFT × 预算 × 并发） | 本地 |
| experiments/g3/cost-model.json | 冻结的成本模型（标定后生成） | 共用 |
| experiments/g3/config.yaml | G3 配置 | — |

---

## 11. 执行步骤（experiment-designs.md §G3.9）

| 步骤 | 内容 | 环境 | 产物 |
|---|---|---|---|
| 11.1 | 成本标定（云端实测 5 类成本） | 云端 | `cost-model.json` |
| 11.2 | 实现无损三层动作 cache manager + 迁移路径 | 本地 | `cache_manager.py` |
| 11.3 | 实现 heuristic reuse 估计器 | 本地 | `reuse_estimator.py` |
| 11.4 | 实现 FlowCache-Lossless controller（含安全水位与回退） | 本地 | `controller.py` |
| 11.5 | W8 冒烟：主 cell × 4 对照 × 100 workflow 子集 | 云端 | 冒烟结果 |
| 11.6 | 9 cell × 6 对照 × 495 episodes × 3 种子全网格运行 | 云端 | `raw_results.csv` |
| 11.7 | 无损质量抽检（20 episodes closed-loop） | 云端 | 质量核验记录 |
| 11.8 | 统计判定与绘图 | 本地 | `g3-verdict.md`、`figures/g3-*.png` |
| 11.9 | 文档同步（冻结 G3，更新 IDEA/ccfa） | 本地 | 更新后的文档 |

---

## 12. 预期产物（experiment-designs.md §G3.11）

| 产物 | 路径 |
|---|---|
| 无损 cache manager + controller v1 | `experiments/g3/` |
| 成本模型 | `experiments/g3/cost-model.json` |
| 结果表与图（p95 TTFT × 预算 × 并发） | `experiments/g3/results/`、`figures/g3-*.png` |
| 判定报告 | `experiments/g3/g3-verdict.md` |

### 12.1 结果表格模板

**表 G3-1：主 cell（1 GiB、c=4，τ-bench 495 episodes × 3 种子）**

| 策略 | p95 TTFT (ms) | vs FlowCache 相对差 | 吞吐 (req/s) | 吞吐相对变化 | block hit | saved-prefill ms | controller 开销 ms |
|---|---|---|---|---|---|---|---|
| No-Cache | TBD | TBD | TBD | TBD | TBD | TBD | — |
| APC-LRU | TBD | TBD | TBD | TBD | TBD | TBD | — |
| GDSF | TBD | TBD | TBD | TBD | TBD | TBD | — |
| SizeCost-LRU | TBD | TBD | TBD | TBD | TBD | TBD | — |
| FlowCache-Lossless | TBD | — | TBD | — | TBD | TBD | TBD |
| Oracle-Cost | TBD | TBD | TBD | TBD | TBD | TBD | — |

**表 G3-2：全 9 cell 摘要**

| 容量 | 并发 | FlowCache p95 TTFT | 最佳简单策略 p95 TTFT | 相对改善 [95% CI] | 吞吐变化 [95% CI] | 判定达标? |
|---|---|---|---|---|---|---|
| 1 GiB | c=1 | TBD | TBD | TBD | TBD | TBD |
| 1 GiB | c=4（主 cell） | TBD | TBD | TBD | TBD | TBD |
| 1 GiB | c=8 | TBD | TBD | TBD | TBD | TBD |
| 2 GiB | c=1 | TBD | TBD | TBD | TBD | TBD |
| 2 GiB | c=4 | TBD | TBD | TBD | TBD | TBD |
| 2 GiB | c=8 | TBD | TBD | TBD | TBD | TBD |
| 4 GiB | c=1 | TBD | TBD | TBD | TBD | TBD |
| 4 GiB | c=4 | TBD | TBD | TBD | TBD | TBD |
| 4 GiB | c=8 | TBD | TBD | TBD | TBD | TBD |

**表 G3-3：无损质量抽检（closed-loop，20 episodes）**

| 检查项 | 结果 |
|---|---|
| BF16 缓存路径 vs 无缓存的任务成功率差 | TBD（要求 = 0） |
| 输出文本逐 token 一致率 | TBD |
| 工具调用序列一致率 | TBD |

---

## 13. 与 G1′ 的衔接

| 维度 | G1′（已完成） | G3（本设计） |
|---|---|---|
| 问题 | 复用机会是否存在 | 无损驻留是否优于简单淘汰 |
| 动作 | 无淘汰 vs Oracle 淘汰 | GPU ↔ CPU ↔ evict（A₀） |
| 数据 | 1320 episodes 物理前缀访问流 | 复用 G1′ 访问流，495 × 3 种子 |
| 容量 | 1/2/4/6 GiB | 1/2/4 GiB（聚焦通过 cell） |
| 并发 | c=1/4/8 | c=1/4/8 |
| Baseline | LRU/GDSF/SizeCost/APC-LRU + Oracle | + No-Cache + FlowCache-Lossless |
| 通过条件 | headroom ≥ 10% AND CI > 0 | p95 TTFT ≥ 15% AND 吞吐 ≤ 5% AND CI > 0 |
| 通过后 | → G3 | → G4（量化） → G2（联合 R-D） |

G3 通过后，controller 框架与成本模型将直接被 G4（量化）和 G2（联合 R-D）复用。
