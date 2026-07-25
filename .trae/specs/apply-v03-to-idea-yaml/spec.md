# Apply v0.3 Scope Redesign to IDEA and ccfa.yaml Spec

## Why

`experiment-scope-redesign/spec.md`（v0.3）已批准，将实验体系从 13 章（6 Gate + 7 E）精简为 5 章 + 2 个小判定。但 `IDEA.rewritten.md` 和 `ccfa.yaml` 仍反映 v0.2 的 13 章体系，导致上游源文档与已批准设计不一致，后续实施会基于过时信息。

## What Changes

### IDEA.rewritten.md

- **§6.1 最小 workload 组合**：主表严格限定 τ-bench + BFCL 两个主工具 workload；StableToolBench、SWE 轨迹、Toolathlon 降级为 Ch.5 鲁棒性压力面；CATraces、Mooncake、MuSiQue、2WikiMultihopQA 从 workload 组合中删除
- **§7 可行性门槛**：G5（Learning）标记为已删除（设计选择，非失败）；G1/G2/G4 标注运行方式改为复用正式实验数据；G3 增加"冒烟"前置（W8）与主表最终确认两时点
- **§8 正式实验计划**：E1–E7 章节重构为 Ch.1–Ch.5 描述（画像/Pilot/估计器/主表/鲁棒性），保留原文档的指标和成功标准，但按 v0.3 的对照/cell/数据集规模重写
- **§11 路线切换**：route A 的 requires_gates 去掉 G5；G5 删除不再触发路线切换
- **§12 14 周执行计划**：周次表按 v0.3 §6 执行顺序更新（W7–W8 Pilot 提前、W8 G3 冒烟、W9 末样本量标定、W10–W11 主表、W12 鲁棒性）
- **§14 写作就绪条件**：移除 G5 相关条件；G2/G4 通过条件不变

### ccfa.yaml

- **gates**：删除 G5 条目；G1/G2/G3/G4 的 `design_doc` 更新指向新章节锚点；`week` 按 v0.3 更新；G3 增加 `smoke_check` 字段描述 W8 冒烟
- **experiments**：E1–E7 重构为 Ch.1–Ch.5 条目，更新 `description`/`week`/`gates`/`design_doc`
- **routes[A].requires_gates**：已经是 `["G0","G1","G2","G3","G4"]`，无需改动（G5 本不在其中）
- **stage**：`updated_at` 更新；新增 `scope_spec` 字段指向 `experiment-scope-redesign/spec.md`
- **claims**：不变（C1/C2/C3 不变）

## Impact

- Affected specs: `experiment-scope-redesign`（上游设计，本变更是其落地）
- Affected code: 无代码改动；仅文档与配置
  - `IDEA.rewritten.md`（§6.1, §7, §8, §11, §12, §14）
  - `ccfa.yaml`（gates, experiments, stage 段）
- 不影响：`experiments/g2-pilot-design.md`（保持有效）、`experiments/experiment-designs.md`（将在后续单独 spec 中重构为 v0.3 5 章结构）、`execution-plan.md`（将在后续单独 spec 中更新）

## MODIFIED Requirements

### Requirement: IDEA §6.1 最小 workload 组合

原 §6.1 列出 5 个 workload 角色（主 A/B、合成 DAG、QA、LMSYS 负对照），并在末段说明"14 周版本只保留一个主模型和两个主工具 workload"。v0.3 进一步收紧：

- **主表 workload**：τ-bench（495）+ BFCL v3 multi-turn（800），严格两个
- **Ch.5 鲁棒性压力面**：StableToolBench（500，family-out）、SWE 轨迹（200）、Toolathlon（200）
- **Ch.3 fidelity 质量面**：LongBench（1000）+ GSM8K（100）
- **辅助**：BurstGPT 窗口（到达证据，不计样本）、LMSYS-Chat-1M（500，负对照附注）
- **删除**：CATraces、Mooncake、MuSiQue、2WikiMultihopQA、合成 DAG（已有用户禁令）

合成 DAG 的结构因果敏感性角色由 SWE 轨迹分层分析替代（已在 v0.2 §0.4.6 映射，v0.3 保留此映射）。

### Requirement: IDEA §7 可行性门槛

- **G0**：不变
- **G1**：判定逻辑不变；数据来源改为复用 Ch.1 画像数据；`design_doc` 指向 `experiments/experiment-designs.md#ch1`
- **G2**：判定逻辑不变；R–D 错位用 Ch.2 Pilot 数据；"joint > 解耦组合"用 Ch.4 主表判定；`design_doc` 指向 `experiments/g2-pilot-design.md` + `experiments/experiment-designs.md#ch4`
- **G3**：增加 W8 冒烟前置（主 cell × 4 无损对照 × 100 子集）；最终阈值判定用 Ch.4 主表无损对照行；`design_doc` 指向 `experiments/experiment-designs.md#ch4`
- **G4**：判定逻辑不变；数据来源改为复用 Ch.3 fidelity 侧数据；`design_doc` 指向 `experiments/experiment-designs.md#ch3`
- **G5**：**删除**。GNN 不启用是设计选择；论文主张"简单可解释 controller 足够"

### Requirement: IDEA §8 正式实验计划

E1–E7 重构为 Ch.1–Ch.5：

- **Ch.1 工作负载画像**（原 E1+G1）：τ-bench 495 + BFCL 800；overlap/LCP/next-use/working-set 画像 + oracle headroom
- **Ch.2 R–D 错位 Pilot**（原 G2-Pilot）：τ-bench 80 子集；Spearman ρ + 四象限；保持 `g2-pilot-design.md` 不变
- **Ch.3 估计器有效性**（原 E2+E3）：reuse 侧 2 变体（heuristic vs survival）；fidelity 侧 2 变体（uniform vs norm/range proxy）；GNN 删除；fidelity 用 LongBench 1000 + GSM8K 100
- **Ch.4 端到端主结果**（原 E4+E5 核心）：10 对照 × 6 cell；核心 4 变体 + 2 设计消融（无 parent-closure、无 CPU tier）同表；开销透明账并入表列
- **Ch.5 鲁棒性与失败分析**（原 E6+E7）：3 轴（family-out、到达扰动、branch 噪声）；失败模式从 Ch.4 负结果 cell 提取；STB 500 + SWE 200 + Toolathlon 200

### Requirement: IDEA §12 14 周执行计划

周次表更新为：

| 周次 | 目标 | Gate / 产物 |
|---|---|---|
| W1–W2 | 冻结模型/后端/主机；Q-storage codec spike | G0 |
| W3–W5 | τ-bench 495 + BFCL 800 轨迹录制 | 可重放 trace |
| W6–W7 | Ch.1 画像 + G1 判定（复用 trace） | E1 画像 |
| W7–W8 | Ch.2 Pilot + Ch.3 reuse 侧（并行）→ G2 存在性判定 | G2 Pilot |
| W8 | G3 冒烟（主 cell × 4 无损对照 × 100 子集） | G3 冒烟 |
| W9 | Ch.3 fidelity 侧 + G4 判定 | G4 |
| W9 末 | 用实测效应量标定 Ch.4 样本量（封顶 495/800） | 样本量冻结 |
| W10–W11 | Ch.4 主表（G2/G3 最终确认复用主表） | E4 主表 |
| W12 | Ch.5 鲁棒性（STB 500 录制在此窗口） | E5 鲁棒性 |
| W13 | 复跑冻结 / W14 写作 | 冻结结果 |

### Requirement: ccfa.yaml gates/experiments/stage

- `gates`：删除 G5；G1/G2/G3/G4 的 `design_doc` 和 `week` 更新；G3 增加 `smoke_check` 子字段
- `experiments`：E1–E7 → Ch.1–Ch.5；更新 `description`/`week`/`gates`/`design_doc`
- `stage`：`updated_at` 更新为 `2026-07-25`；新增 `scope_spec: ".trae/specs/experiment-scope-redesign/spec.md"`

## REMOVED Requirements

### Requirement: IDEA §7 G5 Learning Gate

**Reason**: v0.3 将 GNN 不启用定为设计选择而非 gate 失败。论文主张"简单可解释 controller 足够"，G5 的存在会暗示 GNN 是必需的。
**Migration**: G5 条目从 §7 和 ccfa.yaml 中删除。§4.3 的预测器选择顺序（heuristic → survival → GNN 仅在前两者与 oracle 仍有明显差距时）保留作为方法描述，但不再作为 gate。论文可将"简单 controller 足够"写为发现。

### Requirement: IDEA §8 E1–E7 独立实验章节

**Reason**: v0.3 合并为 5 章，E2+E3→Ch.3、E4+E5 核心→Ch.4、E6+E7→Ch.5。
**Migration**: E1–E7 的指标和成功标准保留并迁入对应 Ch.x 章节；`experiments/experiment-designs.md` 的 13 子节结构将在后续单独 spec 中重构为 5 章，本变更只更新 IDEA 和 yaml 的引用。
