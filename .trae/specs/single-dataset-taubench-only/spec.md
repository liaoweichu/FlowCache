# 单数据集 τ-bench 1,320 episodes Spec

## Why

FlowCache 当前规划 7 核心数据集（τ-bench 495 + BFCL 800 + LongBench 1000 + GSM8K 100 + StableToolBench 500 + SWE 200 + Toolathlon 200，核心样本总量 ~3,300）。用户要求将数据集全部修改为 τ-bench 1,320 episodes（165 tasks × 8 seeds）单一数据集，与 τ-bench 原论文（ICLR 2025）的 pass^k 评测方法论完全对齐，与同领域 CCF-A/B 论文样本量中位数（~1,320）一致，并将 4090D 上的 GPU 机时从 ~14 小时进一步压缩到 ~7 小时，聚焦 14 周日历预算内完成主表。

## What Changes

- **BREAKING**: 主表 workload 从 τ-bench 495 + BFCL 800 双数据集改为 τ-bench 1,320 episodes（165 tasks × 8 seeds）单数据集
- **BREAKING**: 删除 BFCL v3、LongBench、GSM8K、StableToolBench、SWE 轨迹、Toolathlon 6 个数据集
- **BREAKING**: Ch.3 fidelity 质量面从 LongBench 1000 + GSM8K 100 改为复用 τ-bench 1,320 episodes（量化质量直接在主 workload 上验证）
- **BREAKING**: Ch.5 鲁棒性从 3 轴（family-out / 到达扰动 / branch 噪声）缩减为 1 轴（到达扰动），删除 family-out 和 branch 噪声轴
- τ-bench 任务数从 495（165 × 3 seeds）改为 1,320（165 × 8 seeds），与原论文 pass^k 对齐
- 核心样本总量从 ~3,300 降至 1,320（降幅 60%）
- 删除 LMSYS-Chat-1M 负对照附注（不再需要多 workload 对照）
- BurstGPT 窗口保留为到达证据（Ch.4 replay 参数），不产生 workflow 样本

## Impact

- Affected specs:
  - [experiment-scope-redesign/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/experiment-scope-redesign/spec.md) §5 样本量封顶
  - [trim-dataset-portfolio/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/trim-dataset-portfolio/spec.md): 全册核心样本总量从 ~3,720 降至 1,320
  - [reconsider-g1-sample-size/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/reconsider-g1-sample-size/spec.md): BFCL 单 seed 决策作废，BFCL 整体删除
  - [gsm8k-only-feasibility-analysis/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/gsm8k-only-feasibility-analysis/spec.md): GSM8K 整体删除
  - [experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.4 数据集组合、Ch.1/2/3/4/5 各章数据来源、§0.8 功效分析、§12 周次表
- Affected code:
  - [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml): workload.datasets 改为 ["tau-bench"]，删除 bfcl_v3 配置，seeds 保持 8 个
  - [experiments/e1/record_trajectories.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/record_trajectories.py): 删除 _record_bfcl_g1 路径或保留但默认不调用
  - [experiments/e1/bfcl_adapter.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/bfcl_adapter.py): 保留代码但不在主流程调用
  - [experiments/g0/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/g0/config.yaml): 无变更（G0 结构用例不依赖数据集规模）
  - [ccfa.yaml](file:///d:/00MyProject/Prefix%20Caching/ccfa.yaml): 无直接数据集字段，但 stage.week 可能调整
- 受益：
  - GPU 机时从 ~14 小时降至 ~7 小时（4090D），14 周日历预算更宽裕
  - 与 τ-bench 原论文 pass^k (k≤8) 方法论完全对齐，无需解释 BFCL 单 seed 的方法论不对等问题
  - 样本量（1,320）与同领域 CCF-A/B 明确披露类中位数（~1,320）完全一致
  - 消除 BFCL/GSM8K/LongBench/STB/SWE/Toolathlon 6 个数据集的集成维护负担
- 风险：
  - workflow-family 泛化证据减弱（原 τ-bench↔BFCL 交叉验证改为 τ-bench 内 retail↔airline 两域对照）
  - fidelity 质量面在长上下文任务上的覆盖减弱（τ-bench 平均上下文长度 < LongBench）
  - family-out 鲁棒性轴消失，rebuttal 时若审稿人要求跨工具家族证据需补做

## ADDED Requirements

### Requirement: τ-bench 单数据集 1,320 episodes

FlowCache 全部实验章节（Ch.1/2/3/4/5）使用 τ-bench 单数据集，165 tasks × 8 seeds = 1,320 episodes。所有 Gate 判定（G1/G2/G3/G4）和主表对照均复用同一 trace。

#### Scenario: τ-bench 单数据集录制
- **WHEN** 用户运行 `python record_trajectories.py --dataset tau-bench`
- **THEN** 系统对 τ-bench 165 tasks 用 8 seeds（user simulator seed）录制 1,320 episodes
- **AND** τ-bench trace 文件数 = 1,320
- **AND** 不录制 BFCL/LongBench/GSM8K/STB/SWE/Toolathlon 任何 trace

#### Scenario: pass^k 分析
- **WHEN** 录制完成
- **THEN** 支持 pass^k（k∈{1,2,4,8}）分析，与 τ-bench 原论文对齐
- **AND** 不对 BFCL 报告 pass^k（BFCL 已删除）

### Requirement: Ch.3 fidelity 质量面复用 τ-bench

Ch.3 fidelity 侧（量化非劣检验）在 τ-bench 1,320 episodes 上验证，不再使用 LongBench/GSM8K。量化质量（logit KL、top-k change、任务成功率变化）直接在主 workload 上测量。

#### Scenario: Ch.3 fidelity 侧数据来源
- **WHEN** 运行 Ch.3 fidelity 侧实验
- **THEN** 数据来源为 τ-bench 1,320 episodes（与 Ch.1/Ch.4 共用 trace）
- **AND** 不录制 LongBench 1000 或 GSM8K 100
- **AND** 量化质量指标（logit KL、top-k change、任务成功率 Δsuccess）在 τ-bench 任务上测量

### Requirement: Ch.5 鲁棒性单轴

Ch.5 鲁棒性从 3 轴缩减为 1 轴（到达扰动），删除 family-out 和 branch 噪声轴。

#### Scenario: Ch.5 鲁棒性轴
- **WHEN** 运行 Ch.5 鲁棒性实验
- **THEN** 仅评估到达扰动轴（BurstGPT 窗口 replay）
- **AND** 不评估 family-out 轴（StableToolBench 500 已删除）
- **AND** 不评估 branch 噪声轴（SWE/Toolathlon 已删除）
- **AND** retail↔airline 两域对照可作为 τ-bench 内部的弱 family-out 证据

### Requirement: 核心样本总量 1,320

FlowCache 全册核心样本总量封顶为 1,320 episodes（τ-bench 165 × 8），与 τ-bench 原论文对齐。

#### Scenario: 核心样本总量检查
- **WHEN** 录制完成
- **THEN** τ-bench trace 文件数 = 1,320
- **AND** 总核心样本量 = 1,320（不超过封顶）
- **AND** BurstGPT 窗口不计入核心样本（仅 replay 参数）

## MODIFIED Requirements

### Requirement: 主表 workload 组合

**原（v0.3）**：主表 workload 为 τ-bench 495 + BFCL v3 multi-turn 800，严格两个工具 workload。

**现（本 spec）**：主表 workload 为 τ-bench 1,320 episodes（165 × 8 seeds）单数据集。删除 BFCL v3，删除"严格两个工具 workload"约束。

### Requirement: Ch.4 主表 cell

**原（v0.3）**：6 cell（主-1/主-2 τ-bench/BFCL × 25%/50% × 8 并发 × 1-3 seeds + 边界-1/边界-2 τ-bench/BFCL × 10% × 16 并发）。

**现（本 spec）**：cell 简化为 τ-bench 单 workload 上的预算 × 并发网格，具体 cell 设计在 Ch.4 pilot 后冻结，但总 replay 数不超过 v0.3 的 100 replay 上限。

### Requirement: §0.4 数据集组合

**原（v0.3）**：7 核心数据集（τ-bench 495、BFCL 800、LongBench 1000、GSM8K 100、STB 500、SWE 200、Toolathlon 200），核心样本总量 ~3,300（trim spec 修正为 ~3,720）。

**现（本 spec）**：1 核心数据集（τ-bench 1,320），核心样本总量 1,320。删除 LongBench/GSM8K/STB/SWE/Toolathlon/LMSYS-Chat-1M 6 个数据集。

### Requirement: §12 14 周执行计划

**原（v0.3）**：W3–W5 录制 τ-bench 495 + BFCL 800；W9 末封顶 495/800；W12 STB 500 录制。

**现（本 spec）**：W3–W4 录制 τ-bench 1,320（GPU 机时减半，窗口压缩 1 周）；W9 末封顶 1,320；W12 仅到达扰动 replay（无 STB 录制）。

## REMOVED Requirements

### Requirement: BFCL v3 multi-turn 800 episodes

**Reason**: 用户要求单数据集 τ-bench 1,320。BFCL scripted user turns 的方法论与 τ-bench LLM user simulator 不对等，删除后避免解释成本。

**Migration**: 已录制的 BFCL trace（如有）保留为可选附注；BFCL 适配器代码保留但不在主流程调用；rebuttal 时若审稿人要求第二工具家族证据可补做。

### Requirement: LongBench 1000 + GSM8K 100（Ch.3 fidelity 质量面）

**Reason**: 用户要求单数据集。fidelity 质量面直接在 τ-bench 主 workload 上验证，避免数据集切换导致的 trace 不一致。

**Migration**: LongBench/GSM8K 量化质量指标改为在 τ-bench 任务上测量；若需长上下文量化质量证据可在附录补 LongBench。

### Requirement: StableToolBench 500（Ch.5 family-out 轴）

**Reason**: 用户要求单数据集。family-out 轴的证据力弱于到达扰动轴，且 STB 500 样本量不足以单独成章。

**Migration**: retail↔airline 两域对照作为 τ-bench 内部弱 family-out 证据；rebuttal 时可补 STB。

### Requirement: SWE 轨迹 200 + Toolathlon 200（Ch.5 压力面）

**Reason**: 用户要求单数据集。SWE/Toolathlon 与 FlowCache C1–C3 主线关联弱，200 样本不足以单独成章。

**Migration**: 压力面证据由到达扰动轴（BurstGPT replay）单独承载；rebuttal 时可补。

### Requirement: LMSYS-Chat-1M 500 负对照附注

**Reason**: 用户要求单数据集。负对照附注非核心证据，删除以简化实验体系。

**Migration**: 删除附注，不补做。
