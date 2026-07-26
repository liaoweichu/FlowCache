# 重新审视 G1 样本量 Spec

## Why

G1 实验当前规划 7,720 episodes（τ-bench 1,320 + BFCL 6,400），显著超过同领域 KV cache 复用/管理论文的样本量。对比同类论文：

| 论文 | 会议 | 数据集数 | 样本总量 |
|---|---|---|---|
| CacheGen | SIGCOMM 2024 | 4 | 662 contexts |
| EvicPress | arXiv 2025-12 | 12 | 600 contexts |
| vLLM × Mooncake agentic | vLLM blog 2026-05 | 1 (SWE-bench Pro) | 610 traces |
| KVFlow | NeurIPS 2025 | ~2 (synthetic + PEER) | 未明确（合成 workflow） |
| PBKV | arXiv 2026-05 | 3 workflow benchmarks | 未明确 |
| vLLM/PagedAttention | SOSP 2023 | 2 (ShareGPT/Alpaca) | ~1,000s requests/batch |
| τ-bench 原论文 | ICLR 2025 | 1 (165 tasks) | 165 × 8 = 1,320 episodes |

FlowCache G1 的 7,720 episodes 是 CacheGen (662) 的 **11.7×**、EvicPress (600) 的 **12.9×**、vLLM×Mooncake (610) 的 **12.7×**。

核心问题：**BFCL 8 decode seeds 的边际价值远低于 τ-bench 8 user simulator seeds**。BFCL 的 user turns 是 scripted 固定字符串，8 个 decode seeds 只变 agent 输出，不变对话轨迹；而 τ-bench 的 8 user simulator seeds 变整个对话轨迹。两者方法论不对等，BFCL 不需要 8 seeds。

## What Changes

- **BFCL seeds 从 8 降到 1**（greedy decode, do_sample=False），回到 BFCL 官方默认设定
- **τ-bench 保持 8 seeds**（与原论文 pass^k 对齐）
- 总样本量从 7,720 降到 **2,120**（τ-bench 1,320 + BFCL 800），与 spec v0.3 原始封顶对齐
- 节省 ~36 GPU 小时（4090D），原 50 GPU 小时预算降至 ~14 GPU 小时
- 更新 [experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.2/G1.3 表格回退到 v0.3 原始封顶
- 更新 [config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) BFCL seeds=1
- 更新 [g1-experiment-implementation.md](file:///d:/00MyProject/Prefix%20Caching/.trae/documents/g1-experiment-implementation.md) §2.1 算力预算表
- **BREAKING**: 撤销上一个 spec 的 BFCL 8 seeds 决定，回到 spec v0.3 封顶 800

## Impact

- Affected specs:
  - [experiment-scope-redesign/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/experiment-scope-redesign/spec.md) §5 样本量封顶
  - [experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.2/G1.3
  - [trim-dataset-portfolio/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/trim-dataset-portfolio/spec.md): 本 spec 的 2,120 主表封顶保持不变；全册核心样本总量从 ~4,120 调整为 ~3,720（删除 SWE 200 + Toolathlon 200）
- Affected code:
  - [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) workload.bfcl_v3
  - [experiments/e1/record_trajectories.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/record_trajectories.py) `_record_bfcl_g1` seed 循环逻辑
- 受益：节省 36 GPU 小时，与同领域论文样本量对齐，避免"为了多 seed 而多 seed"的过度实验

## ADDED Requirements

### Requirement: BFCL 单 seed 录制（与 BFCL 官方对齐）

BFCL v3 multi-turn 默认 `temperature=0`（greedy decode），单 seed 即可复现官方结果。FlowCache G1 采用 BFCL 官方默认设定，不引入额外 decode seeds。

#### Scenario: BFCL 单 seed 录制
- **WHEN** 用户运行 `python record_trajectories.py --dataset bfcl_v3`
- **THEN** 系统对每个 BFCL subset 用单 seed（seed=0, do_sample=False, temperature=0）录制 200 episodes
- **AND** 总 BFCL episodes = 4 subsets × 200 = 800
- **AND** 不产生 pass^k（k>1）分析（BFCL 官方也不提供）

#### Scenario: τ-bench 保持 8 seeds
- **WHEN** 用户运行 `python record_trajectories.py --dataset tau-bench`
- **THEN** 系统对 τ-bench 用 8 seeds（user simulator seed）录制 165 × 8 = 1,320 episodes
- **AND** 支持 pass^k（k∈{1,2,4,8}）分析，与 τ-bench 原论文对齐

### Requirement: 总样本量封顶 2,120

G1 主表样本量封顶为 τ-bench 1,320 + BFCL 800 = 2,120 episodes，与 spec v0.3 原始封顶对齐，与同领域论文（CacheGen 662、EvicPress 600、vLLM×Mooncake 610）样本量级一致。

#### Scenario: 总样本量检查
- **WHEN** 录制完成
- **THEN** τ-bench trace 文件数 = 1,320
- **AND** BFCL trace 文件数 = 800
- **AND** 总文件数 = 2,120（不超过 spec 封顶）

注：2,120 为主表封顶。全册核心样本总量为 ~3,720（主表 2,120 + 质量面 1,100 + 鲁棒性 STB 500），详见 trim-dataset-portfolio spec。

## MODIFIED Requirements

### Requirement: BFCL seeds 数

**原（上一个 spec）**：BFCL 用 8 decode seeds（do_sample=True, temperature=0.7），与 τ-bench 8 user simulator seeds 对齐。总量 6,400 episodes。

**现（本 spec）**：BFCL 用单 seed（do_sample=False, temperature=0），与 BFCL 官方默认对齐。总量 800 episodes。BFCL scripted user turns 不需要 multi-seed；8 decode seeds 只变 agent 输出，不变对话轨迹，边际价值远低于 τ-bench 的 8 user simulator seeds。

## REMOVED Requirements

### Requirement: BFCL 8 decode seeds

**Reason**: 与同领域论文样本量不对齐（CacheGen 662、EvicPress 600、vLLM×Mooncake 610），且 BFCL scripted user turns 的 8 decode seeds 只变 agent 输出不变对话轨迹，边际价值低。

**Migration**: 已录制的 BFCL trace（如有）保留为 seed=0 子集；若已有 8 seeds 录制，保留 seed=0 子集，其余作废。报告 pass^k 分析时仅对 τ-bench 报告，BFCL 报告 pass^1。
