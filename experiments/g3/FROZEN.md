# G3 Frozen: PROTOCOL-INVALID (Verdict 不可信)

**Frozen Date**: 2026-07-27
**Status**: PROTOCOL-INVALID — `g3-verdict.*` 的 NO-GO 结论**不可作为路线切换依据**
**Verdict File**: `g3-verdict.json`, `g3-verdict.md`（保留作为问题诊断证据，不作为判定）

## Verdict 摘要（无效）

- Verdict: ❌ NO-GO（**作废**）
- Main cell: 1 GiB, c=4
- raw_results.csv: 162 行 = 9 cells × 6 baselines × 3 seeds（G3 原始设计）
- p95 TTFT 改善: vs gdsf −40.04%, vs sizecost −69.11%
- CI 退化为单点 `[−40.04%, −40.04%]`（bootstrap 无效）

## Protocol 失效证据（三大问题）

### 1. Seed 映射失败 → bootstrap 退化

- `run_g3_grid.py` 原设计中 `replay_seeds = [0, 1, 2]`，但 τ-bench trace 中的 seed
  是原始值（如 `101112`），`filter_by_seed` 过滤后返回空集。
- 结果：3 个 seed 的行**完全相同**，bootstrap CI 退化为单点
  `[mean, mean]`，统计检验无效。
- 证据：`g3-verdict.json` 中所有 CI `ci_low == ci_high == mean`。

### 2. CPU 层从未使用 → controller 退化

- `controller.py` 原迁移阈值 `migrate_threshold = 0.1` 过高，几乎所有 GPU
  victim 的 R 值都 ≤ 0.1，直接从 GPU 淘汰，从不迁移到 CPU。
- 证据：`raw_results.csv` 中所有 `flowcache_lossless` 行的
  `migrate_count = 0`，`restore_count = 0`，`migrate_ms_total = 0.0`。
- 后果：FlowCache-Lossless 退化为某种纯 GPU 策略，且 `safety_margin=0.10`
  导致 GPU 有效容量缩容 10%，性能反而不如纯 GPU 启发式（sizecost/gdsf）。

### 3. 统计单位错误 → per-seed bootstrap 无效

- 原按 3 个 seed 做 bootstrap，样本量 n=3 且值相同，无法支持 95% CI。
- 正确做法：以 165 个 `task_id` 为聚类单元做 paired bootstrap。

## 主 cell 异常解读（无效，仅诊断用）

Cell (1 GiB, c=4): FlowCache p95 = 168,558 ms vs sizecost 99,676 ms
（差 69%）。这是 CPU 层未使用 + GPU 缩容 10% 的叠加效应，**不代表价值感知
controller 本身的真实性能**。不能据此推断"无损 residency 主张不成立"。

## 后继实验：G3′（已修复上述问题）

G3′ 在**同一目录** `experiments/g3/` 下重跑（代码已就地修复）：

| 问题 | G3 原 | G3′ 修复 |
|---|---|---|
| Seed 映射 | `replay_seeds=[0,1,2]` 过滤失败 | 移除 seed 过滤，单次跑全部 1320 episodes |
| 统计单位 | per-seed (n=3, 退化) | per-task_id (n=165, paired bootstrap) |
| 迁移阈值 | `migrate_threshold=0.1`（CPU 层从未使用） | `migrate_threshold=0.01` |
| CPU 层淘汰 | CPU 满时直接淘汰 GPU 块 | 新增 `_select_cpu_victim`，按 R 值淘汰 CPU 块腾位 |
| 安全水位 | `safety_margin=0.10`（GPU 缩容 10%） | `safety_margin=0.05` |
| 输出格式 | per-seed 行（含 `seed` 字段） | per-task 行（含 `task_id` 字段），165 行/cell×baseline |

**预期产物（G3′ 重跑后）**：
- `results/raw_results.csv`：9 cells × 6 baselines × 165 tasks = 8,910 行
- `g3-verdict.md` / `g3-verdict.json`：基于 task_id 聚类 bootstrap 的有效判定

## Do NOT modify

本目录下除以下文件外的**历史文件不得修改**（保留作为问题诊断证据）：
- `results/raw_results.csv`（G3 原始 162 行）— 将被 G3′ 重跑覆盖
- `g3-verdict.md` / `g3-verdict.json`（G3 原始无效判定）— 将被 G3′ 重跑覆盖
- `figures/g3-*.png`（G3 原始图）— 将被 G3′ 重跑覆盖

代码文件（`controller.py`、`run_g3_grid.py`、`g3_verdict.py`、`config.yaml`）
**已就地更新为 G3′ 修复版**，可在云端直接重跑。G3 原始代码状态见 git 历史。
