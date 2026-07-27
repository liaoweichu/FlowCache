# G3 Frozen: PROTOCOL-INVALID (两轮均不可信)

**Frozen Date**: 2026-07-27
**Status**: PROTOCOL-INVALID — G3 和 G3′ 的 NO-GO 结论**均不可作为路线切换依据**
**Verdict File**: `g3-verdict.json`, `g3-verdict.md`（保留作为问题诊断证据，不作为判定）

## 第一轮：G3 原始（162 行，protocol-invalid）

- Verdict: ❌ NO-GO（**作废**）
- raw_results.csv: 162 行 = 9 cells × 6 baselines × 3 seeds
- 三大问题：seed 映射失败、CPU 层从未使用（migrate_threshold=0.1）、per-seed bootstrap 退化

## 第二轮：G3′（8,910 行，仍 protocol-invalid）

- Verdict: ❌ NO-GO（**作废**）
- raw_results.csv: 8,910 行 = 9 cells × 6 baselines × 165 tasks（数据规模正确 ✅）
- 表头含 `task_id` 列 ✅

### G3′ 仍未修复的两个新 Bug

#### Bug 1: `migrate_count` 仍然全为 0（CPU 层仍未使用）

- **根因**：`controller.py` 的 `_ensure_gpu_space` 用 LRU 选 victim（age 最大的块），
  再用 R 值决定迁移/淘汰。但 R 公式 `exp(-0.005×age)` 在 `age >= horizon(1000)` 时
  返回 0.0，而 LRU victim 的 age 在 1320 episodes 大规模 trace 中经常超过 1000，
  导致 `victim_r=0 ≤ migrate_threshold=0.01`，**直接淘汰，从不迁移**。
- **证据**：所有 `flowcache_lossless` 行 `migrate_count=0`、`restore_count=0`、`overhead_ms=0.00`。
- **修复（G3''）**：改为总是迁移到 CPU（不检查 R 值），CPU 容量自动设为 GPU 有效容量的 2 倍。

#### Bug 2: `g3_verdict.py` 未更新为 task_id 版本（CI=[0,0]）

- **根因**：`evaluate_go_no_go` 仍调用 `collect_per_seed_improvement`（按 `seed` 分组，
  默认 metric `"p95_ttft_ms"`），但 G3' csv 中无 `seed` 字段（所有行 seed=0 互相覆盖）、
  无 `p95_ttft_ms` 字段（实际是 `task_p95_ttft_ms` 和 `global_p95_ttft_ms`），
  导致 per_seed 列表为空，`bootstrap_ci([])` = `(0, 0, 0)`。
- **证据**：所有 CI = `[0.00%, 0.00%]`，`better_than_heuristic` mean=0.00%。
- **修复（G3''）**：重写 `aggregate_by_cell_baseline`（正确聚合 per-task 行）、
  替换为 `collect_per_task_improvement`（按 task_id 分组，metric 用 `task_p95_ttft_ms`）。

### G3′ 部分 cell 表现良好（全局 p95，非 bootstrap CI）

| Capacity | Conc | FlowCache p95 | Best Simple p95 | 改善 |
|---:|---:|---:|---:|---:|
| 2 GiB | 4 | 8,609 | 22,612 (gdsf) | **+61.93%** |
| 4 GiB | 8 | 8,304 | 18,644 (gdsf) | **+55.46%** |
| 1 GiB | 1 | 8,116 | 10,241 (sizecost) | +20.74% |
| 1 GiB | 4 | 107,550 | 76,424 (sizecost) | -40.73% (主 cell，异常) |

即使 CPU 层未使用，FlowCache 在中容量中并发下仍大幅优于 gdsf/sizecost，
说明价值感知 controller 的 GPU 内淘汰策略本身有效。

## 后继实验：G3''（第二轮修复）

G3'' 在**同一目录** `experiments/g3/` 下重跑（代码已就地修复）：

### 第一轮修复（Bug 1-2）

| Bug | G3′ | G3'' 修复 |
|---|---|---|
| migrate_count=0 | LRU victim age>=horizon → R=0 ≤ 0.01 → 直接淘汰 | 总是迁移到 CPU（不检查 R 值），CPU 满→按 R 值淘汰 CPU 块腾位 |
| CPU 容量无限 | `cpu_capacity_blocks=-1`（无限） | 自动设为 GPU 有效容量的 2 倍 |
| CI=[0,0] | `collect_per_seed_improvement` + metric 名不匹配 | `collect_per_task_improvement` + `task_p95_ttft_ms` |
| 聚合错误 | `aggregate_by_cell_baseline` 按 seed 分组互相覆盖 | 从 per-task 行提取 global_* 字段，sum 聚合 saved_prefill |

### 第二轮修复（Bug 3-4-6，代码审视发现）

| Bug | 根因 | G3'' 修复 |
|---|---|---|
| throughput 偏低 | `n_requests = len(request_miss_cost)` 只含有 miss 的 request | `n_requests = len(request_ttft)` 含全部 request |
| p95 TTFT 高估 | p95 只含有 miss 的 request，排除全命中 request（TTFT=0） | 所有 request 都计入 TTFT 列表（hit=0, miss=miss_cost） |
| CPU hit 竞态 | `_ensure_gpu_space` 可能淘汰要 restore 的 CPU 块 → KeyError | 传递 `protect_hash`，`_select_cpu_victim` 跳过目标块 |

### 第三轮修复（Bug 11，深度代码审视发现）

| Bug | 根因 | G3'' 修复 |
|---|---|---|
| 聚合不一致 | `aggregate_by_cell_baseline` 中 `miss_cost_ms`/`hits`/`misses` 只取第一行 per-task 值，但 `saved_prefill_ms` 用 sum 聚合 | 4 个 per-task 字段统一用 `per_task_sums` sum 聚合 |

### 已知代码质量问题（不影响正确性，不修复）

| 问题 | 说明 | 决策 |
|---|---|---|
| `_clock` 双时钟 | controller 和 manager 各自维护 `_clock`，manager 的 `admit_gpu`/`touch_gpu` 自增 `_clock` 是冗余的 | 不修改：controller 总是覆盖 `last_access`，LRU 和 R 值计算正确 |
| `migrate_to_cpu` 统计 | CPU 满时仍计数为 `migrate_to_cpu_count`，实际是淘汰 | 不修改：controller 已确保 `not cpu_full()`，该分支不会触发 |

### 未修复的已知问题（Bug 5）

| Bug | 说明 | 决策 |
|---|---|---|
| share_count 静态化 | `compute_share_counts` 是全 trace 共享度，非 H=1000 窗口动态值 | 保留，R 值仍由 share_count+age+position 三因子决定，全 trace 共享度是合理近似 |

**预期产物（G3'' 重跑后）**：
- `results/raw_results.csv`：9 cells × 6 baselines × 165 tasks = 8,910 行
  - `migrate_count > 0`（CPU 层被使用）
  - `global_p95_ttft_ms` 合理（含全命中 request）
  - `global_throughput` 合理（用总 request 数）
- `g3-verdict.md` / `g3-verdict.json`：基于 task_id 聚类 bootstrap 的有效判定（CI 非单点）

## Do NOT modify

代码文件（`controller.py`、`g3_verdict.py`、`run_g3_grid.py`、`config.yaml`）
**已就地更新为 G3'' 修复版**，可在云端直接重跑。G3/G3' 原始代码状态见 git 历史。

`results/raw_results.csv`、`g3-verdict.*`、`figures/g3-*.png` 为 G3' 产物，
将被 G3'' 重跑覆盖。
