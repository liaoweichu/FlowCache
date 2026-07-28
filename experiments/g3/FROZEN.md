# G3 Frozen: PROTOCOL-INCOMPLETE（历史结果均不得触发路线切换）

**Frozen Date**: 2026-07-27
**Status**: PROTOCOL-INCOMPLETE — G3、G3′ 与当前 G3″ 的结果**均不可作为路线切换依据**
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

### 已修复的已知问题（Bug 5）

| Bug | 说明 | 决策 |
|---|---|---|
| share_count 静态化 | 旧 `compute_share_counts` 使用完整 future trace 的最终共享度 | 已改为 trailing access window 内、截至当前访问的因果共享度；输出 feature scope，checker fail-closed |

**预期产物（G3'' 重跑后）**：
- `results/raw_results.csv`：9 cells × 6 baselines × 165 tasks = 8,910 行
  - `migrate_count > 0`（CPU 层被使用）
  - `global_p95_ttft_ms` 合理（含全命中 request）
  - `global_throughput` 合理（用总 request 数）
- `g3-verdict.md` / `g3-verdict.json`：基于 task_id 聚类 bootstrap 的有效判定（CI 非单点）

## 2026-07-27 G3-P0 修复记录

原 “Do NOT modify / 可直接全网格重跑” 指令已被本节取代。深度复核发现：

- GPU/CPU victim selection 每次线性扫描，导致 FlowCache replay wall 1,283.1 s；
- 单 block D2H 线性拟合可产生负值；
- 旧 p95 未逐请求加入 migrate/restore；
- arrival-window rate 被误标为 throughput；
- always-migrate 只有容量扩展语义，不是 selective value-aware controller；
- GPU-only Oracle 与额外拥有 CPU tier 的方法不构成公平主比较。

代码现已进入 G3-P0：O(1) GPU LRU、lazy-heap CPU victim、非负查表 transfer cost、
逐请求 movement/policy 计费、fail-closed verdict。下一步只跑
`experiments/g3-next-experiment.md` 定义的 2 GiB、c=4 单 cell；通过前不得覆盖历史结果，
不得恢复 9-cell 网格，也不得将当前结果解释为 G3 NO-GO。

## 2026-07-28 G3-P1 因果 GPU 准入与选择性迁移记录

- `flowcache_lossless` 使用 `oracle_cost_proxy + selective_value`；
  `flowcache_selective_migrate_only` 与 `flowcache_always_migrate` 分别
  隔离 GPU admission 和 CPU migration 的增量收益。
- admission 使用容量归一化 recency、因果 share、跨驱逐历史频率与位置先验形成有界 likelihood proxy；该 proxy 未校准，不得称为概率。
- 候选净价值显式扣除 H2D、D2H 与 CPU hold；CPU 满时与最低价值 incumbent 比较。
- 满 GPU miss 先比较 incoming 与 incumbent 的因果 cost value；bypass
  只是不保留 reusable cache 副本，不跳过当前请求的计算。
- 完整未来索引只提供给 `oracle_cost`/Belady；在线策略若收到
  `future_accesses` 立即报错，相同历史接不同未来的前缀不变性测试通过。
- transfer 标定查值和静态块信号缓存，CPU victim 使用可压缩 lazy heap；
  28 个回归测试通过。
- 20,000-access hot/cold 机制 trace 在相同 9,999 hits 下将 modeled
  movement 从 selective-migrate-only 的 1,100.943 ms 降至 0 ms；
  本轮 replay wall 从约 0.105 s 降至 0.078 s，但这不是 workload 结果。
- 200,000-access 等成本循环负对照中，因果 doorkeeper 令完整策略与
  selective-migrate-only 都得到 3,213 hits、10,481.116 ms movement，
  不再以少量搬运节省换取大量额外 miss。
- 23,107-access 局部物理工程复测中，cold-start doorkeeper 版本 bypass
  1,055/18,307（5.76%），相对 selective-migrate-only：
  hit rate +1.091 个百分点、migration −4.56%、transfer −2.81%、
  miss cost −0.99%；仍低于 5% transfer 门槛，不得冻结参数或写入主结果。
- 100-request / 4-workflow 局部物理 smoke 是新增 GPU admission 前的旧
  结果；selection rate=99.98%、migration reduction=0.017% 只能作为
  机制重构动机，不能证明新组合有效。
- validation 扫描复用固定基线与相同 CPU 参数的 selective-only 结果，
  54 组候选由 324 次降为 76 次 baseline replay；不复用 FlowCache 候选结果。
- 下一步只能在 task-grouped validation 上冻结 admission 参数，再运行一次 held-out test；详细协议见 `experiments/g3-next-experiment.md`。
