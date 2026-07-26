# Tasks

## Phase 1：G1 核心代码（verdict + grid runner + 绘图）

- [x] Task 1: 创建 `experiments/g1/` 目录结构与 `__init__.py`
  - [x] SubTask 1.1: 创建 `experiments/g1/`、`experiments/g1/tests/`、`experiments/g1/results/`、`experiments/g1/figures/` 目录
  - [x] SubTask 1.2: 创建 `experiments/g1/__init__.py`（空文件，包标识）
  - [x] SubTask 1.3: 创建 `experiments/g1/tests/__init__.py`（空文件）

- [x] Task 2: 实现 `experiments/g1/config.yaml`
  - [x] SubTask 2.1: 定义 `budgets: [0.10, 0.25, 0.50, 1.00]`
  - [x] SubTask 2.2: 定义 `replay_seeds: [1, 2, 3]`
  - [x] SubTask 2.3: 定义 `datasets: ["tau_bench"]`
  - [x] SubTask 2.4: 定义 `baselines:` 段，引用 `experiments/e1/config.yaml` 的 baseline 列表（或直接复制 enabled baseline 名单）
  - [x] SubTask 2.5: 定义 `trace_source: "experiments/e1/traces/bf16/tau_bench/"`

- [x] Task 3: 实现 `experiments/g1/run_grid.py`
  - [x] SubTask 3.1: 读取 `experiments/g1/config.yaml` 获取 baseline 列表、预算档位、种子数
  - [x] SubTask 3.2: 复用 `experiments/e1/compare_oracle.py` 的 `build_access_trace` 与 8 个 baseline 类
  - [x] SubTask 3.3: 对每个 (baseline, budget, dataset, seed) 组合运行 baseline，收集 hits/misses/hit_rate/evictions/saved_prefill_ms/miss_cost_ms/p95_ttft_ms
  - [x] SubTask 3.4: KVFlow faithful 遇到时输出 `status=pending, reason=adapter_not_implemented`，不阻塞
  - [x] SubTask 3.5: trace 数 < 1320 时在 CSV 顶部注释标注 "pilot: N/1320 episodes available" + stdout 警告
  - [x] SubTask 3.6: 输出 `experiments/g1/results/raw_results.csv`，含列：baseline, budget, dataset, seed, hits, misses, hit_rate, evictions, saved_prefill_ms, miss_cost_ms, p95_ttft_ms, status

- [x] Task 4: 实现 `experiments/g1/verdict.py`
  - [x] SubTask 4.1: 读取 `experiments/g1/results/raw_results.csv`
  - [x] SubTask 4.2: 对每个 (budget, dataset) 组合计算 headroom = Oracle-Cost.miss_cost − max(LRU, GDSF, SizeCost, APC-LRU).miss_cost
  - [x] SubTask 4.3: 计算 headroom 相对差 = headroom / Oracle-Cost.miss_cost
  - [x] SubTask 4.4: 用 paired workflow-level bootstrap（1000 次）计算 95% CI
  - [x] SubTask 4.5: 多预算档位 Bonferroni 校正
  - [x] SubTask 4.6: 对照 G1.8 判定阈值（headroom ≥ 10% → pass；closest baseline 可比性 → pass/fail）
  - [x] SubTask 4.7: 输出 `experiments/g1/g1-verdict.md`（含 G1.11.1 表 G1-1 模板填充结果）
  - [x] SubTask 4.8: 输出 `experiments/g1/g1-verdict.json`（含 `go_no_go: {headroom, comparability}`）

- [x] Task 5: 实现 `experiments/g1/plot_headroom.py`
  - [x] SubTask 5.1: 读取 `experiments/g1/results/raw_results.csv`
  - [x] SubTask 5.2: 绘制策略 × 预算的 miss_cost 图（8 baseline + 4 预算档位）
  - [x] SubTask 5.3: 绘制策略 × 预算的 p95 TTFT 图
  - [x] SubTask 5.4: 输出 `experiments/g1/figures/g1-headroom.png`

## Phase 2：G1 单元测试

- [x] Task 6: 实现 `experiments/g1/tests/test_run_grid.py`
  - [x] SubTask 6.1: 测试配置加载（budgets/seeds/datasets/baselines 正确解析）
  - [x] SubTask 6.2: 测试网格展开（8 baseline × 4 budget × 1 dataset × 3 seed = 96 组合）
  - [x] SubTask 6.3: 测试 CSV 输出格式（列名、数据类型）
  - [x] SubTask 6.4: 测试 KVFlow faithful 跳过逻辑（输出 status=pending）

- [x] Task 7: 实现 `experiments/g1/tests/test_g1_verdict.py`
  - [x] SubTask 7.1: 测试 headroom 计算（Oracle-Cost − max 简单策略）
  - [x] SubTask 7.2: 测试 headroom 相对差计算
  - [x] SubTask 7.3: 测试 bootstrap 95% CI（用固定随机种子验证可复现）
  - [x] SubTask 7.4: 测试 Bonferroni 校正（α / 预算档位数）
  - [x] SubTask 7.5: 测试 Go/No-Go 判定逻辑（headroom ≥ 10% → pass）

## Phase 3：文档同步

- [ ] Task 8: 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.4 主表对照 #4 脚注
  - [ ] SubTask 8.1: 定位 G1.11.1 表 G1-1 下方脚注（line 789）
  - [ ] SubTask 8.2: 将 "KVFlow 为 faithful reproduction（待 WSL2 adapter 实现）" 改为 "KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活（`config.yaml: kvflow_faithful.enabled: true`），adapter 实现中"

- [ ] Task 9: 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.9 / G1.11
  - [ ] SubTask 9.1: G1.9 步骤 1.5 产物列改为 "原始结果表（`experiments/g1/run_grid.py` → `results/raw_results.csv`）"
  - [ ] SubTask 9.2: G1.9 步骤 1.6 产物列改为 "`experiments/g1/verdict.py` → `experiments/g1/g1-verdict.md` + `g1-verdict.json`"
  - [ ] SubTask 9.3: G1.11 预期产物表补充代码路径（run_grid.py / verdict.py / plot_headroom.py）

- [ ] Task 10: 修改 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) baselines 段顶部注释
  - [ ] SubTask 10.1: 在注释中补充 "G1 verdict 模块通过 `experiments/g1/run_grid.py` 调用本配置"

## Phase 4：验证

- [ ] Task 11: 验证 G1 代码可运行
  - [ ] SubTask 11.1: `python -m pytest experiments/g1/tests/ -v` 全部通过
  - [ ] SubTask 11.2: `python experiments/g1/run_grid.py` 在现有 28 τ-bench trace 上产出 raw_results.csv（pilot 模式）
  - [ ] SubTask 11.3: `python experiments/g1/verdict.py` 产出 g1-verdict.md 与 g1-verdict.json
  - [ ] SubTask 11.4: `python experiments/g1/plot_headroom.py` 产出 g1-headroom.png

- [ ] Task 12: 验证文档一致性
  - [ ] SubTask 12.1: grep "experiments/g1/run_grid.py" experiment-designs.md 命中 G1.9 步骤 1.5
  - [ ] SubTask 12.2: grep "experiments/g1/verdict.py" experiment-designs.md 命中 G1.9 步骤 1.6
  - [ ] SubTask 12.3: grep "AutoDL Linux 激活" IDEA.rewritten.md 命中 §8 Ch.4 脚注
  - [ ] SubTask 12.4: git diff --name-only 确认新增 experiments/g1/ 目录 + 3 个文档修改

# Task Dependencies

## Phase 1（核心代码）
- Task 1 (目录) → Task 2 (config.yaml) → Task 3 (run_grid.py) → Task 4 (verdict.py) → Task 5 (plot_headroom.py)
- Task 3, 4, 5 严格顺序：verdict 依赖 grid 输出，plot 依赖 grid 输出

## Phase 2（单元测试）
- Task 6 依赖 Task 3（测试 grid runner）
- Task 7 依赖 Task 4（测试 verdict）
- Task 6, 7 可并行

## Phase 3（文档同步）
- Task 8, 9, 10 互相独立，可并行
- 与 Phase 1/2 无依赖，可并行

## Phase 4（验证）
- Task 11 依赖 Phase 1 + Phase 2 全部完成
- Task 12 依赖 Phase 3 全部完成
