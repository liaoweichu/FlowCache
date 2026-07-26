# Checklist

## Phase 1 — Task 1: G1 目录结构
- [x] `experiments/g1/` 目录存在
- [x] `experiments/g1/tests/` 目录存在
- [x] `experiments/g1/results/` 目录存在
- [x] `experiments/g1/figures/` 目录存在
- [x] `experiments/g1/__init__.py` 存在（空文件）
- [x] `experiments/g1/tests/__init__.py` 存在（空文件）

## Phase 1 — Task 2: G1 config.yaml
- [x] `experiments/g1/config.yaml` 存在
- [x] 含 `budgets: [0.10, 0.25, 0.50, 1.00]`
- [x] 含 `replay_seeds: [1, 2, 3]`
- [x] 含 `datasets: ["tau_bench"]`
- [x] 含 `baselines:` 段（引用或复制 E1 enabled baseline 名单）
- [x] 含 `trace_source: "experiments/e1/traces/bf16/tau_bench/"`

## Phase 1 — Task 3: run_grid.py
- [x] `experiments/g1/run_grid.py` 存在
- [x] 读取 `experiments/g1/config.yaml`
- [x] 复用 `experiments/e1/compare_oracle.py` 的 build_access_trace 与 8 个 baseline 类
- [x] 对每个 (baseline, budget, dataset, seed) 组合运行并收集指标
- [x] KVFlow faithful 输出 status=pending，不阻塞
- [x] trace < 1320 时标注 "pilot: N/1320" + stdout 警告
- [x] 输出 `experiments/g1/results/raw_results.csv`，含 12 列

## Phase 1 — Task 4: verdict.py
- [x] `experiments/g1/verdict.py` 存在
- [x] 读取 `experiments/g1/results/raw_results.csv`
- [x] 计算 headroom = Oracle-Cost.miss_cost − max(LRU, GDSF, SizeCost, APC-LRU).miss_cost
- [x] 计算 headroom 相对差
- [x] paired workflow-level bootstrap 1000 次，95% CI
- [x] Bonferroni 校正（α / 预算档位数）
- [x] Go/No-Go 判定（headroom ≥ 10% → pass）
- [x] 输出 `experiments/g1/g1-verdict.md`（含 G1.11.1 表 G1-1 填充）
- [x] 输出 `experiments/g1/g1-verdict.json`（含 go_no_go 字段）

## Phase 1 — Task 5: plot_headroom.py
- [x] `experiments/g1/plot_headroom.py` 存在
- [x] 读取 `experiments/g1/results/raw_results.csv`
- [x] 绘制策略 × 预算的 miss_cost 图
- [x] 绘制策略 × 预算的 p95 TTFT 图
- [x] 输出 `experiments/g1/figures/g1-headroom.png`

## Phase 2 — Task 6: test_run_grid.py
- [x] `experiments/g1/tests/test_run_grid.py` 存在
- [x] 测试配置加载
- [x] 测试网格展开（96 组合）
- [x] 测试 CSV 输出格式
- [x] 测试 KVFlow faithful 跳过逻辑
- [x] `python -m pytest experiments/g1/tests/test_run_grid.py -v` 全部通过

## Phase 2 — Task 7: test_g1_verdict.py
- [x] `experiments/g1/tests/test_g1_verdict.py` 存在
- [x] 测试 headroom 计算
- [x] 测试 headroom 相对差
- [x] 测试 bootstrap 95% CI（固定随机种子）
- [x] 测试 Bonferroni 校正
- [x] 测试 Go/No-Go 判定逻辑
- [x] `python -m pytest experiments/g1/tests/test_g1_verdict.py -v` 全部通过

## Phase 3 — Task 8: IDEA.rewritten.md §8 Ch.4 脚注
- [ ] G1.11.1 表 G1-1 下方脚注（line 789）已更新
- [ ] 含 "KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活"
- [ ] 含 "config.yaml: kvflow_faithful.enabled: true"

## Phase 3 — Task 9: experiment-designs.md G1.9 / G1.11
- [ ] G1.9 步骤 1.5 产物列含 `experiments/g1/run_grid.py`
- [ ] G1.9 步骤 1.6 产物列含 `experiments/g1/verdict.py`
- [ ] G1.11 预期产物表含代码路径

## Phase 3 — Task 10: config.yaml baselines 段注释
- [ ] `experiments/e1/config.yaml` baselines 段顶部注释含 "G1 verdict 模块通过 `experiments/g1/run_grid.py` 调用本配置"

## Phase 4 — Task 11: G1 代码可运行验证
- [ ] `python -m pytest experiments/g1/tests/ -v` 全部通过
- [ ] `python experiments/g1/run_grid.py` 产出 raw_results.csv（pilot 模式，28 trace）
- [ ] `python experiments/g1/verdict.py` 产出 g1-verdict.md 与 g1-verdict.json
- [ ] `python experiments/g1/plot_headroom.py` 产出 g1-headroom.png

## Phase 4 — Task 12: 文档一致性验证
- [ ] `grep "experiments/g1/run_grid.py" experiment-designs.md` 命中 G1.9 步骤 1.5
- [ ] `grep "experiments/g1/verdict.py" experiment-designs.md` 命中 G1.9 步骤 1.6
- [ ] `grep "AutoDL Linux 激活" IDEA.rewritten.md` 命中 §8 Ch.4 脚注
- [ ] `git diff --name-only` 含新增 experiments/g1/ 目录 + 3 个文档修改
