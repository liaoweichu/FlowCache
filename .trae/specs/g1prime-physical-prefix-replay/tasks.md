# Tasks

## Phase 0: 准备与验证

- [x] Task 1: 验证现有轨迹结构，确认重编译所需字段齐全
  - [ ] SubTask 1.1: 读取 3-5 条 `experiments/e1/traces/bf16/tau_bench/*.json`，列出 meta/steps/messages 字段
  - [ ] SubTask 1.2: 确认 G0 冻结的 Qwen tokenizer + chat template 路径（从 `experiments/e1/config.yaml` 读取）
  - [ ] SubTask 1.3: 确认每个 assistant step 的实测 prefill_ms 字段位置
  - [ ] SubTask 1.4: 编写 `experiments/g1prime/verify_trace_fields.py`，输出字段完整性报告

## Phase 1: 物理前缀重编译器

- [ ] Task 2: 实现 `experiments/g1prime/recompile_prefixes.py`
  - [ ] SubTask 2.1: 加载 G0 冻结的 Qwen tokenizer + chat template
  - [ ] SubTask 2.2: 对每条轨迹的每个 assistant step，重建该 step 之前的完整 message history（system + user + assistant + tool）
  - [ ] SubTask 2.3: 使用 chat template 序列化为完整 prompt，tokenize 得到 token IDs + positions
  - [ ] SubTask 2.4: 按 16-token block 跨 message 连续切分，计算 `block_hash = H(parent_hash, tokenIds, positions)` 和 `parent_hash`
  - [ ] SubTask 2.5: 输出 `experiments/g1prime/physical_traces/request_prefixes.jsonl`（每行一个 request event）
  - [ ] SubTask 2.6: 编写单元测试 `tests/test_recompile_prefixes.py`（跨 message 连续分块、parent chain 正确性、跨 workflow 共享前缀）

- [ ] Task 3: 验证重编译输出规模符合预期
  - [ ] SubTask 3.1: 统计 request 总数（预期 ~25,653）
  - [ ] SubTask 3.2: 统计 block access 总数（预期 ~8.27M）
  - [ ] SubTask 3.3: 统计 episode 内前缀重访问比例（预期 ~92.4%）
  - [ ] SubTask 3.4: 统计跨 workflow 共享 block 数（预期 > 721，远大于当前 G1 的 0.1%）

## Phase 2: 访问流构建与并发模拟

- [ ] Task 4: 实现 `experiments/g1prime/build_physical_access_trace.py`
  - [ ] SubTask 4.1: 从 `request_prefixes.jsonl` 读取所有 request events
  - [ ] SubTask 4.2: 每个 request 展开为完整前缀 block 访问序列
  - [ ] SubTask 4.3: 记录 request_id、workflow_id、step_id、arrival_time、prefill_ms
  - [ ] SubTask 4.4: 输出 `experiments/g1prime/physical_traces/access_trace.jsonl`

- [ ] Task 5: 实现 `experiments/g1prime/simulate_concurrency.py`
  - [ ] SubTask 5.1: 从现有轨迹的工具等待时间合成全局到达时间
  - [ ] SubTask 5.2: 实现并发度 c ∈ {1, 4, 8} 的交错调度
  - [ ] SubTask 5.3: 并发度 1 = 顺序基线（等价于当前 G1）
  - [ ] SubTask 5.4: 输出 `experiments/g1prime/physical_traces/access_trace_c{1,4,8}.jsonl`
  - [ ] SubTask 5.5: 编写单元测试 `tests/test_simulate_concurrency.py`（并发度正确性、工具等待期间 inactive prefix 与其他 workflow 竞争）

## Phase 3: 成本模型与容量定义修正

- [ ] Task 6: 实现修正的成本模型
  - [ ] SubTask 6.1: 从 G0 冻结的模型配置计算 bytes_per_block（层数 × 头数 × 维度 × 2 × 2 / 16）
  - [ ] SubTask 6.2: 实现 per-token prefill rate（从实测 prefill_ms / total_tokens 计算）
  - [ ] SubTask 6.3: 禁用固定 8ms 回退，改为 per-token rate × block token 数
  - [ ] SubTask 6.4: request 级成本归因：miss_prefill_ms = sum(missed_block_i.prefill_ms_share)
  - [ ] SubTask 6.5: 编写单元测试 `tests/test_cost_model.py`

- [ ] Task 7: 实现绝对 KV 容量定义
  - [ ] SubTask 7.1: 容量档位 1 / 2 / 4 / 6 GiB
  - [ ] SubTask 7.2: capacity_blocks = floor(C × 1024³ / bytes_per_block)
  - [ ] SubTask 7.3: 100%（无限制）仅作 sanity check
  - [ ] SubTask 7.4: 编写 `experiments/g1prime/config.yaml`

## Phase 4: G1′ verdict 模块

- [ ] Task 8: 实现 `experiments/g1prime/run_grid.py`
  - [ ] SubTask 8.1: 读取 config.yaml（容量档位、并发度、baseline 列表）
  - [ ] SubTask 8.2: 全网格运行：6 baselines × 4 容量 × 3 并发 × 1320 episodes
  - [ ] SubTask 8.3: 复用 `experiments/e1/compare_oracle.py` 的 baseline 类
  - [ ] SubTask 8.4: 收集 request 级指标（miss-prefill token/ms、p50/p95 TTFT、resume hit rate）
  - [ ] SubTask 8.5: 输出 `experiments/g1prime/results/raw_results.csv`

- [ ] Task 9: 实现 `experiments/g1prime/verdict.py`
  - [ ] SubTask 9.1: headroom_abs = Oracle-Cost miss cost − max(LRU, GDSF, SizeCost, APC-LRU) miss cost
  - [ ] SubTask 9.2: headroom_rel = headroom_abs / Oracle-Cost miss cost
  - [ ] SubTask 9.3: 165 task group 聚类 bootstrap（1000 次）
  - [ ] SubTask 9.4: 95% CI lower bound
  - [ ] SubTask 9.5: Go/No-Go 判定：headroom_rel ≥ 10% AND CI lower > 0
  - [ ] SubTask 9.6: 输出 `g1prime-verdict.md` + `g1prime-verdict.json`

- [ ] Task 10: 实现 `experiments/g1prime/plot_headroom.py`
  - [ ] SubTask 10.1: 双面板图：miss cost + p95 TTFT vs 容量
  - [ ] SubTask 10.2: 按并发度分面（c=1/4/8）
  - [ ] SubTask 10.3: 输出 `figures/g1prime-headroom.png`

## Phase 5: 测试与验证

- [ ] Task 11: 编写 G1′ verdict 模块单元测试
  - [ ] SubTask 11.1: `tests/test_run_grid.py` — 网格展开、CSV 输出格式
  - [ ] SubTask 11.2: `tests/test_g1prime_verdict.py` — headroom 计算、bootstrap CI、Go/No-Go 判定逻辑

- [ ] Task 12: 端到端验证
  - [ ] SubTask 12.1: 用小样本（10 episodes）跑通全流程
  - [ ] SubTask 12.2: 验证 request 级指标合理性
  - [ ] SubTask 12.3: 验证 headroom 符号正确（Oracle 应优于 simple）

## Phase 6: 全量运行与文档同步

- [ ] Task 13: 全量运行 G1′
  - [ ] SubTask 13.1: 1320 episodes 全网格运行
  - [ ] SubTask 13.2: 生成 verdict 报告
  - [ ] SubTask 13.3: 生成 headroom 图

- [ ] Task 14: 文档同步
  - [ ] SubTask 14.1: 冻结 `experiments/g1/` 为 "diagnostic negative result (protocol-invalid)"
  - [ ] SubTask 14.2: IDEA.rewritten.md §7 添加 G1′ 章节
  - [ ] SubTask 14.3: experiment-designs.md 添加 G1′ 设计
  - [ ] SubTask 14.4: 更新 ccfa.yaml 状态

# Task Dependencies

- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 2]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 1]
- [Task 7] depends on [Task 6]
- [Task 8] depends on [Task 5, Task 6, Task 7]
- [Task 9] depends on [Task 8]
- [Task 10] depends on [Task 8]
- [Task 11] depends on [Task 8, Task 9]
- [Task 12] depends on [Task 11]
- [Task 13] depends on [Task 12]
- [Task 14] depends on [Task 13]

# Parallelizable Work

- [Task 6] 和 [Task 7] 可与 [Task 2] 并行（都只依赖 Task 1）
- [Task 10] 可与 [Task 9] 并行（都依赖 Task 8）
