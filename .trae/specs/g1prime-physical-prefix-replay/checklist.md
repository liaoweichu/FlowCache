# Checklist

## Phase 0: 准备与验证
- [ ] 现有轨迹的 meta/steps/messages 字段已列出，确认包含完整 message history
- [ ] G0 冻结的 Qwen tokenizer + chat template 路径已从 config.yaml 确认
- [ ] 每个 assistant step 的实测 prefill_ms 字段位置已确认
- [ ] `verify_trace_fields.py` 输出字段完整性报告，无缺失字段

## Phase 1: 物理前缀重编译器
- [ ] `recompile_prefixes.py` 加载 G0 冻结的 Qwen tokenizer + chat template
- [ ] 每个 assistant step 重建该 step 之前的完整 message history
- [ ] 使用 chat template 序列化为完整 prompt，tokenize 得到 token IDs + positions
- [ ] 按 16-token block 跨 message 连续切分（不产生 fragment block）
- [ ] `block_hash = H(parent_hash, tokenIds, positions)` 正确计算
- [ ] `parent_hash` 正确指向上一 block 的 hash（根节点 parent_hash = ""）
- [ ] 输出 `request_prefixes.jsonl`，每行一个 request event
- [ ] 单元测试 `test_recompile_prefixes.py` 通过：跨 message 连续分块、parent chain 正确性、跨 workflow 共享前缀
- [ ] request 总数 ≈ 25,653（±10%）
- [ ] block access 总数 ≈ 8.27M（±10%）
- [ ] episode 内前缀重访问比例 ≈ 92.4%（±5%）
- [ ] 跨 workflow 共享 block 数 > 721（远大于当前 G1 的 0.1%）

## Phase 2: 访问流构建与并发模拟
- [ ] `build_physical_access_trace.py` 正确展开 request 为 block 访问序列
- [ ] 每个 block 访问记录 request_id、workflow_id、step_id、arrival_time、prefill_ms
- [ ] 输出 `access_trace.jsonl`
- [ ] `simulate_concurrency.py` 合成全局到达时间
- [ ] 并发度 c=1/4/8 的交错调度正确实现
- [ ] c=1 等价于顺序基线（与当前 G1 行为一致）
- [ ] 工具等待期间的 inactive prefix 与其他 workflow 的 prefix 竞争缓存
- [ ] 输出 `access_trace_c{1,4,8}.jsonl`
- [ ] 单元测试 `test_simulate_concurrency.py` 通过

## Phase 3: 成本模型与容量定义修正
- [ ] bytes_per_block 从 G0 冻结的模型配置计算（不硬编码）
- [ ] per-token prefill rate 从实测 prefill_ms / total_tokens 计算
- [ ] 固定 8ms 回退已禁用，改为 per-token rate × block token 数
- [ ] request 级成本归因：miss_prefill_ms = sum(missed_block_i.prefill_ms_share)
- [ ] 单元测试 `test_cost_model.py` 通过
- [ ] 容量档位为 1 / 2 / 4 / 6 GiB
- [ ] capacity_blocks = floor(C × 1024³ / bytes_per_block)
- [ ] 100%（无限制）仅作 sanity check，不参与 Go/No-Go 判定
- [ ] `config.yaml` 已编写

## Phase 4: G1′ verdict 模块
- [ ] `run_grid.py` 读取 config.yaml（容量档位、并发度、baseline 列表）
- [ ] 全网格运行：6 baselines × 4 容量 × 3 并发 × 1320 episodes
- [ ] 复用 `experiments/e1/compare_oracle.py` 的 baseline 类
- [ ] 收集 request 级指标：miss-prefill token/ms、p50/p95 TTFT、resume hit rate
- [ ] 输出 `results/raw_results.csv`
- [ ] `verdict.py` 正确计算 headroom_abs 和 headroom_rel
- [ ] 165 task group 聚类 bootstrap（1000 次）
- [ ] 95% CI lower bound 正确计算
- [ ] Go/No-Go 判定逻辑：headroom_rel ≥ 10% AND CI lower > 0
- [ ] 输出 `g1prime-verdict.md` + `g1prime-verdict.json`
- [ ] `plot_headroom.py` 生成双面板图（miss cost + p95 TTFT vs 容量）
- [ ] 按并发度分面（c=1/4/8）
- [ ] 输出 `figures/g1prime-headroom.png`

## Phase 5: 测试与验证
- [ ] `test_run_grid.py` 通过：网格展开、CSV 输出格式
- [ ] `test_g1prime_verdict.py` 通过：headroom 计算、bootstrap CI、Go/No-Go 判定逻辑
- [ ] 小样本（10 episodes）端到端跑通
- [ ] request 级指标合理性已验证
- [ ] headroom 符号正确（Oracle 应优于 simple，headroom_abs ≥ 0）

## Phase 6: 全量运行与文档同步
- [ ] 1320 episodes 全网格运行完成
- [ ] verdict 报告已生成
- [ ] headroom 图已生成
- [ ] `experiments/g1/` 已冻结为 "diagnostic negative result (protocol-invalid)"
- [ ] IDEA.rewritten.md §7 已添加 G1′ 章节
- [ ] experiment-designs.md 已添加 G1′ 设计
- [ ] ccfa.yaml 状态已更新

## 关键不变量
- [ ] 现有 `experiments/e1/` 代码只读，未被修改
- [ ] 现有 `experiments/g1/` 冻结，未被修改
- [ ] G1′ 不重新运行 τ-bench 对话（复用现有 1,320 轨迹）
- [ ] G1′ 不需要 GPU（纯 CPU replay）
- [ ] 所有新增代码在 `experiments/g1prime/` 目录下
