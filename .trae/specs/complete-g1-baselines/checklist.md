# Checklist

## Phase 1 — Task 1: cost-aware access trace
- [x] `build_access_trace` 读取每个 step 的 `prefill_ms` 字段
- [x] 块级 `prefill_ms` 按-token 范围比例分摊
- [x] access dict 含 `block_hash`、`parent_hash`、`prefill_ms`、`workflow_id`、`step_id`
- [x] step 缺 `prefill_ms` 时回退到 `block_size × 0.5`

## Phase 1 — Task 2: SizeCostCache
- [x] 类定义存在且结构与 GDSFCache 一致
- [x] `access(block_hash, prefill_ms=0.0)` 命中时 freq+=1、priority 重算
- [x] miss 时 priority = clock + prefill_ms / size
- [x] 驱逐逻辑用 heap lazy deletion
- [x] size 取 token_range_end - token_range_start

## Phase 1 — Task 3: APCLRUCache
- [x] 类定义存在且维护 parent_to_children / child_to_parent
- [x] 命中时更新 last_access，saved_prefill_ms += prefill_ms
- [x] miss 时若 parent 在缓存中，parent 视为 hit（更新 last_access），当前 block 计 miss
- [x] 驱逐时选 last_access 最旧的块 X，递归淘汰 X 的所有 descendant
- [x] evictions 累加 X + descendant 数
- [x] 容量按"块数"计（与 LRU/GDSF 一致）

## Phase 1 — Task 4: OracleCostCache
- [x] 类定义存在且结构与 BeladyOracle 一致
- [x] 额外存储 `block_cost: Dict[str, float]`
- [x] 首次访问时记录 `block_cost[block_hash] = prefill_ms`
- [x] 驱逐时选 `block_cost[h] / next_use_distance(h, current_idx)` 最小的块
- [x] distance=sys.maxsize 时比率视为 0（优先淘汰）
- [x] tie-break：比率相同时回退到 next_use_distance 最大者优先

## Phase 1 — Task 5: main() 集成
- [x] 每个 budget 循环中实例化 SizeCost / APC-LRU / Oracle-Cost
- [x] APC-LRU replay 传入 parent_hash
- [x] Oracle-Cost replay 传入 access_idx
- [x] results JSON 含 `sizecost` / `apc_lru` / `oracle_cost` 三字段
- [x] 旧字段 `lru` / `gdsf` / `oracle` 保留

## Phase 1 — Task 6: _print_summary
- [x] 表格 header 含 SizeCost / APC-LRU / Oracle-Cost 三列
- [x] headroom 改为 `oracle_cost_hr − max(lru, gdsf, sizecost, apc_lru)`
- [x] 同时显示 `oracle_hr`（旧 Belady）与 `oracle_cost_hr`

## Phase 1 — Task 7: plot_characterization.py
- [x] oracle comparison 图扩展为 6 柱
- [x] 不同颜色区分简单策略 / Oracle / closest baseline
- [x] 图标题或图例标注 headroom

## Phase 1 — Task 8: test_baselines.py
- [x] LRUCache capacity=0/1/N 测试通过
- [x] GDSFCache capacity=0/1/N 测试通过
- [x] SizeCostCache cost-aware 驱逐测试通过（cost=10 优先于 cost=100 淘汰）
- [x] APCLRUCache prefix chain 命中测试通过（访问 B 时 A 的 last_access 更新）
- [x] APCLRUCache prefix chain 淘汰测试通过（淘汰 X 时 descendant 也被淘汰）
- [x] BeladyOracle capacity=0/1/N 测试通过
- [x] OracleCostCache cost-aware 驱逐测试通过
- [x] OracleCostCache 与 Belady 对比测试通过（Oracle-Cost miss_cost_ms ≤ Belady）
- [x] build_access_trace cost-aware 测试通过（step prefill_ms=500 含 10 块 → 每块 ≈ 50）

## Phase 1 — Task 9: 重跑与验证
- [x] `python experiments/e1/compare_oracle.py` 成功运行
- [x] `e1-oracle-comparison.json` 含 6 个 baseline 字段
- [x] `oracle_cost.miss_cost_ms ≤ oracle.miss_cost_ms`（Oracle-Cost 不劣于 Belady）
- [x] `oracle_cost.saved_prefill_ms ≥ oracle.saved_prefill_ms`
- [x] headroom 计算结果记录（是否 ≥ 10% 是 G1 判定关键证据）
- [x] `python experiments/e1/plot_characterization.py` 成功生成新图

## Phase 1 — Task 10: 文档同步
- [x] `IDEA.rewritten.md` §8 Ch.1 baseline 描述已具体化
- [x] `experiments/experiment-designs.md` Ch.1 追加 baseline 细节
- [x] cost model 说明（step prefill_ms 按比例分摊）
- [x] APC-LRU prefix chain 语义说明
- [x] Oracle-Cost 比率公式说明

## Phase 2 — Task 11: PBKV 调研
- [x] WebSearch 已搜索 arxiv 2605.06472 GitHub repo
- [x] 调研结论记录到 `experiments/e1/baselines/RESEARCH_NOTES.md`
- [x] 结论含"可用 / 不可用 / 需 inspired variant"之一

## Phase 2 — Task 12: KVFlow 调研
- [x] WebSearch 已搜索 arxiv 2507.07400 GitHub repo
- [x] 调研结论记录到 `experiments/e1/baselines/RESEARCH_NOTES.md`
- [x] 结论含"可用 / 不可用 / 需 inspired variant"之一

## Phase 2 — Task 13: closest baseline 实现
- [x] `experiments/e1/baselines/` 目录已创建
- [x] 至少 1 个 adapter / inspired variant 文件已创建
- [x] 文件 docstring 顶部标注 "INSPIRED VARIANT" 或 "FAITHFUL REPRODUCTION"
- [x] 若为 inspired variant，差异说明已写入 docstring

## Phase 2 — Task 14: closest baseline 集成
- [x] `compare_oracle.py` main() 调用 closest baseline
- [x] results JSON 含 `closest_baseline` 字段
- [x] `_print_summary` 追加 closest baseline 列
- [x] closest baseline 的 hit% 介于最佳简单策略与 Oracle 之间（合理性检查）

## Phase 2 — Task 15: 验证与文档
- [x] 完整 `compare_oracle.py` 运行成功，7+ baseline 均产生结果
- [x] `IDEA.rewritten.md` §8 Ch.1 记录 closest baseline 实现方式
- [x] `experiments/experiment-designs.md` Ch.1 记录 closest baseline 差异说明

## 跨 Phase 验证
- [x] Phase 1 完成后即可独立验收 G1 headroom（Oracle-Cost vs 4 个简单策略）
- [x] Phase 2 完成后满足 G1 第 2 项条件（closest baseline 可比性）
- [x] `pytest experiments/e1/tests/` 全部通过
- [x] `experiments/e1/outputs/e1-oracle-comparison.json` schema 向后兼容（旧字段保留）
- [x] `experiments/e1/figures/e1-oracle-comparison.png` 已重新生成

## 设计一致性验证
- [x] 6 个 baseline 与 IDEA.rewritten.md §8 Ch.4 主表对照（APC-LRU/GDSF/PBKV†/KVFlow†/Oracle-Cost）的实现保持一致
- [x] APC-LRU 的实现与 [vLLM APC](https://docs.vllm.ai/en/v0.14.0/design/prefix_caching/) 的 prefix-aware 语义对齐
- [x] Oracle-Cost 与 Belady 的关系符合"cost-aware 优于 distance-aware"理论预期
- [x] SizeCost 与 GDSF 的关系符合"cost weighting 优于 frequency weighting"理论预期
