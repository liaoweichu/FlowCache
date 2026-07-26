# Tasks

## Phase 1：纯代码 baseline（SizeCost + APC-LRU + Oracle-Cost）

- [x] Task 1: 修改 `experiments/e1/compare_oracle.py` 的 `build_access_trace` 函数
  - [x] SubTask 1.1: 读取每个 step 的 `prefill_ms` 字段（若不存在则回退到 `block_size × 0.5`）
  - [x] SubTask 1.2: 按 `token_range_end - token_range_start` 比例分摊 step 级 `prefill_ms` 到该 step 的每个 block
  - [x] SubTask 1.3: 在每个 access dict 中存 `prefill_ms`（块级）与 `parent_hash`、`block_hash`（供 APC-LRU 使用）
  - [x] SubTask 1.4: 保留 `workflow_id`、`step_id` 字段不变

- [x] Task 2: 新增 `SizeCostCache` 类到 `experiments/e1/compare_oracle.py`
  - [x] SubTask 2.1: 类结构与 `GDSFCache` 一致（capacity / cache dict / heap / clock / 计数器）
  - [x] SubTask 2.2: `access(block_hash, prefill_ms=0.0)` 方法：命中时增加 freq、重算 priority = clock + prefill_ms_accumulated / size；miss 时插入 freq=1、priority = clock + prefill_ms / size
  - [x] SubTask 2.3: size 取块 token 数（来自 token_range_end - token_range_start），uniform 时退化为 GDSF + cost weighting
  - [x] SubTask 2.4: 驱逐逻辑与 GDSF 相同（heap lazy deletion）

- [x] Task 3: 新增 `APCLRUCache` 类到 `experiments/e1/compare_oracle.py`
  - [x] SubTask 3.1: 类结构：capacity / cache dict (block_hash → {parent_hash, children: set, last_access}) / 计数器
  - [x] SubTask 3.2: 维护 `parent_to_children: Dict[str, Set[str]]` 与 `child_to_parent: Dict[str, str]`
  - [x] SubTask 3.3: `access(block_hash, parent_hash, prefill_ms=0.0)`：若 block 在缓存中，命中，更新 last_access，saved_prefill_ms += prefill_ms；若不在缓存中且 parent_hash 在缓存中，视为"prefix 命中"，但仍需为当前 block 计 miss（仅 parent 视为 hit 并更新 last_access）；若 parent 也不在，整链 miss
  - [x] SubTask 3.4: 驱逐时选 last_access 最旧的块 X，递归淘汰 X 的所有 descendant（通过 parent_to_children），evictions += 淘汰数
  - [x] SubTask 3.5: 容量按"块数"计（与 LRU/GDSF 一致），不按 chain 数计

- [x] Task 4: 新增 `OracleCostCache` 类到 `experiments/e1/compare_oracle.py`
  - [x] SubTask 4.1: 类结构与 `BeladyOracle` 一致（capacity / future_accesses / cache set / 计数器）
  - [x] SubTask 4.2: 额外存储 `block_cost: Dict[str, float]`（首次访问时记录该块的 saved_prefill_ms）
  - [x] SubTask 4.3: `access(block_hash, access_idx, prefill_ms=0.0)`：若首次见到该 block，记录 `block_cost[block_hash] = prefill_ms`
  - [x] SubTask 4.4: 驱逐时选 `block_cost[h] / next_use_distance(h, current_idx)` 最小的块（distance=sys.maxsize 时比率视为 0，优先淘汰）
  - [x] SubTask 4.5: tie-break：比率相同时回退到 Belady 的 next_use_distance 最大者优先淘汰

- [x] Task 5: 修改 `experiments/e1/compare_oracle.py` 的 `main()` 函数
  - [x] SubTask 5.1: 在每个 budget 循环中，追加 SizeCost / APC-LRU / Oracle-Cost 的实例化与 replay
  - [x] SubTask 5.2: APC-LRU replay 需传入 `parent_hash`（从 access dict 读取）
  - [x] SubTask 5.3: Oracle-Cost replay 需传入 `access_idx`（与 Belady 一致）
  - [x] SubTask 5.4: results JSON 在每个 budget 下新增 `sizecost` / `apc_lru` / `oracle_cost` 三个字段
  - [x] SubTask 5.5: 保留旧字段 `lru` / `gdsf` / `oracle`（向后兼容）

- [x] Task 6: 修改 `experiments/e1/compare_oracle.py` 的 `_print_summary` 函数
  - [x] SubTask 6.1: 表格 header 追加 SizeCost / APC-LRU / Oracle-Cost 三列
  - [x] SubTask 6.2: headroom 计算改为 `oracle_cost_hr − max(lru_hr, gdsf_hr, sizecost_hr, apc_lru_hr)`
  - [x] SubTask 6.3: 同时打印 `oracle_hr`（旧 Belady）与 `oracle_cost_hr` 两列，便于对比

- [x] Task 7: 修改 `experiments/e1/plot_characterization.py`
  - [x] SubTask 7.1: 找到 oracle comparison 柱状图相关函数
  - [x] SubTask 7.2: 将 3 柱（LRU/GDSF/Oracle）扩展为 6 柱（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）
  - [x] SubTask 7.3: 用不同颜色区分简单策略 / Oracle / closest baseline 三类
  - [x] SubTask 7.4: 在图标题或图例中标注 headroom（Oracle-Cost − 最佳简单策略）

- [x] Task 8: 新增 `experiments/e1/tests/test_baselines.py`
  - [x] SubTask 8.1: 测试 `LRUCache`（capacity=0/1/N、全 miss、全 hit）—— 已有实现，仅补测
  - [x] SubTask 8.2: 测试 `GDSFCache`（同上）
  - [x] SubTask 8.3: 测试 `SizeCostCache`：cost-aware 驱逐（cost=10 vs cost=100，优先淘汰 cost=10）
  - [x] SubTask 8.4: 测试 `APCLRUCache`：prefix chain 命中（访问 root→A→B，A 已缓存时 A 的 last_access 更新）；prefix chain 淘汰（淘汰 X 时 descendant 也被淘汰）
  - [x] SubTask 8.5: 测试 `BeladyOracle`（已有实现，仅补测）
  - [x] SubTask 8.6: 测试 `OracleCostCache`：cost-aware 驱逐（cost=10/distance=2 vs cost=100/distance=2，优先淘汰 cost=10）；与 Belady 对比，Oracle-Cost 的 miss_cost_ms ≤ Belady
  - [x] SubTask 8.7: 测试 `build_access_trace` 的 cost-aware 行为：step prefill_ms=500 含 10 块时，每块 prefill_ms ≈ 50

- [x] Task 9: 重跑 E1 oracle comparison 并验证 headroom
  - [x] SubTask 9.1: 运行 `python experiments/e1/compare_oracle.py` 重新生成 `e1-oracle-comparison.json`
  - [x] SubTask 9.2: 验证 `oracle_cost.miss_cost_ms ≤ oracle.miss_cost_ms`（Oracle-Cost 不劣于 Belady）
  - [x] SubTask 9.3: 计算 headroom = `oracle_cost hit% − max(lru, gdsf, sizecost, apc_lru) hit%`，记录是否 ≥ 10%
  - [x] SubTask 9.4: 运行 `python experiments/e1/plot_characterization.py` 重新生成对比图

- [x] Task 10: 同步文档
  - [x] SubTask 10.1: 修改 `IDEA.rewritten.md` §8 Ch.1：将"比较 LRU、size-aware heuristic 与离线 Belady/cost-aware oracle"改为"6 个 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）+ ≥1 个 closest baseline"
  - [x] SubTask 10.2: 修改 `experiments/experiment-designs.md` Ch.1：追加 baseline 实现细节、cost model 说明、APC-LRU prefix chain 语义、Oracle-Cost 比率公式

## Phase 2：closest baseline 评估（PBKV + KVFlow）

- [x] Task 11: 调研 PBKV 官方代码可用性
  - [x] SubTask 11.1: 搜索 arxiv 2605.06472 对应的 GitHub repo（用 WebSearch）
  - [x] SubTask 11.2: 若找到，检查是否支持 τ-bench trace 输入格式（JSON with block_assignments）
  - [x] SubTask 11.3: 检查依赖（PyTorch / GraphSAGE 库等）是否可在 RTX 4090D + 24GB 显存满足
  - [x] SubTask 11.4: 输出调研结论到 `experiments/e1/baselines/RESEARCH_NOTES.md`（"PBKV 官方代码可用/不可用/需 inspired variant"）

- [x] Task 12: 调研 KVFlow 官方代码可用性
  - [x] SubTask 12.1: 搜索 arxiv 2507.07400 对应的 GitHub repo
  - [x] SubTask 12.2: 若找到，检查是否支持 τ-bench trace 输入格式
  - [x] SubTask 12.3: 检查依赖
  - [x] SubTask 12.4: 输出调研结论到 `experiments/e1/baselines/RESEARCH_NOTES.md`

- [x] Task 13: 实现 closest baseline（基于 Task 11/12 调研结果）
  - [x] SubTask 13.1: 若 PBKV 官方代码可用，实现 `experiments/e1/baselines/pbkv_adapter.py`，包装官方代码接受 τ-bench trace
  - [x] SubTask 13.2: 若 KVFlow 官方代码可用，实现 `experiments/e1/baselines/kvflow_adapter.py`
  - [x] SubTask 13.3: 若两者均不可用，实现 `experiments/e1/baselines/pbkv_inspired.py`（GraphSAGE 风格的 reuse score 预测，明确标注与原论文差异）
  - [x] SubTask 13.4: 在文件 docstring 顶部标注 "INSPIRED VARIANT" 或 "FAITHFUL REPRODUCTION"

- [x] Task 14: 集成 closest baseline 到 `compare_oracle.py`
  - [x] SubTask 14.1: 在 `main()` 中追加调用 closest baseline（如可用）
  - [x] SubTask 14.2: results JSON 追加 `closest_baseline` 字段
  - [x] SubTask 14.3: `_print_summary` 追加 closest baseline 列
  - [x] SubTask 14.4: 验证 closest baseline 的 hit% 介于最佳简单策略与 Oracle 之间（否则说明实现有问题）

- [x] Task 15: Phase 2 验证与文档
  - [x] SubTask 15.1: 运行完整 `compare_oracle.py`，确认 7+ baseline 均产生结果
  - [x] SubTask 15.2: 在 `IDEA.rewritten.md` §8 Ch.1 中记录 closest baseline 的实现方式（faithful / inspired）
  - [x] SubTask 15.3: 在 `experiments/experiment-designs.md` Ch.1 中记录 closest baseline 的差异说明

# Task Dependencies

## Phase 1（串行 + 并行混合）
- Task 1 必须先完成（其他 Task 依赖 cost-aware access trace）
- Task 2, 3, 4 可并行（独立类实现）
- Task 5 依赖 Task 2, 3, 4
- Task 6 依赖 Task 5
- Task 7 依赖 Task 5（读取新 results JSON）
- Task 8 依赖 Task 2, 3, 4（测试新类）
- Task 9 依赖 Task 5, 6, 7, 8
- Task 10 依赖 Task 9（确认结果后再同步文档）

## Phase 2（依赖 Phase 1 完成）
- Task 11, 12 可并行（独立调研）
- Task 13 依赖 Task 11, 12（基于调研结果选实现路径）
- Task 14 依赖 Task 13
- Task 15 依赖 Task 14

## 跨 Phase 依赖
- Phase 2 不阻塞 Phase 1 验收：Phase 1 完成后即可重跑 G1 headroom 验证
- Phase 2 完成后才满足 G1 第 2 项条件（closest baseline 可比性）
