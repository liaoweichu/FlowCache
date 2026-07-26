# Complete G1 Baselines Spec

## Why

G1 通过条件（[IDEA.rewritten.md §7 G1](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md)）需要**两项同时满足**：
1. **headroom ≥ 10%**：Oracle 相对最佳简单策略的 miss-cost 或 p95 TTFT 改进空间 → 需要 Oracle-Cost + SizeCost/APC-LRU
2. **closest baseline 可比性**：≥1 个 PBKV/KVFlow 能在公平协议下忠实运行 → 需要 PBKV 或 KVFlow

当前 [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py) 仅实现了 3 个 baseline（LRU、GDSF、Belady），缺 5 个：SizeCost、APC-LRU、Oracle-Cost（Phase 1，纯代码）+ PBKV、KVFlow（Phase 2，需先找官方代码）。三类角色不同：
- **简单启发式**（LRU/GDSF/SizeCost/APC-LRU）：对照组，证明"简单策略不够好"
- **Oracle 上界**（Belady/Oracle-Cost）：理论最优，衡量"headroom 有多大"
- **Closest baseline**（PBKV/KVFlow）：和 FlowCache 最接近的已发表工作，证明"方法比已有的强"

## What Changes

### Phase 1：补全纯代码 baseline（SizeCost + APC-LRU + Oracle-Cost）

- 在 `experiments/e1/compare_oracle.py` 中新增 `SizeCostCache` 类：cost-aware GDSF 变体，priority = clock + saved_prefill_ms / size（用步骤级 `prefill_ms` 而非统一 block_size × 0.5）
- 在 `experiments/e1/compare_oracle.py` 中新增 `APCLRUCache` 类：vLLM APC 风格的 prefix-aware LRU，缓存单元为完整 prefix chain，命中时整链保留，驱逐时整链淘汰（参考 [vLLM APC](https://docs.vllm.ai/en/v0.14.0/design/prefix_caching/)）
- 在 `experiments/e1/compare_oracle.py` 中新增 `OracleCostCache` 类：cost-aware Belady，驱逐时选择 `saved_prefill_ms / next_use_distance` 最小的块（即"重算代价低且下次访问远"的块优先淘汰）
- 修改 `build_access_trace`：从步骤级 `prefill_ms` 估算每块的 saved-prefill cost（按 token_range 长度比例分摊步骤级 prefill_ms），替换当前统一的 `block_size × 0.5`
- 修改 `main()`：在所有预算级别下追加运行 SizeCost / APC-LRU / Oracle-Cost，纳入 results JSON
- 修改 `_print_summary`：表格追加 3 列（SizeCost / APC-LRU / Oracle-Cost 的 hit%），headroom 计算改为 `Oracle-Cost hit% − max(LRU, GDSF, SizeCost, APC-LRU) hit%`
- 修改 [experiments/e1/plot_characterization.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/plot_characterization.py)：oracle vs heuristic 对比柱状图追加 3 个 baseline
- 新增 `experiments/e1/tests/test_baselines.py`：6 个 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）的单元测试，覆盖 capacity=0/1/N、全 miss、全 hit、prefix chain 命中、cost-aware 驱逐等场景
- **BREAKING**：`e1-oracle-comparison.json` 输出 schema 扩展，每个 budget 下新增 `sizecost` / `apc_lru` / `oracle_cost` 三个字段（旧字段 `lru`/`gdsf`/`oracle` 保留以保持向后兼容）

### Phase 2：closest baseline 评估（PBKV + KVFlow）

- 新增 `experiments/e1/baselines/` 目录，存放 closest baseline 的独立适配器
- Phase 2 子任务先调研，后实现：
  - 调研 PBKV（[arxiv 2605.06472](https://arxiv.org/abs/2605.06472)）官方代码可用性：是否有 GitHub repo、是否支持 τ-bench trace 输入、依赖是否可在 4090D 满足
  - 调研 KVFlow（[arxiv 2507.07400](https://arxiv.org/abs/2507.07400)）官方代码可用性：同上
  - 基于调研结果在 `experiments/e1/baselines/` 下实现 `pbkv_adapter.py` 或 `kvflow_adapter.py`（≥1 个）
  - 若官方代码不可用或不兼容，实现明确标注的 `pbkv_inspired.py` / `kvflow_inspired.py`，并在 spec.md 中记录差异
- 在 `compare_oracle.py` 的 `main()` 中追加调用 closest baseline（如可用），纳入 results JSON 与 summary 表
- Phase 2 不阻塞 Phase 1：Phase 1 完成后即可重跑 G1 headroom 验证；Phase 2 用于补全 closest baseline 可比性

### 文档同步

- 在 [IDEA.rewritten.md §8 Ch.1](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) 工作负载画像章节中，将"比较 LRU、size-aware heuristic 与离线 Belady/cost-aware oracle"具体化为"6 个 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）+ ≥1 个 closest baseline（PBKV 或 KVFlow）"
- 在 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) Ch.1 段落中追加 baseline 实现细节与 cost model 说明

## Impact

- Affected specs:
  - [e1-workload-characterization/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/e1-workload-characterization/spec.md): 本 spec 扩展其 "E1 Oracle vs Heuristic 对比" requirement，从 3 baseline 扩展到 6 baseline + closest baseline
- Affected code:
  - [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py): 新增 3 个 cache 类、修改 build_access_trace、main、_print_summary
  - [experiments/e1/plot_characterization.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/plot_characterization.py): 修改 oracle comparison 图
  - 新增 [experiments/e1/tests/test_baselines.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_baselines.py): 6 baseline 单元测试
  - 新增 `experiments/e1/baselines/` 目录（Phase 2）
  - [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.1: baseline 描述具体化
  - [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) Ch.1: 追加 baseline 细节
- 受益：
  - G1 headroom 验证完整：Oracle-Cost 相对 4 个简单策略（LRU/GDSF/SizeCost/APC-LRU）的最佳值，可计算真实 cost-aware headroom
  - G1 closest baseline 可比性满足：PBKV 或 KVFlow 实现后可参与 G1 主表对照
  - 与 §8 Ch.4 主表对照（10 个对照，包括 APC-LRU/GDSF/PBKV†/KVFlow†/Oracle-Cost）的 baseline 实现保持一致
- 风险：
  - PBKV/KVFlow 官方代码可能不可用或不兼容 τ-bench trace 格式 → Phase 2 实现为 inspired variant 并明确标注
  - APC-LRU 的 prefix-chain 命中语义与 vLLM 实际 APC 可能有差异 → 在 spec.md 中记录近似程度
  - Oracle-Cost 的 cost model 基于 step-level `prefill_ms` 按比例分摊，可能不完全等同于真实 per-block 重算代价 → 在 spec.md 中记录近似

## ADDED Requirements

### Requirement: SizeCost baseline（cost-aware GDSF）

系统 SHALL 在 `compare_oracle.py` 中提供 `SizeCostCache` 类，实现 cost-aware GDSF 驱逐策略。

#### Scenario: 块按 cost/size 优先级驱逐
- **WHEN** 缓存满且需驱逐
- **THEN** 优先驱逐 `priority = clock + saved_prefill_ms / size` 最小的块
- **AND** `saved_prefill_ms` 来自步骤级 `prefill_ms` 按 token 范围比例分摊，而非统一 `block_size × 0.5`

#### Scenario: 与 LRU/GDSF 在相同 trace 与预算下对比
- **WHEN** 运行 `main()` 在 budget=0.25 下
- **THEN** results JSON 包含 `sizecost` 字段，记录 hits/misses/hit_rate/evictions/saved_prefill_ms/miss_cost_ms

### Requirement: APC-LRU baseline（prefix-aware LRU）

系统 SHALL 在 `compare_oracle.py` 中提供 `APCLRUCache` 类，实现 vLLM APC 风格的 prefix-aware LRU。

#### Scenario: prefix chain 整链命中
- **WHEN** 访问块 B（其父链为 root→A→B），且 A 已在缓存中
- **THEN** 整链 root→A→B 均视为命中，A 与 B 的 LRU 时间戳更新为当前
- **AND** `saved_prefill_ms` 累加 A 与 B 两块的 cost

#### Scenario: prefix chain 整链淘汰
- **WHEN** 缓存满且需驱逐，选择 LRU 块 X
- **THEN** 块 X 及其所有后裔（descendant）整链淘汰
- **AND** `evictions` 累加淘汰的块数（X + descendant 数）

### Requirement: Oracle-Cost baseline（cost-aware Belady 上界）

系统 SHALL 在 `compare_oracle.py` 中提供 `OracleCostCache` 类，实现 cost-aware Belady 离线最优驱逐策略。

#### Scenario: 块按 cost/distance 比率驱逐
- **WHEN** 缓存满且需驱逐
- **THEN** 优先驱逐 `saved_prefill_ms / next_use_distance` 最小的块（即"重算代价低且下次访问远"的块）
- **AND** 若多个块比率相同，回退到 Belady 的 next_use_distance 最大者优先驱逐

#### Scenario: 离线最优上界
- **WHEN** 运行完整 trace
- **THEN** Oracle-Cost 的 `miss_cost_ms` ≤ Belady 的 `miss_cost_ms`（cost-aware 优于 distance-aware）
- **AND** Oracle-Cost 的 `saved_prefill_ms` ≥ Belady 的 `saved_prefill_ms`

### Requirement: cost-aware access trace

系统 SHALL 修改 `build_access_trace`，使每个 access 的 `prefill_ms` 来自步骤级 `prefill_ms` 按 token 范围比例分摊。

#### Scenario: 块 cost 反映真实 prefill 开销
- **WHEN** 步骤 S 的 `prefill_ms=500`，含 10 个块
- **THEN** 每个块的 `prefill_ms ≈ 50`（按 token_range 长度比例分摊，若所有块等长则为 50）
- **AND** 不再使用 `block_size × 0.5` 的统一估算

### Requirement: 6 baseline 单元测试

系统 SHALL 在 `experiments/e1/tests/test_baselines.py` 中提供 6 个 baseline 类的单元测试。

#### Scenario: capacity 边界
- **WHEN** capacity=0 或 capacity=1
- **THEN** 所有 6 个 baseline 类不抛出异常，且 capacity=0 时所有访问均 miss

#### Scenario: 全 miss 与全 hit
- **WHEN** 访问序列为全不重复块（全 miss）或全相同块（全 hit）
- **THEN** 6 个 baseline 类的 hits/misses 计数正确

#### Scenario: prefix chain 语义（仅 APC-LRU）
- **WHEN** 访问 root→A→B 链，A 已缓存
- **THEN** APC-LRU 视为 A 与 B 均命中；LRU/GDSF/SizeCost 仅 B 命中（A 不更新）

#### Scenario: cost-aware 驱逐（SizeCost / Oracle-Cost）
- **WHEN** 缓存满，存在 cost=10/distance=2 与 cost=100/distance=2 两个候选淘汰
- **THEN** SizeCost 优先淘汰 cost=10 的块；Oracle-Cost 优先淘汰 cost=10/distance=2 的块（比率较小）

### Requirement: closest baseline 评估（Phase 2）

系统 SHALL 在 `experiments/e1/baselines/` 下实现 ≥1 个 closest baseline（PBKV 或 KVFlow），或在官方代码不可用时实现明确标注的 inspired variant。

#### Scenario: PBKV 或 KVFlow 忠实运行
- **WHEN** Phase 2 调研确认 PBKV 或 KVFlow 官方代码可用且兼容 τ-bench trace
- **THEN** 实现对应 adapter（如 `pbkv_adapter.py`），在 `main()` 中调用并纳入 results JSON

#### Scenario: inspired variant 标注
- **WHEN** 官方代码不可用或不兼容
- **THEN** 实现 `pbkv_inspired.py` 或 `kvflow_inspired.py`，在文件 docstring 顶部明确标注 "INSPIRED VARIANT — 与原论文实现的差异如下：..."
- **AND** 在 [IDEA.rewritten.md §8 Ch.1](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) 中记录"inspired variant"事实

## MODIFIED Requirements

### Requirement: E1 Oracle vs Heuristic 对比

**原（[e1-workload-characterization/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/e1-workload-characterization/spec.md)）**：
系统 SHALL 提供脚本 `experiments/e1/compare_oracle.py`，在 open-loop replay 下实现 LRU、size-aware GDSF 和离线 Belady oracle，比较 saved-prefill ms 和 miss-cost。

**现（本 spec）**：
系统 SHALL 提供脚本 `experiments/e1/compare_oracle.py`，在 open-loop replay 下实现 **6 个 baseline**（LRU、GDSF、SizeCost、APC-LRU、Belady、Oracle-Cost）+ **≥1 个 closest baseline**（PBKV 或 KVFlow），比较 saved-prefill ms 和 miss-cost。headroom 计算改为 `Oracle-Cost − max(LRU, GDSF, SizeCost, APC-LRU)`。

### Requirement: E1 可视化

**原**：oracle vs LRU/GDSF saved-prefill 对比柱状图

**现**：oracle vs 4 个简单策略（LRU/GDSF/SizeCost/APC-LRU）saved-prefill 对比柱状图，外加 Oracle-Cost 上界线

## REMOVED Requirements

（无移除项；本 spec 为纯增量扩展）
