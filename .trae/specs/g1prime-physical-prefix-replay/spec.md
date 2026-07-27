# G1′: 物理前缀重编译与正确回放 Spec

## Why

当前 G1 判定为 fail（headroom ≈ 0%），但该判定 **protocol-invalid / inconclusive**：回放协议与 FlowCache 的真实研究对象不一致。具体地：
- 轨迹按单条 message 独立分词，未使用完整 chat template，未跨 message 连续分块；
- 把"块创建"当成"块访问"——每次 assistant 恢复生成时实际访问完整历史前缀，但轨迹只记录每块一次；
- 容量定义脱离 24GB GPU 现实（10% budget ≈ 41.4 GiB KV）；
- 并发、TTFT、bootstrap 统计单位均不正确。

因此需要用已有 1,320 条 τ-bench 轨迹重做 **G1′**：重编译为物理精确前缀访问流，用正确的容量、并发和统计重新判定 Route A/B。**不重新运行 τ-bench 对话。**

## What Changes

### 新增（ADDED）

- **物理前缀重编译器** `experiments/g1prime/recompile_prefixes.py`：
  - 输入：现有 `experiments/e1/traces/bf16/tau_bench/*.json` 轨迹
  - 使用 G0 冻结的 Qwen tokenizer + chat template，重建每次 assistant 调用的完整序列化 prompt
  - 按全局 token 位置重新划分 16-token KV 块，建立精确 parent chain（`block_hash = H(parent_hash, tokenIds, positions)`）
  - 输出：`experiments/g1prime/physical_traces/request_prefixes.jsonl`（每行一个 request event，含完整 block 列表）

- **请求级访问流构建器** `experiments/g1prime/build_physical_access_trace.py`：
  - 每个 assistant 调用生成一个"访问完整前缀"的 request event
  - 预期规模：~25,653 requests、~8.27M block accesses、~7.64M episode 内前缀重访问
  - 输出：`experiments/g1prime/physical_traces/access_trace.jsonl`

- **并发模拟器** `experiments/g1prime/simulate_concurrency.py`：
  - 使用现有 episode 顺序和工具等待时间，合成全局到达时间
  - 测试并发度 1 / 4 / 8（同时活跃 workflow 数）
  - 输出：`experiments/g1prime/physical_traces/access_trace_c{1,4,8}.jsonl`

- **绝对容量定义**：替换百分比 budget 为绝对 KV 容量
  - 容量档位：1 / 2 / 4 / 6 GiB（对应 24GB GPU 的合理 KV 预算）
  - 100%（无限制）仅作 sanity check，**不参与 Go/No-Go 判定**
  - 块大小 = 16 token，BF16 KV 每 block ≈ 0.5 MiB（按 Qwen-7B 层数×头数×维度实测）

- **修正的成本模型**：
  - prefill_ms 归因到 request（而非 output block）
  - 禁用 system/user/tool 块的固定 8ms 回退，改为按 token 数 × 实测 per-token prefill rate
  - 报告 request 级 miss-prefill token / ms

- **G1′ verdict 模块** `experiments/g1prime/verdict.py`：
  - headroom = Oracle-Cost miss cost − max(LRU, GDSF, SizeCost, APC-LRU) miss cost
  - 通过条件：任一 (容量, 并发) 组合下 headroom_rel ≥ 10% AND bootstrap CI lower > 0
  - 统计单位：165 task group × 8 seeds = 1320 个配对样本
  - bootstrap：以 task group 为重采样单位（165 个聚类），1000 次

- **报告产物**：
  - `experiments/g1prime/g1prime-verdict.md` — 判定报告
  - `experiments/g1prime/g1prime-verdict.json` — 机器可读判定
  - `experiments/g1prime/figures/g1prime-headroom.png` — headroom 图
  - `experiments/g1prime/results/raw_results.csv` — 全网格原始结果

### 修改（MODIFIED）

- **G1 状态冻结**：现有 `experiments/g1/` 标注为"diagnostic negative result"，不再修改
- **IDEA.rewritten.md §7 G1**：添加 G1′ 章节，说明 G1 fail 为 protocol-invalid，G1′ 为修正后的判定
- **experiment-designs.md**：添加 G1′ 实验设计（数据集、容量、并发、统计、判定阈值）

### 不做（NOT IN SCOPE）

- 不重新运行 τ-bench 对话（复用现有 1,320 轨迹）
- 不实现联合 R-D 控制器（G1′ 通过后才做，P1-A）
- 不实现 Q8/Q4 量化实验（G2 范围）
- 不训练复用预测器（G1′ 只用 oracle + 简单策略）
- 不恢复 τ-bench reward/termination 标签（P3，非 G1′ 范围）

## Impact

- **Affected specs**:
  - `g1-verdict-module-and-final-sync` — 原 G1 verdict 冻结为诊断性负结果
  - `complete-g1-baselines` — baseline 实现复用（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）
  - `single-dataset-taubench-only` — 数据集不变（τ-bench 1320 episodes）
- **Affected code**:
  - `experiments/e1/record_trajectories.py` — 只读（复用轨迹，不修改）
  - `experiments/e1/compare_oracle.py` — 只读（复用 baseline 类）
  - `experiments/e1/trace_utils.py` — 只读（复用 load_all_trajectories）
  - `experiments/g1/` — 冻结，不再修改
  - `experiments/g1prime/` — **新建目录**
  - `IDEA.rewritten.md` — 添加 G1′ 章节
  - `experiments/experiment-designs.md` — 添加 G1′ 设计
- **Hardware**: 单卡 RTX 4090D 24GB（G1′ 不需要 GPU，纯 CPU replay）

## ADDED Requirements

### Requirement: 物理前缀重编译

系统 SHALL 提供一个重编译器，从现有 τ-bench 轨迹重建物理精确前缀访问流。

#### Scenario: 重建完整请求前缀
- **WHEN** 重编译器处理一条轨迹的某个 assistant step
- **THEN** 系统使用 G0 冻结的 Qwen tokenizer + chat template，将该 step 之前的全部 message history（system + user + assistant + tool）序列化为完整 prompt
- **AND** 按 16-token block 切分，计算每个 block 的 `block_hash = H(parent_hash, tokenIds, positions)` 和 `parent_hash`
- **AND** 输出一个 request event，包含完整 block 列表和该 step 的实测 prefill_ms

#### Scenario: 跨 message 连续分块
- **WHEN** 上一条 message 的末尾 block 未填满 16 token
- **THEN** 下一条 message 的 token 拼接到上一 block，直到填满再开新 block
- **AND** 跨 message 边界不产生 fragment block（符合 vLLM/SGLang 的实际行为）

#### Scenario: 跨 workflow 共享前缀识别
- **WHEN** 两个 request 的 system prompt + tool schema 完全相同
- **THEN** 这两个 request 的根 block 序列产生相同的 block_hash
- **AND** 系统在 access_trace 中记录这些 block 被多次访问

### Requirement: 请求级访问流构建

系统 SHALL 从重编译的 request events 构建请求级访问流，每个 assistant 调用对应一次完整前缀访问。

#### Scenario: 请求展开为 block 访问
- **WHEN** 构建器处理一个 request event（含 N 个 prefix blocks）
- **THEN** 该 request 的所有 N 个 block 按前缀顺序加入 access_trace
- **AND** 每个 block 访问记录 request_id、workflow_id、step_id、arrival_time、prefill_ms

#### Scenario: episode 内前缀复用
- **WHEN** 同一 episode 的第 k 次 assistant 调用访问前缀
- **THEN** 该前缀的 block 与第 1..k-1 次调用的前缀 block 共享 block_hash
- **AND** 系统统计 episode 内前缀重访问比例（预期 ~92.4%）

### Requirement: 并发模拟

系统 SHALL 模拟多 workflow 并发，测试缓存竞争。

#### Scenario: 并发度配置
- **WHEN** 用户指定并发度 c ∈ {1, 4, 8}
- **THEN** 系统合成全局到达时间，使任意时刻最多 c 个 workflow 活跃
- **AND** 活跃 workflow 的 request 按 arrival_time 交错
- **AND** 工具等待期间的 inactive prefix 与其他 workflow 的 prefix 竞争缓存

#### Scenario: 并发度 1（顺序基线）
- **WHEN** c = 1
- **THEN** workflow 完全顺序执行（等价于当前 G1 的行为，作为对照）

### Requirement: 绝对 KV 容量

系统 SHALL 使用绝对 KV 容量（GiB）替代百分比 budget。

#### Scenario: 容量档位
- **WHEN** 运行 G1′ 全网格
- **THEN** 容量档位为 1 / 2 / 4 / 6 GiB
- **AND** 100%（无限制）仅作 sanity check，不参与 Go/No-Go 判定
- **AND** 块大小 = 16 token，BF16 KV 每 block 容量按 Qwen-7B 实测（层数 × 头数 × 维度 × 2 × 2 bytes / 16）

#### Scenario: 容量转 block 数
- **WHEN** 指定容量 C GiB
- **THEN** capacity_blocks = floor(C × 1024³ / bytes_per_block)
- **AND** bytes_per_block 从 G0 冻结的模型配置计算，不硬编码

### Requirement: 修正的成本模型

系统 SHALL 正确归因 prefill 成本。

#### Scenario: request 级成本归因
- **WHEN** 一个 request miss 了 k 个 block
- **THEN** 该 request 的 miss_prefill_ms = sum(missed_block_i.prefill_ms_share)
- **AND** prefill_ms_share 按 block token 数占 request 总 token 数的比例分摊

#### Scenario: 禁用固定回退
- **WHEN** 某个 block 的实测 prefill_ms 缺失或为零
- **THEN** 系统使用 per-token prefill rate（从 G0 实测）× block token 数估算
- **AND** 不使用固定 8ms 回退

### Requirement: G1′ 统计判定

系统 SHALL 以 165 task group × 8 seeds 为统计单位进行判定。

#### Scenario: headroom 计算
- **WHEN** 对每个 (容量, 并发) 组合计算 headroom
- **THEN** headroom_abs = Oracle-Cost miss cost − max(LRU, GDSF, SizeCost, APC-LRU) miss cost
- **AND** headroom_rel = headroom_abs / Oracle-Cost miss cost

#### Scenario: bootstrap 置信区间
- **WHEN** 计算 headroom 的 95% CI
- **THEN** 以 165 task group 为重采样单位（聚类 bootstrap）
- **AND** 1000 次重采样
- **AND** 报告 CI lower bound

#### Scenario: Go/No-Go 判定
- **WHEN** 评估 G1′ 通过条件
- **THEN** 通过条件：存在 (容量, 并发) 组合使 headroom_rel ≥ 10% AND CI lower > 0
- **AND** 若通过：进入 P1-A（联合 R-D 控制器）
- **AND** 若不通过：冻结 G1′ 为负结果，转 Route B（Cacheability Gap Benchmark）

### Requirement: 完整指标报告

系统 SHALL 报告以下 request 级指标：
- miss-prefill token 数 / ms
- p50 / p95 TTFT（request 级，非 block 级）
- resume hit rate（episode 内前缀复用命中率）
- 迁移成本（GPU↔CPU，按 block-seconds）
- GPU block-seconds（缓存占用 × 驻留时间）

## MODIFIED Requirements

### Requirement: G1 状态冻结
现有 `experiments/g1/` 标注为"diagnostic negative result (protocol-invalid)"。所有相关文档（IDEA §7、experiment-designs.md）添加说明：G1 fail 为协议无效结论，G1′ 为修正后的判定。

### Requirement: baseline 复用
G1′ 复用 `experiments/e1/compare_oracle.py` 中的 baseline 类（LRU / GDSF / SizeCost / APC-LRU / Belady / Oracle-Cost）。PBKV-Inspired / ThunderAgent-Inspired 在 G1′ 中为可选项（若 G1′ 通过则补充）。

## REMOVED Requirements

### Requirement: 百分比 budget
**Reason**: 10% budget ≈ 41.4 GiB，脱离 24GB GPU 现实；100% budget 下改善按定义为零，门槛结构上不可能通过。
**Migration**: 替换为绝对 KV 容量（1/2/4/6 GiB），100% 仅作 sanity check。

### Requirement: 3 replay seed bootstrap
**Reason**: 3 个 seed 的聚合值无法提供足够统计功效；应以 165 task group × 8 seeds 为配对样本。
**Migration**: 改为 165 task group 聚类 bootstrap，1000 次。
