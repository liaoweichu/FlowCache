# E1 工作负载画像与缓存机会分析 Spec

## Why
G1 gate 要求证明真实 Agent workload 中存在非平凡的 exact-prefix 再访问，且离线 oracle 明显优于 LRU/简单启发式。E1 是 G1 的中心证据，必须在主论文中呈现（不能放附录）。当前 E1 只有 IDEA Section 8 的概要描述，缺乏可执行的代码和与 ccfa.yaml 的同步。

## What Changes
- 细化 E1 实验设计：具体化数据采集协议、指标计算方式、输出格式
- 生成 E1 实验代码：轨迹录制、block 索引、overlap 统计、oracle vs LRU 对比、可视化
- 同步 ccfa.yaml：更新 E1 experiment 字段，关联 G1 gate

## Impact
- Affected specs: ccfa.yaml（E1 experiment 字段更新）
- Affected code: `experiments/e1/` 目录（新建）
- Affected docs: `experiments/g2-pilot-design.md`（E1 的轨迹录制代码可被 G2 Pilot Phase 1 复用）

## ADDED Requirements

### Requirement: E1 轨迹录制脚本
系统 SHALL 提供一个脚本 `experiments/e1/record_trajectories.py`，使用 Qwen3-8B-Instruct 对 τ-bench 子集（80 workflows）逐个录制 BF16 完整轨迹，记录每一步的 token IDs、工具调用、工具结果、block 分配日志。

#### Scenario: 成功录制所有 workflow
- **WHEN** 用户运行 `python experiments/e1/record_trajectories.py`
- **THEN** 80 个 workflow 的轨迹存入 `experiments/e1/traces/bf16/`，每个 workflow 一个 JSON 文件

#### Scenario: 轨迹包含 block 信息
- **WHEN** 录制完成
- **THEN** 每条轨迹包含 token IDs（按 step 组织）、block hash 列表、prefill 时间测量

### Requirement: E1 工作负载画像脚本
系统 SHALL 提供一个脚本 `experiments/e1/characterize_workload.py`，从录制的轨迹计算并输出 E1 规定的所有指标。

#### Scenario: 输出画像报告
- **WHEN** 用户运行 `python experiments/e1/characterize_workload.py`
- **THEN** 输出包含以下指标的 JSON 报告和 Markdown 表格：
  - workflow 长度/深度/宽度/分支率/工具等待时长
  - exact-prefix overlap ratio、LCP tokens 分布
  - next-use distance 分布（按 block 统计）
  - block working-set size、KV/总显存占比
  - oracle vs LRU heuristic 的 saved-prefill headroom

### Requirement: E1 Oracle vs Heuristic 对比
系统 SHALL 提供脚本 `experiments/e1/compare_oracle.py`，在 open-loop replay 下实现 LRU、size-aware GDSF 和离线 Belady oracle，比较 saved-prefill ms 和 miss-cost。

#### Scenario: Oracle 明显优于 LRU
- **WHEN** 执行对比
- **THEN** oracle 相对最佳简单策略存在约 10% 的 miss-cost 或 p95 TTFT 改进空间（否则触发 G1 失败条件）

### Requirement: E1 可视化
系统 SHALL 提供脚本 `experiments/e1/plot_characterization.py` 生成以下图表：
  - exact-prefix overlap 分布直方图
  - next-use distance CDF
  - KV working-set size 时序图
  - oracle vs LRU/GDSF saved-prefill 对比柱状图

### Requirement: ccfa.yaml 同步
E1 experiment 的 status 和 metadata SHALL 更新到 ccfa.yaml 中，关联 G1 gate。

#### Scenario: ccfa.yaml 包含 E1 字段
- **WHEN** spec 实现完成
- **THEN** ccfa.yaml 中 E1 experiment 包含 id、description、status、week、gates、scripts 字段，G1 gate 的 depends_on 更新为包含 E1
