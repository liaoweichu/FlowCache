# G1 Verdict Module & Final Sync Spec

## Why

G1 实验复用 Ch.1（E1）画像数据，不独立运行（IDEA §8 Ch.1 line 617）。当前 `experiments/e1/compare_oracle.py` 已集成 8 个 baseline（LRU/GDSF/SizeCost/APC-LRU/PBKV-inspired/ThunderAgent-inspired/Belady/Oracle-Cost），但 **G1.9 步骤 1.5（全网格运行）和 1.6（统计分析与判定）缺少代码**——`experiments/g1/` 目录尚不存在，无法产出 G1.11 预期产物（`results/*.csv` / `figures/g1-headroom.png` / `g1-verdict.md`）。同时 IDEA §8 Ch.1 与 experiment-designs.md G1 章节的文档已在前几次会话中同步，但 §8 Ch.4 主表对照 #4 的脚注仍需补充 KVFlow faithful 状态。

本 spec 创建 G1 verdict 模块（grid runner + verdict 报告生成器 + headroom 绘图），完成 G1 阶段所有剩余代码，并同步 IDEA/md/yaml 文档的最终细节。

## What Changes

### 新增代码（G1 阶段剩余代码）

- 新增 [experiments/g1/](file:///d:/00MyProject/Prefix%20Caching/experiments/g1/) 目录及以下文件：
  - `__init__.py`：包标识
  - `run_grid.py`：全网格运行器（策略 × 预算档位 × 数据集 × 3 replay 种子），复用 `experiments/e1/compare_oracle.py` 的 baseline 实现，输出 `experiments/g1/results/raw_results.csv`
  - `verdict.py`：G1 判定报告生成器，读取 `raw_results.csv`，计算 headroom（Oracle-Cost − max(LRU, GDSF, SizeCost, APC-LRU)）、paired workflow-level bootstrap 95% CI、Bonferroni 校正，产出 `experiments/g1/g1-verdict.md` 与 `experiments/g1/g1-verdict.json`
  - `plot_headroom.py`：headroom 图绘制（策略 × 预算），产出 `experiments/g1/figures/g1-headroom.png`
  - `config.yaml`：G1 实验配置（复用 E1 trace、3 replay 种子、预算档位 10%/25%/50%/100%）
  - `tests/test_g1_verdict.py`：verdict 模块单元测试（headroom 计算、bootstrap CI、Bonferroni 校正）
  - `tests/test_run_grid.py`：grid runner 单元测试（配置加载、网格展开、CSV 输出格式）

### 文档更新（md / yaml）

- 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.4 主表对照 #4 脚注：补充 "KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活，adapter 实现中"
- 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.9 步骤 1.5/1.6：从 "TBD" 改为指向 `experiments/g1/run_grid.py` 与 `experiments/g1/verdict.py`
- 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.11 预期产物表：补充 `experiments/g1/run_grid.py` 等代码路径
- 修改 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) baselines 段：在顶部注释中补充"G1 verdict 模块通过 `experiments/g1/run_grid.py` 调用本配置"

## Impact

- Affected specs:
  - [complete-g1-baselines/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/complete-g1-baselines/spec.md): 已完成 PBKV/ThunderAgent inspired 实现，本 spec 提供 G1 verdict 模块以完成 G1.9 步骤 1.5/1.6
  - [sync-thunderagent-baseline-choice/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/sync-thunderagent-baseline-choice/spec.md): 已完成 baseline 文档同步，本 spec 补充 G1 代码层
- Affected code:
  - 新增 `experiments/g1/` 目录及 7 个文件（run_grid.py / verdict.py / plot_headroom.py / config.yaml / __init__.py + 2 测试）
  - 修改 `IDEA.rewritten.md` §8 Ch.4 脚注
  - 修改 `experiments/experiment-designs.md` G1.9 / G1.11
  - 修改 `experiments/e1/config.yaml` baselines 段注释
- 受益：
  - G1.9 步骤 1.5/1.6 可执行：`python experiments/g1/run_grid.py` 一键产出 raw_results.csv，`python experiments/g1/verdict.py` 产出 g1-verdict.md
  - G1.11 预期产物全部有代码支撑
  - 文档与代码一致，IDEA §8 Ch.1 "Gate 复用" 声明可验证
- 风险：
  - G1 grid runner 复用 E1 trace（28/1320 episodes），当前 trace 不足，verdict 报告会标注 "trace 不足，结果为 pilot"
  - KVFlow faithful adapter 未实现，grid runner 中 kvflow_faithful 行标记为 "pending"，不阻塞其他 baseline 运行

## ADDED Requirements

### Requirement: G1 grid runner

系统 SHALL 提供 `experiments/g1/run_grid.py`，全网格运行 G1.4 表中所有 enabled baseline × 预算档位 × 数据集 × 3 replay 种子。

#### Scenario: 全网格运行
- **WHEN** 执行 `python experiments/g1/run_grid.py`
- **THEN** 读取 `experiments/g1/config.yaml` 获取 baseline 列表、预算档位、种子数
- **AND** 对每个 (baseline, budget, dataset, seed) 组合，调用 `experiments/e1/compare_oracle.py` 的 baseline 实现
- **AND** 输出 `experiments/g1/results/raw_results.csv`，含列：baseline, budget, dataset, seed, hits, misses, hit_rate, evictions, saved_prefill_ms, miss_cost_ms, p95_ttft_ms

#### Scenario: KVFlow faithful 跳过
- **WHEN** grid runner 遇到 `kvflow_faithful`（adapter 未实现）
- **THEN** 在 CSV 中输出一行 `kvflow_faithful, status=pending, reason=adapter_not_implemented`
- **AND** 不阻塞其他 baseline 运行

#### Scenario: trace 不足警告
- **WHEN** 可用 τ-bench trace 数 < 1320
- **THEN** 在 CSV 顶部注释中标注 "pilot: N/1320 episodes available"
- **AND** 在 stdout 输出警告

### Requirement: G1 verdict 报告生成器

系统 SHALL 提供 `experiments/g1/verdict.py`，读取 raw_results.csv 并产出 G1 判定报告。

#### Scenario: headroom 计算
- **WHEN** 执行 `python experiments/g1/verdict.py`
- **THEN** 读取 `experiments/g1/results/raw_results.csv`
- **AND** 对每个 (budget, dataset) 组合，计算 headroom = Oracle-Cost.miss_cost − max(LRU, GDSF, SizeCost, APC-LRU).miss_cost
- **AND** 计算 headroom 相对差 = headroom / Oracle-Cost.miss_cost
- **AND** 输出 `experiments/g1/g1-verdict.md`，含 G1.11.1 表 G1-1 模板填充结果

#### Scenario: bootstrap 95% CI
- **WHEN** verdict 模块计算 headroom
- **THEN** 用 paired workflow-level bootstrap（1000 次）计算 95% CI
- **AND** 多预算档位视为同一家族检验，Bonferroni 校正

#### Scenario: Go/No-Go 判定
- **WHEN** verdict 模块生成报告
- **THEN** 对照 G1.8 判定阈值：headroom ≥ 10% → 通过第一项；closest baseline 可比性 → 通过第二项
- **AND** 输出 `experiments/g1/g1-verdict.json`，含 `go_no_go: {headroom: pass/fail, comparability: pass/fail}`

### Requirement: G1 headroom 绘图

系统 SHALL 提供 `experiments/g1/plot_headroom.py`，绘制 headroom 图。

#### Scenario: 绘制 headroom 图
- **WHEN** 执行 `python experiments/g1/plot_headroom.py`
- **THEN** 读取 `experiments/g1/results/raw_results.csv`
- **AND** 绘制策略 × 预算的 miss_cost 与 p95 TTFT 图
- **AND** 输出 `experiments/g1/figures/g1-headroom.png`

### Requirement: G1 config.yaml

系统 SHALL 提供 `experiments/g1/config.yaml`，定义 G1 实验配置。

#### Scenario: 配置含预算档位与种子
- **WHEN** 读取 `experiments/g1/config.yaml`
- **THEN** 含 `budgets: [0.10, 0.25, 0.50, 1.00]`
- **AND** 含 `replay_seeds: [1, 2, 3]`
- **AND** 含 `datasets: ["tau_bench"]`
- **AND** 含 `baselines:` 段引用 `experiments/e1/config.yaml` 的 baseline 列表

### Requirement: G1 单元测试

系统 SHALL 提供单元测试覆盖 verdict 与 grid runner 的核心逻辑。

#### Scenario: verdict 单元测试
- **WHEN** 运行 `python -m pytest experiments/g1/tests/test_g1_verdict.py`
- **THEN** 测试 headroom 计算、bootstrap CI、Bonferroni 校正、Go/No-Go 判定逻辑
- **AND** 所有测试通过

#### Scenario: grid runner 单元测试
- **WHEN** 运行 `python -m pytest experiments/g1/tests/test_run_grid.py`
- **THEN** 测试配置加载、网格展开、CSV 输出格式、KVFlow 跳过逻辑
- **AND** 所有测试通过

## MODIFIED Requirements

### Requirement: IDEA §8 Ch.4 主表对照 #4 脚注

**原**：†：G1.4.1 判定后填入实际可用的 closest baseline 名称。PBKV-inspired 与 ThunderAgent-inspired 均为 inspired variant（无官方代码 / API 级代理非块级缓存），KVFlow 为 faithful reproduction（待 WSL2 adapter 实现）。

**现**：†：G1.4.1 判定后填入实际可用的 closest baseline 名称。PBKV-inspired 与 ThunderAgent-inspired 均为 inspired variant（无官方代码 / API 级代理非块级缓存），KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活（`config.yaml: kvflow_faithful.enabled: true`），adapter 实现中。

### Requirement: experiment-designs.md G1.9 步骤 1.5/1.6

**原**：
| 1.5 | 全网格运行（策略 × 预算 × 数据集 × 3 种子） | 原始结果表 |
| 1.6 | 统计分析与判定 | `experiments/g1/g1-verdict.md` |

**现**：
| 1.5 | 全网格运行（策略 × 预算 × 数据集 × 3 种子） | 原始结果表（`experiments/g1/run_grid.py` → `results/raw_results.csv`） |
| 1.6 | 统计分析与判定 | `experiments/g1/verdict.py` → `experiments/g1/g1-verdict.md` + `g1-verdict.json` |

## REMOVED Requirements

（无移除项）
