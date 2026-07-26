# Sync ThunderAgent-Inspired Baseline Choice Spec

## Why

[reviews/closest-baseline-code-search.md](file:///d:/00MyProject/Prefix%20Caching/reviews/closest-baseline-code-search.md) 对 IDEA.rewritten.md §3.1 表中 12 项 closest baseline 候选论文做了系统开源代码调研，筛选出 **ThunderAgent**（ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑，NVIDIA Dynamo 2.0 集成）作为补充 closest baseline，并已在代码层实现 [experiments/e1/baselines/thunderagent_inspired.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/thunderagent_inspired.py) + 9 个单元测试 + 集成到 [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py)。

但设计层文档（IDEA.rewritten.md §8、experiments/experiment-designs.md §0.10 通用 baseline 名录、experiments/e1/config.yaml）尚未同步这一选用改变——仍只列 PBKV/KVFlow 作为 closest baseline。本 spec 把这一选用改变正式写入 yaml 与 md 文件，使设计文档与代码实现一致。

## What Changes

### 文档同步（md 文件）

- 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.1（line 615）：将"6 个 baseline + ≥1 个 closest baseline（PBKV 或 KVFlow）"扩展为"6 个 baseline + 2 个 inspired closest baseline（PBKV-inspired + ThunderAgent-inspired）+ KVFlow faithful（待 WSL2 adapter）"
- 修改 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.4 主表（line 673，对照 #4）：将"KVFlow† 或 PBKV†"扩展为"KVFlow† / PBKV† / ThunderAgent†"，脚注补充 ThunderAgent-inspired 的差异说明
- 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10 通用 baseline 名录（line 416）：在 KVFlow† / PBKV† 行后新增 ThunderAgent-inspired 行，明确标注 inspired variant 与差异
- 修改 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10 脚注（line 424）：补充 ThunderAgent-inspired 的可比性判定结论（API 级代理非块级缓存 → inspired variant）

### 配置同步（yaml 文件）

- 修改 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml)：新增 `baselines` 配置段，列出 compare_oracle.py 中启用的全部 baseline 及其类型标签（simple_heuristic / oracle / closest_inspired / closest_faithful_pending），使配置文件成为 baseline 选用的单一事实源

### 已完成项（不在本 spec 范围内，仅记录事实）

- [experiments/e1/baselines/thunderagent_inspired.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/thunderagent_inspired.py)（258 行，已实现）
- [experiments/e1/baselines/test_thunderagent_inspired.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/test_thunderagent_inspired.py)（9 个单元测试，全部通过）
- [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py) 集成 thunderagent_inspired 字段
- [experiments/e1/baselines/RESEARCH_NOTES.md](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/RESEARCH_NOTES.md) ThunderAgent 调研章节
- [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.4 / G1.4.1 / G1.9 / G1.11.1 已更新（前一会话完成）

## Impact

- Affected specs:
  - [complete-g1-baselines/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/complete-g1-baselines/spec.md): 本 spec 是其后续同步——complete-g1-baselines 已完成 PBKV-inspired 实现，本 spec 把 ThunderAgent-inspired 的选用同步到设计文档与配置
- Affected code:
  - [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.1 / Ch.4: baseline 描述扩展
  - [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10: 通用 baseline 名录新增 ThunderAgent-inspired
  - [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml): 新增 baselines 配置段
- 受益：
  - 设计文档与代码实现一致：读者从 IDEA / experiment-designs / config.yaml 任一入口都能看到 ThunderAgent-inspired 是已启用的 closest baseline
  - config.yaml 成为 baseline 选用的单一事实源，后续新增/停用 baseline 只改 yaml 即可
  - G1 第 2 项条件（closest baseline 可比性）的判定证据更充分：2 个 inspired variant + 1 个 faithful pending
- 风险：
  - ThunderAgent-inspired 是 inspired variant（API 级代理非块级缓存），不能等同于 faithful ThunderAgent → 所有文档明确标注 "inspired" 与差异说明
  - config.yaml 新增 baselines 段是纯描述性配置，compare_oracle.py 当前不读取该段（baseline 列表硬编码在 main() 中）→ 在 spec 中明确说明这一点，避免误以为改 yaml 即可增减 baseline

## ADDED Requirements

### Requirement: ThunderAgent-inspired 出现在通用 baseline 名录

系统 SHALL 在 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.10 通用 baseline 名录中新增 ThunderAgent-inspired 行。

#### Scenario: 名录含 ThunderAgent-inspired
- **WHEN** 读者查阅 §0.10 通用 baseline 名录
- **THEN** 表格中存在 ThunderAgent-inspired 行
- **AND** 类型列标注为"最近工作 baseline（inspired variant）"
- **AND** 描述列含"ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；API 级代理非块级缓存，提取 program-aware + 2^{-t} time decay 核心 idea"

### Requirement: IDEA §8 Ch.1 / Ch.4 baseline 描述扩展

系统 SHALL 在 [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) §8 Ch.1 与 Ch.4 中将 closest baseline 描述从"PBKV 或 KVFlow"扩展为含 ThunderAgent-inspired。

#### Scenario: Ch.1 headroom 描述含 ThunderAgent-inspired
- **WHEN** 读者查阅 §8 Ch.1 报告指标
- **THEN** baseline 描述为"6 个 baseline + 2 个 inspired closest baseline（PBKV-inspired + ThunderAgent-inspired）+ KVFlow faithful（待 WSL2 adapter）"

#### Scenario: Ch.4 主表对照 #4 含 ThunderAgent
- **WHEN** 读者查阅 §8 Ch.4 主表对照 #4
- **THEN** 行标题为"KVFlow† / PBKV† / ThunderAgent†"
- **AND** 说明列含"≥1 个可公平运行的 closest baseline；不兼容项使用明确标注的 inspired variant"

### Requirement: config.yaml 含 baselines 配置段

系统 SHALL 在 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) 中新增 `baselines` 配置段，列出 compare_oracle.py 启用的全部 baseline。

#### Scenario: baselines 段列出 8 个 baseline
- **WHEN** 读取 config.yaml 的 baselines 段
- **THEN** 含 8 个 entry：lru / gdsf / sizecost / apc_lru / belady / oracle_cost / pbkv_inspired / thunderagent_inspired
- **AND** 每个 entry 含 `type` 字段，值为 simple_heuristic / oracle / closest_inspired 之一
- **AND** 每个 entry 含 `enabled: true` 字段
- **AND** pbkv_inspired 与 thunderagent_inspired 含 `variant: inspired` 与 `source_paper` 字段

#### Scenario: kvflow_faithful 标记为 pending
- **WHEN** 读取 config.yaml 的 baselines 段
- **THEN** 存在 kvflow_faithful entry
- **AND** `enabled: false`
- **AND** `note: "requires WSL2 + CUDA + Rust toolchain; deferred"`

### Requirement: 文档同步不修改任何代码

系统 SHALL 在本 spec 执行过程中不修改任何 .py 文件。

#### Scenario: 仅文档与配置变更
- **WHEN** 本 spec 完成后检查 git diff
- **THEN** 变更仅出现在 .md 与 .yaml 文件中
- **AND** 不存在 .py 文件变更

## MODIFIED Requirements

### Requirement: §0.10 通用 baseline 名录 closest baseline 行

**原（[experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) line 416）**：
| KVFlow† / PBKV† | 至少一个在公平协议下忠实运行的 closest baseline；无法忠实复现的标 `*-inspired` 并先解决可比性 | 最近工作 baseline |

**现（本 spec）**：
| KVFlow† / PBKV† | 至少一个在公平协议下忠实运行的 closest baseline；无法忠实复现的标 `*-inspired` 并先解决可比性 | 最近工作 baseline |
| ThunderAgent-inspired† | ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；API 级代理非块级缓存，提取 program-aware + 2^{-t} time decay 核心 idea | 最近工作 baseline（inspired variant） |

### Requirement: §0.10 脚注

**原（line 424）**：
†：PBKV/KVFlow 的可比性判定流程见 G1 章；若两者均无法忠实运行，按 IDEA §11"所有 closest baseline 均无法忠实比较"风险处理，inspired variant 只能作次要补充。

**现（本 spec）**：
†：PBKV/KVFlow/ThunderAgent 的可比性判定流程见 G1 章 G1.4.1 检查清单。PBKV 无官方代码 → inspired variant；ThunderAgent 官方代码可用但为 API 级代理非块级缓存 → inspired variant；KVFlow 官方代码可用但需 WSL2 + CUDA + Rust → faithful 待 adapter。若三者均无法忠实运行，按 IDEA §11"所有 closest baseline 均无法忠实比较"风险处理，inspired variant 只能作次要补充。

## REMOVED Requirements

（无移除项；本 spec 为纯文档与配置同步）
