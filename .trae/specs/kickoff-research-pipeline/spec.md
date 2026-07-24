# FlowCache 研究执行管线启动 Spec

## Why
FlowCache 的 idea 评审（ccf-idea-reviewer）结论为 pivot-with-rescue-route，存在三个前置阻塞项：中心 claim（reuse value 与 fidelity risk 的错位）未经验证、6 篇核心 2026 arXiv prior art 未独立核验、14 周执行计划未形式化为 stage-gate 管理工件。本 spec 启动三条并行轨道以解除这些阻塞，为后续论文写作奠定可执行基础。

## What Changes
- 新增 14 周执行计划与 stage gate 管理工件（基于 ccfa.yaml 的 G0-G5 + E1-E7）
- 新增 6 篇核心 2026 arXiv prior art 的独立核验报告（PBKV / ARKV / QKVShare / GraphFlow / CacheWise / ThunderAgent）
- 新增 G2 pilot 实验设计文档（R-D 相关性测试协议），填充 experiments/ 目录
- 更新 ccfa.yaml 的 gate / experiment 状态字段以反映核验与设计结果

## Impact
- Affected specs: ccfa.yaml（stage gate 追踪字段）
- Affected artifacts: experiments/ 目录、reviews/ 目录、项目根目录的 14 周计划文档
- 不修改 IDEA.rewritten.md 的研究内容
- 不修改 manuscript/ 下的论文草稿

## ADDED Requirements

### Requirement: 14 周执行计划与 Stage Gate 管理
系统 SHALL 基于 IDEA.rewritten.md Section 12 和 ccfa.yaml 的 G0-G5 / E1-E7，产出一份可执行的 14 周计划文档，包含每周目标、gate 产物、失败动作、依赖关系，并提供 stage gate 状态追踪机制。

#### Scenario: 计划制定完成
- **WHEN** pipeline-orchestrator 读取 ccfa.yaml 和 IDEA Section 12
- **THEN** 产出 14 周计划文档，每周映射到对应 gate / experiment，并标注关键路径和失败回退路线

#### Scenario: Gate 状态可追踪
- **WHEN** 用户查询某个 gate 的状态
- **THEN** 能从 ccfa.yaml 或追踪文件中获取当前状态（not_started / in_progress / passed / failed）

#### Scenario: 失败回退路线明确
- **WHEN** G0 / G1 / G2 / G3 / G4 中任一关键门槛失败
- **THEN** 计划文档明确指向路线 A→B 切换条件和后续动作

### Requirement: 2026 arXiv Prior Art 独立核验
系统 SHALL 对 6 篇核心 2026 arXiv prior art 进行独立存在性与内容核验，判断其是否已实现联合 precision + residency 控制，产出核验报告。

#### Scenario: 核验完成且 novelty 安全
- **WHEN** literature-searcher 核验 6 篇论文
- **AND** 全部论文均未实现联合 precision + residency 控制
- **THEN** 产出核验报告，标记 novelty delta 保留，推荐进入主线 A

#### Scenario: 核验完成且 novelty 崩塌
- **WHEN** 任一论文已实现联合 precision + residency 控制
- **THEN** 产出核验报告，标记 novelty 崩塌风险，推荐转路线 B

#### Scenario: 论文不存在或无法访问
- **WHEN** arXiv ID 无法解析或论文不存在
- **THEN** 标记为"未确认存在"，不假设其内容，降级处理并记录对 novelty 评估的影响

### Requirement: G2 Pilot 实验设计
系统 SHALL 设计一个最小可行的 G2 pilot 实验协议，用于测试 reuse value (R) 与 fidelity risk (D) 在真实 workload 上的相关性，产出实验设计文档并填充 experiments/ 目录。

#### Scenario: 实验设计完成
- **WHEN** experiment-designer 读取 IDEA Section 7 G2 和 Section 8 E1-E3
- **THEN** 产出实验设计文档，包含数据集选择、R / D 标签采集协议、统计检验方法、样本量与功效分析、判定阈值

#### Scenario: 实验设计可执行
- **WHEN** 用户按设计文档执行 pilot
- **THEN** 能在 50-100 个 workflow 上完成 R-D 相关性测试并得出 go / no-go 结论

#### Scenario: 设计遵守用户偏好
- **WHEN** 选择数据集和样本量
- **THEN** 使用真实数据集子集（50 样本级别），不使用伪造数据，符合 RTX 4090D 硬件约束
