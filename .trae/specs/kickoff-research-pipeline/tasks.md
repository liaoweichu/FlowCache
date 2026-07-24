# Tasks

## Track 1: 14 周执行计划与 Stage Gate 管理（ccf-pipeline-orchestrator）
- [x] Task 1.1: 读取 ccfa.yaml 和 IDEA.rewritten.md Section 12，形式化 14 周计划文档
  - [x] SubTask 1.1.1: 提取 IDEA Section 12 的 W1-W14 每周目标、gate、产物
  - [x] SubTask 1.1.2: 产出 14 周计划文档到项目根目录
- [x] Task 1.2: 映射每周到 G0-G5 gates 和 E1-E7 experiments，标注关键路径
  - [x] SubTask 1.2.1: 标注 G0→G1→G2→G3→G4→G5 的依赖链与关键路径
  - [x] SubTask 1.2.2: 标注 E1-E7 与 gate 的对应关系
- [x] Task 1.3: 为每个 gate 定义失败动作和回退路线（路线 A→B 切换条件）
  - [x] SubTask 1.3.1: 从 IDEA Section 7 和 Section 11 提取每个 gate 的失败动作
  - [x] SubTask 1.3.2: 整合路线 A / B / C 的切换触发条件
- [x] Task 1.4: 建立 stage gate 状态追踪机制，更新 ccfa.yaml

## Track 2: 2026 arXiv Prior Art 独立核验（ccf-literature-searcher）
- [x] Task 2.1: 核验 PBKV (arXiv:2605.06472) 的存在性、内容、是否含 fidelity-aware precision
- [x] Task 2.2: 核验 ARKV (arXiv:2603.08727) 的存在性、内容、是否含 workflow next-use
- [x] Task 2.3: 核验 QKVShare (arXiv:2605.03884) 的存在性、内容、是否含长生命周期驻留
- [x] Task 2.4: 核验 GraphFlow (arXiv:2605.22566, claimed ICML 2026) 的存在性、venue、内容
- [x] Task 2.5: 核验 CacheWise (arXiv:2606.16824) 和 ThunderAgent (arXiv:2602.13692)
- [x] Task 2.6: 汇总核验报告，判定 novelty delta 是否保留，更新 references.bib 的 unverified 标注
  - [x] SubTask 2.6.1: 产出核验报告文件
  - [x] SubTask 2.6.2: 更新 manuscript/references.bib，移除已核验条目的 unverified 标注或标注确认结果

## Track 3: G2 Pilot 实验设计（ccf-experiment-designer）
- [x] Task 3.1: 选定 pilot 数据集（BFCL multi-turn 或 τ-bench，50-100 workflow 子集）
  - [x] SubTask 3.1.1: 评估 BFCL 与 τ-bench 在 exact-prefix reuse 上的适用性
  - [x] SubTask 3.1.2: 定义 50-100 workflow 的子集选择规则
- [x] Task 3.2: 设计 R 标签采集协议（next-use 时间、是否被复用、saved-prefill tokens）
- [x] Task 3.3: 设计 D 标签采集协议（Q8/Q4 恢复后 logit KL、任务成功率变化）
  - [x] SubTask 3.3.1: 选定量化精度（Q8/Q4）和恢复后质量指标
  - [x] SubTask 3.3.2: 设计离线干预回放协议
- [x] Task 3.4: 选定统计检验方法（Spearman/Kendall 相关、显著性、功效分析、样本量）
- [x] Task 3.5: 定义 go/no-go 判定阈值（如 Spearman < 0.4 → go，> 0.7 → no-go）
- [x] Task 3.6: 产出实验设计文档到 experiments/ 目录

# Task Dependencies
- Track 2 和 Track 3 可并行执行，无相互依赖
- Track 1 可基于现有 IDEA Section 12 先起草，不阻塞 Track 2 / Track 3
- Track 1 Task 1.4（状态追踪）应整合 Track 2 / Track 3 的结论以设定初始 gate 状态
- Track 2 Task 2.6（核验报告汇总）依赖 Task 2.1-2.5 全部完成
- Track 3 Task 3.6（文档产出）依赖 Task 3.1-3.5 全部完成
