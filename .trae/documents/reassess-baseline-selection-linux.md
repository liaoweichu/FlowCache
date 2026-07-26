# 重新评估 Baseline 选取（Linux 可行性变化）

> **背景**：前次调研把 Windows 可行性作为评分维度（3 分制），导致 KVFlow（需 WSL2/Linux+CUDA+Rust）被标记为 "deferred"，InferCept/Continuum/Helium 等因 Windows=0 被排除。用户指出云端环境为 **AutoDL Linux 平台**（有 GPU + 通常 root + CUDA），需重新评估这些 baseline 的可行性，并考虑新增。

---

## Summary

基于 AutoDL Linux 环境，重新评估 [reviews/closest-baseline-code-search.md](file:///d:/00MyProject/Prefix%20Caching/reviews/closest-baseline-code-search.md) 中因 Windows 不可行被 EXCLUDE/降权的 baseline。核心决策：

1. **新增 KVFlow faithful reproduction**（NeurIPS 2025，官方代码可用）——从 `enabled: false / deferred` 改为 `enabled: true / active`，作为 G1 第二项条件（"≥1 个 closest baseline 忠实运行"）的最强证据。需在 AutoDL Linux 上编译魔改 SGLang + Rust + CUDA，并写 τ-bench adapter。
2. **不新增 InferCept / Continuum / Helium / LPC**——理由：InferCept 复现困难（GitHub Issue #2 + 1 年未更新）；Continuum 仅 9 commits preview version；Helium/LPC 与 FlowCache 问题域重叠有限。
3. **不新增 CacheWise**——venue 不明（arXiv preprint，无正式 proceedings），coding agent traces 不直接适用 τ-bench，留作后续备选。
4. **更新 4 份文档**反映 Linux 可行性变化与 KVFlow faithful 的状态升级。

---

## Current State Analysis

### 当前 baseline 配置（[experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml)）

| Baseline | type | enabled | 状态 |
|---|---|---|---|
| lru / gdsf / sizecost / apc_lru | simple_heuristic | true | 已实现 |
| belady / oracle_cost | oracle | true | 已实现 |
| pbkv_inspired | closest_inspired | true | 已实现（无官方代码） |
| thunderagent_inspired | closest_inspired | true | 已实现（API 级代理） |
| **kvflow_faithful** | closest_faithful_pending | **false** | **deferred（需 WSL2）** |

### G1 第二项条件现状（[experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.8）

> "≥1 个 PBKV/KVFlow 在公平协议下忠实运行，**或**不兼容原因已清楚解释且 inspired variant 仅作次要补充"

当前状态：只有 inspired variant（PBKV-inspired + ThunderAgent-inspired），KVFlow faithful 标记 "deferred"。G1 第二项条件只能通过 fallback（"不兼容原因已清楚解释"）满足，**不是最强证据**。

### Linux 下可行性变化（[reviews/closest-baseline-code-search.md](file:///d:/00MyProject/Prefix%20Caching/reviews/closest-baseline-code-search.md) §5 EXCLUDE 表）

| Baseline | 原 EXCLUDE 原因 | Linux 下可行性 | venue | 与 G1 相关性 |
|---|---|---|---|---|
| **KVFlow** | deferred（WSL2/CUDA/Rust） | **YES**（AutoDL Linux 可编译） | NeurIPS 2025 | 高（agent workflow KV） |
| InferCept | Windows=0（CUDA 编译） | YES 但风险高（Issue #2 复现困难 + 1 年未更新） | ICML 2024 | 中（interruption-aware） |
| Continuum | Windows=0（vLLM fork） | YES 但 preview version（9 commits，不含论文估计逻辑） | arXiv 2511.02230 | 中（TTL+reload） |
| Helium | Windows=1（vLLM 依赖） | YES | SIGMOD 2026 | 中（workflow-as-query-plan） |
| LPC | Windows=1（vLLM+CUDA） | YES | NeurIPS 2025 | 低（非 workflow-aware） |
| Agent Memory | Apple Silicon only | NO（硬依赖 MLX） | arXiv 2603.04428 | 低 |

### KVFlow 官方代码详情（[experiments/e1/baselines/RESEARCH_NOTES.md](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/RESEARCH_NOTES.md) line 56-82）

- Repo: https://github.com/PanZaifeng/KVFlow（Apache-2.0，末次 commit 2026-03-13）
- 语言：Python 75.4% / Rust 10.5% / C++ 6.7% / CUDA 6.7%
- 后端：魔改 SGLang + SScheduler PFEngine（非 vLLM）
- ASG（Agent Step Graph）抽象与 τ-bench `block_hash`/`parent_hash` DAG 天然契合
- G1.4.1 已判定："faithful（待 adapter 实现）"——官方代码可用，τ-bench trace 需 adapter 但语义兼容

---

## Proposed Changes

### Change 1: KVFlow faithful reproduction 状态升级

**文件**：[experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml)

**what**：将 `kvflow_faithful` entry 从 `enabled: false` 改为 `enabled: true`，更新 `note` 字段反映 AutoDL Linux 可行性。

**why**：AutoDL Linux 平台有 GPU + root + CUDA，可编译魔改 SGLang + Rust。KVFlow 是 G1 第二项条件的最强证据（NeurIPS 2025 + 官方代码 + ASG 与 τ-bench DAG 契合）。

**how**：
```yaml
  - name: kvflow_faithful
    type: closest_faithful
    enabled: true  # was: false
    source_paper: "arXiv 2507.07400 (NeurIPS 2025)"
    note: "AutoDL Linux + CUDA + Rust toolchain; requires τ-bench adapter (SGLang prefix-tree + PlanManager.update_agent_timestep)"
```

### Change 2: 更新 closest-baseline-code-search.md 调研文档

**文件**：[reviews/closest-baseline-code-search.md](file:///d:/00MyProject/Prefix%20Caching/reviews/closest-baseline-code-search.md)

**what**：
1. 在文档顶部新增 "§0.1 Linux 可行性重评（2026-07-26）" 小节，说明 AutoDL Linux 环境下的可行性变化
2. 更新 §3 PRIORITY_IMPLEMENT 总览表的 "Windows" 列名为 "OS 可行性"，并区分 Windows/Linux 两个子列（或加注 Linux 可行性）
3. 在 §5 EXCLUDE 表中，对 InferCept/Continuum/Helium/LPC 加注 "Linux 下可行但未新增（理由见 §0.1）"
4. 在 §6 实现建议中，新增 "§6.2 次选实现：KVFlow faithful reproduction（AutoDL Linux）" 小节

**why**：调研文档是 baseline 选取决策的单一事实源，需反映 Linux 环境变化与决策依据。

**how**：保持原文档结构不变，新增小节 + 加注，不删除原有 Windows 评分（保留作为历史记录）。

### Change 3: 更新 RESEARCH_NOTES.md 中 KVFlow 调研章节

**文件**：[experiments/e1/baselines/RESEARCH_NOTES.md](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/RESEARCH_NOTES.md)

**what**：
1. 更新 KVFlow 章节的 "Windows feasibility" 字段为 "Linux feasibility"（或新增 Linux 字段）
2. 更新 "Recommendation" 字段：从 "faithful reproduction via official repo, **when WSL2/CUDA environment available**" 改为 "faithful reproduction via official repo, **AutoDL Linux environment available, adapter implementation in progress**"
3. 更新 Overall Recommendation：KVFlow 从 "deferred" 升级为 "active faithful reproduction"

**why**：RESEARCH_NOTES.md 是 G1.4.1 检查清单的证据来源，需反映 KVFlow faithful 的状态升级。

### Change 4: 更新 experiment-designs.md G1 章节

**文件**：[experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md)

**what**：
1. G1.4.1 检查清单 KVFlow 列：将 "faithful（待 adapter 实现）" 改为 "faithful（AutoDL Linux adapter 实现中）"，更新 "所需引擎钩子" 行反映 AutoDL Linux 可行性
2. G1.8 判定阈值第二项：保留原文（"≥1 个 PBKV/KVFlow 在公平协议下忠实运行"），但在备注中说明 "KVFlow faithful reproduction 已激活，AutoDL Linux adapter 实现中"
3. G1.11.1 结果表：将 "KVFlow（faithful，待 adapter）" 行的状态从 TBD 改为 "in progress"

**why**：G1 章节是实验执行的权威指南，需反映 KVFlow faithful 的状态升级。

---

## Assumptions & Decisions

### Assumptions

1. **AutoDL Linux 环境能力**：AutoDL 平台提供 root 权限 + CUDA toolkit + 足够磁盘空间（魔改 SGLang + Rust 编译需 ~20GB）。若实际环境受限（如无 root、CUDA 版本不兼容），KVFlow faithful 可能仍需 fallback 到 inspired variant。
2. **KVFlow 代码可编译性**：假设 2026-03-13 的末次 commit 在 AutoDL Linux 上可编译（NeurIPS 2025 论文已发表，代码应稳定）。若编译失败，需回退到 inspired variant。
3. **τ-bench adapter 工程量**：假设 adapter（τ-bench block_assignments → SGLang prefix-tree + PlanManager.update_agent_timestep）可在 1-2 周内完成。RESEARCH_NOTES.md 已确认 ASG 与 block_hash/parent_hash DAG 天然契合。

### Decisions

1. **只新增 KVFlow faithful，不新增其他 baseline**：
   - **理由**：G1 第二项条件只需 ≥1 个 faithful closest baseline。KVFlow 是最佳候选（venue 最强 + 官方代码 + 问题域高度重叠）。
   - **InferCept**：虽 ICML 2024，但 GitHub Issue #2 显示复现困难 + 1 年未更新，风险高于收益。
   - **Continuum**：仅 9 commits preview version，不含论文估计逻辑，无法忠实复现。
   - **Helium**：SIGMOD 2026，但 workflow-as-query-plan 与 FlowCache exact-prefix block 问题域重叠有限。
   - **LPC**：NeurIPS 2025，但非 workflow-aware，与 FlowCache 距离较远。
   - **CacheWise**：venue 不明（arXiv preprint），coding agent traces 不直接适用 τ-bench。

2. **保留 Windows 评分作为历史记录**：不删除 closest-baseline-code-search.md 中的 Windows 评分，只在新增小节中说明 Linux 可行性变化。这样文档既反映当前决策，又保留历史推理过程。

3. **KVFlow faithful reproduction 的 adapter 工程不在本 plan 范围内**：本 plan 只做文档与配置更新（反映 Linux 可行性变化与状态升级）。实际的 adapter 工程（clone repo + 编译 + 写 adapter + 运行）是后续独立任务，需单独 spec。

---

## Verification Steps

1. **config.yaml 验证**：
   - 读取 [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml)，确认 `kvflow_faithful` entry 的 `enabled: true` 且 `note` 字段反映 AutoDL Linux
   - 确认 baselines 段仍含 9 个 entry（8 原有 enabled + kvflow_faithful 现也 enabled = 9 enabled）

2. **closest-baseline-code-search.md 验证**：
   - grep "Linux 可行性" 命中 §0.1 新增小节
   - grep "AutoDL" 命中 KVFlow 相关段落
   - 确认 §5 EXCLUDE 表中 InferCept/Continuum/Helium/LPC 有 Linux 加注

3. **RESEARCH_NOTES.md 验证**：
   - grep "AutoDL Linux" 命中 KVFlow 章节
   - 确认 Overall Recommendation 中 KVFlow 从 "deferred" 升级为 "active"

4. **experiment-designs.md 验证**：
   - grep "AutoDL" 命中 G1.4.1 / G1.8 / G1.11.1
   - 确认 G1.4.1 KVFlow 列判定从 "待 adapter 实现" 改为 "adapter 实现中"

5. **一致性验证**：
   - config.yaml 的 kvflow_faithful.enabled 与 RESEARCH_NOTES.md 的 KVFlow Recommendation 一致
   - experiment-designs.md G1.4.1 的 KVFlow 判定与 config.yaml 一致
   - 所有文档中 KVFlow faithful 均标注 "AutoDL Linux adapter 实现中"，不误标为"已运行"

6. **无代码变更验证**：
   - `git diff --name-only` 确认变更仅出现在 .md 与 .yaml 文件中，无 .py 变更
