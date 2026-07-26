# 第 2 组 13 篇论文样本量核实结果汇编计划

## Summary

将已完成的 WebFetch 核实结果（第 2 组 13 篇 CCF-A/B 论文）汇编为结构化输出，按用户指定的严格标准（只接受论文正文/表格中明确写出的数字）逐篇报告样本量披露情况。WebFetch 操作已完成，本计划聚焦于结果汇编与输出。

## Current State Analysis

### 已完成的工作

通过 WebFetch 已访问 13 篇论文的 arXiv abs/HTML/PDF 页面，结果存放在临时文件中。已核实的论文及初步发现：

| # | 论文 | Venue | CCF | arXiv ID | WebFetch 结果 |
|---|------|-------|-----|----------|---------------|
| 1 | eLLM | EuroSys 2026 | B | (待核实) | 未明确披露样本量（自适应 KV caching，正文未提） |
| 2 | KV Cache in the Wild | USENIX ATC 2025 | B | 2506.02634 | 未明确（两条生产 trace to-C/to-B，无具体请求数） |
| 3 | τ-bench | ICLR 2025 | A(视同) | 2406.12045 | **已确认：165 tasks × 8 seeds = 1,320 episodes** |
| 4 | COMET | ASPLOS 2025 | A | (待核实) | 未明确（LLaMA family 评估，无样本数） |
| 5 | KVFlow | NeurIPS 2025 | A | 2507.07400 | 未明确（合成 workflow 参数化配置） |
| 6 | GraphFlow | ICML 2026 | A | 2605.22566 | 未明确（5 benchmark，无具体样本数） |
| 7 | SAGA | HPDC 2026 | B | 2605.00528 | 部分明确（SWE-bench 500 verified + WebArena 812，需核实原文是否显式写出） |
| 8 | ARKV | CCGRID 2025 | B | 2603.08727 | 未明确（LLaMA3/Qwen3 长上下文，无样本数） |
| 9 | HILOS | ASPLOS 2026 | A | (待核实) | 未明确 |
| 10 | TokenFlow | EuroSys 2026 | B | (待核实) | 未明确 |
| 11 | CacheBlend | EuroSys 2025 | B | (待核实) | 未明确 |
| 12 | Ada-KV | NeurIPS 2025 | A | 2407.11550 | 未明确（Ruler 13 + LongBench 16 子任务，无每子任务样本数） |
| 13 | Cake | ICML 2025 | A | (OpenReview WOyOtaO6lQ) | 未明确（多种硬件/数据集/存储配置） |

### 严格标准（用户明确要求）

- ✅ 纳入：论文正文/表格明确写出"we use N samples/problems/episodes"
- ❌ 排除：仅给数据集名称无数字（如"we evaluate on LongBench"）
- ❌ 排除：估算值（如"~2,900 based on 29 tasks × 100"）
- ❌ 排除：合成参数化配置（如"branches=1, depth=32"）
- ❌ 排除：生产 trace 无具体样本数（如"百万级 requests/天"）
- ❌ 排除：数据集标准容量（如 SWE-bench Verified 标称 500、WebArena 标称 812）若论文未显式写出实验用量

### 当前结论

- 13 篇中仅 **1 篇（τ-bench）** 明确披露样本量：165 tasks × 8 seeds = 1,320 episodes
- 1 篇（SAGA）需进一步核实原文是否显式写出 "500" 和 "812"（还是仅引用数据集名称）
- 11 篇未明确披露样本量

## Proposed Changes

### 步骤 1：汇编结构化结果（仅汇编，无需额外 WebFetch）

按用户指定的输出格式，逐篇输出 13 篇论文的核实结果。每篇包含 6 个字段：

```markdown
### 论文 N: [标题]
- **是否明确披露**: ✅ 是 / ❌ 否
- **精确数字**: [数字 或 "未明确披露"]
- **数据集列表**: [数据集名称列表]
- **总样本量**: [数字 或 "未明确"]
- **数据来源**: [arXiv abs / arXiv HTML / arXiv PDF / OpenReview / 会议官网]
- **原文引用**: [原文片段 或 "无明确引用"]
```

### 步骤 2：对 SAGA 的 "500" 和 "812" 做严格判定

根据已有 WebFetch 内容（temp file `47cd17bd-c313-4fa9-b6dc-a352e55e4398.txt`），SAGA 提到 SWE-bench 和 WebArena。需要判定：
- 若原文写 "we use 500 verified tasks from SWE-bench" → 纳入
- 若原文仅写 "we evaluate on SWE-bench and WebArena" → 排除（仅数据集名称）
- 若原文写 "SWE-bench (500 tasks)" 作为数据集介绍而非实验用量 → 边界情况，标注"数据集标称容量非实验用量"

### 步骤 3：输出统计汇总

| 统计维度 | 数值 |
|---|---|
| 核实论文总数 | 13 |
| 明确披露样本量 | 1（τ-bench） |
| 部分明确（数据集标称容量） | 1（SAGA，待判定） |
| 未明确披露 | 11 |
| 披露率 | 7.7%（严格标准）/ 15.4%（含标称容量） |

### 步骤 4：返回结构化结果给父代理

将完整汇编结果作为最终响应返回给父代理，由父代理转交给用户。**不创建新文件**（除非用户后续要求落盘到 `reviews/` 目录）。

## Assumptions & Decisions

### 假设
1. WebFetch 返回的内容已足够判定样本量披露情况（无需再次访问）
2. τ-bench 的 165×8=1,320 已通过 PDF 全文确认（temp file `bee60b56`）
3. SAGA 的 SWE-bench/WebArena 数字来自 HTML 内容，需按严格标准判定是否算"实验用量明确披露"

### 决策
1. **τ-bench 165×8=1,320 纳入**：原论文 PDF 明确写 "165 tasks" 和 "pass^k for k=8"，乘积 1,320 是 pass^k 评估的标准样本量
2. **Ada-KV 13+16 子任务不纳入**：仅给子任务数不给每子任务样本数，不符合严格标准
3. **KVFlow 合成 workflow 不纳入**：参数化配置（branches/depth/width）非真实样本量
4. **KV Cache in the Wild 不纳入**：生产 trace 无具体请求数
5. **SAGA 边界判定**：若原文仅引用数据集名称（"SWE-bench", "WebArena"）无数字 → 排除；若有显式数字 → 纳入并标注"数据集标称容量"
6. **输出方式**：直接在响应中返回结构化结果，不落盘新文件（避免创建非必要文件）

## Verification Steps

1. **τ-bench 验证**：确认 165 tasks (115 retail + 50 airline) × 8 seeds = 1,320 episodes 有原文佐证
2. **SAGA 验证**：判定 "500" 和 "812" 是论文显式写出还是数据集标称容量
3. **格式验证**：每篇论文输出 6 个字段（是否明确披露/精确数字/数据集列表/总样本量/数据来源/原文引用）
4. **严格标准验证**：所有"未明确"判定必须有原文缺失佐证（即 WebFetch 返回内容中确实找不到数字）
5. **统计验证**：汇总数字与单篇判定一致

## 执行顺序

```
Step 1: 基于 WebFetch 已返回内容，逐篇判定样本量披露情况
Step 2: 对 SAGA 的 "500"/"812" 做严格标准判定
Step 3: 按用户指定格式汇编 13 篇结构化结果
Step 4: 输出统计汇总
Step 5: 返回完整结果给父代理
```
