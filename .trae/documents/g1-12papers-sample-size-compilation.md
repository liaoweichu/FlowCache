# 第 1 组 12 篇论文样本量提取结果汇编计划

## Summary

将已完成的 WebFetch 核实结果（第 1 组 12 篇 KV cache 优化论文）汇编为结构化输出，按用户指定的严格标准（只接受论文正文/表格中明确写出的数字）逐篇报告样本量披露情况。WebFetch 与 WebSearch 核实操作已完成，本计划聚焦于结果汇编与输出。**关键发现**：12 篇中仅 1 篇（PM-KVQ）明确披露样本量（512 calibration samples，已通过 arXiv PDF 原文验证），其余 11 篇均未明确披露。

## Current State Analysis

### 已完成的工作

通过 WebFetch 已访问 12 篇论文的 arXiv abs/HTML/PDF/OpenReview/会议页面，结果存放在临时文件中。已通过 WebSearch 对 PM-KVQ 的 "512 calibration samples" 进行了交叉验证（来源：arXiv PDF https://arxiv.org/pdf/2505.18610）。

### 12 篇论文核实结果汇总

| # | 论文 | Venue | CCF | arXiv ID | 样本量披露 | 关键发现 |
|---|------|-------|-----|----------|-----------|---------|
| 1 | KVCOMM | NeurIPS 2025 | A | OpenReview Vem6FQvRvq | ❌ 否 | 参数化配置（1K input + 512 prefix + 512 output tokens, 5 agents），非样本量 |
| 2 | DMS (Hyper-Scaling) | NeurIPS 2025 | A | OpenReview 8ZiElzQxf1 | ❌ 否 | AIME24/GPQA/LiveCodeBench/QASPER，无样本数 |
| 3 | AQUA-KV | ICML 2025 | A | (ICML page) | ❌ 否 | LongBench 14 任务，无每子任务样本数 |
| 4 | KVTuner | ICML 2025 | A | (ICML page) | ❌ 否 | GSM8K/GPQA，无样本数 |
| 5 | PM-KVQ | ICLR 2026 | A | 2505.18610 | ✅ 是 | **512 calibration samples**（arXiv subset of RedPajama，每样本 2,048 tokens） |
| 6 | Mustafar | NeurIPS 2025 | A | 2505.22913 | ❌ 否 | LongBench + RULER，无样本数 |
| 7 | KVTC (Transform Coding) | ICLR 2026 | A | (arXiv PDF) | ❌ 否 | 8 个 benchmark（AIME25/GSM8K/LongBench/MATH-500/MMLU/Qasper/RULER/LiveCodeBench），无样本数 |
| 8 | HiFC | NeurIPS 2025 | A | (NeurIPS PDF) | ❌ 否 | NarrativeQA，无样本数 |
| 9 | Mooncake | FAST 2025 | A | 2407.00079 | ❌ 否 | Conversation/Tool&Agent/Synthetic trace，~120K+ requests 为估算值非明确写出 |
| 10 | DirectKV | OSDI 2026 | A | (USENIX PDF) | ❌ 否 | 无样本数细节 |
| 11 | CacheSlide | FAST 2026 | A | (USENIX PDF) | ❌ 否 | 无样本数细节 |
| 12 | Jenga | ACM (SOSP/ASPLOS?) | A | (ACM PDF) | ❌ 否 | 无样本数细节 |

### 严格标准（用户明确要求）

- ✅ 纳入：论文正文/表格明确写出"we use N samples/problems/episodes"
- ❌ 排除：仅给数据集名称无数字（如"we evaluate on LongBench"）
- ❌ 排除：估算值（如"~120K based on trace aggregation"）
- ❌ 排除：合成参数化配置（如"1K input tokens, 512 prefix tokens, 5 agents"）
- ❌ 排除：生产 trace 无具体样本数（如"百万级 requests/天"）
- ❌ 排除：数据集标准容量（如 LongBench 标称容量）若论文未显式写出实验用量

### 关键验证：PM-KVQ 的 512 calibration samples

**验证来源**：arXiv PDF (https://arxiv.org/pdf/2505.18610)
**原文引用**："For the calibration dataset, we use the arXiv subset of RedPajama [22] as our calibration dataset. This subset consists of academic papers in LaTeX format, containing mathematical formulas and the reasoning process. Specifically, we randomly select 512 samples, each with a length of 2,048 tokens, for calibration."
**判定**：✅ 纳入。论文正文明确写出 "512 samples"，符合严格标准。

### 当前结论

- 12 篇中仅 **1 篇（PM-KVQ）** 明确披露样本量：512 calibration samples
- 11 篇未明确披露样本量（仅给数据集名称或参数化配置）
- 披露率：8.3%（严格标准）

## Proposed Changes

### 步骤 1：按用户指定格式汇编 12 篇结构化结果

按用户指定的输出格式，逐篇输出 12 篇论文的核实结果。每篇包含 7 个字段：

```markdown
### 论文 N: [标题]
- **CCF等级**: A
- **样本量是否明确披露**: ✅ 是 / ❌ 否
- **精确样本数字**: [数字 或 "未明确披露"]
- **数据集列表**: [数据集名称列表]
- **总样本量**: [数字 或 "未明确"]
- **数据来源**: [arXiv abs / arXiv HTML / arXiv PDF / OpenReview / 会议官网]
- **原文引用**: [原文片段 或 "无明确引用"]
```

### 步骤 2：对 11 篇"未明确"论文逐篇说明原因

每篇"未明确披露"的论文需说明：
- WebFetch 返回内容中确实找不到数字（非未访问）
- 仅有的数字属于排除类别（参数化配置/估算值/数据集名称）
- 原文引用片段佐证缺失

### 步骤 3：输出统计汇总

| 统计维度 | 数值 |
|---|---|
| 核实论文总数 | 12 |
| 明确披露样本量 | 1（PM-KVQ: 512 calibration samples） |
| 未明确披露 | 11 |
| 披露率 | 8.3%（严格标准） |

### 步骤 4：返回结构化结果给父代理

将完整汇编结果作为最终响应返回给父代理，由父代理转交给用户。**不创建新文件**（避免创建非必要文件；除非用户后续要求落盘到 `reviews/` 目录）。

## Assumptions & Decisions

### 假设
1. WebFetch 返回的内容已足够判定样本量披露情况（无需再次访问）
2. PM-KVQ 的 "512 samples" 已通过 arXiv PDF 原文交叉验证（WebSearch 第三方索引佐证）
3. Mooncake 的 "~120K+ requests" 为估算值，不符合严格标准，不纳入
4. KVCOMM 的 "1K input tokens / 512 prefix tokens / 5 agents" 为参数化配置，非样本量，不纳入

### 决策
1. **PM-KVQ 512 calibration samples 纳入**：arXiv PDF 正文明确写 "we randomly select 512 samples, each with a length of 2,048 tokens, for calibration"，符合严格标准
2. **KVCOMM 参数化配置不纳入**：1K/512/5 是 token 数和 agent 数，非评估样本量
3. **Mooncake trace 估算值不纳入**：~120K+ 为基于 trace 聚合的估算，论文未显式写出
4. **DMS/AQUA-KV/KVTuner/Mustafar/KVTC/HiFC 仅数据集名称不纳入**：AIME24/GPQA/LongBench/GSM8K 等为数据集名称，无样本数
5. **DirectKV/CacheSlide/Jenga 无样本数细节不纳入**：WebFetch 返回内容中未找到任何样本相关数字
6. **输出方式**：直接在响应中返回结构化结果，不落盘新文件（遵循"NEVER create files unless absolutely necessary"原则）

## Verification Steps

1. **PM-KVQ 验证**：确认 "512 samples" 有 arXiv PDF 原文佐证（已通过 WebSearch 验证）
2. **格式验证**：每篇论文输出 7 个字段（CCF等级/是否明确披露/精确数字/数据集列表/总样本量/数据来源/原文引用）
3. **严格标准验证**：所有"未明确"判定必须有原文缺失佐证（即 WebFetch 返回内容中确实找不到符合标准的数字）
4. **统计验证**：汇总数字（1 明确 / 11 未明确 / 8.3% 披露率）与单篇判定一致
5. **排除项验证**：KVCOMM 参数化配置、Mooncake 估算值、其余 9 篇仅数据集名称，均符合排除标准

## 执行顺序

```
Step 1: 基于 WebFetch/WebSearch 已返回内容，逐篇判定 12 篇样本量披露情况
Step 2: 对 PM-KVQ 的 "512 calibration samples" 引用 arXiv PDF 原文佐证
Step 3: 按用户指定格式汇编 12 篇结构化结果（含 7 字段）
Step 4: 输出统计汇总（1/12 明确，8.3% 披露率）
Step 5: 返回完整结果给父代理（不创建新文件）
```

## 关键文件路径（仅供汇编参考，不修改）

- 临时文件目录：`C:\Users\lwc\AppData\Local\Temp\trae\toolcall-output\`
  - KVCOMM: `801cba04-cbfc-4de7-a7ec-0f4c510abdf0.txt`
  - DMS: `253f314c-e855-4786-852a-31428979d4e2.txt`
  - AQUA-KV: `b0b32e49-a44c-429e-acfe-d3cc4bc3ee50.txt`
  - KVTuner: `9061610e-e432-4370-91ee-68634c210969.txt`
  - PM-KVQ (PDF text): `54539527-06d9-4903-b95a-732fe6e89278.txt`
  - PM-KVQ (HTML TOC): `8bd67059-72a1-4207-b6b1-0b2160ec743f.txt`
  - Mustafar: `79ef7ea9-c8c5-41c6-a1a1-6bfb085d634c.txt`
  - KVTC: `48fad8f8-0d0d-418c-abb8-64d7cac8586b.txt`
  - HiFC: `e2878eba-9d96-4013-b061-7013d43296d3.txt`
  - Mooncake (v3): `f3747f1c-914f-4ec0-b3ac-2e4bd5f3d0bb.txt`
  - Mooncake (v4 HTML): `1b123b7f-0c8e-4c7b-8930-f4326bf9c1cd.txt`
  - DirectKV: `cb0c3f5b-1eff-441f-b836-ad7fa3e6163e.txt`
  - CacheSlide: `c0a56fdf-93a9-4afc-8f48-262f9a3502af.txt`
  - Jenga: `b8c9130b-4d8e-4174-a08e-c82d28a7d07d.txt`

- 已有相关文档（仅供交叉参考）：
  - `d:\00MyProject\Prefix Caching\.trae\documents\group2-13papers-sample-verification.md`（第 2 组 13 篇，类似格式）
  - `d:\00MyProject\Prefix Caching\.trae\documents\ccf-ab-kv-cache-papers-survey-2025-2026.md`（CCF-A/B 调研计划）
  - `d:\00MyProject\Prefix Caching\reviews\survey-2025-2026-kv-cache-agent-papers.md`（20 篇调研文档）
