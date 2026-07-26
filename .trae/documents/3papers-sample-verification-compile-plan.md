# 3 篇论文样本量核实结果汇编计划（Mooncake / HCache / CacheBlend）

## 摘要 (Summary)

本计划聚焦于将已完成的 WebFetch/WebSearch 核实结果汇编为结构化中文回复。用户要求核实 Mooncake、HCache、CacheBlend 三篇论文实验中使用的**精确样本数量**，必须找到论文原文中明确写出的数字。基于多轮检索（arXiv abs 页面、HTML 全文、PDF、第三方博客、原数据集论文交叉验证），结果已存放在临时文件中。本计划只做结果汇编与判定，不创建任何代码或额外文档文件，最终以单条结构化中文回复返回给父代理。

## 当前状态分析 (Current State Analysis)

### 已确认的论文元数据

| 论文 | arXiv ID | 标题 | 会议 |
|------|----------|------|------|
| Mooncake | 2407.00079 | Mooncake: A KV-Centric-centric Disaggregated Architecture for LLM Serving | FAST 2025 (USENIX) |
| HCache | 2410.05209 | HCache: Heavy-Channel KV Cache Compression for Efficient Large Language Model Inference | ACM 2024（具体会议待定，arXiv 预印本/期刊） |
| CacheBlend | 2405.16444 | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | EuroSys 2025（Best Paper Award） |

### 各论文样本数量披露情况（基于已检索内容）

**1. Mooncake (arXiv:2407.00079, FAST 2025)**
- 论文正文与 GitHub-hosted FAST25 版本均提及 "Sampled Real-world Request Trace" 章节
- 明确披露的数字：**"processing over 100 billion tokens daily"**（生产规模描述）
- 数据集结构：Conversation trace / Tool&Agent trace / Synthetic trace
- **未明确披露**：trace 中具体的请求数 / 样本数（如 "N requests"）
- 第 1 组 12 篇汇编中的判定：❌ 否（~120K+ requests 为估算值，非原文显式写出）
- 判定：❌ **未明确披露样本数**

**2. HCache (arXiv:2410.05209, ACM/期刊）**
- 使用 ShareGPT4 和 L-Eval traces 作为评估负载
- WebFetch 返回的 ACM PDF 内容提及使用 ShareGPT4 和 L-Eval traces，但**无精确样本数字**
- L-Eval 原论文（交叉验证）：20 个子任务、508 文档、2,000+ query-response pairs（但这是 L-Eval 标称容量，**非 HCache 论文显式写出的实验用量**）
- 判定：❌ **未明确披露样本数**（HCache 自身正文未给出具体样本数；L-Eval 的 2,000+ 来自原数据集论文，不符合"论文原文显式写出"的严格标准）

**3. CacheBlend (arXiv:2405.16444, EuroSys 2025 Best Paper)**
- 论文正文提及 "four popular benchmark datasets"（4 个 RAG/QA 基准），但 arXiv PDF 主文未对每个数据集给出精确样本数
- LMCache 官方博客（blog.lmcache.ai, 2025-03-31）**明确写出**：
  > "On the popular RAG dataset of '2WikiMQA' with a chunk size of 512 tokens, the following figure plots the average Time to First Token (TTFT) and average quality (in F1 score) of CacheBlend and standard KV cache reuse (widely used in vLLM), **across 1,500 randomly sampled queries**."
- 该 1,500 数字由 LMCache 官方博客显式披露，是 CacheBlend 作者团队（同 LMCache 维护方）的官方说明
- 严格标准判定：博客为论文作者团队官方发布，但**非论文 arXiv PDF 正文/表格**——属于"作者团队官方博客显式披露"，处于 ✅ 与 ❌ 的边界
- 论文 arXiv PDF 自身：❌ 未在正文显式写出每数据集样本数
- 综合判定：⚠️ **部分明确（1,500 queries for 2WikiMQA，来自作者团队官方博客而非 arXiv PDF 正文）**

### 严格标准（用户明确要求）

- ✅ 纳入：论文正文/表格明确写出 "we use N samples/queries/episodes"
- ⚠️ 边界：作者团队官方博客显式写出 N（非 arXiv PDF 正文，但同作者团队）—— 标注"博客披露"
- ❌ 排除：仅给数据集名称无数字（如 "we evaluate on ShareGPT4"）
- ❌ 排除：估算值（如 "~120K+ based on trace aggregation"）
- ❌ 排除：生产规模描述（如 "processing over 100 billion tokens daily"）
- ❌ 排除：原数据集论文标称容量（如 L-Eval 原文 2,000+ pairs）若本论文未显式写出实验用量

### 当前结论

- 3 篇中：
  - **0 篇**在 arXiv PDF 正文中显式写出每数据集样本数
  - **1 篇（CacheBlend）**通过作者团队官方博客显式披露 2WikiMQA 使用 1,500 queries（边界情况，标注来源）
  - **2 篇（Mooncake、HCache）**未明确披露样本数（仅给数据集名称或生产规模描述）
- 严格披露率（仅算 arXiv PDF 正文）：0%
- 含作者团队博客的放宽披露率：33.3%（仅 CacheBlend 1 项数据集）

## 提议的回复结构 (Proposed Changes)

最终的中文回复将按以下结构组织（不创建任何文件，直接在对话中返回）：

### 总体汇总表

一张汇总表，包含：论文 | arXiv ID | 会议 | 数据集 | 精确样本数 | 是否披露（✅/⚠️/❌） | 数据来源

### 每篇论文详细条目（3 个小节）

每节包含用户要求的 6 个字段：

1. **论文完整标题和 arXiv ID**
2. **发表会议和年份**
3. **研究方向（1-2 句）**
4. **实验数据集表格**（数据集名称 | 精确样本数量 | 用途）
   - 已披露的填具体数字（含来源标注：arXiv PDF / 作者博客 / 原数据集论文）
   - 未披露的统一标注"未明确披露样本数"
5. **原文引用**（带英文原文引用，标注来源：arXiv PDF / 官方博客 / WebSearch 结果）
6. **判定**：
   - ✅ 有精确样本数字（arXiv PDF 正文/表格显式写出）
   - ⚠️ 部分明确（作者团队博客披露，非 arXiv PDF 正文）
   - ❌ 未明确披露

### 关键发现总结

- 3 篇论文中无任何一篇在 arXiv PDF 正文中显式写出每数据集精确样本数
- CacheBlend 通过作者团队官方博客（blog.lmcache.ai, 2025-03-31）披露 2WikiMQA 使用 1,500 queries，是边界情况
- Mooncake 仅披露生产规模（"100 billion tokens daily"），无 trace 样本数
- HCache 使用 ShareGPT4 + L-Eval traces，但正文未给出精确样本数；L-Eval 的 2,000+ pairs 来自原数据集论文，非 HCache 显式实验用量
- 严格披露率（仅算 arXiv PDF 正文）：0%；放宽披露率（含作者团队博客）：33.3%

## 假设与决策 (Assumptions & Decisions)

### 假设

1. WebFetch 与 WebSearch 已返回的内容足够判定样本量披露情况，无需再做额外检索
2. Mooncake 的 "100 billion tokens daily" 是生产规模描述，非实验样本数
3. HCache 的 L-Eval 2,000+ query-response pairs 来自 L-Eval 原论文（非 HCache 自身披露），不纳入 ✅ 判定
4. CacheBlend 的 1,500 queries 来自 LMCache 官方博客，作者团队与论文一致，但博客非 arXiv PDF 正文，判定为 ⚠️ 边界情况

### 决策

1. **Mooncake 判定为 ❌**：arXiv PDF 与 FAST25 版本均未显式写出 trace 样本数
2. **HCache 判定为 ❌**：arXiv/ACM PDF 未给出 ShareGPT4/L-Eval 的精确样本数
3. **CacheBlend 判定为 ⚠️**：2WikiMQA 的 1,500 queries 由作者团队官方博客显式披露，但非 arXiv PDF 正文
4. **输出方式**：直接在响应中返回结构化中文结果，不创建新文件（遵循"NEVER create files unless absolutely necessary"原则）
5. **语言**：最终回复用中文，原文引用保留英文
6. **不创建文件**：遵循用户"纯研究任务，不需要写代码或修改任何文件，只需返回检索结果"的指示

## 验证步骤 (Verification steps)

1. **格式验证**：每篇论文输出 6 个字段（标题/arXiv ID、会议/年份、研究方向、数据集表格、原文引用、判定）
2. **判定一致性**：
   - Mooncake: ❌（与第 1 组 12 篇汇编中 Mooncake 判定一致）
   - HCache: ❌（HCache 未在第 1/2 组中，本次新增判定）
   - CacheBlend: ⚠️（第 2 组 13 篇中 CacheBlend 判定为"未明确"，本次因找到 LMCache 官方博客的 1,500 queries，升级为"部分明确"）
3. **原文引用准确性**：
   - Mooncake 引用 "processing over 100 billion tokens daily"（FAST25 PDF）
   - CacheBlend 引用 "across 1,500 randomly sampled queries"（LMCache 官方博客）
   - HCache 引用相关内容并标注"未找到精确样本数"
4. **严格标准验证**：所有 ❌ 判定必须有原文缺失佐证（即 WebFetch 返回内容中确实找不到符合标准的数字）
5. **输出形式验证**：单一中文回复，无多余文件创建

## 执行顺序

```
Step 1: 基于 WebFetch/WebSearch 已返回内容，逐篇判定 3 篇论文样本量披露情况
Step 2: 对 CacheBlend 的 1,500 queries 引用 LMCache 官方博客原文佐证，标注 ⚠️ 边界判定
Step 3: 对 Mooncake 的 "100 billion tokens daily" 引用 FAST25 PDF，标注 ❌（生产规模非样本数）
Step 4: 对 HCache 的 ShareGPT4/L-Eval 引用相关内容并标注 ❌（无精确样本数）
Step 5: 按用户指定格式汇编 3 篇结构化结果（含 6 字段）
Step 6: 输出统计汇总（0/3 严格披露，1/3 部分明确，2/3 未明确）
Step 7: 返回完整中文结果给父代理（不创建新文件）
```

## 关键文件路径（仅供汇编参考，不修改）

### 已检索的临时文件

- 临时文件目录：`C:\Users\lwc\AppData\Local\Temp\trae\toolcall-output\`
  - Mooncake (FAST25 PDF, GitHub-hosted): `915a23c0-3e17-49fa-b38b-93a9159a5b07.txt`
  - Mooncake (HTML): `212d9c9a-60d1-4d7f-a818-55ae1056dc6d.txt`
  - Mooncake (v3): `f3747f1c-914f-4ec0-b3ac-2e4bd5f3d0bb.txt`（第 1 组 12 篇引用）
  - Mooncake (v4 HTML): `1b123b7f-0c8e-4c7b-8930-f4326bf9c1cd.txt`（第 1 组 12 篇引用）
  - HCache (ACM PDF): `bfa496ef-0a9a-4dad-9478-75b12e2857b7.txt`
  - HCache (HTML): `c081696a-5b8e-41af-95b6-7186a59f6d39.txt`
  - CacheBlend (arXiv PDF): `72d905ff-d978-4b69-8b4b-0f132bafedbc.txt`
  - CacheBlend (LMCache 官方博客): `8fb7b855-a5b2-4baa-b0c9-bf7d88d34d31.txt`
  - CacheBlend (HTML): `630919d9-2316-4162-b1b6-219f14d5a6b2.txt`

### 已有相关文档（仅供交叉参考，不修改）

- `d:\00MyProject\Prefix Caching\.trae\documents\g1-12papers-sample-size-compilation.md`（第 1 组 12 篇，Mooncake 已在其中判定为 ❌）
- `d:\00MyProject\Prefix Caching\.trae\documents\group2-13papers-sample-verification.md`（第 2 组 13 篇，CacheBlend 已在其中判定为"未明确"，本次因新发现 LMCache 博客的 1,500 queries 而升级判定）
- `d:\00MyProject\Prefix Caching\reviews\survey-2025-2026-kv-cache-agent-papers.md`（20 篇调研文档，Mooncake/CacheBlend 在其中作为对比基线被提及）
