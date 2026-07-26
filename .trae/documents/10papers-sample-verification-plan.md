# 10 篇论文样本量核实计划

## Summary

用户要求核实 10 篇 KV cache / LLM serving / agent workflow 领域论文实验中使用的精确样本数量。这是**纯研究任务**——不需要写代码或修改任何文件，只需通过 WebSearch / WebFetch 检索 arXiv abs / OpenReview / 会议官网，对每篇论文返回 6 项结构化信息。本计划描述核实策略、严格判定标准与最终输出格式。

## Current State Analysis

### 已有研究基础

通过 Phase 1 探索，已发现以下已完成的工作：

1. **`.trae/documents/g1-12papers-sample-size-compilation.md`**（第 1 组 12 篇）
   - 已确认：**PM-KVQ = 512 calibration samples**（arXiv 2505.18610 PDF 原文佐证）
   - 已确认未明确披露：KVCOMM、DMS、AQUA-KV、KVTuner、Mustafar、KVTC、HiFC、Mooncake、DirectKV、CacheSlide、Jenga

2. **`.trae/documents/group2-13papers-sample-verification.md`**（第 2 组 13 篇）
   - 已确认：**τ-bench = 165 tasks × 8 seeds = 1,320 episodes**（arXiv 2406.12045 PDF 原文佐证）
   - 部分明确：**SAGA**（提到 SWE-bench 500 verified + WebArena 812，但需严格判定是论文显式写出还是仅数据集标称容量）

3. **`reviews/survey-2025-2026-kv-cache-agent-papers.md`**（20 篇调研）
   - 已确认 KVFlow、GraphFlow 等论文未明确披露样本量

### 10 篇论文核实状态（任务清单）

| # | 论文 | Venue | arXiv ID | 当前状态 | 待核实动作 |
|---|------|-------|----------|----------|-----------|
| 1 | PM-KVQ | ICLR 2026 | 2505.18610 | ✅ 已确认 512 samples | 引用 arXiv PDF 原文 |
| 2 | KVTC (Transform Coding) | ICLR 2026 | (待核实) | ❌ 初步未明确 | 重新 WebFetch arXiv PDF，搜 "samples/problems/instances" |
| 3 | DirectKV | OSDI 2026 | (USENIX PDF) | ❌ 初步未明确 | WebFetch USENIX proceedings 页面，搜实验章节 |
| 4 | CacheSlide | FAST 2026 | (USENIX PDF) | ❌ 初步未明确 | WebFetch FAST 2026 proceedings，搜实验样本 |
| 5 | Mooncake | FAST 2025 | 2407.00079 | ❌ 初步未明确 | WebFetch arXiv PDF，搜 "requests/episodes/tasks" |
| 6 | KV Cache in the Wild | USENIX ATC 2025 | 2506.02634 | ❌ 初步未明确 | WebFetch USENIX ATC 2025 proceedings，搜 trace 大小 |
| 7 | KVFlow | NeurIPS 2025 | 2507.07400 | ❌ 初步未明确 | WebFetch arXiv PDF，搜 PEER benchmark 样本数 |
| 8 | GraphFlow | ICML 2026 | 2605.22566 | ❌ 初步未明确 | WebFetch arXiv PDF，搜 5 benchmark 样本数 |
| 9 | SAGA | HPDC 2026 | 2605.00528 | ⚠️ 部分明确 | WebFetch arXiv PDF，严格判定 "500"/"812" 是否显式写出 |
| 10 | τ-bench | ICLR 2025 | 2406.12045 | ✅ 已确认 1,320 episodes | 引用 PDF 原文 |

### 严格判定标准（用户明确要求）

- ✅ **纳入**：论文正文/表格明确写出 "we use N samples/problems/episodes/instances"
- ❌ **排除**：仅给数据集名称无数字（如 "we evaluate on LongBench"）
- ❌ **排除**：估算值（如 "~120K based on trace aggregation"）
- ❌ **排除**：合成参数化配置（如 "1K input tokens, 512 prefix tokens, 5 agents"）
- ❌ **排除**：生产 trace 无具体样本数（如 "百万级 requests/天"）
- ❌ **排除**：数据集标准容量（如 SWE-bench Verified 标称 500、WebArena 标称 812）若论文未显式写出实验用量

### SAGA 边界判定规则

根据已有的 SAGA HTML 内容（temp file `6f4be156-...`）：
- 摘要写 "On a 64-GPU cluster serving SWE-bench coding agents and WebArena browser tasks"
- 未在已读内容中看到 "500" 或 "812" 显式数字

判定规则：
- 若 §9 Evaluation 显式写 "we evaluate on 500 SWE-bench Verified tasks" → 纳入
- 若仅写 "we evaluate on SWE-bench" → 排除（仅数据集名称）
- 若写 "SWE-bench Verified (500 tasks)" 作为数据集介绍而非实验用量 → 标注"数据集标称容量非实验用量"

## Proposed Changes

### 步骤 1：并行 WebFetch 核实 8 篇待核实论文

按以下优先级并行（每批 ≤ 5 个）：

**批次 A（5 篇并行）**：
1. **KVTC**：`https://arxiv.org/abs/2502.06510`（若 ID 不对则 WebSearch "KVTC Transform Coding KV cache ICLR 2026"）
2. **Mooncake**：`https://arxiv.org/pdf/2407.00079`
3. **KVFlow**：`https://arxiv.org/pdf/2507.07400`
4. **GraphFlow**：`https://arxiv.org/pdf/2605.22566`
5. **SAGA**：`https://arxiv.org/html/2605.00528v1#S9`（直接定位到 Evaluation 章节）

**批次 B（3 篇并行）**：
6. **DirectKV**：WebSearch "DirectKV zero-copy KV cache offloading OSDI 2026 USENIX"
7. **CacheSlide**：`https://www.usenix.org/conference/fast26/presentation/liu-yang`
8. **KV Cache in the Wild**：`https://www.usenix.org/conference/atc25`（或 WebSearch 论文标题）

**搜索策略**：
- 优先访问 arXiv abs / PDF 页面（最权威）
- 若 arXiv 不可用，访问会议官网 proceedings
- 在 WebFetch 返回内容中搜索关键词：`samples|problems|instances|episodes|tasks|requests|traces|N=|we use|we evaluate`
- 对 SAGA，专门定位 §9 Evaluation 章节，搜索 "500"、"812"、"verified"、"tasks"

### 步骤 2：对每篇论文做严格判定

每篇论文按以下流程判定：

```
1. WebFetch 返回内容中是否含样本数字？
   - 是 → 进入步骤 2
   - 否 → 标记 "未明确披露样本数"，跳到步骤 4
2. 数字是否符合严格标准（"we use N samples"）？
   - 是 → 纳入，记录原文引用
   - 否 → 进入步骤 3
3. 数字属于排除类别？
   - 数据集标称容量 → 标注"数据集标称容量非实验用量"
   - 估算值 → 标注"估算值非明确披露"
   - 合成参数化 → 标注"参数化配置非样本量"
   - 生产 trace 无具体数 → 标注"生产 trace 无具体样本数"
4. 输出该篇结构化结果
```

### 步骤 3：按用户指定格式输出结构化结果

每篇论文输出 6 项字段：

```markdown
### 论文 N: [完整标题]
1. **完整标题**: [标题]
2. **arXiv ID**: [ID 或会议页]
3. **发表会议和年份**: [Venue Year]
4. **研究方向**: [1-2 句描述]
5. **实验数据集表格**:
   | 数据集名称 | 精确样本数量 | 用途 |
   |-----------|-------------|------|
   | [名称]    | [数字 或 "未明确披露样本数"] | [用途] |
6. **原文引用**: ["..." 或 "未明确披露样本数"]
```

### 步骤 4：输出统计汇总

| 统计维度 | 数值 |
|---------|------|
| 核实论文总数 | 10 |
| 明确披露样本量 | [N]（PM-KVQ + τ-bench + 其他确认的） |
| 未明确披露 | [N] |
| 披露率 | [N/10] |

### 步骤 5：返回结构化结果给父代理

将完整核实结果作为最终响应返回给父代理，由父代理转交给用户。

**输出方式决策**：
- 不创建新文件（遵循 "NEVER create files unless absolutely necessary" 原则）
- 直接在响应中返回结构化结果
- 若用户后续要求落盘到 `reviews/` 目录再考虑

## Assumptions & Decisions

### 假设
1. WebFetch 能成功访问 arXiv / USENIX / OpenReview 页面（已验证大部分可用）
2. 已确认的 PM-KVQ (512 samples) 和 τ-bench (1,320 episodes) 不需要重新核实
3. SAGA 的 §9 Evaluation 章节能通过 `#S9` 锚点直接定位
4. 若论文 PDF 不可访问，arXiv HTML 版本或会议官网页可作 fallback

### 决策
1. **PM-KVQ 512 samples 引用 arXiv PDF 原文**：已通过 WebSearch 交叉验证，符合严格标准
2. **τ-bench 1,320 episodes 引用 PDF 原文**：165 tasks × 8 seeds = 1,320，符合 pass^k 评估标准
3. **SAGA 边界判定**：以 §9 Evaluation 章节是否显式写出 "500"/"812" 为准，而非摘要
4. **未明确披露的论文**：若 WebFetch 返回内容确实找不到样本数字，标记 "未明确披露样本数"
5. **不落盘新文件**：直接在响应中返回结构化结果，遵循最小变更原则
6. **响应语言**：中文（用户明确要求 "Respond in 中文"）

## Verification Steps

1. **PM-KVQ 验证**：确认 "we randomly select 512 samples, each with a length of 2,048 tokens, for calibration" 有 arXiv PDF 原文佐证（已验证）
2. **τ-bench 验证**：确认 "165 tasks × 8 seeds = 1,320 episodes" 有 PDF 原文佐证（已验证）
3. **SAGA 验证**：在 §9 Evaluation 章节中搜索 "500"、"812"、"verified"、"tasks" 关键词
4. **格式验证**：每篇论文输出 6 项字段（标题/arXiv ID/会议年份/研究方向/数据集表/原文引用）
5. **严格标准验证**：所有 "未明确" 判定必须有原文缺失佐证（即 WebFetch 返回内容中确实找不到符合标准的数字）
6. **统计验证**：汇总数字（确认数 / 未明确数 / 披露率）与单篇判定一致
7. **语言验证**：最终响应用中文

## 执行顺序

```
Step 1: 批次 A 并行 WebFetch（5 篇：KVTC/Mooncake/KVFlow/GraphFlow/SAGA）
Step 2: 批次 B 并行 WebFetch（3 篇：DirectKV/CacheSlide/KV Cache in the Wild）
Step 3: 对 8 篇待核实论文逐篇做严格判定（含 SAGA 边界判定）
Step 4: 按 6 项字段格式输出 10 篇结构化结果
Step 5: 输出统计汇总（确认数 / 未明确数 / 披露率）
Step 6: 返回完整结果给父代理（不创建新文件）
```

## 关键文件路径（仅供参考，不修改）

### 临时文件目录（已存放的 WebFetch 结果）
`C:\Users\lwc\AppData\Local\Temp\trae\toolcall-output\`

### 已有研究文档（仅供交叉参考）
- `d:\00MyProject\Prefix Caching\.trae\documents\g1-12papers-sample-size-compilation.md`（第 1 组 12 篇汇编结果）
- `d:\00MyProject\Prefix Caching\.trae\documents\group2-13papers-sample-verification.md`（第 2 组 13 篇汇编结果）
- `d:\00MyProject\Prefix Caching\reviews\survey-2025-2026-kv-cache-agent-papers.md`（20 篇调研文档）
- `d:\00MyProject\Prefix Caching\experiments\experiment-designs.md`（FlowCache 实验设计）

## 关键 URL 列表

### arXiv 论文（优先访问）
- PM-KVQ: `https://arxiv.org/abs/2505.18610`（已确认 512 samples）
- τ-bench: `https://arxiv.org/abs/2406.12045`（已确认 1,320 episodes）
- Mooncake: `https://arxiv.org/abs/2407.00079`
- KVFlow: `https://arxiv.org/abs/2507.07400`
- GraphFlow: `https://arxiv.org/abs/2605.22566`
- SAGA: `https://arxiv.org/html/2605.00528v1#S9`（直接定位 Evaluation 章节）
- KVTC: 需 WebSearch 确定 arXiv ID（候选：2502.06510）

### USENIX 会议页面
- DirectKV (OSDI 2026): `https://www.usenix.org/conference/osdi26`
- CacheSlide (FAST 2026): `https://www.usenix.org/conference/fast26/presentation/liu-yang`
- KV Cache in the Wild (ATC 2025): `https://www.usenix.org/conference/atc25`

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| WebFetch 返回内容截断 | 优先访问 PDF 文本版本（更完整）；分章节访问 |
| arXiv ID 不准确 | 通过 WebSearch 先确定准确 ID |
| 论文确实未披露样本数 | 严格按标准标记 "未明确披露样本数"，不强行估算 |
| SAGA 边界判定困难 | 以 §9 Evaluation 章节显式数字为准，若仅数据集介绍则标注 "数据集标称容量非实验用量" |
| USENIX 页面访问受限 | 优先 arXiv 版本，fallback 到会议官网 |
