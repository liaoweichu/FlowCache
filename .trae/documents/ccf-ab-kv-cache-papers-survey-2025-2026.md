# 2025-2026 年 CCF-A/B KV cache+serving+agent 论文调研计划

## Summary

检索 2025-2026 年 CCF-A/B 会议中 KV cache + LLM serving + agent 推理领域的 20 篇论文，详细记录每篇的研究方向、数据集和精确样本量。**严格标准**：只纳入论文正文/表格中明确给出样本数字的论文，估算值和"未明确"一律排除。输出文档供 FlowCache 实验设计引用。

## Current State Analysis

### 已有基础

[reviews/survey-2025-2026-kv-cache-agent-papers.md](file:///d:/00MyProject/Prefix%20Caching/reviews/survey-2025-2026-kv-cache-agent-papers.md) 已有 20 篇论文调研，但：
- 仅 6 篇（30%）是 CCF-A/B（KVFlow/GraphFlow/Cake/Ada-KV/SAGA/ARKV）
- 仅 1 篇（Ada-KV）有样本量（且为估算值）
- 14 篇为 arXiv 预印本，不满足本次 CCF-A/B 要求

### 用户要求（Phase 2 已确认）

| 维度 | 决策 |
|---|---|
| 检索范围 | 中等：KV cache 管理 + LLM serving 系统 + agent 推理系统 |
| 样本量披露 | **严格**：必须论文正文/表格中明确给出精确数字，估算值和"未明确"一律排除 |
| 年份范围 | 混合：会议发表年份或 arXiv 提交年份任一落在 2025-2026 |
| 数量 | 20 篇，全部 CCF-A 或 CCF-B |

### Phase 1 初步检索发现的候选论文

通过 WebSearch 已发现以下 CCF-A/B 候选论文（需进一步核实样本量）：

| # | 论文 | Venue | CCF | 研究方向 | 初步数据集线索 |
|---|---|---|---|---|---|
| 1 | KVCOMM | NeurIPS 2025 | A | 多 agent KV cache 跨上下文复用 | RAG/math/coding，需核实样本数 |
| 2 | Mustafar | NeurIPS 2025 | A | KV cache 非结构化稀疏剪枝 | LongBench，需核实样本数 |
| 3 | DMS (Hyper-Scaling) | NeurIPS 2025 | A | KV cache 压缩 + 推理超扩 | AIME24/GPQA/LiveCodeBench/MATH500，需核实 |
| 4 | AQUA-KV | ICML 2025 | A | 自适应 KV 量化 | LongBench 14 任务，需核实 |
| 5 | KVTuner | ICML 2025 | A | 层级混合精度 KV 量化 | GSM8K/GPQA，需核实 |
| 6 | PM-KVQ | ICLR 2026 | A | 长链 KV 渐进式混合精度量化 | 512 校准样本（已确认） |
| 7 | KVTC (Transform Coding) | ICLR 2026 | A | KV cache 变换编码压缩 | AIME25/GSM8K/LongBench 等 8 个，需核实 |
| 8 | DirectKV | OSDI 2026 | A | 零拷贝 KV cache offloading | 需核实 |
| 9 | CacheSlide | FAST 2026 | A | 跨位置感知 KV cache 复用 | agent benchmarks，需核实 |
| 10 | KV Cache in the Wild | USENIX ATC 2025 | B | 生产 trace KV cache 特征分析 | Trace A + Trace B，需核实 |
| 11 | KVFlow | NeurIPS 2025 | A | 工作流感知前缀缓存 | 合成 workflow，需核实 |
| 12 | GraphFlow | ICML 2026 | A | 图结构工作流 KV 管理 | 5 benchmark，需核实 |
| 13 | Cake | ICML 2025 | A | KV cache 加载系统 | 需核实 |
| 14 | Ada-KV | NeurIPS 2025 | A | 自适应 KV cache | Ruler+LongBench 29 子任务，需核实 |
| 15 | SAGA | HPDC 2026 | B | 工作流原子调度 | SWE-bench+WebArena，需核实 |
| 16 | ARKV | CCGRID 2025 | B | 自适应 KV 管理（长上下文） | 需核实 |
| 17 | Mooncake | FAST 2025 | A | KV cache 分离式服务 | 生产 trace，需核实 |
| 18 | τ-bench | ICLR 2025 | A | agent 工具调用 benchmark | 165×8=1,320（已确认） |

还需检索至少 2-4 篇补足 20 篇，候选检索方向：
- ASPLOS 2025/2026（CCF-A）KV cache 相关
- HPCA 2025/2026（CCF-A）KV cache 相关
- EuroSys 2025/2026（CCF-B）LLM serving 相关
- SOSP 2025（CCF-A）LLM serving 相关

## Proposed Changes

### 步骤 1：补充 WebSearch 检索（4 轮并行）

**目标**：补足候选论文池至 25+ 篇，确保筛选后有 20 篇满足严格标准。

| 轮次 | 检索查询 | 目标会议 |
|---|---|---|
| 1a | "ASPLOS 2025 2026 KV cache LLM serving inference" | ASPLOS（CCF-A） |
| 1b | "SOSP 2025 LLM serving KV cache system" | SOSP（CCF-A） |
| 1c | "EuroSys 2025 2026 LLM inference KV cache serving" | EuroSys（CCF-B） |
| 1d | "HPCA 2025 2026 KV cache GPU LLM inference" | HPCA（CCF-A） |

### 步骤 2：逐篇 WebFetch 核实样本量（25+ 篇并行）

**目标**：对每篇候选论文进行 WebFetch，访问 arXiv abs/OpenReview/会议官网页面，提取：
- 精确样本数字（如 "150 problems"、"1,320 episodes"、"512 calibration samples"）
- 数据集名称和子任务数
- 模型列表
- 实验配置（seeds、hop 数等）

**严格筛选标准**：
- ✅ 纳入：论文正文/表格明确写出"we use N samples/problems/episodes"
- ❌ 排除：仅给数据集名称无数字（如"we evaluate on LongBench"）
- ❌ 排除：估算值（如"~2,900 based on 29 tasks × 100"）
- ❌ 排除：合成参数化配置（如"branches=1, depth=32"）
- ❌ 排除：生产 trace 无具体样本数（如"百万级 requests/天"）

### 步骤 3：编写最终调研文档

**输出文件**：[reviews/ccf-ab-kv-cache-papers-2025-2026.md](file:///d:/00MyProject/Prefix%20Caching/reviews/ccf-ab-kv-cache-papers-2025-2026.md)

**文档结构**：

```markdown
# 2025-2026 CCF-A/B KV cache + LLM serving + Agent 论文调研

## 1. 调研方法
### 1.1 纳入标准
- 会议：CCF-A 或 CCF-B
- 年份：会议发表或 arXiv 提交任一在 2025-2026
- 领域：KV cache 管理/压缩/量化/调度 + LLM serving 系统 + agent 推理系统
- 样本量：论文正文/表格明确给出精确数字

### 1.2 排除标准
- arXiv 预印本未被 CCF-A/B 会议接收
- 仅给数据集名称无样本数字
- 估算值或合成参数化配置
- 生产 trace 无具体样本数

### 1.3 CCF 等级对照
| 会议 | CCF 等级 | 2025-2026 届次 |
|---|---|---|
| NeurIPS | A | 2025, 2026 |
| ICML | A | 2025, 2026 |
| ICLR | A（视同） | 2025, 2026 |
| SOSP | A | 2025 |
| OSDI | A | 2026 |
| ASPLOS | A | 2025, 2026 |
| ISCA | A | 2025, 2026 |
| HPCA | A | 2025, 2026 |
| FAST | A | 2025, 2026 |
| SIGCOMM | A | 2025 |
| HPDC | B | 2025, 2026 |
| CCGRID | B | 2025, 2026 |
| EuroSys | B | 2025, 2026 |
| USENIX ATC | B | 2025, 2026 |

## 2. 论文总览表
| # | 论文 | Venue | CCF | 研究方向 | 数据集数 | 样本量 | 精确度 |

## 3. 每篇论文详细信息
### 论文 1: [标题]
- 标题/Venue/arXiv ID/作者/提交日期
- 研究方向（1-2 句）
- 关键技术（bullet list）
- 实验数据集表（数据集名称 | 样本数量 | 用途）
- 总样本量（精确数字）
- 模型列表
- 性能结果
- 数据来源（arXiv abs / OpenReview / 会议官网）

## 4. 统计汇总
- CCF 等级分布（A vs B）
- 会议分布
- 数据集数统计（中位数、范围）
- 样本量统计（中位数、范围）
- 数据集频率排行（top 5）

## 5. 对 FlowCache 的启示
- FlowCache 数据集数 vs 同领域 CCF-A/B 中位数
- FlowCache 样本量 vs 同领域 CCF-A/B 中位数
- FlowCache 数据集选择是否对标领域标配
```

### 步骤 4：交叉验证与质量检查

- 每篇论文的 CCF 等级通过 [CCF 推荐目录](https://www.ccf.org.cn/Academic_Evaluation/By_category) 核实
- 每篇论文的样本量数字必须能在 arXiv abs/OpenReview 页面或 PDF 正文中找到原文佐证
- 若某篇论文核实后不满足严格标准，从候选池中剔除并补检索
- 最终 20 篇全部满足"CCF-A/B + 2025-2026 + 精确样本量"三重标准

## Assumptions & Decisions

### 假设
1. 2025-2026 年 CCF-A/B 会议中 KV cache + LLM serving + agent 领域有足够多的论文（≥25 篇）供筛选
2. 至少 20 篇论文在正文/表格中明确披露了样本数字
3. WebFetch 能获取 arXiv abs/OpenReview 页面内容用于核实

### 决策
1. **ICLR 视同 CCF-A**：虽未入 CCF 推荐目录，但学术界普遍视为 A 类顶会，且 τ-bench 原论文发表于此
2. **USENIX ATC 视同 CCF-B**：虽 USENIX 官方未列入 CCF 目录，但学术界普遍视为 B 类
3. **样本量严格标准**：用户明确要求"必须精确数字"，估算值一律排除（即使基于合理推断）
4. **不更新现有 survey 文档**：新建独立文档 `ccf-ab-kv-cache-papers-2025-2026.md`，避免污染现有 20 篇调研
5. **若最终不足 20 篇**：如实报告实际找到的篇数，并说明原因（领域太新或样本量披露不足）

## Verification Steps

1. **CCF 等级验证**：每篇论文的 venue 必须能在 CCF 推荐目录中找到对应等级
2. **年份验证**：每篇论文的会议发表年份或 arXiv 提交年份必须在 2025-2026
3. **样本量验证**：每篇论文的样本数字必须有原文引用（WebFetch 返回内容或 PDF 正文截图）
4. **领域相关性验证**：每篇论文必须涉及 KV cache / LLM serving / agent 推理中的至少一项
5. **去重验证**：与现有 survey 20 篇不重复（若重复，标注并引用现有条目）
6. **最终数量验证**：文档必须包含恰好 20 篇满足全部标准的论文

## 执行顺序

```
Step 1: 4 轮并行 WebSearch（ASPLOS/SOSP/EuroSys/HPCA）→ 补足候选池
Step 2: 25+ 篇并行 WebFetch → 核实每篇样本量
Step 3: 筛选 20 篇满足严格标准的论文
Step 4: 编写 reviews/ccf-ab-kv-cache-papers-2025-2026.md
Step 5: 交叉验证 CCF 等级、年份、样本量、领域相关性
Step 6: 返回最终文档给用户
```
