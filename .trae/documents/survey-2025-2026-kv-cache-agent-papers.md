# 调研计划：2025-2026 年 KV cache + Agent 工作流领域 20 篇论文数据集调研

## Summary

系统检索 2025-2026 年发表的、与 FlowCache 研究范围（**狭义：KV cache 管理 / 前缀缓存 / Agent 工作流 KV 管理 / 联合 residency 控制**）最接近的 20 篇论文，对每篇提取：研究方向、实验评估数据集名称、单数据集样本数量、总样本量。最终产出一份结构化调研文档，用于支撑 FlowCache 实验设计的数据集选择论证。

**说明**：用户原文说"训练集"，但 FlowCache 是推理/缓存系统，不训练模型。本调研中"数据集"均指**实验评估用的数据集**（evaluation datasets / benchmarks），非模型训练集。

## Current State Analysis

### 项目已有基础
- `reviews/prior-art-verification.md` 已核验 6 篇 2026 年 arXiv 论文（PBKV/ARKV/QKVShare/GraphFlow/CacheWise/ThunderAgent）
- `IDEA.rewritten.md §3.1` 列出 15 篇最接近工作（含 2025-2026 论文）
- `.trae/specs/trim-dataset-portfolio/spec.md` 已对比 14 篇同领域论文的数据集数和样本量
- 项目已引用的 2025-2026 狭义论文约 15 篇，但数据集信息分散、部分未核实

### 调研范围界定（狭义）
**纳入范围**（必须满足至少一项）：
- KV cache 管理策略（驱逐/准入/替换/分层）
- 前缀缓存 / RadixAttention / block-level prefix reuse
- Agent / 工作流感知的 KV 管理（tool call 暂停/恢复、DAG 调度）
- KV cache 量化/压缩与 residency 联合控制
- LLM serving 中的 KV cache 调度（TTFT/尾延迟/SLO）

**排除范围**：
- 纯长上下文注意力压缩（MInference 等，无 cache 管理成分）
- 纯模型训练/微调（非推理系统）
- 纯多模态 KV（无文本 LLM serving 关联）
- 2024 年及更早的论文（用户明确要求 2025-2026）

### 关键约束
- 狭义范围下 2025-2026 论文可能不足 20 篇纯新论文；允许包含项目已引用的，但**所有数据集信息必须重新核实**（不直接复用项目内已有描述）
- 优先级：NeurIPS/ICML/ICLR/OSDI/SOSP/ASPLOS/ATC/EuroSys/SIGCOMM 2025-2026 > arXiv 2025-2026 预印本
- 每篇论文必须给出**数据集名称**和**样本数量**；若论文未明确披露，标注"未明确"并给出估算依据

## Proposed Changes

### 产出文件
- **新建**：`d:\00MyProject\Prefix Caching\reviews\survey-2025-2026-kv-cache-agent-papers.md`
  - 一份结构化调研文档，包含 20 篇论文的详细信息

### 文档结构

```
# 2025-2026 KV Cache + Agent 工作流领域论文数据集调研

## 1. 调研范围与方法
- 纳入/排除标准
- 检索源（arXiv、Google Scholar、venue proceedings）
- 检索关键词组合

## 2. 论文总览表（20 篇）
| # | 论文 | Venue/Year | 研究方向 | 数据集数 | 总样本量 |
|---|------|-----------|---------|---------|---------|
| 1 | ... | NeurIPS 2025 | ... | ... | ... |

## 3. 每篇论文详细信息
### 论文 1: [标题]
- **Venue/Year**: 
- **arXiv ID**: 
- **研究方向**: （1-2 句概述核心贡献）
- **关键技术**: （KV cache 管理策略/前缀缓存/Agent 感知等）
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | ... | ... | ... |
- **总样本量**: 
- **数据来源**: （论文哪一节引用 / WebSearch 结果链接）

...（重复 20 次）

## 4. 统计汇总
- 数据集数分布（中位数、范围）
- 样本量分布（中位数、范围）
- 常见数据集排名（ShareGPT、LongBench、τ-bench 等出现频次）

## 5. 对 FlowCache 的启示
- FlowCache 5 数据集 / ~3,720 样本 vs 同领域论文的对比定位
- 哪些数据集是领域标配（FlowCache 必须包含）
- 哪些数据集是 FlowCache 独有（差异化）
```

### 检索策略

**Step 1: 已有项目文档梳理**（快速获得 ~10 篇候选）
- 读 `reviews/prior-art-verification.md` → 6 篇已核验论文
- 读 `IDEA.rewritten.md §3.1` → 15 篇最接近工作列表
- 读 `.trae/specs/trim-dataset-portfolio/spec.md` 研究对比表 → 10 篇
- 去重后得到候选清单（约 15-18 篇 2025-2026 狭义论文）

**Step 2: WebSearch 补充检索**（补齐到 20 篇 + 核实数据集）
并行执行以下 WebSearch 查询（每个 5 条结果）：
1. `"KV cache" "agent" 2025 OR 2026 arxiv`
2. `"prefix caching" "LLM serving" 2025 OR 2026`
3. `"KV cache eviction" "workflow" 2025 OR 2026`
4. `"KV cache" "tool call" 2025 OR 2026`
5. `"KV cache" "quantization" "residency" 2025 OR 2026`
6. `"agent serving" "KV cache" 2025 OR 2026 NeurIPS OR ICML OR OSDI`
7. `"next-use prediction" "KV cache" 2025 OR 2026`
8. `"RadixAttention" OR "prefix tree" "KV cache" 2025 OR 2026`

**Step 3: WebFetch 核实关键论文**
对每篇候选论文，若数据集信息不明确，WebFetch 其 arXiv abstract 或 PDF：
- `https://arxiv.org/abs/{arxiv_id}` 获取摘要
- 必要时 `https://arxiv.org/pdf/{arxiv_id}` 获取实验章节

**Step 4: 数据集信息提取**
对每篇论文提取：
- 数据集名称（如 ShareGPT、LongBench、τ-bench、GSM8K）
- 单数据集样本数量（如 500 conversations、1000 questions）
- 总样本量（所有数据集求和）
- 若论文只给 "requests" 不给 "samples"，按上下文换算

### 候选论文种子清单（基于项目已有信息，需重新核实）

以下论文已在项目中出现，作为检索种子（**数据集信息必须重新核实**）：

| # | 论文 | arXiv/Venue | 备注 |
|---|------|------------|------|
| 1 | KVFlow | arXiv 2507.07400 (NeurIPS 2025) | Agent Step Graph + 未来感知驱逐 |
| 2 | PBKV | arXiv 2605.06472 (2026-05) | GraphSAGE reuse 预测 |
| 3 | CacheWise | arXiv 2606.16824 (2026-06) | Coding agent KV |
| 4 | ThunderAgent | arXiv 2602.13692 (2026-02) | LLM Programs pause/restore |
| 5 | TokenCake | arXiv 2510.18586 (2025-10) | Function-call-aware offload |
| 6 | Helium | arXiv 2603.16104 (2026-03) | Workflow-as-query-plan |
| 7 | ARKV | arXiv 2603.08727 (CCGRID 2025) | 三态 KV 管理 |
| 8 | QKVShare | arXiv 2605.03884 (2026-05) | Multi-agent DAG handoff |
| 9 | GraphFlow | arXiv 2605.22566 (ICML 2026) | wGraph + base KV |
| 10 | Agent Memory | arXiv 2603.04428 (2026-03) | 持久化 Q4 KV |
| 11 | HybridFlow | arXiv 2512.22137 (2025-12) | DAG 子任务端云路由 |
| 12 | Continuum | arXiv 2511.02230 (2025-11) | 工具暂停 TTL |
| 13 | EvicPress | arXiv 2512.14946 (2025-12) | 12 数据集评估 |
| 14 | FlowKV | arXiv 2504.03775 (2025-04) | LongBench KV 压缩 |
| 15 | Ada-KV | NeurIPS 2025 | Ruler + LongBench |
| 16 | Learned Prefix Caching | NeurIPS 2025 | 对话内容预测驱逐 |
| 17 | InferCept | ICML 2024 | 工具间断 KV（2024 边界，可作 2025 camera-ready） |
| 18 | CacheGen | SIGCOMM 2024 | 2024 但 venue 2024-08（边界） |
| 19 | Mooncake | FAST 2025 | 生产 trace |
| 20 | τ-bench | ICLR 2025 | Agent benchmark |

**说明**：#17-18 严格说是 2024 工作，但 venue 在 2024 下半年/2025 初。若用户严格要求 2025-2026，则替换为 Step 2 新检索的论文。最终 20 篇以 Step 2 检索结果为准。

## Assumptions & Decisions

### 假设
1. **"训练集" = "实验评估数据集"**：FlowCache 是推理系统不训练模型，用户说的"训练集"理解为评估用的 benchmark/workload 数据集
2. **狭义范围可能不足 20 篇**：若 Step 2 检索后仍不足 20 篇纯狭义论文，允许纳入 1-3 篇中义范围（LLM serving 系统、KV 压缩）论文补齐，但会在文档中标注
3. **数据集信息不完整的论文**：若某篇论文只给 "requests" 不给具体样本数，标注"未明确"并按典型 batch size 估算
4. **arXiv ID 2026 开头（如 2603.xxxxx）**：项目 prior-art 文档中出现的 2026 arXiv ID，可能是虚构或真实预印本，需 WebSearch/WebFetch 核实真实性

### 决策
1. **输出语言**：中文（与用户最新消息一致）
2. **输出位置**：`reviews/survey-2025-2026-kv-cache-agent-papers.md`（与现有 `reviews/prior-art-verification.md` 同目录）
3. **不修改现有文件**：本调研仅产出新文档，不修改 experiment-designs.md 等现有文件（用户未要求）
4. **并行检索**：Step 2 的 8 个 WebSearch 查询并行执行，最大化效率
5. **核实优先级**：优先核实 arXiv abstract（快），仅对数据集信息不明确的论文 fetch PDF

## Verification Steps

执行完成后，验证以下检查点：

1. **论文数量**：文档包含恰好 20 篇论文（不多不少）
2. **时间范围**：所有论文均为 2025 或 2026 年发表（venue/arXiv 时间）
3. **领域相关**：每篇论文均在狭义范围内（KV cache + Agent 工作流）
4. **信息完整性**：每篇论文都有：
   - [ ] 论文标题
   - [ ] Venue/Year
   - [ ] arXiv ID 或链接
   - [ ] 研究方向（1-2 句）
   - [ ] 数据集名称列表
   - [ ] 单数据集样本数量
   - [ ] 总样本量
5. **数据真实性**：数据集信息来自 WebSearch/WebFetch 核实，非编造
6. **统计汇总**：文档末尾有数据集数和样本量的统计汇总
7. **FlowCache 对比**：文档末尾有 FlowCache（5 数据集 / ~3,720 样本）与同领域论文的对比定位

## 执行步骤

1. **读取项目已有文档**（获取种子清单和数据集信息基础）
   - Read `reviews/prior-art-verification.md`
   - Read `IDEA.rewritten.md §3.1`（用 Grep 定位）
   - Read `.trae/specs/trim-dataset-portfolio/spec.md`

2. **并行 WebSearch 检索**（8 个查询并行）
   - 执行上述 Step 2 的 8 个 WebSearch 查询
   - 汇总结果，去重，筛选 2025-2026 狭义论文

3. **WebFetch 核实关键论文**（对数据集信息不明确的论文）
   - 并行 fetch 5-10 篇论文的 arXiv abstract
   - 提取实验章节的数据集信息

4. **生成调研文档**
   - 按文档结构组织 20 篇论文信息
   - 填充总览表和每篇详细信息
   - 撰写统计汇总和 FlowCache 对比

5. **验证**
   - 检查 20 篇论文信息完整性
   - 确认所有数据集信息有来源
