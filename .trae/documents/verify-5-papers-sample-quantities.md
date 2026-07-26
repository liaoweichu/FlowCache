# 核实 5 篇论文实验精确样本数量 — 研究报告计划

## 摘要 (Summary)
本任务为纯研究任务，目标是核实 KVLink、NanoFlow、Cake、Ada-KV、KVCOMM 五篇 2025 年顶会论文实验中使用的**精确样本数量**。基于对 arXiv abs 页面、HTML 全文、PDF 内容以及 Web 搜索的多轮检索，整理每篇论文的标题/arXiv ID、会议年份、研究方向、数据集表格（含样本数或"未明确披露"标注）、原文引用与 ✅/❌ 判定。最终输出为单条结构化中文回复，不创建任何代码或额外文档文件。

## 当前状态分析 (Current State Analysis)

通过 Phase 1 的探索，已确认以下事实：

### 已确认的论文元数据
| 论文 | arXiv ID | 标题 | 会议 |
|---|---|---|---|
| KVLink | 2502.16002 | KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse | NeurIPS 2025 |
| NanoFlow | 2408.12757 | NanoFlow: Towards Optimal Large Language Model Serving Throughput | OSDI 2025 |
| Cake | 2410.03065 | Compute Or Load KV Cache? Why not Both? | ICML 2025 |
| Ada-KV | 2407.11550 | Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference | NeurIPS 2025 |
| KVCOMM | 2510.12872 | KVComm: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems | NeurIPS 2025 |

### 各论文样本数量披露情况（基于已检索内容）

**1. KVLink (arXiv:2502.16002)**
- 摘要明确指出"Experiments across 7 datasets"
- 训练数据包括 2WikiMQA、TriviaQA、FineWeb、Tülu 3（来自检索到的实验设置片段）
- 评估数据集提及 NQ、HotpotQA 等，但**未在已检索内容中明确披露每个评估数据集的精确样本数**
- 仅给出性能提升数字（如 NQ +6.6%、HotpotQA +7.3%），未给出样本数

**2. NanoFlow (arXiv:2408.12757)**
- 使用 ShareGPT、LMSys-Chat、Splitwise 三个工作负载
- Web 搜索结果中找到请求长度统计表（Avg. Input/Output + 标准差）：
  - Splitwise: Avg. Input 1155 (Std 1109), Avg. Output 211 (Std 163)
  - LMSYS-Chat: Avg. Input 102 (Std 169), Avg. Output 222 (Std 210)
  - ShareGPT: Avg. Input 246 (Std 547), Avg. Output 322 (Std 244)
- 但**未明确披露每个数据集的请求/样本总数**

**3. Cake (arXiv:2410.03065)**
- 使用 LongAlpaca-7B 和 LongAlpaca-13B 模型进行评估
- 测试上下文长度 5/9/14k，带宽 2000-32000 mbps，GPU 资源 10/50/90%
- **未明确披露 LongAlpaca 评估样本总数**

**4. Ada-KV (arXiv:2407.11550)**
- 摘要明确指出"13 datasets from Ruler and 16 datasets from LongBench"
- 总计 29 个数据集
- v3 版本曾提到"16 datasets within LongBench"和"Needle-in-a-Haystack"
- **未明确披露每个子任务的样本数**

**5. KVCOMM (arXiv:2510.12872)**
- **明确披露**：GSM8K 在四 agent 系统中使用 1,319 个样本（原文："as the reuse rate reaches 95% across 1,319 samples in a four-agent system for GSM8K"）
- 三个任务：RAG（MMLU）、数学推理（GSM8K）、协同编程（HumanEval）
- RAG 和编程任务的样本数**未明确披露**

## 提议的回复结构 (Proposed Changes)

最终的中文回复将按以下结构组织（不创建任何文件，直接在对话中返回）：

### 总体表格
一张汇总表，包含：论文 | arXiv ID | 会议 | 数据集数量 | 是否披露精确样本数（✅/❌）

### 每篇论文详细条目（5 个小节）

每节包含 6 个字段，严格遵循用户要求的格式：

1. **论文完整标题和 arXiv ID**
2. **发表会议和年份**
3. **研究方向（1-2 句）**
4. **实验数据集表格**（数据集名称 | 精确样本数量 | 用途）
   - 已披露的填具体数字
   - 未披露的统一标注"未明确披露样本数"
5. **原文引用**（带英文原文引用，标注来源章节）
6. **判定**：✅ 有精确样本数字 / ❌ 未明确披露

### 关键发现总结
- 5 篇论文中仅 KVCOMM 的 GSM8K (1,319 样本) 明确披露样本数
- 其余 4 篇论文（KVLink、NanoFlow、Cake、Ada-KV）均未明确披露精确样本数
- 多数论文仅给出数据集名称、子任务数或长度统计，未给出具体样本/请求数量

## 假设与决策 (Assumptions & Decisions)

1. **数据来源**：以已检索的 arXiv HTML 全文、PDF 内容、Web 搜索结果为准；不再进行额外检索（已经过多轮充分检索）
2. **判定标准**：只有原文中明确出现"X samples"、"X requests"、"X test examples"等具体数字时才标记 ✅；仅给出数据集名称或子任务数不算披露
3. **KVCOMM 特例**：GSM8K 单项 ✅，但 RAG/编程任务 ❌；整体判定为部分披露
4. **语言**：最终回复用中文，原文引用保留英文
5. **不创建文件**：遵循用户"纯研究任务，不需要写代码或修改任何文件"的指示

## 验证步骤 (Verification steps)

1. 检查每篇论文的 6 个字段是否完整
2. 确认 ✅/❌ 判定与原文引用一致
3. 确认未将"数据集名称数量"误判为"样本数量"（如 Ada-KV 的 29 个数据集 ≠ 29 个样本）
4. 确认 KVCOMM 的 1,319 样本引用准确（来自原文 Introduction 段）
5. 确认输出为单一中文回复，无多余文件创建
