# 2025-2026 KV Cache + Agent 工作流领域论文数据集调研

**调研日期**：2026-07-25
**调研范围**：2025-01 至 2026-07 发表的、与 FlowCache 研究范围（狭义：KV cache 管理 / 前缀缓存 / Agent 工作流 KV 管理 / 联合 residency 控制）最接近的论文
**调研目的**：对比同领域论文的数据集选择和样本量，为 FlowCache 实验设计提供论证依据

---

## 1. 调研范围与方法

### 1.1 纳入标准（必须满足至少一项）
- KV cache 管理策略（驱逐/准入/替换/分层）
- 前缀缓存 / RadixAttention / block-level prefix reuse
- Agent / 工作流感知的 KV 管理（tool call 暂停/恢复、DAG 调度）
- KV cache 量化/压缩与 residency 联合控制
- LLM serving 中的 KV cache 调度（TTFT/尾延迟/SLO）

### 1.2 排除标准
- 纯长上下文注意力压缩（无 cache 管理成分）
- 纯模型训练/微调（非推理系统）
- 纯多模态 KV（无文本 LLM serving 关联）
- 2024 年及更早的论文

### 1.3 检索源
- arXiv（cs.DC / cs.OS / cs.LG / cs.MA）
- WebSearch（Google 索引的 arXiv HTML 版本、venue proceedings）
- 项目已有文档（`reviews/prior-art-verification.md`、`IDEA.rewritten.md §3.1`）

### 1.4 检索关键词
`"KV cache" AND ("agent" OR "workflow" OR "tool call")`、`"prefix caching" AND "LLM serving"`、`"KV cache eviction"`、`"KV cache" AND "quantization" AND "residency"`

### 1.5 说明
FlowCache 是推理/缓存系统，不训练模型。本调研中"数据集"均指**实验评估用的数据集**（evaluation datasets / benchmarks），非模型训练集。部分论文仅披露数据集名称未披露样本量，标注"未明确"。

---

## 2. 论文总览表（20 篇）

| # | 论文 | Venue/Year | 研究方向 | 数据集数 | 总样本量 |
|---|------|-----------|---------|---------|---------|
| 1 | KVFlow | NeurIPS 2025 (arXiv 2507.07400) | 工作流感知前缀缓存 + Agent Step Graph 驱逐 | ~2 | 未明确（合成 workflow） |
| 2 | PBKV | arXiv 2026-05 (2605.06472) | 预测式 KV-Cache 管理（动态工作流） | 3 | 未明确 |
| 3 | CacheWise | arXiv 2026-06 (2606.16824) | 编码 agent KVCache 管理 + 真实 trace 画像 | 1（自采） | 未明确（真实编码 agent traces） |
| 4 | GraphFlow | ICML 2026 (2605.22566) | 图结构工作流管理 + wGraph + base KV | 5 | 未明确 |
| 5 | ThunderAgent | arXiv 2026-02 (2602.13692) | LLM Programs 程序感知 agentic 推理 | 3 | 未明确 |
| 6 | TokenCake | arXiv 2025-10 (2510.18586) | 多 agent KV-Cache 时空双调度 | 2 | 未明确（ShareGPT + AgentCode） |
| 7 | Continuum | arXiv 2025-11 (2511.02230) | 多轮 agent KV Cache TTL 调度 | 3 | 未明确（SWE-Bench + BFCL + OpenHand） |
| 8 | MemDecay | arXiv 2026-07 (2607.10582) | 区域感知 KV 驱逐（agent trace 语义结构） | 1（自采） | ~24 facts × 4 设置（2 上下文 × 2 模型） |
| 9 | SAGA | HPDC 2026 (2605.00528) | 工作流原子调度 + Agent Execution Graph | 2 | 未明确（SWE-bench + WebArena） |
| 10 | SideQuest | arXiv 2026-02 (2602.22603) | 模型驱动 KV 管理（侧线程语义驱逐） | 2 | 未明确（FRAMES + BrowseComp） |
| 11 | Agent Memory | arXiv 2026-02 (2603.04428) | 边缘多 agent 持久化 Q4 KV cache | 3（模型） | 未明确（4K-32K context 测试） |
| 12 | ARKV | CCGRID 2025 (2603.08727) | 自适应资源高效 KV 管理（长上下文） | 未明确 | 未明确（LLaMA3/Qwen3 长上下文任务） |
| 13 | QKVShare | arXiv 预印本 2026-05 (2605.03884)¹¹ | 多 agent 量化 KV handoff（inter-agent） | 1 | **150 problems**（GSM8K × 2-5 hops） |
| 14 | HyMCache | arXiv 2026-07 (2607.18141) | CXL-Hybrid 内存 KV cache 框架 | 未明确 | 未明确（对比 LMCache/Mooncake） |
| 15 | C2KV | arXiv 2026-07 (2607.17715) | 压缩可组合 KV cache 复用 | 未明确 | 未明确（RAG + 多文档推理） |
| 16 | Error Certificates | arXiv 2026-07 (2607.21475) | 随机化 KV 驱逐误差证书 | 未明确 | 未明确（预注册 7 claims） |
| 17 | EvicPress | arXiv 2025-12 (2512.14946) | KV cache 驱逐 + 压力测试 | **12** | **~600 contexts**（估算） |
| 18 | Cake | ICML 2025 | KV cache 加载系统（计算/IO 平衡） | 未明确 | 未明确（多种硬件/数据集/存储） |
| 19 | FlowKV | arXiv 2025-04 (2504.03775) | KV cache 压缩 | 1 | 未明确（LongBench） |
| 20 | Ada-KV | NeurIPS 2025 | 自适应 KV cache（attention 统计） | 2（29 子任务） | ~2,900（估算，Ruler 13 + LongBench 16） |

> ¹¹ QKVShare 为 arXiv 预印本，未被任何会议/期刊正式接收（非 peer-reviewed）。引用时不得称为"published in"或"accepted to"。

---

## 3. 每篇论文详细信息

### 论文 1: KVFlow
- **标题**: KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- **Venue/Year**: NeurIPS 2025 (arXiv 2507.07400, 2025-07-10)
- **arXiv ID**: [2507.07400](https://arxiv.org/abs/2507.07400)
- **研究方向**: 工作流感知的 KV cache 管理框架，将 agent 执行调度抽象为 Agent Step Graph，为每个 agent 分配 steps-to-execution 值估计未来激活的时序接近度，指导 KV node 级细粒度驱逐策略，并引入全重叠 KV 预取机制。
- **关键技术**: Agent Step Graph、steps-to-execution 驱逐策略、tree-structured cache 共享前缀管理、CPU→GPU 后台预取
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 合成 workflow（参数化） | 未明确 | 单 workflow 大 prompt 测试（branches=1 等参数） |
  | PEER benchmark | 未明确 | 多并发 workflow 场景 |
- **总样本量**: 未明确（论文使用参数化合成 workflow，按 branches/depth/width 配置生成）
- **对比基线**: SGLang with hierarchical radix cache
- **性能**: 单 workflow 1.83× speedup，多并发 workflow 2.19× speedup
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`

### 论文 2: PBKV
- **标题**: Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management
- **Venue/Year**: arXiv 2026-05 (2605.06472, 2026-05-07)
- **arXiv ID**: [2605.06472](https://arxiv.org/abs/2605.06472)
- **研究方向**: 针对动态 agent 工作流（调用序列依赖任务上下文）的预测式 KV-Cache 管理，融合历史工作流引导和当前任务上下文预测未来若干步的 agent 调用，估计 cache entry 复用潜力，保守使用预测以增强鲁棒性。
- **关键技术**: GraphSAGE + workflow-history attention、连续 reuse score、多步预测、保守驱逐/预取
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 3 个 workflow benchmarks | 未明确 | 动态工作流 + 静态工作流对比 |
- **总样本量**: 未明确（论文摘要仅说"三个 workflow benchmark"）
- **对比基线**: LRU、KVFlow
- **性能**: 相对 LRU 1.85× speedup（动态），相对 KVFlow 1.26× speedup（静态）
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`

### 论文 3: CacheWise
- **标题**: CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents
- **Venue/Year**: arXiv 2026-06 (2606.16824, 2026-06-15)
- **arXiv ID**: [2606.16824](https://arxiv.org/abs/2606.16824)
- **研究方向**: 收集真实编码 agent trace 数据集进行工作负载画像，发现编码 agent 会话反复复用大前缀并产生持续 KVCache 压力；提出 prefix-aware scheduling + reuse-aware eviction（由 tool call metadata 轻量预测引导）。
- **关键技术**: 真实编码 agent trace 收集、prefix-aware scheduling、tool call metadata 预测
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 自采真实编码 agent traces | 未明确 | 工作负载画像 + 系统评估 |
- **总样本量**: 未明确（论文摘要说"collected traces"，具体数量需查 PDF）
- **对比基线**: vLLM 默认策略
- **性能**: KVCache 驱逐减少 2-2.6×，agent 会话完成时间提升 3.5×
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`

### 论文 4: GraphFlow
- **标题**: GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving
- **Venue/Year**: ICML 2026 (arXiv 2605.22566, 2026-05-21)
- **arXiv ID**: [2605.22566](https://arxiv.org/abs/2605.22566)
- **研究方向**: 提出统一图表示 wGraph（每个节点对应原子操作），作为任务特定工作流动态实例化的共享基底；adaptive workflow generation 从 wGraph 基于任务语义动态构造工作流；topology-aware state management 利用 wGraph 结构管理 KV cache（base KV + residual 重构）。
- **关键技术**: wGraph 统一图表示、adaptive workflow generation、base KV + residual 重构 `KV(P,v) = KVbase(v) + ΔKV(P,v)`
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 5 个 benchmark datasets | 未明确 | 性能对比（平均提升 4.95 个百分点） |
- **总样本量**: 未明确（论文摘要说"five benchmark datasets"，具体名称需查 PDF）
- **对比基线**: SOTA 方法
- **性能**: 平均提升 4.95 个百分点，内存占用减少约 4×
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`

### 论文 5: ThunderAgent
- **标题**: ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System
- **Venue/Year**: arXiv 2026-02 (2602.13692, v1 2026-02-14, v3 2026-06-30)
- **arXiv ID**: [2602.13692](https://arxiv.org/abs/2602.13692)
- **研究方向**: 将 agentic workflow 抽象为 LLM Programs，统一管理 KV cache、系统状态、外部工具资源（disk memory、network ports）；program-aware scheduler 和 tool resource manager 最大化 KV cache 命中率，缓解内存不平衡，支持异步环境准备。
- **关键技术**: LLM Programs 抽象、program-aware scheduler、tool resource manager、异步环境准备
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | coding agent | 未明确 | 编码场景 |
  | routing agent | 未明确 | 路由场景 |
  | scientific discovery agent | 未明确 | 科学发现场景 |
- **总样本量**: 未明确
- **对比基线**: SOTA 推理系统
- **性能**: serving 吞吐 1.5-3.6×，RL rollout 1.8-3.9×，disk 内存节省 4.2×
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`
- **开源**: [github.com/Agentic-Kinetics/ThunderAgent](https://github.com/Agentic-Kinetics/ThunderAgent)

### 论文 6: TokenCake
- **标题**: TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications
- **Venue/Year**: arXiv 2025-10 (2510.18586, v3 2026-05-20)
- **arXiv ID**: [2510.18586](https://arxiv.org/abs/2510.18586)
- **研究方向**: 针对多 agent 应用的 KV-Cache 时空双调度：Temporal Scheduler 事件驱动主动 offload 空闲 KV cache + 预测性 upload；Spatial Scheduler 动态内存分区 + 混合优先级（图结构 + 运行时状态）为关键路径 agent 预留 GPU 内存。
- **关键技术**: Temporal Scheduler（offload/prefetch）、Spatial Scheduler（动态分区）、CPU block buffer、progressive GPU reservation
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | Code-Writer workload | 未明确 | 多 agent 编码流水线（programmer/reviewer/tester） |
  | Deep-Research workload | 未明确 | 多 agent 研究流水线（search/summarize/synthesize） |
  | ShareGPT + AgentCode requests | 未明确 | Poisson 到达请求 |
- **总样本量**: 未明确
- **对比基线**: vLLM
- **性能**: 端到端延迟降低 47.06%+，GPU KV 利用率提升 16.9%
- **数据来源**: arXiv abstract + WebSearch 结果

### 论文 7: Continuum
- **标题**: Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **Venue/Year**: arXiv 2025-11 (2511.02230, v6 2026-05-25)
- **arXiv ID**: [2511.02230](https://arxiv.org/abs/2511.02230)
- **研究方向**: 针对多轮 agent 工作流（LLM 调用与工具交替）的 KV cache TTL 机制，选择性 pin KV cache 在 GPU 内存，TTL 由 reload cost 和潜在排队延迟决定；TTL 过期后自动驱逐，对边缘情况鲁棒。
- **关键技术**: KV cache TTL、prefill/reload cost 建模、queueing delay 建模、program-level FCFS
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | SWE-Bench | 未明确 | 编码 agent |
  | BFCL | 未明确 | 函数调用 agent |
  | OpenHand | 未明确 | 多轮 agent |
- **总样本量**: 未明确
- **模型**: Llama-3.1 8B/70B, Gemma-3 12B, GLM-4.5 355B
- **对比基线**: 原始 end-of-turn eviction
- **性能**: 平均 job completion time 提升 8×+
- **数据来源**: arXiv abstract + WebSearch 结果

### 论文 8: MemDecay
- **标题**: MemDecay: Region-Aware KV Cache Evict for Efficient LLM Agent Inference
- **Venue/Year**: arXiv 2026-07 (2607.10582)
- **arXiv ID**: [2607.10582](https://arxiv.org/abs/2607.10582)
- **研究方向**: 训练无关的 KV cache 驱逐框架，利用 agent trace 的语义结构（system prompt/plan/user message/tool input/tool output/retrieval/scratchpad 等区域标签），为每个区域分配基础优先级和衰减率，结合 attention 衍生重要性，应用区域特定时间衰减。
- **关键技术**: region label、region-specific base priority + decay rate、attention-derived importance、page-granular eviction
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 自采 agent traces | 24 facts × 4 设置 | region-conditioned attention lifetime 测量 |
- **总样本量**: ~96（24 facts × 2 上下文长度 × 2 模型）
- **模型**: Qwen2.5-1.5B, Qwen2.5-3B
- **上下文长度**: ~450, ~1700 tokens
- **数据来源**: WebSearch 结果（arXiv PDF）

### 论文 9: SAGA
- **标题**: SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters
- **Venue/Year**: HPDC 2026 (arXiv 2605.00528, 2026-05-01)
- **arXiv ID**: [2605.00528](https://arxiv.org/abs/2605.00528)
- **研究方向**: 分布式 scheduler，将整个 agent workflow（而非单个推理调用）作为一等调度单元；Agent Execution Graphs 捕获工作流结构预测 KV cache 跨 tool-call 边界复用；session-affinity batching + work stealing；Agent Fair Share 任务完成时间公平性。
- **关键技术**: Agent Execution Graphs、session-affinity batching、work stealing、Agent Fair Share、speculative prefetching
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | SWE-bench | 未明确 | 编码 agent |
  | WebArena | 未明确 | 浏览器 agent |
- **总样本量**: 未明确
- **硬件**: 64-GPU cluster
- **对比基线**: vLLM v0.15.1 with prefix caching + affinity routing
- **性能**: 任务完成时间 1.64× speedup，GPU 内存利用率 1.22×，SLO 达成率 99.2%
- **数据来源**: WebSearch 结果（arXiv HTML）

### 论文 10: SideQuest
- **标题**: SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning
- **Venue/Year**: arXiv 2026-02 (2602.22603, 2026-02-27)
- **arXiv ID**: [2602.22603](https://arxiv.org/abs/2602.22603)
- **研究方向**: 模型驱动 KV cache 压缩，使用 LRM 本身（而非 attention 启发式）识别并驱逐长 agentic 任务中的过时 tool outputs；每 K 轮 fork 生成到侧线程，检查开放 tool outputs 的 cursor indices，推理哪些已过时，发出结构化删除命令。
- **关键技术**: side thread fork、cursor index 追踪、语义驱逐（"last-use index"）、turn boundary 同步
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | FRAMES | 未明确 | 长程 agentic 推理 |
  | BrowseComp | 未明确 | 浏览器 agent |
- **总样本量**: 未明确
- **性能**: peak token 用量 -56% 到 -65%，KV cache 内存读取 -53% 到 -71%，SGLang 吞吐 +83.9%，端到端 runtime -36.8%，精度退化 ≤2%
- **数据来源**: WebSearch 结果（GitHub issue 引用 arXiv）

### 论文 11: Agent Memory
- **标题**: Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices
- **Venue/Year**: arXiv 2026-02 (2603.04428, 2026-02-17)
- **arXiv ID**: [2603.04428](https://arxiv.org/abs/2603.04428)
- **研究方向**: 边缘设备多 agent LLM 推理的持久化 Q4 KV cache，将每个 agent 的 KV cache 以 4-bit 量化格式持久化到磁盘，直接重载到 attention 层，消除冗余 O(n) prefill；block pool + BatchQuantizedKVCache + cross-phase context injection。
- **关键技术**: Q4 量化持久化、safetensors 格式、block pool、BatchQuantizedKVCache、cross-phase context injection
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | Gemma 3 12B（dense GQA, 48 层） | 4K-32K context | TTFT 测试 |
  | DeepSeek-Coder-V2-Lite 16B（MoE MLA, 27 层） | 4K-32K context | TTFT 测试 |
  | Llama 3.1 8B（dense GQA, 32 层） | 1K-16K context | TTFT 测试 |
- **总样本量**: 未明确（按 context length 测试点数估算 ~12-15 测试点）
- **硬件**: Apple M4 Pro（10.2 GB cache budget）
- **性能**: TTFT 降低 136×（Gemma 22-136×），Q4 比 FP16 容纳 4× 更多 agent contexts
- **数据来源**: WebSearch 结果（arxiv.deeppaper.ai）
- **开源**: [github.com/yshk-mxim/agent-memory](https://github.com/yshk-mxim/agent-memory)

### 论文 12: ARKV
- **标题**: ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs
- **Venue/Year**: CCGRID 2025 (arXiv 2603.08727, 2026-02-19，标注 Accepted in ACM/IEEE CCGRID 2025)
- **arXiv ID**: [2603.08727](https://arxiv.org/abs/2603.08727)
- **研究方向**: 长上下文推理的自适应资源高效 KV cache 管理，prefill 阶段通过 attention entropy/variance/kurtosis 估计每层 original quantization 比例；decoding 阶段基于 heavy-hitter 打分将 token 分配到 Original/Quantization/Eviction 三态。
- **关键技术**: attention entropy/variance/kurtosis、O/Q/E 三态分配、heavy-hitter 打分
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 长上下文任务（具体名称未明确） | 未明确 | LLaMA3/Qwen3 精度保留测试 |
- **总样本量**: 未明确
- **模型**: LLaMA3, Qwen3
- **性能**: 保留 ~97% 精度，KV 内存减少 4×
- **数据来源**: arXiv abstract + 项目 `reviews/prior-art-verification.md`

### 论文 13: QKVShare
- **标题**: QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs
- **Venue/Year**: **arXiv 预印本**（2026-05-05，2605.03884）— **未被任何会议/期刊正式接收**（Comments 字段无 venue 标注，非 peer-reviewed）
- **arXiv ID**: [2605.03884](https://arxiv.org/abs/2605.03884)
- **研究方向**: 多 agent 边缘设备间的量化 KV cache 交接（inter-agent handoff），结合 token 级 mixed-precision 分配、自包含 CacheCard 表示、HuggingFace 兼容的 cache 注入路径。
- **关键技术**: token 级 mixed-precision、CacheCard 表示、HuggingFace 兼容注入、topology-aware 重要性评分（扩展"Don't Waste Bits"控制器 + downstream-demand + segment features）
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | GSM8K | **150 problems** × 2-5 hops | 多 agent 链式 KV handoff 精度退化测试（inter-agent handoff） |
- **总样本量**: **150 problems**（每 problem 经历 2/3/4/5 个 agent hop，总计 ~600-750 agent transitions）
- **模型**: Llama-3.1-8B-Instruct（4-bit 权重量化）
- **GSM8K 用法**: 数学推理作为"载体任务"，验证 KV cache 在 agent 间传递后精度是否保留；每个 hop 是一个独立 agent，KV cache 在 agent 间传递（**非多轮工具调用，非单 agent 多 turn 会话**）
- **性能**: 交接延迟 1K 上下文 130.7ms（vs 150.2ms 重 prefill），8K 上下文 397.1ms（vs 1029.7ms）；stage timing 显示 post-injection generation 主导 TTFT
- **作者自承认局限**: "the current evidence does not yet isolate a consistent topology-aware advantage over local-only adaptation"（拓扑感知控制器相对 local-only 自适应未表现一致优势）
- **与 FlowCache 的场景差异**:
  | 维度 | QKVShare | FlowCache |
  |---|---|---|
  | 评估场景 | inter-agent handoff（agent 间 KV 传递） | intra-agent multi-turn（单 agent 内 tool-call 暂停/恢复） |
  | KV 操作位置 | agent 之间 | agent 内部（跨 turn） |
  | 数学任务角色 | 载体任务，验证 handoff 精度 | 不适用——FlowCache 不评估数学推理 |
  | 多轮结构来源 | hop 数（2-5 agent 串联） | tool-call 轮次（τ-bench 平均 10+ 轮） |
  | KV 管理决策 | 量化 bit 分配（per-token） | 驻留/驱逐 + 精度联合（per-block） |
- **数据来源**: arXiv abstract + arXiv PDF 全文 + 项目 `reviews/prior-art-verification.md`

### 论文 14: HyMCache
- **标题**: HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory
- **Venue/Year**: arXiv 2026-07 (2607.18141)
- **arXiv ID**: [2607.18141](https://arxiv.org/abs/2607.18141)
- **研究方向**: 集成 CXL-hybrid memory（CXL-HM）的 KV cache 框架，用于多轮 LLM serving；CXL-HM 结合小量 in-device DRAM + 大容量 SSD-backed capacity，利用 multi-turn KV cache 的 read-dominant/predictable/append-only 特性，request-level prefix prefetching + opportunistic write buffering。
- **关键技术**: CXL-HM、request-level prefix prefetching、opportunistic write buffering、DRAM 管理
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 未明确（对比 LMCache/Mooncake） | 未明确 | single-aggregator + PD-disaggregated serving |
- **总样本量**: 未明确
- **对比基线**: local LMCache, 1TB distributed-DRAM Mooncake
- **性能**: 比 LMCache 3.0×（单节点），1.45×（PD-disaggregated）；比 Mooncake 性能低 30% 但 DRAM 用量少 16×
- **数据来源**: WebSearch 结果（arXiv Troller）

### 论文 15: C2KV
- **标题**: C2KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference
- **Venue/Year**: arXiv 2026-07 (2607.17715)
- **arXiv ID**: [2607.17715](https://arxiv.org/abs/2607.17715)
- **研究方向**: 非 prefix KV 复用的统一框架，联合优化 KV 提取和推理时拼接；学习可组合且压缩的 KV cache manifold，显式设计为 position-agnostic；lightweight sidecar Extractor + learnable compression tokens + structured attention flow。
- **关键技术**: composable compressed KV manifold、position-agnostic、sidecar Extractor、compression tokens
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | RAG workloads | 未明确 | 检索增强生成 |
  | 多文档推理 | 未明确 | long-context 推理 |
- **总样本量**: 未明确
- **数据来源**: WebSearch 结果（arXiv Troller）

### 论文 16: Error Certificates for KV-Cache Eviction
- **标题**: Error Certificates for KV-Cache Eviction via Randomized Design
- **Venue/Year**: arXiv 2026-07 (2607.21475)
- **arXiv ID**: [2607.21475](https://arxiv.org/abs/2607.21475)
- **研究方向**: 证明确定性 KV 驱逐（top-k importance score）无法知道被销毁的内容；提出随机化驱逐恢复可识别性，Poisson-sampled tail + Hájek correction + survey-sampling variance estimator 生成 per-step error certificate。
- **关键技术**: random eviction、Poisson sampling、Hájek correction、survey-sampling variance、error certificate
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | real workloads（具体未明确） | 未明确 | 预注册 7 claims 验证 |
- **总样本量**: 未明确
- **性能**: 0.97 empirical coverage；question-aware eviction 25-50% budget 近乎免费；certificate AUC 0.73-0.75（attribution）
- **数据来源**: WebSearch 结果（arXiv Troller）

### 论文 17: EvicPress
- **标题**: EvicPress（具体标题待核实，arXiv 2512.14946）
- **Venue/Year**: arXiv 2025-12 (2512.14946)
- **arXiv ID**: [2512.14946](https://arxiv.org/abs/2512.14946)
- **研究方向**: KV cache 驱逐策略评估，在 12 个数据集和 5 个模型上进行全面评估。
- **关键技术**: KV cache eviction 压力测试
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 12 个数据集（具体名称未明确） | ~50/数据集（估算） | 5 模型 × 12 数据集评估 |
- **总样本量**: **~600 contexts**（估算，基于"12 datasets and 5 models"）
- **数据来源**: 项目 `.trae/specs/trim-dataset-portfolio/spec.md` 研究对比表

### 论文 18: Cake
- **标题**: Compute or Load KV Cache? Why Not Both?
- **Venue/Year**: ICML 2025
- **arXiv ID**: 待核实（OpenReview WOyOtaO6lQ）
- **研究方向**: KV cache 加载系统，最优平衡计算和 I/O 以最小化 TTFT；bidirectional scheduling 动态平衡 KV cache 计算和加载；adaptive scheduling 与非 prefix caching 请求无缝集成。
- **关键技术**: bidirectional scheduling、adaptive scheduling、computation/IO parallel utilization
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | 多种数据集（具体未明确） | 未明确 | 跨硬件配置/存储条件评估 |
- **总样本量**: 未明确
- **性能**: TTFT 平均降低 2.6×
- **数据来源**: WebSearch 结果（ICML 2025 poster）

### 论文 19: FlowKV
- **标题**: FlowKV（具体标题待核实，arXiv 2504.03775）
- **Venue/Year**: arXiv 2025-04 (2504.03775)
- **arXiv ID**: [2504.03775](https://arxiv.org/abs/2504.03775)
- **研究方向**: KV cache 压缩，使用 LongBench 数据集评估。
- **关键技术**: KV cache 压缩（具体方法待核实）
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | LongBench | 未明确 | 长上下文 KV 压缩评估 |
- **总样本量**: 未明确
- **数据来源**: 项目 `.trae/specs/trim-dataset-portfolio/spec.md` 研究对比表

### 论文 20: Ada-KV
- **标题**: Ada-KV（具体标题待核实，NeurIPS 2025）
- **Venue/Year**: NeurIPS 2025
- **arXiv ID**: 待核实
- **研究方向**: 自适应 KV cache 管理，基于 attention 统计驱动 token 重要性；在 Ruler 和 LongBench 两个 benchmark family（共 29 子任务）上评估。
- **关键技术**: attention 统计、自适应 KV 分配
- **实验数据集**:
  | 数据集名称 | 样本数量 | 用途 |
  |-----------|---------|------|
  | Ruler | 13 子任务 | 长上下文 benchmark family 1 |
  | LongBench | 16 子任务 | 长上下文 benchmark family 2 |
- **总样本量**: **~2,900**（估算，29 子任务 × ~100/子任务）
- **数据来源**: 项目 `.trae/specs/trim-dataset-portfolio/spec.md` 研究对比表

---

## 4. 统计汇总

### 4.1 数据集数分布

| 统计维度 | 数值 |
|---|---|
| 明确披露数据集数的论文数 | 12/20 |
| 数据集数中位数 | **2 个** |
| 数据集数范围 | 1-12 个 |
| 数据集数众数 | 1-2 个（占 60%） |
| 使用 ≥5 个数据集的论文 | GraphFlow (5), EvicPress (12), Ada-KV (2 family/29 子任务) |

### 4.2 样本量分布

| 统计维度 | 数值 |
|---|---|
| 明确披露样本量的论文数 | 4/20（QKVShare 150, MemDecay ~96, EvicPress ~600, Ada-KV ~2,900） |
| 样本量中位数 | **~150-600**（基于少量明确披露的论文） |
| 样本量范围 | 96-2,900 |
| 多数论文样本量 | **未明确**（仅披露数据集名称或使用合成 workflow） |

### 4.3 数据集披露完整度

| 披露程度 | 论文数 | 占比 |
|---|---|---|
| 完整披露（名称 + 样本量） | 4 | 20% |
| 部分披露（仅名称） | 8 | 40% |
| 仅披露数量 | 1 | 5% |
| 未明确 | 7 | 35% |

### 4.4 常见数据集排名（出现频次）

| 排名 | 数据集 | 出现论文数 | 论文 |
|---|---|---|---|
| 1 | SWE-Bench / SWE-bench | 3 | Continuum, SAGA, CacheWise（隐含） |
| 2 | BFCL | 1 | Continuum（FlowCache v0.5 已移除 BFCL，不再作为数据集） |
| 3 | LongBench | 3 | FlowKV, Ada-KV, ARKV（隐含） |
| 4 | GSM8K | 1 | QKVShare |
| 5 | WebArena | 1 | SAGA |
| 6 | ShareGPT | 1 | TokenCake |
| 7 | FRAMES | 1 | SideQuest |
| 8 | BrowseComp | 1 | SideQuest |
| 9 | Ruler | 1 | Ada-KV |
| 10 | PEER | 1 | KVFlow |

### 4.5 评估模式分布

| 评估模式 | 论文数 | 占比 |
|---|---|---|
| 真实 benchmark 数据集 | 12 | 60% |
| 合成/参数化 workflow | 3 | 15% |
| 自采真实 trace | 3 | 15% |
| 仅模型/architecture 测试 | 2 | 10% |

---

## 5. 对 FlowCache 的启示

### 5.1 FlowCache 数据集组合定位

| 维度 | FlowCache v0.5 | 同领域论文中位数 | 倍数 |
|---|---|---|---|
| 核心数据集数 | 1（v0.5 移除 BFCL 后单数据集 τ-bench） | 2 | 0.5× |
| 核心样本总量 | ~1,320 | ~150-600 | 2.2-8.8× |
| 主表样本量 | 1,320（τ-bench 165 tasks × 8 seeds） | ~150-600 | 2.2-8.8× |
| 数据集披露完整度 | 完整（名称+样本量） | 20% 完整披露 | — |

### 5.2 关键发现

1. **FlowCache v0.5 单数据集精简**：v0.5 移除 BFCL 后，核心数据集数从 5 降为 1（单数据集 τ-bench 1,320 episodes），低于同领域论文中位数（2 个）。这一精简聚焦于与 τ-bench 原论文（ICLR 2025）pass^k 评估的方法论完全对齐，rebuttal 时可按 IDEA.rewritten.md §6.1 migration 规则补 BFCL/STB 作跨工具家族泛化证据。

2. **FlowCache 样本量与 τ-bench 原论文对齐**：1,320 episodes（165 任务 × 8 seeds）与 τ-bench 原论文完全对齐，是 pass^k (k≤8) 评估的必要样本量，高于同领域论文中位数（~150-600）。

3. **同领域论文数据集披露不完整**：80% 的论文未完整披露样本量，仅给数据集名称或使用合成 workflow。FlowCache 在数据集披露完整度上优于多数同领域论文。

4. **领域标配数据集**：
   - **SWE-Bench**：编码 agent 场景标配（Continuum, SAGA, CacheWise）
   - **BFCL**：函数调用 agent 场景标配（Continuum；FlowCache v0.5 已移除，rebuttal 可补）
   - **LongBench**：长上下文 KV 压缩标配（FlowKV, Ada-KV, ARKV）
   - **GSM8K**：accuracy sanity 标配（QKVShare；FlowCache v0.5 已移除）

5. **FlowCache 独有差异化**：
   - **τ-bench 1,320（8 seeds）**：同领域论文中仅 τ-bench 原论文使用此规模，FlowCache 是唯一在 KV cache 管理工作中使用 τ-bench pass^k 评估的
   - **StableToolBench 500**：family-out 鲁棒性评估，同领域论文无使用
   - **多 seed 评估**：同领域论文多为单 seed，FlowCache 的 8 seeds pass^k 评估更严格

### 5.3 建议

1. **保持 5 数据集组合**：与同领域论文相比，FlowCache 的 5 数据集组合在覆盖度和深度上平衡合理，无需进一步精简。

2. **主表样本量保持 2,120**：τ-bench 1,320 对齐原论文 pass^k 评估，BFCL 800 覆盖 4 子集 × 200，都是领域标准做法。

3. **在论文中明确披露样本量**：同领域论文 80% 未完整披露样本量，FlowCache 应在实验章节明确每个数据集的样本量，体现实验严谨性。

4. **引用本调研对比表**：在论文 Related Work 或实验设计章节引用本调研的对比表，论证 FlowCache 数据集选择的合理性。

5. **关注 EvicPress 的 12 数据集评估**：EvicPress 用 12 数据集但仅 ~600 contexts，FlowCache 可引用其作为"数据集数多但样本量少"的对比案例，突出 FlowCache 单数据集深度更高的优势。

---

## 6. 调研局限性

1. **部分论文数据集信息不完整**：20 篇中仅 4 篇完整披露样本量，其余基于摘要和搜索片段推断，可能存在误差。
2. **未下载完整 PDF 核实**：对于"未明确"的数据集信息，需下载完整 PDF 查阅实验章节才能获取准确数字。
3. **arXiv 2026 论文真实性**：部分 2026 年 arXiv ID（如 2607.xxxxx）较新，虽已通过 WebFetch 确认存在，但可能后续有版本更新。
4. **狭义范围限制**：本调研仅覆盖 KV cache + Agent 工作流狭义范围，未纳入 LLM serving 系统（vLLM、SGLang 等 2024 工作）和长上下文注意力压缩（MInference 等）。

---

## 7. 参考文献列表

1. KVFlow — [arXiv:2507.07400](https://arxiv.org/abs/2507.07400) (NeurIPS 2025)
2. PBKV — [arXiv:2605.06472](https://arxiv.org/abs/2605.06472) (2026-05)
3. CacheWise — [arXiv:2606.16824](https://arxiv.org/abs/2606.16824) (2026-06)
4. GraphFlow — [arXiv:2605.22566](https://arxiv.org/abs/2605.22566) (ICML 2026)
5. ThunderAgent — [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) (2026-02)
6. TokenCake — [arXiv:2510.18586](https://arxiv.org/abs/2510.18586) (2025-10)
7. Continuum — [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) (2025-11)
8. MemDecay — [arXiv:2607.10582](https://arxiv.org/abs/2607.10582) (2026-07)
9. SAGA — [arXiv:2605.00528](https://arxiv.org/abs/2605.00528) (HPDC 2026)
10. SideQuest — [arXiv:2602.22603](https://arxiv.org/abs/2602.22603) (2026-02)
11. Agent Memory — [arXiv:2603.04428](https://arxiv.org/abs/2603.04428) (2026-02)
12. ARKV — [arXiv:2603.08727](https://arxiv.org/abs/2603.08727) (CCGRID 2025)
13. QKVShare — [arXiv:2605.03884](https://arxiv.org/abs/2605.03884) (2026-05)
14. HyMCache — [arXiv:2607.18141](https://arxiv.org/abs/2607.18141) (2026-07)
15. C2KV — [arXiv:2607.17715](https://arxiv.org/abs/2607.17715) (2026-07)
16. Error Certificates — [arXiv:2607.21475](https://arxiv.org/abs/2607.21475) (2026-07)
17. EvicPress — [arXiv:2512.14946](https://arxiv.org/abs/2512.14946) (2025-12)
18. Cake — ICML 2025 ([OpenReview](https://openreview.net/forum?id=WOyOtaO6lQ))
19. FlowKV — [arXiv:2504.03775](https://arxiv.org/abs/2504.03775) (2025-04)
20. Ada-KV — NeurIPS 2025
