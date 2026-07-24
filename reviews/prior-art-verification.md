# Prior-Art 核验报告：6 篇 2026 年 arXiv 论文联合 precision+residency 控制审查

**核验日期**：2026-07-24
**核验者**：文献核验子代理（ccf-integrity-auditor 模式）
**核验对象**：FlowCache 项目 IDEA.rewritten.md Section 3.1 所列 6 篇最接近公开工作
**核验目的**：判断任一 prior art 是否已实现"workflow 预测复用 + 保真感知精度 + 联合驻留控制"，评估 FlowCache 主线 A 的 novelty 风险

---

## 1. 核验方法

### 1.1 三维度判定标准

对每篇论文独立判断以下三个维度是否成立：

| 维度 | 判定标准 |
|---|---|
| **workflow next-use 预测** | 是否从工作流结构（DAG/图/程序）预测 KV block 的未来复用时机或概率 |
| **fidelity-aware precision** | 是否根据任务质量风险（而非 attention 统计或固定规则）决定 KV 量化精度 |
| **联合 precision+residency 控制** | 是否在统一框架下同时决定 KV 的精度层级和驻留位置（GPU/CPU/evict） |

### 1.2 证据来源

1. **arXiv abs 页面**：通过 WebFetch 访问每篇论文的 `https://arxiv.org/abs/<id>`，确认论文存在性、标题、作者、提交日期、venue 标注
2. **arXiv PDF 内容**：通过 WebSearch 检索获取 GraphFlow 的 PDF 正文片段，验证 "base KV + residual" 方法
3. **第三方索引**：papernotes.org ICML 2026 收录列表、themoonlight.io 文献综述作为交叉验证

### 1.3 约束

- 不发明论文内容；若 WebFetch 返回的内容不足以判断某维度，标记为"无法从摘要确认"
- arXiv ID 年份为 2026（编号 2602-2606），超出知识截止日期，所有判断均基于 WebFetch 实际返回内容
- 若论文不存在，如实记录"未确认存在"

---

## 2. 逐篇核验结果

### 2.1 PBKV（arXiv:2605.06472）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management |
| **作者** | Haoyu Zheng, Fangcheng Fu, Jia Wu, Binhang Yuan, Yongqiang Zhang, Hao Wang, Yuanyuan Zhu, Xiao Yan, Jiawei Jiang |
| **提交日期** | 2026-05-07 |
| **Venue** | arXiv 预印本（cs.LG），未见正式会议标注 |
| **内容摘要** | PBKV 预测动态工作流中未来若干步的 agent 调用，融合历史工作流与当前任务上下文。基于预测估计 cache entry 的复用潜力，将高潜力 entry 保留在 GPU 内存。在驱逐和预取阶段保守使用预测以增强鲁棒性。三个 workflow benchmark 上相对 LRU 加速 1.85×，相对 KVFlow 加速 1.26×。 |
| **workflow next-use 预测** | ✅ 是（核心贡献，融合历史与上下文预测 agent 调用） |
| **fidelity-aware precision** | ❌ 否（摘要无任何量化或质量风险建模） |
| **联合 precision+residency** | ❌ 否（仅基于复用预测做 GPU 驻留/驱逐/预取，无精度决策） |
| **对 FlowCache novelty 的影响** | 仅覆盖复用预测维度，未触及保真风险或精度-驻留联合控制 |

### 2.2 ARKV（arXiv:2603.08727）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs |
| **作者** | Jianlong Lei, Shashikant Ilager |
| **提交日期** | 2026-02-19 |
| **Venue** | arXiv 预印本；**Comments 字段标注 "Accepted in ACM/IEEE CCGRID 2025 conference"**（注意：arXiv 提交于 2026，但标注已被 2025 年会议接收，疑为 camera-ready 或扩展版） |
| **内容摘要** | ARKV 在 prefill 阶段通过 attention entropy、variance、kurtosis 估计每层的 original quantization (OQ) 比例。decoding 阶段基于 heavy-hitter 打分将 token 分配到 Original（全精度）、Quantization（低精度）、Eviction 三态。LLaMA3/Qwen3 上长上下文任务保留 ~97% 精度，KV 内存减少 4×。 |
| **workflow next-use 预测** | ❌ 否（基于 attention 统计，非工作流结构） |
| **fidelity-aware precision** | ⚠️ 部分（attention 统计驱动精度，但**非任务质量风险驱动**；使用 attention entropy/variance/kurtosis 作为 token 重要性代理，未建模 ΔQ 或任务成功率） |
| **联合 precision+residency** | ⚠️ 部分（Original/Quantization/Eviction 三态是一种精度+驻留联合，但由 attention 启发式驱动，非工作流预测或质量风险约束下的联合优化） |
| **对 FlowCache novelty 的影响** | 最接近的"精度+驻留"工作，但缺乏工作流预测和质量风险解耦；其三态分配是 attention 启发式，非 R-D 联合控制 |

### 2.3 QKVShare（arXiv:2605.03884）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs |
| **作者** | Pratik Honavar, Tejpratap GVSL |
| **提交日期** | 2026-05-05 |
| **Venue** | arXiv 预印本（cs.AI, cs.MA），12 pages, 1 figure, 3 tables |
| **内容摘要** | QKVShare 研究多 agent 边缘设备间的量化 KV cache 交接。结合 token 级 mixed-precision 分配、自包含 CacheCard 表示、HuggingFace 兼容的 cache 注入路径。在 150 个 GSM8K 问题上用 Llama-3.1-8B-Instruct 测试，自适应量化在深 hop、高预算场景下相对 uniform 量化有清晰收益；交接延迟从 1K 上下文的 130.7ms（vs 150.2ms 重 prefill）到 8K 上下文的 397.1ms（vs 1029.7ms）。 |
| **workflow next-use 预测** | ❌ 否（agent handoff，非未来访问预测） |
| **fidelity-aware precision** | ⚠️ 部分（自适应量化分配，但未建模任务质量风险；以 handoff 效率为目标） |
| **联合 precision+residency** | ❌ 否（聚焦 agent 间 handoff，非长生命周期驻留/驱逐管理） |
| **对 FlowCache novelty 的影响** | 覆盖多 agent 量化交接，但未涉及工作流预测或长期驻留控制；与 FlowCache 的 exact-prefix residency 场景正交 |

### 2.4 GraphFlow（arXiv:2605.22566）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving |
| **作者** | Ao Li, Shangpeng Yang, Fahao Chen, Tianheng Xu, Peng Li, Zhou Su |
| **提交日期** | 2026-05-21 |
| **Venue** | ✅ **ICML 2026 已确认**（arXiv Comments 字段标注 "Accepted to ICML 2026"；PDF 正文首页标注 "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"） |
| **内容摘要** | GraphFlow 提出 wGraph 统一图表示，每个节点对应一个原子操作，作为任务特定工作流动态实例化的共享基底。两个核心设计：(1) adaptive workflow generation 从 wGraph 基于任务语义动态构造工作流；(2) topology-aware state management 利用 wGraph 结构管理 KV cache。**PDF 正文确认使用 base KV + residual 重构**：`KV(P, v) = KVbase(v) + ΔKV(P, v)`，其中 `KVbase(v)` 来自独立操作描述，`ΔKV(P, v)` 为前缀特定残差。五个 benchmark 上平均提升 4.95 个百分点，内存占用减少约 4×。 |
| **workflow next-use 预测** | ⚠️ 部分（wGraph 结构用于状态管理，但非显式 next-use 时间/概率预测；更偏向 workflow 实例化与 KV 复用） |
| **fidelity-aware precision** | ❌ 否（无量化或质量风险建模） |
| **联合 precision+residency** | ❌ 否（使用 base KV + residual 重构 context-aware KV，**非 exact-prefix + 精度/驻留联合控制**） |
| **对 FlowCache novelty 的影响** | ICML 2026 正式论文，压缩了"图结构 workflow KV 管理"的宽泛新颖性；但其 base+residual 重构与 FlowCache 的 fail-closed exact-prefix + precision/residency 路径不同，未覆盖 R-D 解耦 |

### 2.5 CacheWise（arXiv:2606.16824）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents |
| **作者** | Shubham Tiwari, Tapan Chugh, Nash Rickert, Simon Peter, Ratul Mahajan, Haiying Shen |
| **提交日期** | 2026-06-15 |
| **Venue** | arXiv 预印本（cs.DC, cs.OS），未见正式会议标注 |
| **内容摘要** | CacheWise 收集真实编码 agent trace 数据集，分析显示编码 agent 会话反复复用大前缀并产生持续 KVCache 压力。提出 CacheWise KVCache 管理层：prefix-aware scheduling + reuse-aware eviction，由 tool call metadata 的轻量预测引导。在 vLLM 中实现，KVCache 驱逐减少 2-2.6×，agent 会话完成时间提升 3.5×。 |
| **workflow next-use 预测** | ✅ 是（tool call metadata 预测 reuse order） |
| **fidelity-aware precision** | ❌ 否（无量化或质量风险建模） |
| **联合 precision+residency** | ❌ 否（仅 reuse 预测 + 驱逐，无精度决策） |
| **对 FlowCache novelty 的影响** | 覆盖工具元数据驱动的复用预测，但完全未触及精度或质量风险；与 PBKV 类似仅占一维 |

### 2.6 ThunderAgent（arXiv:2602.13692）

| 字段 | 内容 |
|---|---|
| **存在性** | ✅ 已确认存在 |
| **标题** | ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System |
| **作者** | Hao Kang, Ziyang Li, Weili Xu, Xinyu Yang, Yinfang Chen, Junxiong Wang, Beidi Chen, Tushar Krishna, Chenfeng Xu, Simran Arora |
| **提交日期** | 2026-02-14（v1），2026-06-30（v3） |
| **Venue** | arXiv 预印本（cs.OS, cs.MA）；已开源 github.com/Agentic-Kinetics/ThunderAgent |
| **内容摘要** | ThunderAgent 将 agentic workflow 抽象为 LLM Programs，统一视图管理 KV cache、系统状态、外部工具资源（disk memory、network ports）。引入 program-aware scheduler 和 tool resource manager，最大化 KV cache 命中率，缓解内存不平衡，支持异步环境准备。coding/routing/scientific discovery agent 上吞吐提升 1.5-3.6×，RL rollout 1.8-3.9×，disk 内存节省 4.2×。 |
| **workflow next-use 预测** | ⚠️ 部分（program-aware 调度利用 workflow 程序结构，但非显式 next-use 概率预测；更偏向调度与资源管理） |
| **fidelity-aware precision** | ❌ 否（无量化或质量风险建模） |
| **联合 precision+residency** | ❌ 否（program-aware 调度与资源生命周期管理，无精度/驻留联合决策） |
| **对 FlowCache novelty 的影响** | 压缩"program-aware KV 调度"新颖性，但未涉及精度或保真风险；其贡献在调度与资源协调，非 cache 精度控制 |

---

## 3. 三维度汇总矩阵

| 论文 | workflow next-use 预测 | fidelity-aware precision | 联合 precision+residency | Novelty 威胁等级 |
|---|:---:|:---:|:---:|---|
| PBKV | ✅ 是 | ❌ 否 | ❌ 否 | 低（仅复用预测） |
| ARKV | ❌ 否 | ⚠️ 部分（attention 驱动，非质量风险） | ⚠️ 部分（O/Q/E 三态，非联合优化） | 中（最接近精度+驻留，但缺工作流与质量风险） |
| QKVShare | ❌ 否 | ⚠️ 部分（自适应量化，非质量风险） | ❌ 否 | 低（handoff 场景） |
| GraphFlow | ⚠️ 部分（wGraph 状态管理） | ❌ 否 | ❌ 否（base+residual，非 precision/residency） | 中（ICML 2026 正式论文，压缩图结构 workflow KV 新颖性） |
| CacheWise | ✅ 是 | ❌ 否 | ❌ 否 | 低（仅复用预测） |
| ThunderAgent | ⚠️ 部分（program-aware 调度） | ❌ 否 | ❌ 否 | 低（调度与资源管理） |

---

## 4. Novelty Delta 判定

### 4.1 判定结论：**保留（PRESERVED）**

FlowCache 的核心 novelty claim——"reuse value (R) 与 fidelity risk (D) 的错位可被联合控制利用"——在 6 篇 prior art 中**未被任何一篇同时覆盖**。

**关键证据**：

1. **无任何论文同时满足三维度**：最高覆盖数为 ARKV 满足"部分"的两维（precision + residency），但其精度由 attention 启发式驱动，非任务质量风险；且完全缺乏 workflow next-use 预测。

2. **R-D 解耦无人触及**：
   - PBKV 和 CacheWise 仅做复用预测（R 维），无保真风险（D 维）
   - ARKV 的精度由 attention entropy/variance/kurtosis 决定，是 token 重要性代理，**非任务质量风险 ΔQ**
   - QKVShare 的自适应量化以 handoff 效率为目标，非质量风险约束
   - 没有论文独立估计不同精度的质量损伤并联合优化

3. **联合控制未被实现**：
   - ARKV 的 Original/Quantization/Eviction 三态是最接近的，但由 attention 启发式驱动，非工作流预测或质量风险约束下的联合优化
   - GraphFlow 使用 base KV + residual 重构，与 FlowCache 的 exact-prefix + precision/residency 路径正交
   - ThunderAgent 的 program-aware 调度不涉及精度决策

4. **exact-prefix 语义未被挑战**：
   - GraphFlow 的 base+residual 是操作级重构，非 fail-closed exact-prefix block
   - 其他论文未明确讨论 cache compatibility 的严格语义

### 4.2 残余风险

虽然 novelty 保留，但以下风险仍需注意：

| 风险 | 说明 | 建议 |
|---|---|---|
| **机械组合风险** | PBKV（复用预测）+ ARKV（attention 精度）的机械组合可能近似 FlowCache 的部分收益 | G2 门槛必须直接对比"PBKV 式预测 + uniform 量化"和"PBKV 式预测 + ARKV 式 attention 精度"两类解耦基线 |
| **宽泛新颖性被压缩** | GraphFlow（ICML 2026）已压缩"图结构 workflow KV 管理"的宽泛 claim；PBKV/CacheWise 已覆盖工作流预测复用 | FlowCache 不可声称"首次工作流感知 KV 管理"或"首次图结构复用预测"；必须聚焦 R-D 解耦与联合控制 |
| **ARKV 三态的边界** | ARKV 的 O/Q/E 三态在形式上接近 precision+residency 联合，需在论文中明确区分 attention 启发式 vs 质量风险驱动 | 论文 Related Work 必须明确区分"attention 统计驱动精度"与"任务质量风险驱动精度" |
| **GraphFlow venue 权重** | ICML 2026 正式论文，审稿更严格，其 base+residual 方法可能在 ICML 受众中被视为 workflow KV 管理的 SOTA | FlowCache 需在实验中与 GraphFlow 在相同 workload 上对比，或明确解释为何 base+residual 与 exact-prefix residency 不可直接比较 |

---

## 5. 对 FlowCache 主线 A 的推荐

### 5.1 推荐结论：**继续主线 A，但强化对比协议**

**理由**：

1. 核心 novelty（R-D 解耦 + 联合控制）未被任何 prior art 覆盖
2. 可行性门槛 G2（Two-Axis Necessity）的设计正好检验 R-D 错位是否可被利用
3. 转路线 B（benchmark/characterization）会浪费已验证的 novelty delta

### 5.2 必须执行的强化措施

1. **扩充解耦基线**：在 E4 端到端主结果中，除原有的"reuse-only + uniform quantization"外，必须增加：
   - "PBKV 式预测 + ARKV 式 attention 精度"解耦组合
   - "PBKV 式预测 + QKVShare 式 mixed-precision"解耦组合
   
2. **明确区分 precision 驱动来源**：在 Related Work 和 Method 中明确：
   - ARKV：attention entropy/variance/kurtosis → token 重要性 → 精度（非质量风险）
   - FlowCache：离线干预回放 ΔQ → 任务质量风险 → 精度（质量风险驱动）

3. **与 GraphFlow 的边界声明**：
   - 不声称"首次图结构 workflow KV 管理"
   - 明确 GraphFlow 的 base+residual 重构与 FlowCache 的 exact-prefix + precision/residency 是不同技术路径
   - 若 GraphFlow 可在相同 workload 运行，应作为 reference baseline

4. **venue 声明清理**：
   - ARKV 标注 "CCGRID 2025"（注意年份异常，需进一步核验是否为 2026 笔误或扩展版）
   - GraphFlow 确认为 ICML 2026
   - 其余 4 篇均为 arXiv 预印本，不得在论文中称为"已发表"

---

## 6. 未确认存在的论文列表

**无**。所有 6 篇论文的 arXiv ID 均成功解析，论文均存在且可访问。

---

## 7. 核验元数据

| 项目 | 内容 |
|---|---|
| 核验工具 | WebFetch（arXiv abs 页面）、WebSearch（PDF 正文与第三方索引） |
| 核验日期 | 2026-07-24 |
| 知识截止提醒 | 核验者知识截止于 2025-08，所有 2026 年 arXiv 论文判断均基于 WebFetch 实际返回内容 |
| 局限性 | (1) 未下载完整 PDF 逐节阅读，部分维度判断基于摘要和搜索片段；(2) ICML 2026 官网下载页未提供可检索的论文列表，GraphFlow 的 ICML 收录依据 arXiv Comments 字段和 PDF 正文首页标注；(3) ARKV 的 "CCGRID 2025" 标注存在年份异常，需进一步核验 |
| 建议后续动作 | (1) 下载 PBKV、ARKV、GraphFlow 的完整 PDF，核验实验设置与 FlowCache 的可比性；(2) 核验 ARKV 的 CCGRID 收录年份；(3) 在 ICML 2026 官方 PMLR 卷宗发布后再次确认 GraphFlow 收录状态 |
