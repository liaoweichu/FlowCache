# FlowCache：面向内存受限 Agent 工作流的复用价值–保真风险解耦前缀缓存

> **英文题目**：FlowCache: Decoupling Reuse Value and Fidelity Risk for Prefix Caching in Memory-Constrained Agent Workflows  
> **项目阶段**：重构后的 idea / feasibility-gated system proposal  
> **当前决策**：Conditional Go——只有通过第 7 节的可行性门槛后，才进入完整实现  
> **主投方向**：IEEE ICWS 2027（具体 CFP 与截止时间待官方发布）  
> **备选方向**：IEEE EDGE；具备真实双节点部署和长期服务证据后再考虑 IEEE TSC  
> **硬件约束**：单卡 NVIDIA RTX 4090D 24GB  
> **结论边界**：memory-constrained GPU emulation，不直接等同于真实移动/边缘设备  
> **最后更新**：2026-07-24  

---

## 0. 一页摘要

### 0.1 问题

多工作流 Agent 服务会在推理、工具等待、分支切换和会话恢复之间反复暂停与继续。暂停后的 KV 前缀若全部留在 GPU，会挤占新请求的显存；若全部卸载或驱逐，则恢复时需要传输或重新 prefill。已有工作已经覆盖工作流感知驱逐、GPU/CPU 分层、前缀缓存和 KV 量化，但通常分别优化“未来是否复用”和“缓存应保留何种精度”。

FlowCache 研究以下问题：

> 在固定 GPU 缓存预算和任务质量非劣约束下，分别估计 exact-prefix KV block 的未来复用价值与量化敏感度，并联合决定其精度和驻留位置，能否比“预测式驱逐 + 统一量化”等解耦方案获得更好的 TTFT、尾延迟和 SLO goodput？

### 0.2 核心修正

本版本不再把“工作流语义依赖”当作“KV 可复用关系”：

- **缓存兼容性由确定性规则判断**：模型、版本、tokenizer、chat template、adapter、cache 配置、位置、父块哈希、token block 和 compute lineage 必须兼容；storage encoding 也必须可被当前运行时正确解码。
- **DAG 只预测未来访问**：工作流结构用于预测某个已经兼容的 prefix block 何时会再次被访问，不负责判断它能否复用。
- **只管理 inactive prefix cache**：不对当前 decode 正在逐 token 使用的 active KV 做任意内部删块或拼接。
- **复用价值与保真风险解耦**：未来访问概率高，不代表该块必须保持高精度；未来访问概率低，也不代表该块可以低精度近似。
- **主论文不包含异构边云路由**：4B 与 12B 模型的 KV 不兼容。边云联合路由只作为后续扩展。

### 0.3 核心假设

FlowCache 只有在以下事实同时成立时才值得继续：

1. 真实 Agent workload 中存在非平凡的 exact-prefix 再访问，且离线 oracle 明显优于 LRU/简单启发式。
2. 未来复用价值与量化敏感度并非高度一致，单一分数会产生系统性错误分配。
3. 联合控制在扣除预测、迁移和量化开销后，优于最强的同后端解耦基线。
4. 质量损失能够被预先设定的非劣区间约束，而不是只获得更高 cache hit rate。

若任一核心假设失败，应按第 11 节执行收缩或转向，而不是继续堆叠 GNN、路由器和层级。

---

## 1. 研究对象与严格语义

### 1.1 服务场景

目标场景是单个受限 GPU 上的并发 Agent 服务：

- 多个工作流共享一个推理引擎和 KV cache pool。
- 工作流可能因工具调用、外部 API、用户输入或调度而暂停。
- 恢复请求可能继续使用原有前缀，或与其他请求共享系统提示、工具规范等公共前缀。
- GPU 显存不足以同时保留全部 inactive KV，需要在 GPU、CPU 和驱逐之间选择。
- 量化缓存可能提高容量，但会引入数值误差和编解码开销。

本研究关注的是**推理引擎中可验证的物理 KV 前缀复用**，而不是语义相似缓存、RAG 文档缓存或模型输出复用。

本研究中的预测不确定性主要来自条件分支是否执行、工具时长、重试/取消、会话是否恢复以及多租户缓存竞争。对于始终恢复、纯追加的单会话，next-use 接近确定，学习预测器不应比简单规则更有优势；该场景将作为负对照。

### 1.2 缓存单元

缓存单元 $b$ 是一个固定大小的 token block，其物理 KV 表示包含该 token 范围在全部 Transformer 层上的 K/V 张量；主设计对整个 block 选择统一存储精度，per-layer precision 不在首篇论文范围内。可复用对象不是孤立 block，而是从根开始的连续父链。

block 的逻辑 token-prefix 身份至少包含：

$$
I_b=(m,r,\tau,c,a,h_{\mathrm{parent}},\mathrm{tokenIds},\mathrm{positions})
$$

其中：

- $m,r$：模型及其 revision；
- $\tau$：tokenizer 与 chat template；
- $c$：cache/rope/attention 配置；
- $a$：adapter 或 LoRA 标识；
- $h_{\mathrm{parent}}$：完整父前缀哈希；
- `tokenIds` 与 `positions`：当前完整 block 的 token 和位置。

数值血缘必须拆成“生成时使用的父数值状态”和“当前如何存储”两部分：

$$
C_b=H(C_{\mathrm{parent}},E_{\mathrm{parent}}^{used}),\qquad
E_b=(q_b,\mathrm{codecVersion},\mathrm{scaleLayout},\mathrm{checksum})
$$

其中 $C_b$ 是 immutable `computeLineage`，记录生成 block $b$ 时实际消费的父表示；$E_b$ 是可变 `storageEncoding`。重新编码一个已有 block 只改变 $E_b$，不能追溯修改已有 child 的 $C$。新 child 的 $C$ 必须包含生成时父 block 的 $C$ 与当时实际使用的 $E$。缓存表示键为 $K_b=(I_b,C_b,E_b)$。

从量化祖先恢复后生成的新 child KV 会继承 approximate/tainted compute lineage，不能与 canonical BF16 lineage 错误别名。首篇实现采用以下保守规则：

- inactive block 可以压缩存储；
- 恢复时先解码为 active BF16，不要求混合精度 attention kernel；
- 从 approximate lineage 继续生成的 child block 保持 lineage 隔离，默认只供同一 workflow 使用；
- approximate child 若要进入跨 workflow 共享池，必须完成 canonical BF16 重算或保留可审计的独立 lineage；
- 任何 lineage 缺失或不兼容都触发 fail-closed 重算。

因此，block 只有在 token-prefix 身份兼容、父链连续、lineage 可接受，且运行时能够正确解码其物理表示时才可恢复。相同文本若前序 token、位置或模型不同，不视为可复用 KV。

### 1.3 三类状态必须分开

| 状态 | 定义 | 可采取的动作 | 是否影响质量 |
|---|---|---|---|
| Active KV | 当前请求 decode 正在使用的连续前缀 | 由引擎正常维护；不做任意内部删块 | 不适用 |
| Inactive exact-prefix cache | 暂停/结束请求留下、未来可能恢复的连续 block 父链 | GPU 驻留、CPU offload、驱逐；通过门槛后可量化存储 | BF16 offload/驱逐重算无损；量化–解码可能有损 |
| Semantic history | 工作流事件、工具结果、历史答案的文本或结构 | 用于重新序列化 prompt | 不是可直接消费的 KV |

FlowCache 只优化第二类。分支汇合时只能复用共同最长前缀，不能把两个分支的 KV 直接拼接。

### 1.4 工作流结构的正确作用

工作流 DAG 描述数据依赖、控制依赖、工具等待、分支和重试。它可以帮助估计：

- 某个暂停分支是否可能很快恢复；
- 哪个已声明后继会访问相同的规范化前缀；
- 工具等待时间和 next-use distance；
- 不同工作流之间的到达竞争。

DAG 不决定缓存兼容性。兼容性始终由 tokenizer 输出和父块哈希做 fail-closed 检查。

对于静态工作流，只能使用调度时已经声明的图结构；对于动态图，只能使用时刻 $t$ 已揭示的节点和边。未来执行后才出现的边、答案引用或 attention 信息不得作为时刻 $t$ 的特征。

---

## 2. 问题形式化

在决策时刻 $t$，系统拥有 inactive block 集合 $\mathcal{B}_t$、GPU 缓存预算 $B_G$、CPU pinned-memory 预算 $B_C$，以及决策时可见的工作流状态 $X_t$。

### 2.1 复用价值

对 block $b$ 和驻留动作 $a$，以 Evict 为零基线定义 action-relative saving：

$$
S_{b,a}(t)=\mathbb{E}\left[
e^{-\beta T_b^{next}}\mathbf{1}(T_b^{next}\le H)
\cdot\left(C^{res}_{b,\mathrm{evict}}-C^{res}_{b,a}\right)
\mid X_t
\right]
$$

其中：

- $T_b^{next}$ 是下一次 exact-prefix 访问时间；
- $H$ 是调度时间窗；
- $\beta$ 控制对较晚访问的折扣；
- $C^{res}_{b,\mathrm{evict}}$ 是驱逐后重新 prefill 当前增量 block 的代价；
- $C^{res}_{b,a}$ 是从动作 $a$ 对应层级恢复该 block 的传输、解码和材料化代价。

因此 $S_{b,\mathrm{evict}}=0$，GPU-BF16、GPU-Q-storage、CPU-BF16 和 CPU-Q-storage 会得到不同价值。所有成本按父前缀长度、block 大小、batch/concurrency、PCIe 状态和引擎状态实测建模。

为避免父子 block 重复计算，block 的 saving 只作为**增量价值**，并仅在完整祖先链被选为可恢复状态时生效。端到端成本最终按最长可恢复前缀 $\ell_t$ 计算，而不是独立 block saving 的无条件求和。

### 2.2 保真风险

对精度 $q$，定义：

$$
D_{b,q}=\mathbb{E}[\Delta Q\mid b\text{ 被恢复使用},q,X_t]
$$

$\Delta Q$ 可由离线干预回放中的 logit divergence、答案/工具调用变化和任务成功率变化共同标定。$D_{b,q}$ 描述 block 被实际恢复时的条件量化风险，与未来是否访问分开建模；未被访问的量化 block 不产生任务质量损失。

多个量化 block 的误差不假设线性可加。系统在独立 calibration split 上对选中近似 lineage 集合 $A_t$ 标定总风险上界 $\widehat D^{UCB}(A_t)$，目标满足：

$$
\Pr\!\left(\Delta Q(A_t)\le\widehat D^{UCB}(A_t)\right)\ge 1-\delta
$$

可使用 split-conformal 或其他明确报告覆盖率的方法；单 block 的 $D_{b,q}$ 主要用于候选排序和解释。若独立校准集上的经验覆盖率、样本量或 exchangeability 条件不足，则该机制只能称为“经验风险预算”，不能宣称硬质量非劣保证。

### 2.3 动作空间

第一阶段仅实现无损动作：

$$
\mathcal{A}_0=\{\text{GPU-BF16},\ \text{CPU-BF16},\ \text{Evict}\}
$$

通过量化可行性门槛后，扩展为：

$$
\mathcal{A}_1=\{\text{GPU-BF16},\ \text{GPU-Q-storage},\ \text{CPU-BF16},\ \text{CPU-Q-storage},\ \text{Evict}\}
$$

`Q-storage` 表示使用后端实际支持的 Q8/Q4 等格式保存 inactive block；恢复时必须先材料化为 active BF16，并预留 staging 与 active-cache 空间。首篇实现不要求 attention kernel 直接消费混合精度 KV。

“Evict”本身不损失任务质量，只在再次访问时增加重算延迟；只有量化、截断或近似稀疏化进入质量损失项。

### 2.4 优化目标

令 $x_{b,a}\in\{0,1\}$ 表示 block $b$ 选择动作 $a$。对于未来请求序列，目标是在硬质量约束下最小化：

$$
\min_x\ \mathbb{E}\left[
\sum_t
C^{res}_t+C^{place}_t+C^{hold}_t+
C^{policy}_t+C^{SLO}_t
\right]
$$

约束包括：

$$
\sum_a x_{b,a}=1,\qquad
\sum_{b,a:\operatorname{tier}(a)=k}M_{b,a}x_{b,a}\le B_k,\qquad
u_b=\sum_{a\ne\mathrm{evict}}x_{b,a},\qquad
u_b\le u_{\mathrm{parent}(b)},\qquad u_b\le g_b
$$

$$
\widehat D^{UCB}(A_t)\le\epsilon
$$

其中 $B_k$ 是 GPU/CPU 容量，$u_b$ 表示 block 是否能够作为连续父链的一部分被恢复，$g_b\in\{0,1\}$ 表示 token identity、compute lineage、storage encoding 和运行时 decoder 是否兼容；根节点约定其虚拟 parent 可用。$\epsilon,\delta$ 是预先设定的质量非劣风险预算。项目先通过 pilot 和功效/精度分析确定所需 workflow 数量；若无法获得足够窄的置信区间或校准覆盖率，则不能提出质量非劣 claim。

实际控制器可使用 $\lambda$ 作为上述约束的对偶变量进行近似求解，但最终评价必须按硬质量约束筛选结果。控制器还需满足 staging/active-cache 空间和迁移带宽约束。本文采用可解释的滚动时域启发式或近似求解，不在没有证明的情况下声称全局最优或 Pareto optimal。

---

## 3. 与最近工作的边界

### 3.1 最接近的公开工作

| 工作 | 已覆盖内容 | FlowCache 不能再声称的内容 | 尚需验证的差异 |
|---|---|---|---|
| [vLLM APC](https://docs.vllm.ai/en/v0.14.0/design/prefix_caching/) | 完整 block 的 token/父前缀哈希与 LRU | “首次前缀复用” | 在相同 exactness 语义上增加未来价值和精度决策 |
| [KVFlow](https://arxiv.org/abs/2507.07400) | Agent Step Graph、未来感知驱逐、CPU tier 与预取 | “首次工作流感知 KV 管理” | 动态 next-use 与保真风险联合控制 |
| [PBKV](https://arxiv.org/abs/2605.06472) | GraphSAGE、workflow-history attention、连续 reuse score、多步预测、GPU/host 驱逐和预取 | “首次图+attention 预测复用”“首次连续 reuse score” | PBKV 未联合建模量化损伤；必须与其策略或忠实近似进行比较 |
| [CacheWise](https://arxiv.org/abs/2606.16824) | 工具元数据预测 reuse order、prefix-aware 调度与驱逐 | “首次预测未来 reuse order” | 更一般的 workflow family 与精度联合决策 |
| [Learned Prefix Caching](https://papers.neurips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html) | 从对话内容预测 continuation probability 并指导前缀驱逐 | “首次学习式前缀驱逐” | 工作流 next-use cost 与量化风险的联合控制 |
| [InferCept](https://proceedings.mlr.press/v235/abhyankar24a.html) | 工具/API 间断期间的 KV preserve/discard/swap、GPU memory pressure 与 swap budget | “首次 interruption-aware KV 管理” | partial-DAG-informed block next-use 与 fidelity-aware precision；不把 memory budget 本身作为差异 |
| [Continuum](https://arxiv.org/abs/2511.02230) | 工具暂停期间的 TTL、reload cost 与调度 | “首次工具等待生命周期管理” | block 级 future value 与 fidelity risk |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | program-aware pause/restore、time decay、迁移和资源生命周期 | “首次 program-aware KV scheduler” | memory-constrained exact-prefix precision/residency |
| [TokenCake](https://arxiv.org/abs/2510.18586) | function-call-aware proactive offload/predictive upload、graph+runtime priority、动态内存分区与调度/KV 协同 | “首次 graph/runtime-aware lifecycle 或 offload” | exact-prefix next-use 与 fidelity-aware precision/residency |
| [Helium](https://arxiv.org/abs/2603.16104) | workflow-as-query-plan、跨 prompt/KV/workflow 的 proactive caching 与 cache-aware scheduling | “首次 workflow-aware proactive caching/scheduling” | fail-closed exact-prefix block、next-use value 与 fidelity-aware precision/residency |
| [ARKV](https://arxiv.org/abs/2603.08727) | attention statistics 驱动 Original/Quantized/Evicted 三态 | “首次 attention-guided 混合精度 KV” | 工作流 next-use 与质量风险解耦 |
| [QKVShare](https://arxiv.org/abs/2605.03884) | multi-agent topology、attention statistics、2/4/8/16-bit CacheCard；当前 prototype 在接收端重构 FP16 | “首次 DAG+attention 混合精度” | 长生命周期驻留/驱逐而非 agent handoff；需比较其 topology-aware 分配而非只比 uniform quantization |
| [GraphFlow（ICML 2026）](https://arxiv.org/abs/2605.22566) | graph-structured workflow、operation-level base KV、prefix residual 与 hot-path 物化 | “首次图结构 workflow KV state 管理” | GraphFlow 以 base KV + residual 重构 context-aware KV；FlowCache 限定 fail-closed exact-prefix，并研究 future value/fidelity-aware residency |
| [Agent Memory](https://arxiv.org/abs/2603.04428) | 在 Apple M4 Pro 上演示持久化 Q4 KV 与直接恢复（arXiv preprint） | “首次边缘低比特 KV” | 非均匀、未来价值感知的精度与驻留 |
| [HybridFlow](https://arxiv.org/abs/2512.22137) | DAG 子任务的端云 benefit–cost 路由 | “首次工作流端云路由” | 主论文不再把路由作为贡献；扩展中只研究 cache-state-aware routing |

没有正式 venue/proceedings 证据的项目按预印本处理；arXiv 编号本身不代表已同行评审。GraphFlow 已列入 [ICML 2026 官方论文列表](https://icml.cc/Downloads/2026)；其余项目在最终写作前逐项记录并复核正式 venue、版本与代码状态。

### 3.2 可辩护但仍需检索的 gap

截至 2026-07-24 的公开核验尚未发现一篇工作同时满足以下四点：

1. 严格使用 exact-prefix block 作为缓存对象；
2. 从决策时可见的 partial workflow state 估计 next-use 和 saved-prefill value；
3. 独立估计不同 KV 精度的质量损伤；
4. 在统一容量和 SLO 目标下联合决定 precision 与 residency。

这个 gap 仍标记为 **needs-full-paper-search**。即使没有完全相同的工作，若联合控制只等价于 PBKV 和统一量化的机械组合，仍不足以构成强贡献。

### 3.3 新颖性成立条件

新颖性不由“使用 GNN”“三层缓存”或“首次”声明成立，而由以下非显然交互成立：

> 复用价值与保真风险在真实 workload 中存在可测的错位；利用这种错位进行联合分配，可在相同质量约束下获得解耦组合无法达到的延迟–容量 frontier。

若实验显示两者高度相关，或简单的“PBKV 式复用预测 + 全部 Q4”已经达到相同结果，则 FlowCache 应删除双估计器主张并转向第 11 节的保守路线。

---

## 4. 方法蓝图

### 4.1 总体架构

```text
Agent / Tool Runtime
        │ partial workflow state + canonical events
        ▼
Prefix-Stable Workflow Compiler
        │ token ids + parent block hashes + invalidation events
        ▼
Exact-Prefix Cache Index
        │ candidate inactive blocks
        ├──────────────► Reuse-Value Estimator R
        └──────────────► Fidelity-Risk Estimator D
                               │
                               ▼
                 Joint Residency Controller
                    │       │        │
                    ▼       ▼        ▼
                 GPU KV   CPU KV   Evicted
                 BF16/Q*  BF16/Q*  Recompute
                               │
                               ▼
                       Inference Engine
```

`Q*` 只在量化门槛通过后启用。

### 4.2 Prefix-Stable Workflow Compiler

编译器将工作流历史转换为不可变、仅追加的规范事件日志：

```text
system → user request → planner output → tool call → tool result
       → branch/retry event → assistant state → ...
```

职责包括：

1. 固定 role、分隔符、JSON key order、工具 schema 和 chat template。
2. 对每个模型分别 tokenize；不同模型维护完全独立的 prefix namespace。
3. 按实际引擎 block size 生成父链哈希，不硬编码“16 token block”。
4. 记录最长公共前缀、命中 block IDs、next-use、saved-prefill tokens/time。
5. 当模板、adapter、模型 revision、历史摘要或事件内容变化时显式失效。
6. 对分支只共享共同前缀；不尝试合并分支后的 KV。

该模块将“工作流语义”与“物理缓存兼容性”隔离，是所有后续实验的正确性基础。

### 4.3 Reuse-Value Estimator

#### 预测目标

对决策时刻 $t$ 的 block $b$，预测：

- 多时间窗 next-use hazard：$P(T_b^{next}\le h)$；
- 预期 saved-prefill ms；
- 预测置信度或校准区间。

标签来自未来真实 exact-prefix block access，而不是“后续答案是否引用了该节点”。

#### 决策时可用特征

- block：大小、层级、父链深度、最近访问、历史访问次数、实测 prefill cost；
- workflow：已完成节点、已声明未完成后继、步骤类型、当前分支、重试状态；
- service：队列长度、并发度、工具预计等待时间、到达间隔；
- cache：当前位置、迁移成本、GPU/CPU 压力。

禁止使用：

- 未来才产生的 DAG 边或答案引用；
- 目标访问发生后的 attention；
- test workflow 的未来事件；
- 由最终标签直接计算出的“估计剩余步骤”。

#### 模型选择顺序

1. `age + size + measured recompute cost` 启发式；
2. 校准的 survival/hazard 模型；
3. 只有前两者与 oracle 仍存在明显差距时，才使用 GNN 编码 partial DAG。

GNN 是可选实现，不是论文贡献本身。训练使用 PR-AUC、Brier/ECE、成本加权 recall 和 policy regret，而不是只报告 ROC-AUC。

### 4.4 Fidelity-Risk Estimator

#### 离线监督

从训练 workflow 中抽样 block，分别以候选精度恢复并回放相同 continuation，记录：

- token-level logit KL 或 top-k 变化；
- 最终答案 EM/F1 或任务成功状态；
- 工具调用函数名、参数和最终数据库状态是否变化；
- 编解码时间、额外显存和恢复时间。

由这些干预结果构造 $D_{b,q}$。多个量化 block 的误差可能非线性叠加，因此单 block 风险只作为局部 proxy；控制器使用风险上置信界或经组合干预标定的总风险预算，并通过 closed-loop 实验验证。没有干预回放时，不把 attention 权重直接当作质量真值。

#### 在线特征

优先使用低开销特征：

- block 位置、长度、role/type；
- K/V 范数、范围、方差、outlier 比例；
- 跨层统计的 max/quantile/直方图摘要和量化尺度统计；
- 离线标定的模型级与跨层聚合敏感度先验；
- 可选的 kernel 内流式 sketch。

这些特征可以汇总各层差异，但首篇论文的动作仍是“一个 token block 的全层 KV 使用同一存储精度”，不做 per-layer action。

不在线启用完整 `output_attentions=True`。完整 attention materialization 的空间复杂度为 $O(BHL^2)$；例如 $B=1,H=8,L=20K$ 的 BF16 dense attention 单层约为 6.4GB（约 6.0GiB）。实际是否 materialize 或触发 kernel fallback 取决于模型与后端，必须实测，因此它不能被默认当作低开销在线特征。

#### 保守回退

若风险估计不确定，则选择 BF16 或无损驱逐重算。FlowCache 不以质量下降换取“0% OOM”后再忽略任务结果。

### 4.5 Joint Residency Controller

对每个 block/动作计算近似的 action-relative 净收益：

$$
V_{b,a}=S_{b,a}-C^{place}_{b,a}-C^{hold}_{b,a}
$$

其中 $S_{b,a}$ 已包含相对 Evict 的恢复、传输和解码差异，避免重复扣费；$C^{place}$ 是进入该层级的编码/迁移成本，$C^{hold}$ 是占用稀缺内存的机会成本。质量风险不直接按单 block 线性相加，而由 $\widehat D^{UCB}(A_t)$ 进入第 2.4 节的硬约束。

该效用只用于候选排序，最终选择按完整父链和最长可恢复前缀重新计算。控制器每次在请求到达、暂停、恢复、完成或显存压力变化时运行，使用滚动时域 greedy/index policy 或多选择背包近似：

- 优先保留单位 GPU-byte 能节省最多 prefill 且质量风险可控的父链；
- 显式计入 CPU→GPU 恢复、材料化/staging、编解码和控制器开销；
- 在 parent block 不可用时，后继 block 不计为立即可复用；
- approximate lineage 不得与 canonical lineage 错误合并；
- 保留安全水位，避免 allocator reserved memory 导致临界 OOM；
- 预测器失效时回退到 size-aware LRU/GDSF，而不是随机决策。

论文必须同时报告控制器的收益和其自身成本。

### 4.6 可选扩展：Model-Scoped Shadow Frontiers

边云路由不进入主论文。若后续具备两个真实服务端点，可为每个模型维护独立前缀状态：

$$
S_m=(model,\ revision,\ tokenizer,\ template,\ adapter,\ prefixHash,\ p_m,\ tier)
$$

其中 $p_m$ 是模型 $m$ 当前可恢复的最长连续前缀。路由回模型 $m$ 时，只 prefill 文本历史中 $p_m$ 之后的 gap。不同模型之间只传规范事件、prompt 和工具结果，不传 KV。

该扩展需联合优化 route、model-local retention、网络、质量和 catch-up cost，属于高风险 TSC 扩展，不得用单卡轮流加载两个模型替代真实端云证据。

---

## 5. 实现与硬件可行性

### 5.1 后端原则

主实验必须满足：

- 所有主基线使用相同模型、引擎、dtype、请求顺序和缓存预算；
- exact-prefix 命中由引擎真实 block index 产生；
- 忠实复现不了的论文基线必须标为 `*-inspired heuristic`，不能沿用原论文名称；
- 外部引擎结果只能作为独立 reference，不与本引擎的 kernel/scheduler 延迟直接混比；
- W1–W2 先完成最小 Q-storage codec/lineage spike；正式 cache manager 仍按 GPU BF16 ↔ pinned CPU BF16 ↔ evicted 的无损路径实现，再在该正确性基线上接入量化。

候选主后端应支持可观测的 block prefix cache、CPU offload 或可扩展 cache manager。具体选择在 G0 后冻结；不能先写死 `transformers + bitsandbytes`，再假设它同时提供高并发服务、APC 和 INT4 KV。

### 5.2 模型选择

主模型使用 **Qwen2.5-7B-Instruct**（BF16 权重约 15GB，GQA + RoPE 架构，原生 tool calling 支持）。2026-07-25 用户决定从原 Qwen3-8B-Instruct 变更为 Qwen2.5-7B-Instruct，变更记录见 `experiments/experiment-designs.md` Part 0.3。

以下模型被排除：

| 模型 | 排除原因 |
|---|---|
| Qwen3.5 系列（含 9B） | Gated DeltaNet hybrid attention（3 GDN : 1 full attention），75% 层为线性注意力无传统 KV cache，block identity、父链哈希、prefix cache 和 KV quantization 工具链不兼容 |
| Qwen3.6 系列（27B / 35B-A3B） | 同上 Gated DeltaNet 架构不兼容；27B BF16 约 56GB，远超 4090D 24GB |
| Gemma 4 12B | BF16 权重约 24GB，4090D 无法为 KV cache 保留空间；5:1 sliding/global attention 后端支持不成熟 |
| Gemma 4 E4B | 尺寸可行，但 5:1 sliding/global attention 的 KV cache 操作后端支持需 G0 实测；若 Qwen2.5-7B 通过 G0 则不再切换 |
| Llama-3.1-8B-Instruct | 备选（同为 GQA + RoPE），仅当 Qwen2.5-7B 在后端实测中出现兼容性问题时切换 |

模型选择在 G0 阶段冻结。若 Qwen2.5-7B 通过 G0 exactness/loadability 测试，则不再切换到其他模型。

### 5.3 显存预算设置

不再限制“整个 GPU 只能使用 3/4/6/8GB”，而是：

1. 先测量模型加载后的真实 `allocated/reserved`；
2. 预留引擎、临时 activation 和安全水位；
3. 将可控变量定义为 **KV pool budget**；
4. 以 trace 峰值 working set 的比例设置预算，例如 10%/25%/50%/100%；
5. 通过真实并发、长会话和工具暂停产生压力，不为制造 OOM 而使用不自然上下文。

必须报告 GPU allocated、reserved、KV pool、CPU pinned memory 和模型权重，而不是只报告理论位宽。

CPU offload 结果同时依赖主机。正式实验前必须冻结并报告 CPU 型号/核心数、可用 RAM、pinned-memory 上限、PCIe 代际与实际链路宽度、NUMA 位置、CUDA/driver 版本，以及是否存在竞争 CPU/PCIe 负载。若比较不同机器，PCIe 与主机差异必须单独校准。

### 5.4 4090D 能支持的结论

可以支持：

- 同 GPU、同后端下的缓存策略相对比较；
- 固定 trace 上的 prediction、oracle 与 policy regret；
- 经过实测的 PCIe/offload/codec 开销；
- memory-constrained GPU service 的容量与尾延迟结论。

不能单独支持：

- 真实手机或嵌入式设备的能耗、带宽和热约束；
- 云端排队、跨节点计算重叠和真实网络尾延迟；
- 大规模集群调度；
- TSC 级生产部署结论。

---

## 6. 工作负载与无泄漏协议

### 6.1 最小 workload 组合

v0.3 将 workload 体系按实验章节分层，主表严格限定两个工具 workload，其余按角色分配到 Ch.3 质量面、Ch.5 压力面或辅助附注，避免主表规模失控。

**主表 workload（Ch.1/2/3/4 共用，单数据集）**：

| Workload | 样本 | 角色 | 可证明内容 | 不能外推的内容 |
|---|---|---|---|---|
| τ-bench | 1,320 | 主表 workload（原生多轮工具 Agent，165 tasks × 8 seeds） | 工具调用、状态变化、失败和任务成功率；pass^k (k≤8) 一致性；retail↔airline 两域弱 family-out 对照 | 超大规模生产流量；跨工具家族泛化（rebuttal 可补 BFCL/STB） |

**Ch.3 fidelity 质量面**：复用主表 τ-bench 1,320 episodes trace，量化质量（logit KL、top-k change、任务成功率变化）直接在主 workload 上测量。

**辅助角色（不计入核心样本总量）**：

| Workload | 角色 | 用途 |
|---|---|---|
| BurstGPT 窗口 | 到达证据 | Ch.4 到达结构 replay 参数，不产生 workflow 样本 |

**已删除的 workload**：合成可控 DAG（受用户禁令约束）、MuSiQue、2WikiMultihopQA、CATraces（可得性 TBD）、Mooncake（窗口 TBD）、BFCL v3（方法论与 τ-bench LLM user simulator 不对等）、LongBench（fidelity 质量面改在主 workload 验证）、GSM8K（同 LongBench）、StableToolBench（family-out 轴证据力弱）、SWE 轨迹（与 C1–C3 主线关联弱）、Toolathlon（同 SWE）、LMSYS-Chat-1M（负对照附注非核心证据）。其中 BFCL/LongBench/GSM8K/STB/SWE/Toolathlon/LMSYS-Chat-1M 在 v0.4 单数据集精简中删除，rebuttal 时可补。

14 周版本按 v0.4 的 **1 核心 + 1 辅助数据集体系**执行：核心 1 个数据集（τ-bench 1,320 episodes，165 tasks × 8 seeds，核心样本总量 1,320）计入样本总量；辅助 1 个（BurstGPT 窗口）不计入。τ-bench 与原论文（ICLR 2025）pass^k (k≤8) 方法论完全对齐；retail↔airline 两域对照作为弱 family-out 证据。第二模型、BFCL/STB 等跨工具家族泛化、更多 coding workload 移至后续版本或 rebuttal 补做。ShareGPT mirror 不作为主 workload；HotpotQA 不再作为可选短链对照。

### 6.2 Cache-Compatible 序列化

每类 workload 必须给出显式编译规则：

- 哪些事件进入 prompt；
- 事件顺序和格式是否稳定；
- 分支是否重新序列化历史；
- 哪些 block 能形成共同前缀；
- workflow resume 是否保持完全相同的 prefix；
- 历史摘要、截断或模板变化如何触发 invalidation。

如果某 workload 的 exact-prefix overlap 很低，则应报告这一事实，而不是用语义引用制造伪命中。

### 6.3 数据切分

- 以完整 workflow 为最小单位，所有 step 必须在同一 split；
- 进一步按模板、graph signature、源文档、实体或底层问题分组；
- 对跨 split 的 token prefix 做近重复检查；
- validation 选择阈值、模型和预算；test 只运行冻结配置；
- 若使用交叉验证，只在 train 内执行，不与最终 test 混用；
- 统计单位为 workflow，采用 paired workflow-level bootstrap 95% CI。

### 6.4 Open-loop 与 Closed-loop

- **Open-loop replay**：冻结 token IDs、DAG snapshot、工具结果和到达时间，所有策略看到完全相同的未来事件，用于系统性能和 policy 比较。
- **Closed-loop live run**：允许缓存量化影响模型输出和后续工具调用，只用于最终质量、任务成功率和失败分析。

两种模式不可混在同一主表中。Open-loop 不能证明量化后的真实任务质量，Closed-loop 也不能保证每个策略经历完全相同的请求序列。

---

## 7. 可行性门槛

以下数值仅是项目内部 Go/No-Go 阈值，不应作为普遍科学定律写进论文结论。

### G0：Exactness 与 Loadability

- 同模型 BF16 缓存恢复与完整重算在预设数值容差内一致；
- block identity、父链和 invalidation 无错误关联；
- 冻结并记录模型、tokenizer、chat template、后端和 Hugging Face config 的确切 revision/commit；
- 在一个 block 上跑通 inactive Q-storage → active BF16 的编码、解码、staging 与 precision-lineage 隔离；
- 证明目标后端能够拦截和恢复所需 KV，或在 W1–W2 内完成最小扩展点；
- 测量模型加载后的 allocated/reserved 峰值；
- 无法容纳模型、active cache、staging 与运行安全水位的预算直接删除。

**失败动作**：允许切换一次受支持的模型/后端；仍失败则路线 A No-Go，转路线 B，不进入预测器开发。

### G1：Opportunity

- 统计 exact-prefix overlap、next-use distance、可节省 prefill time 和 KV/总显存占比；
- 比较 LRU、size-aware heuristic 与离线 Belady/cost-aware oracle；
- 至少保证 PBKV 或 KVFlow 中一个 closest baseline 能在公平协议下忠实运行；若只能实现 inspired variant，必须先解决可比性；
- 内部参考：oracle 相对最佳简单策略应存在约 10% 的 miss-cost 或 p95 TTFT 改进空间。

**运行方式**：数据来源改为复用 Ch.1 画像数据（τ-bench 1,320 episodes 同 trace），不再独立运行 Gate 实验；判定逻辑与阈值不变。`design_doc` 指向 `experiments/experiment-designs.md#ch1`。

**失败动作**：转向“何时工作流结构产生物理 KV 复用”的 benchmark/characterization 论文。

### G2：Two-Axis Necessity

- 测量 future-use/recompute value 与 $D_{b,q}$ 的 rank correlation，但不把低相关本身当作充分证据；
- 分析四类块：高复用高敏感、高复用低敏感、低复用高敏感、低复用低敏感；
- 在相同质量风险约束下，直接比较 reuse-only、fidelity-only、最强解耦组合和 joint policy；
- joint policy 必须在计入自身开销后形成解耦组合达不到的延迟–容量改进。

**运行方式**：R–D 错位用 Ch.2 Pilot 数据（τ-bench 80 子集，Spearman ρ + 四象限）判定；"joint > 解耦组合"用 Ch.4 主表最终判定。判定逻辑与阈值不变。`design_doc` 指向 `experiments/g2-pilot-design.md` 与 `experiments/experiment-designs.md#ch4`。

**失败动作**：若 joint policy 无净收益，reuse–fidelity 主线不成立，转路线 B；不把低相关分析单独包装成方法贡献。

### G3：Lossless Residency

先只实现 GPU BF16、CPU BF16、evict：

- 恢复和迁移开销必须小于所节省的 prefill；
- 内部参考：固定质量下 p95 TTFT 改善约 15%，吞吐下降不超过约 5%；
- 控制器必须优于 size-aware LRU/GDSF。

**运行方式**：分两时点判定。W8 冒烟前置：主 cell × 4 个无损对照（No-Cache、APC-LRU、GDSF、Reuse-Only）× 约 100 workflow 子集，本质是主表的 pilot run，防止无损驻留不成立时白做量化。最终阈值判定用 Ch.4 主表的无损对照行结果做最终确认。`design_doc` 指向 `experiments/experiment-designs.md#ch4`。

**失败动作**：路线 A No-Go；可保留实现作为工程基线，但不以无损 residency 单独投稿该主张。

### G4：Quantization

- 真实后端支持目标模型的 KV quantization 与恢复；
- active runtime 统一材料化为 BF16，staging 峰值和 tainted lineage 均被正确追踪；
- 量化/反量化不破坏 end-to-end latency；
- pilot 后预注册绝对质量非劣界、$\delta$ 和样本量；95% CI 必须窄到足以检验该界；
- 至少在一个真实工具 workload 上验证，而不只看 logit KL。

**运行方式**：数据来源改为复用 Ch.3 fidelity 侧数据（τ-bench 1,320 episodes，复用 Ch.1 trace）。判定逻辑与阈值不变。`design_doc` 指向 `experiments/experiment-designs.md#ch3`。

**失败动作**：在 G0 已允许的一次模型/后端切换后仍失败，则路线 A No-Go 并转路线 B；不能删除量化后继续使用“reuse–fidelity”主标题投稿。

### G5（Learning）：已删除

G5（Learning）整节已删除。GNN 不启用是设计选择而非 gate 失败；论文主张"简单可解释 controller 足够"，这本身可写为发现（见 §7.2 of `experiment-scope-redesign/spec.md`）。§4.3 的预测器选择顺序（heuristic → survival → GNN 仅在前两者与 oracle 仍有明显差距时）保留作为方法描述，但不再作为 gate。简单 controller 成为默认而非"gate 失败"。

---

## 8. 正式实验计划

v0.3 将原 E1–E7 七个独立实验章节合并为 Ch.1–Ch.5 五章，遵循"一次运行，多处消费"原则：Gate 判定不再独立运行，全部复用本章 trace/数据。各章保留原 v0.2 实验的指标与成功标准，仅按 v0.3 的对照/cell/数据集规模重写。

### Ch.1：工作负载画像（合并原 E1，承载 G1 判定）

**问题**：真实 workload 中是否存在值得管理的 exact-prefix locality？

**数据来源**：τ-bench 1,320 episodes（165 tasks × 8 seeds，与 Ch.4 主表共用 trace，W3–W4 一次录制）。

**报告指标**（保留原 E1）：

- workflow 长度、深度、宽度、分支率和工具等待；
- exact-prefix overlap、LCP tokens、next-use distance；
- block working-set size、KV/总显存占比；
- 6 个 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）+ ≥1 个 closest baseline（PBKV 或 KVFlow）的 headroom；headroom = Oracle-Cost − max(LRU, GDSF, SizeCost, APC-LRU)。

**Gate 复用**：G1 判定（oracle headroom ≥ 10% + closest baseline 可比性）直接复用本章画像数据，不独立运行。

这是中心 claim 的前提，不应被放在附录。

### Ch.2：R–D 错位 Pilot（原 G2-Pilot，承载 G2 存在性判定）

**问题**：复用价值与保真风险是否存在系统性错位？

**数据来源**：τ-bench 80 workflow 子集（从 1,320 episodes 中抽样）。

**方法**：

- 计算 future-use/recompute value 与 $D_{b,q}$ 的 Spearman ρ；
- 四象限分析：高复用高敏感、高复用低敏感、低复用高敏感、低复用低敏感；
- 不把低相关本身当作充分证据，需配合 Ch.4 主表的 joint vs Decoupled-Best 净收益判定。

**Gate 复用**：G2 的 R–D 错位存在性判定复用本章 Pilot 数据；"joint > 解耦组合"最终判定复用 Ch.4 主表。`design_doc` 指向 `experiments/g2-pilot-design.md`（保持有效，不重构）。

### Ch.3：估计器有效性（合并原 E2 + E3，承载 G4 判定）

**问题**：reuse 侧与 fidelity 侧估计器是否各自有效？

**reuse 侧**（2 变体，复用 Ch.1 trace）：

- heuristic：age/LRU、size/recompute-cost；
- survival/hazard 模型。

**删除 GNN 变体**：GNN 不进入主实验（见 §7 G5 删除说明）；§4.3 的预测器选择顺序保留作为方法描述。

**fidelity 侧**（2 变体，数据：τ-bench 1,320 episodes，复用 Ch.1 trace）：

- uniform precision；
- norm/range proxy（FlowCache fidelity estimator 即 norm/range proxy 的校准版本，合并描述，不再单列为独立变体）。

**删除**：静态 layer/position rule 变体、独立的 FlowCache fidelity estimator 变体。

**指标**（保留原 E2/E3）：

- reuse 侧：PR-AUC、Brier、ECE；byte-和 recompute-cost-weighted recall；Precision@budget；policy regret 与 saved-prefill ms；推理开销。
- fidelity 侧：logit KL/top-k change；QA EM/F1；工具调用正确率、最终状态和任务成功率；风险校准；codec latency 与实际容量收益。

准确率提升若不能转换为系统收益，不构成贡献。

**Gate 复用**：G4 判定（量化非劣 + 端到端不破坏延迟）复用本章 fidelity 侧数据，不独立运行。

### Ch.4：端到端主结果（合并原 E4 + E5 核心，承载 G2/G3 最终判定）

**问题**：联合控制在扣除预测、迁移和量化开销后，是否优于最强的同后端解耦基线？

**对照（10 个，原 13 删 LFU、LRU-K/2Q、Uniform-Q4）**：

| # | 对照 | 说明 |
|---|---|---|
| 1 | No-Cache | cold recompute 下界 |
| 2 | APC-LRU | 同引擎实际 APC |
| 3 | GDSF | 强启发式代表（合并 LFU、LRU-K/2Q 的角色） |
| 4 | KVFlow† 或 PBKV† | ≥1 个可公平运行的 closest baseline；另一项若不兼容才使用明确标注的 inspired variant |
| 5 | Uniform-Q8 | 统一 Q8（Q4 仅在 Ch.3 fidelity 侧，不进主表） |
| 6 | Reuse-Only | 核心变体 1：复用价值驱动驻留 + 统一精度 |
| 7 | Fidelity-Only | 核心变体 2：保真风险驱动精度 + 强启发式驻留 |
| 8 | Decoupled-Best | 核心变体 3：最强"reuse policy + uniform quantization"解耦组合 |
| 9 | FlowCache-Joint | 核心变体 4：待验联合 policy |
| 10 | Oracle-Cost | 离线上界 |

**cell（6 个，原 18 删并发 4 档与 100% 预算档）**：

| cell | 预算 | 并发 | workload | seeds |
|---|---|---|---|---|
| 主-1 | 25% | 8 | τ-bench | 3 |
| 主-2 | 50% | 8 | τ-bench | 1 |
| 边界-1 | 10% | 16 | τ-bench | 1 |

注：v0.4 单数据集精简后，原 BFCL cell（主-2/主-4/边界-2）删除。具体 cell 设计在 Ch.4 pilot 后冻结，总 replay 数不超过 v0.3 的 100 replay 上限。

运行量：10 对照 × 3 cell，其中主-1 用 3 seeds = 10×(2×1 + 1×3) = 50 replay（v0.4 单数据集精简，低于 v0.3 的 100 replay 上限）。

**设计消融并入主表**（核心 4 变体 + 2 设计消融同表，同一 cell 仅切开关）：

- 无 parent-closure：后继 block 在父 block 不可用时仍计为可复用；
- 无 CPU tier：仅 GPU + evict，禁用 CPU offload。

**开销透明账**（H2D/D2H、codec、controller、queueing 时间）并入主表列，不单设章节。

**主指标**（保留原 E4）：

- TTFT、JCT p50/p95/p99；
- throughput、SLO goodput、最大 admitted concurrency；
- token/block/byte cache hit；
- saved-prefill tokens/time；
- GPU allocated/reserved、CPU pinned bytes；
- H2D/D2H、codec、controller 和 queueing 开销；
- 任务成功率及质量非劣区间。

主结论必须来自相同引擎、模型、dtype、预算和请求顺序。

**关键问题**（保留原 E5）：不是"删掉模块性能是否下降"，而是联合建模是否解决了单一分数的错误分配。

**降级为附录表**：原 E5 的其余消融轴（无 partial DAG、无成本校准、静态阈值 vs 动态预算、不同 controller 更新频率）降级为附录表或移除，不再作为主表消融主体。

**Gate 复用**：G2 的"joint > 解耦组合"最终判定与 G3 的阈值判定（p95 TTFT 改善 ~15%、吞吐 ≥ −5%、优于 size-aware LRU/GDSF）均复用本章主表的无损对照行与核心变体行，不独立运行。

### Ch.5：鲁棒性与失败分析（合并原 E6 + E7）

**问题**：方法在不同 family、到达扰动和 branch 噪声下是否稳健？负结果能否转化为发现？

**1 个扰动轴**（v0.4 单数据集精简，原 3 轴缩减）：

- 到达扰动：BurstGPT 窗口 replay；
- retail↔airline 两域对照作为 τ-bench 内部弱 family-out 证据（附注，非独立轴）。

**降级为附录**：原 E6 的 CPU 带宽竞争、predictor calibration drift、不同上下文长度、GPU budget 突变降级为附录表。v0.4 单数据集精简后无 SWE/Toolathlon 余量，rebuttal 时若需跨工具家族证据可补 BFCL/STB。

**失败模式**（从 Ch.4 负结果 cell 提取，原 E7 的独立章合并）：

- exact-prefix overlap 过低；
- 量化敏感块误判；
- 高频 GPU↔CPU 抖动；
- parent block 缺失导致后继缓存不可用；
- 模型/template/adapter 变化引发大面积 invalidation；
- controller 开销超过 saved-prefill；
- overload 下的 graceful degradation、拒绝率和 OOM。

**扩展标注**：第二模型和额外 dataset-out 仍标注为"仅资源允许时扩展"，不是 14 周主证据包的前置条件。

---

## 9. 贡献声明

只有相应实验成立后，论文才能提出以下贡献：

1. **Cache-compatible agent workflow abstraction 与 trace protocol**  
   将语义工作流依赖与 exact-prefix KV identity 分离，提供决策时刻、父链、next-use 和 saved-prefill 的可重放记录方式。

2. **复用价值–保真风险解耦的联合 residency controller**  
   分别估计未来 exact-prefix 使用价值和精度损伤，在统一内存、迁移、质量和 SLO 目标下分配缓存。

3. **关于 reuse–fidelity 错位的系统性实证**  
   证明何时联合控制优于预测式驱逐与统一量化的简单组合，并给出无收益区域和失败条件。

贡献类型以**系统设计 + 实证发现**为主，trace/protocol 为辅助评价贡献。不主张：

- 首个工作流感知 KV 管理；
- 首个 GNN/attention reuse predictor；
- 首个三层缓存；
- 首个边缘低比特 KV；
- 首个 cache-aware routing；
- 未经证明的 Pareto optimal。

---

## 10. 与课题组工作的关系

以下关系沿用原稿提供的信息，尚未独立核验论文、代码和已实现模块，因此只作为内部项目定位：

| 课题组工作（原稿提供） | 可复用资产 | 新项目必须保持的边界 |
|---|---|---|
| SCD | KV 表示、缓存转换或兼容性经验 | 不把跨模型蒸馏自动视为 exact-prefix 复用 |
| HeraSys | 工作流图与节点复用经验 | 图关系只预测 next-use，不决定 KV compatibility |
| CONCORD | 异步通信与端云系统经验 | 主论文不把模拟 RTT 当真实端云部署 |
| WPDS | 请求/步骤调度框架 | FlowCache 的中心对象是 inactive prefix residency |
| ConCise | 上下文压缩经验 | prompt 压缩与 KV cache precision 必须分开评价 |

在复用这些资产前，需要核验接口、许可、模型兼容性和可复现实验。本次重写前，项目目录中只看到 IDEA 文档，未发现可用于核验原稿“已实现”表述的代码。

---

## 11. 风险、收缩与转向

| 风险 | 类型 | 触发条件 | 应对 |
|---|---|---|---|
| PBKV/KVFlow/QKVShare 已覆盖大部分机制 | likely-pivot / needs-search | full-paper 对比后只剩组件拼装 | 将贡献收缩为 exact-prefix protocol + reuse–fidelity interaction，或转 benchmark |
| 语义 workflow 几乎不产生非平凡 exact-prefix locality | requires-new-result | G1 oracle headroom 很小 | 做“工作流结构何时转化为物理缓存复用”的 characterization |
| 简单 heuristic 接近 oracle | design-fixable | 简单 cost-aware policy 与 oracle 差距小 | 删除 GNN，强调简单 cost-aware policy（G5 不再作为 gate） |
| 复用价值与保真风险高度一致 | likely-pivot | G2 不支持双轴或 joint 无净收益 | 路线 A No-Go，转路线 B；不以纯 residency 延续原主张 |
| 目标后端不支持所需 KV quantization/offload | feasibility | G0/G4 失败 | 允许一次模型/后端切换；仍失败则转路线 B |
| 量化收益被 codec/PCIe 抵消 | requires-new-result | 端到端无正收益 | 路线 A No-Go；无损 GPU/CPU/evict 仅保留为路线 B 的工程基线 |
| 多 block 量化误差非线性叠加 | design-fixable | 单 block 风险无法预测 closed-loop 质量 | 组合干预标定、风险上界和全任务质量约束 |
| 单卡模拟无法支撑 edge claim | venue-mismatch | 无真实设备/第二端点 | 使用 memory-constrained GPU 表述；转 ICWS/EDGE，TSC 延后 |
| 工作流 split 泄漏模板或底层问题 | evidence-fixable | test 与 train 共享结构/前缀 | group split、prefix dedup、冻结 test |
| 所有 closest baseline 均无法忠实比较 | evidence-fixable / likely-pivot | 代码/引擎不兼容 | 路线 A No-Go；inspired variant 只能作为次要补充，不能替代全部 close baselines |

### 路线 A：推荐主线

**Exact-prefix reuse value + fidelity risk + joint precision/residency**。  
单卡条件下可尝试，创新与实现风险均较高；`requires_gates = [G0, G1, G2, G3, G4]`，任一关键门槛失败即停止并转路线 B（G5 已删除，不再作为 gate，详见 §7）。

### 路线 B：保守回退

**When Does Workflow Structure Create Physical KV Reuse?**  
构建 cache-compatible workflow compiler、trace benchmark、oracle 与简单 cost-aware policy。方法新颖性较低，但语义扎实、单卡可完成，更适合 ICWS/EDGE 的 workload/system characterization。

### 路线 C：高风险扩展

**Model-Scoped Shadow Frontiers for Heterogeneous Edge–Cloud Agents**。  
每个模型维护独立最长前缀，联合优化路由与 model-local cache retention。需要真实双端点、网络 trace 和质量模型，作为 TSC 扩展，不与路线 A 同时塞入首篇会议论文。

---

## 12. 14 周执行计划

v0.4 按"一次运行，多处消费"原则重排周次：单数据集 τ-bench 1,320 episodes，Gate 判定复用正式实验数据，Pilot 提前到 W7–W8，W8 增加 G3 冒烟，W9 末用实测效应量标定 Ch.4 样本量，W10–W11 主表，W12 鲁棒性（仅到达扰动）。

| 周次 | 目标 | Gate / 产物 |
|---|---|---|
| W1–W2 | 冻结模型/后端/主机；Q-storage codec spike | G0 |
| W3–W4 | τ-bench 1,320 episodes 轨迹录制 | 可重放 trace |
| W6–W7 | Ch.1 画像 + G1 判定（复用 trace） | E1 画像 |
| W7–W8 | Ch.2 Pilot + Ch.3 reuse 侧（并行）→ G2 存在性判定 | G2 Pilot |
| W8 | G3 冒烟（主 cell × 4 无损对照 × 100 子集） | G3 冒烟 |
| W9 | Ch.3 fidelity 侧 + G4 判定 | G4 |
| W9 末 | 用实测效应量标定 Ch.4 样本量（封顶 1,320） | 样本量冻结 |
| W10–W11 | Ch.4 主表（G2/G3 最终确认复用主表） | E4 主表 |
| W12 | Ch.5 鲁棒性（到达扰动 replay，无 STB 录制） | E5 鲁棒性 |
| W13 | 复跑冻结 / W14 写作 | 冻结结果 |

如果 G0、G1、G2、G3 或 G4 任一关键门槛失败，路线 A 停止并转路线 B；不能删除量化后继续沿用 reuse–fidelity 主标题（G5 已删除，不再作为 gate，详见 §7）。GNN 和第二模型只有在主结果已稳定且仍有时间时加入。

---

## 13. Venue 与论文叙事

### 13.1 ICWS

[ICWS 2026 官方 CFP](https://services.conferences.computer.org/2026/icws/icws-call-for-papers/)包含 LLM service acceleration、LLM agent service benchmark、cloud–edge–device 协同和 workflow optimization，说明服务/QoS framing 具有较好匹配度。建议论文叙事为：

> A QoS-aware agent-service runtime that exploits workflow-scoped exact-prefix locality and decouples reuse value from fidelity risk under memory constraints.

截至 2026-07-24，本次在 IEEE SERVICES 官方域名的检索未找到 ICWS 2027 CFP，因此截止时间标记为 **TBD**，不得写成已确认的 2027-01。

### 13.2 IEEE EDGE

若最终贡献主要体现为缓存内核、显存层级和 edge resource scheduling，而服务抽象较弱，[IEEE EDGE 2026 官方 CFP](https://services.conferences.computer.org/2026/edge/edge-call-for-papers/)显示 EDGE 可能比 ICWS 更自然；2027 主题和时间仍需等待官方页面。

### 13.3 IEEE TSC

[IEEE TSC scope](https://www.computer.org/digital-library/journals/sc/cfp-services-computing)主要覆盖服务体系结构、业务流程集成、服务性能以及运行管理。本题能否匹配，取决于是否以服务系统和运行管理为核心，而不是只呈现缓存内核。为使期刊扩展具备足够新增性和部署说服力，建议补充：

- 两个真实服务端点；
- 长期、多租户 workload；
- SLA、成本、能耗或资源利用；
- 带宽、RTT、jitter、loss 和云端 queueing；
- 故障、恢复和 operational robustness；
- Shadow Frontiers 或其他具有新增机制的扩展。

---

## 14. 证据状态与待补输入

### 已知

- 用户提供的目标是 ICWS 2027 / TSC，硬件为单张 RTX 4090D。
- 本次重写前，项目目录中只看到 IDEA 文档，未发现可核验实现状态的代码。
- 最近公开工作已经显著压缩“预测式工作流 KV 管理”和“混合精度 KV”的宽泛新颖性。
- 主模型已选定 Qwen2.5-7B-Instruct（GQA + RoPE，BF16 ~15GB），2026-07-25 从原 Qwen3-8B 变更。Qwen3.5/3.6 因 Gated DeltaNet 架构不兼容 KV cache 工具链被排除，Gemma 4 12B 因显存不足被排除。

### 仍需确认

- 最终推理后端及其 cache manager 扩展点；
- 可忠实复现的 PBKV/KVFlow/QKVShare 基线；
- 是否有可使用的课题组代码、真实 Agent trace 或第二台主机；
- KV quantization 对候选模型和 sliding/global attention 的支持；
- ICWS 2027 官方主题、截稿和格式；
- 14 周是否为真实时间约束；
- 任务质量非劣区间应由具体 workload 预先设定。

### 写作就绪条件

满足以下条件后再进入论文写作：

1. G0–G3 通过；
2. G1 证明真实决策空间存在；
3. 最强 close baseline 已能公平运行或清楚解释不兼容；
4. G2 和 G4 均通过，joint policy 在质量约束下胜过最强解耦组合；
5. 主 claim 有一张相同后端的 Pareto 主图和 workflow-level 置信区间；
6. 所有贡献均对应真实结果，不使用预期数字填充结论。

---

## 15. 主要公开来源

- vLLM Automatic Prefix Caching design: <https://docs.vllm.ai/en/v0.14.0/design/prefix_caching/>
- Hugging Face KV cache strategies: <https://huggingface.co/docs/transformers/kv_cache>
- Gemma 4 official overview: <https://ai.google.dev/gemma/docs/core>
- PBKV: <https://arxiv.org/abs/2605.06472>
- KVFlow: <https://arxiv.org/abs/2507.07400>
- CacheWise: <https://arxiv.org/abs/2606.16824>
- Learned Prefix Caching: <https://papers.neurips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html>
- InferCept: <https://proceedings.mlr.press/v235/abhyankar24a.html>
- Continuum: <https://arxiv.org/abs/2511.02230>
- ThunderAgent: <https://arxiv.org/abs/2602.13692>
- TokenCake: <https://arxiv.org/abs/2510.18586>
- Helium: <https://arxiv.org/abs/2603.16104>
- ARKV: <https://arxiv.org/abs/2603.08727>
- QKVShare: <https://arxiv.org/abs/2605.03884>
- GraphFlow: <https://arxiv.org/abs/2605.22566>
- ICML 2026 official paper list: <https://icml.cc/Downloads/2026>
- Agent Memory: <https://arxiv.org/abs/2603.04428>
- HybridFlow: <https://arxiv.org/abs/2512.22137>
- τ-bench: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html>
- NVIDIA GeForce RTX 4090 D: <https://www.nvidia.cn/geforce/graphics-cards/40-series/rtx-4090-d/>
- IEEE ICWS 2026 CFP: <https://services.conferences.computer.org/2026/icws/icws-call-for-papers/>
- IEEE EDGE 2026 CFP: <https://services.conferences.computer.org/2026/edge/edge-call-for-papers/>
- IEEE TSC scope: <https://www.computer.org/digital-library/journals/sc/cfp-services-computing>
