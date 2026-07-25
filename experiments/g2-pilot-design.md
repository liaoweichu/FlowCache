# G2 Pilot 实验设计：Reuse Value (R) 与 Fidelity Risk (D) 相关性验证

> **项目**：FlowCache — 复用价值–保真风险解耦的前缀缓存
> **实验 ID**：G2-Pilot（Two-Axis Necessity 前置验证）
> **关联 Gate**：G2（IDEA.rewritten.md Section 7）
> **硬件**：单卡 NVIDIA RTX 4090D 24GB
> **数据集**：τ-bench（真实多轮工具 Agent benchmark，ICLR 2025）
> **创建日期**：2026-07-24
> **状态**：designed — 等待 G0 通过后执行

---

## 1. 实验目标与 G2 Gate 关系

### 1.1 核心问题

FlowCache 的中心新颖性（IDEA Section 3.3）成立条件为：

> 复用价值与保真风险在真实 workload 中存在可测的错位（misalignment）；利用这种错位进行联合分配，可在相同质量约束下获得解耦组合无法达到的延迟–容量 frontier。

G2 Pilot 的任务是**在投入完整 G2 实验之前，以最小成本验证这一错位是否可测**。具体回答：

1. 真实 Agent workload 中，block 级别的复用价值 R 与保真风险 D 之间是否存在非平凡的相关结构？
2. R 与 D 是否高度一致（→ 联合控制无净收益）或显著错位（→ 联合控制有潜力）？
3. 四类块（高复用高敏感 / 高复用低敏感 / 低复用高敏感 / 低复用低敏感）的分布是否支撑双轴控制的必要性？

### 1.2 与 G2 Gate 的关系

IDEA Section 7 G2 的通过条件包括：

- 测量 future-use/recompute value 与 D_{b,q} 的 rank correlation；
- 分析四类块的分布；
- 在相同质量风险约束下比较 reuse-only、fidelity-only、最强解耦组合和 joint policy。

G2 Pilot 是 G2 的**统计前置**：它只验证 R-D 相关性是否存在错位，**不**实现完整 joint controller 或端到端比较。若 Pilot 显示 R 与 D 高度一致（ρ > 0.7），则直接触发 G2 失败动作，无需投入完整 G2 的 controller 开发。

### 1.3 Pilot 不证明的内容

- Pilot **不**证明 joint policy 在端到端延迟上胜过解耦组合（这是完整 G2 的任务）；
- Pilot **不**训练 Reuse-Value Estimator 或 Fidelity-Risk Estimator（它只采集 oracle 标签）；
- Pilot **不**验证 conformal 风险上界的覆盖率；
- Pilot **不**使用 closed-loop（量化影响模型输出的反馈循环），只做 open-loop 干预回放。

---

## 2. 数据集选择与子集定义

### 2.1 候选评估：τ-bench vs BFCL multi-turn

| 维度 | τ-bench | BFCL V3 Multi-Turn |
|---|---|---|
| 发表状态 | ICLR 2025（正式 proceedings） | BFCL V3 blog + HF dataset |
| 任务规模 | 165 任务（retail 115 + airline 50） | 2000+ question-function-answer pairs |
| 多轮结构 | 用户模拟器驱动多轮策略对话，原生 tool-wait/resume | 多步函数调用，但 tool-wait 模式较弱 |
| 状态ful 后端 | 有（数据库状态决定任务成功） | 有（API state verification，V3 引入） |
| 任务成功判定 | 数据库状态匹配（二值，可复现） | API state 匹配（二值） |
| 确定性可重放 | 工具结果由后端数据库决定，冻结后可重放 | 函数调用结果可冻结 |
| 跨工作流共享前缀 | 同域任务共享 system prompt + policy + tool schema | 不同任务共享函数定义但 system prompt 差异较大 |
| 工具暂停/恢复模式 | 原生（每次工具调用产生 inactive KV） | 较弱（多为单步内连续调用） |
| 上下文长度 | 中等（policy 文档 + 多轮对话，~4-8K tokens） | 较短 |
| 单卡 4090D 可行性 | 80 workflow × 6K context 可行 | 可行 |

### 2.2 选择结果：τ-bench

**主 pilot 数据集**：τ-bench（原版，sierra-research/tau-bench）

**选择理由**：

1. **原生 tool-wait/resume**：τ-bench 的每次工具调用天然产生 KV 前缀暂停与恢复，正是 FlowCache 管理的 inactive exact-prefix cache 场景。
2. **跨工作流共享前缀**：同域（retail/airline）任务共享 system prompt、domain policy 文档和 tool schema，产生真实的 exact-prefix 复用机会，R 标签具有非平凡方差。
3. **确定性可重放**：工具结果由后端数据库状态决定，冻结 token IDs、工具调用序列和工具结果后可完全重放，满足 IDEA Section 6.4 的 open-loop replay 要求。
4. **二值任务成功**：数据库状态匹配给出明确的 workflow-level 质量标签，用于 D 标签的任务成功率维度。
5. **ICLR 2025 正式发表**：数据可信度高，可复现。
6. **规模适中**：165 任务中选 80 个，单卡可完成全部 BF16 轨迹录制 + Q8/Q4 干预回放。

**数据来源**：
- 代码与任务：https://github.com/sierra-research/tau-bench（MIT license）
- 论文：https://arxiv.org/abs/2406.12045（ICLR 2025）
- 任务数据位于仓库 `tau_bench/` 目录下，按域（retail/airline）组织

### 2.3 子集选择规则

**目标规模**：80 个 workflow（满足 50-100 范围）

**分层规则**（确定性、可复现）：

| 域 | 总任务数 | 选取数 | 选取方法 |
|---|---|---|---|
| retail | 115 | 55 | 系统抽样：按 task ID 升序排列，步长 = ceil(115/55) = 3，从 index 0 开始，取前 55 个 |
| airline | 50 | 25 | 系统抽样：按 task ID 升序排列，步长 = ceil(50/25) = 2，从 index 0 开始，取前 25 个 |
| **合计** | 165 | **80** | |

**选取约束**：
- 必须使用真实任务定义（JSON 中的 user instruction、policy、tool set、initial DB state、success criteria），不修改任何任务内容；
- 选取后冻结 task ID 列表并记录到 `experiments/g2-pilot-subset.json`，确保可复现；
- 若某任务在 BF16 录制阶段因模型能力不足无法完成任何工具调用（轨迹为空），则用同域下一个未选任务替换，并记录替换日志。

### 2.4 预期 block 生成量

基于 τ-bench 的多轮结构估算：

| 指标 | 估计值 | 依据 |
|---|---|---|
| 每个.workflow 的工具调用次数 | 5-15 次 | τ-bench retail/airline 任务的典型轮数 |
| 每次工具调用产生的 inactive block 数 | 10-30 个 | system prompt (~2K tokens) + 对话历史 + agent 推理，按 block_size=16 切分 |
| 80 workflow 的总 block 数（含重复） | ~2,400-3,600 | 80 × 15 × 20 |
| 去重后的 unique exact-prefix block 数 | ~300-600 | 跨工作流共享 system prompt 产生大量重复 |
| **有效 R 标签 block 数（去重后）** | **≥ 300** | 远超统计功效所需的最小样本量 |

---

## 3. R 标签采集协议

### 3.1 定义（基于 IDEA Section 2.1）

对决策时刻 t 的 inactive exact-prefix block b，R 标签采集以下分量：

| 分量 | 符号 | 定义 | 类型 |
|---|---|---|---|
| 下次访问时间 | T_b^next | 从 block b 变为 inactive 起到下一次被 exact-prefix 访问的步数（step） | 整数或 ∞ |
| 是否被复用 | reused_b | 1(T_b^next ≤ H)，H 为调度窗口 | 二值 |
| 节省的 prefill tokens | saved_tokens_b | block b 的 token 数（block_size），即若驱逐则需重 prefill 的 token 量 | 整数 |
| 节省的 prefill 时间 | saved_ms_b | 实测的 block_size tokens 在当前引擎/批次下的 prefill 时间（ms） | 浮点 |
| 共享度 | share_count_b | 在 H 窗口内访问 block b 的不同 workflow 数 | 整数 |

**标量 R 值**（用于相关性检验）：

$$
R_b = e^{-\beta T_b^{next}} \cdot \mathbf{1}(T_b^{next} \le H) \cdot \text{saved\_tokens}_b
$$

其中：
- β = 0.005 / step（对应 ~200 step 的折扣半衰期，覆盖单次 replay 的典型跨度）；
- H = 1000 step（调度窗口，覆盖全部 80 workflow 的交错执行）；
- saved_tokens_b = block_size（默认 16）。

**注意**：Spearman 相关是 rank-based，R_b 的单调变换不影响结果，因此 β 和 H 的具体值只影响绝对大小，不影响 rank ordering。

### 3.2 标签来源

**标签来源：未来真实 exact-prefix block access，不是"答案是否引用该节点"。**

- R 标签来自 open-loop replay 中观察到的真实 KV block 访问轨迹；
- 不使用"答案文本是否引用历史节点"等语义信号；
- 不使用 attention 权重推断复用。

### 3.3 采集方法：Open-Loop Replay

**阶段 A — BF16 轨迹录制**：

1. 使用 BF16 模型逐个运行 80 个 τ-bench workflow，记录完整轨迹：
   - 每一步的 token IDs（system prompt、user message、assistant response、tool call、tool result）；
   - 每次工具调用的输入参数和后端返回结果；
   - 用户模拟器的每条消息；
   - 引擎的 block 分配日志（哪些 token 范围对应哪个 block hash）。
2. 冻结所有 token IDs、工具结果和用户消息为 **replay artifact**，存入 `experiments/g2-pilot-traces/`。

**阶段 B — 交错执行模拟**：

1. 为 80 个 workflow 分配到达时间：Poisson 过程，λ = 4（平均每 0.25 step 到达一个新 workflow）；
2. 模拟交错执行（open-loop）：
   - 在每个 step，按到达时间启动新 workflow 或恢复暂停的 workflow；
   - 当 workflow 到达工具调用点时，其当前 prefix 变为 inactive block；
   - 当工具结果返回时（按录制的时间间隔），workflow 恢复，其 prefix 被再次访问；
   - 跨工作流共享的 block（相同 token-prefix identity I_b）记录为同一 block 的多次访问。
3. 对每个 inactive block b，记录：
   - 变为 inactive 的 step（决策时刻 t）；
   - 下次被访问的 step（T_b^next）；
   - 访问它的 workflow 列表（share_count_b）；
   - 该 block 的 token 数和实测 prefill 时间。

**关键约束**：
- 所有策略看到**完全相同**的未来事件（冻结 token IDs 和工具结果），满足 IDEA Section 6.4 的 open-loop 要求；
- 不允许在 replay 中让模型重新生成（不做 closed-loop），确保 R 标签是确定性的。

### 3.4 R 标签特征列表（基于 IDEA Section 4.3）

以下特征在决策时刻 t 可见，将作为未来 Reuse-Value Estimator 的输入特征（Pilot 阶段只采集，不训练）：

| 类别 | 特征 | 说明 |
|---|---|---|
| block | block_size | token 数 |
| block | ancestor_depth | 父链深度（从根到该 block 的 block 数） |
| block | recency_last_access | 上一次被访问到现在的 step 数 |
| block | historical_access_count | 截至时刻 t 的历史访问次数 |
| block | measured_prefill_ms | 实测的该 block prefill 时间 |
| workflow | completed_nodes | 当前 workflow 已完成的节点数 |
| workflow | declared_pending_successors | 已声明但未完成的后继节点数 |
| workflow | current_step_type | 当前步骤类型（user/assistant/tool_call/tool_result） |
| workflow | current_branch_id | 当前所在分支标识 |
| workflow | retry_count | 当前 workflow 的重试次数 |
| service | queue_length | 时刻 t 的请求队列长度 |
| service | active_concurrency | 当前并发执行的 workflow 数 |
| service | avg_tool_wait_ms | 最近 N 次工具调用的平均等待时间 |
| service | arrival_interval_ms | 最近 N 次请求到达间隔 |
| cache | current_tier | block 当前所在层级（GPU/CPU/evicted） |
| cache | migration_cost_ms | 从当前层级迁移到 GPU 的实测成本 |
| cache | gpu_pressure | GPU KV pool 使用率 |

**禁止使用的特征**（IDEA Section 4.3 明确禁止）：
- 未来才产生的 DAG 边或答案引用；
- 目标访问发生后的 attention；
- test workflow 的未来事件；
- 由最终标签直接计算出的"估计剩余步骤"。

---

## 4. D 标签采集协议

### 4.1 定义（基于 IDEA Section 2.2）

对 block b 和精度 q ∈ {Q8, Q4}，D 标签采集以下分量：

| 分量 | 符号 | 定义 | 级别 |
|---|---|---|---|
| token 级 logit KL | KL_{b,q} | 量化恢复后回放相同 continuation 的 token 级 logit KL 散度（与 BF16 基线对比），取 continuation 前 K 个 token 的均值 | token-level |
| top-k token 变化率 | topk_change_{b,q} | continuation 前 K 个 token 中 top-5 token 集合变化比例 | token-level |
| 工具调用函数名变化 | tool_name_changed_{b,q} | 量化后首个工具调用的函数名是否与 BF16 基线一致 | call-level |
| 工具调用参数变化 | tool_params_changed_{b,q} | 量化后首个工具调用的参数 JSON 是否与 BF16 基线一致（精确匹配） | call-level |
| 任务成功率变化 | Δsuccess_{b,q} | 1(BF16 时任务成功) − 1(量化干预 block b 后任务成功) | workflow-level |

**标量 D 值**（用于相关性检验）：

- 主指标：D_{b,q} = KL_{b,q}（token 级 logit KL 均值）
- 辅助指标：D_{b,q}^task = Δsuccess_{b,q}（workflow 级任务成功率变化）

### 4.2 采集方法：离线干预回放（基于 IDEA Section 4.4）

**协议**：

1. **基线录制**：已在 R 标签采集阶段 A 获得 BF16 完整轨迹和对应的 KV cache 快照。
2. **逐 block 干预**：对每个 unique inactive block b（去重后约 300-600 个）：
   a. 将 block b 的 KV 张量从 BF16 编码为精度 q（Q8 或 Q4），再解码回 BF16（模拟量化存储→恢复路径）；
   b. 从 block b 的末尾开始，使用**冻结的 continuation**（BF16 录制阶段的后续 token）进行单次 forward pass；
   c. 记录 forward pass 输出的 logits，与 BF16 基线的 logits 计算逐 token KL 散度；
   d. 取 continuation 前 K=64 个 token 的 KL 均值作为 KL_{b,q}。
3. **工具调用一致性检查**（对包含工具调用的 continuation）：
   a. 比较量化干预后模型"生成"的工具调用（greedy decode 的首个 tool call）与 BF16 基线的工具调用；
   b. 记录 tool_name_changed_{b,q} 和 tool_params_changed_{b,q}。
   c. **注意**：这里允许模型重新 decode（因为我们要检查量化是否改变工具调用决策），但只 decode 首个 tool call，不做完整 closed-loop。
4. **任务成功率变化**（对 block b 所属 workflow）：
   a. 使用量化干预后的工具调用结果执行后端，获取最终数据库状态；
   b. 与 BF16 基线的数据库状态和任务成功标准对比；
   c. 记录 Δsuccess_{b,q}。
   d. **注意**：若 tool_name 和 tool_params 均未变化，则 Δsuccess_{b,q} = 0（无需重新执行后端），节省计算。

### 4.3 量化精度选择

| 精度 | 说明 | 选中理由 |
|---|---|---|
| **Q8** | 8-bit 量化（INT8/FP8） | 温和精度，预期多数 block 低损伤；用于验证"低精度是否真的低风险" |
| **Q4** | 4-bit 量化（INT4/FP4） | 激进精度，预期部分 block 高损伤；用于最大化 D 标签方差，增强统计检验功效 |

**实现要求**：
- 量化/反量化使用 G0 冻结的后端实际支持的 codec（不使用离线 numpy 模拟）；
- 记录编解码时间和额外显存，用于后续 G4 可行性评估；
- 量化是 per-block 全层统一精度（不做 per-layer precision），符合 IDEA Section 1.2 的设计约束。

### 4.4 质量指标选择

| 指标 | 级别 | 用途 |
|---|---|---|
| **logit KL** | token-level | 主 D 指标，连续值，适合相关性检验 |
| top-k 变化率 | token-level | 辅助，验证 KL 是否反映实际 token 选择变化 |
| 工具调用一致性 | call-level | 诊断，区分"logit 变但 tool call 不变"的良性情况 |
| **任务成功率变化** | workflow-level | 辅助 D 指标，验证 token 级风险是否传导到任务级 |

**主相关性检验**使用 logit KL（连续、高分辨率）；**辅助检验**使用任务成功率变化（二值，低分辨率但更贴近最终目标）。

### 4.5 D 标签在线特征列表（基于 IDEA Section 4.4，Pilot 只采集不训练）

| 特征 | 说明 |
|---|---|
| block 位置（ancestor_depth） | 在父链中的深度 |
| block 长度（block_size） | token 数 |
| role/type | block 对应的 prompt 角色（system/user/assistant/tool） |
| K 张量范数（per-layer mean） | 各层 K 张量的 L2 范数均值 |
| V 张量范数（per-layer mean） | 各层 V 张量的 L2 范数均值 |
| K 张量 range（per-layer max-min） | 各层 K 张量的值范围 |
| V 张量 range（per-layer max-min） | 各层 V 张量的值范围 |
| K 张量方差（per-layer） | 各层 K 张量的方差 |
| V 张量方差（per-layer） | 各层 V 张量的方差 |
| outlier 比例 | 超过 3σ 的元素比例 |
| 跨层 max/quantile 摘要 | 跨层统计的 max 和 95/99 分位数 |

**禁止**：不在线启用 `output_attentions=True`（IDEA Section 4.4 明确指出空间复杂度 O(BHL²) 不可接受）。

---

## 5. 统计检验方法

### 5.1 主检验：Spearman 秩相关

**检验问题**：
- H₀: ρ_s = 0（R 与 D 无单调相关，即完全错位）
- H₁: ρ_s ≠ 0（R 与 D 存在单调相关）

**检验方法**：Spearman rank correlation coefficient（非参数，不要求数据线性或正态）

**输入数据**：对每个 block b，配对 (R_b, D_{b,q})，其中 q ∈ {Q8, Q4}

**输出**：
- Spearman ρ_s 及其 p 值
- 95% 置信区间（使用 Fisher z-transform 或 bootstrap）

### 5.2 辅助检验：Kendall τ

**检验方法**：Kendall rank correlation coefficient τ

**用途**：Spearman 对异常值和 ties 敏感时提供稳健性校验；τ 更适合小样本且有 ties 的情况。

### 5.3 显著性水平

- α = 0.05（双侧）
- 所有检验均报告 p 值和 95% CI

### 5.4 功效分析

**目标效应量**：ρ = 0.3（中等效应，对应 G2 判定的"错位边界"附近）

**功效目标**：1 − β ≥ 0.80

**样本量计算**（Fisher z-transform 近似，适用于 Spearman）：

| 目标 ρ | z = atanh(ρ) | 所需 N（α=0.05, power=0.80） |
|---|---|---|
| 0.3 | 0.3095 | 82 |
| 0.4 | 0.4236 | 47 |
| 0.5 | 0.5493 | 29 |
| 0.6 | 0.6931 | 19 |

**结论**：
- N = 50 blocks → 80% 功效可检测 ρ ≥ 0.38
- N = 100 blocks → 80% 功效可检测 ρ ≥ 0.27
- Pilot 预期获得 ≥ 300 unique blocks → 功效充足，可检测 ρ ≥ 0.16

### 5.5 样本量要求

| 要求 | 最小值 | Pilot 预期 |
|---|---|---|
| workflow 数 | 50 | 80 |
| unique block 数 | 50 | 300-600 |
| 统计单位 | workflow（paired bootstrap） | 80 |

**聚类处理**：同一 workflow 内的 blocks 不完全独立。采用以下方法处理：
1. **per-block 分析**：报告所有 block 的 Spearman ρ（作为主结果）；
2. **per-workflow 聚合**：对每个 workflow 取 block 的中位 R 和中位 D，再做 workflow 级 Spearman 相关（N=80，仍有 80% 功效检测 ρ ≥ 0.30）；
3. **workflow-level paired bootstrap 95% CI**（IDEA Section 6.3 要求）：以 workflow 为重采样单位，1000 次 bootstrap。

### 5.6 多重检验校正

对 Q8 和 Q4 两个精度分别检验，共 2 次检验。采用 Bonferroni 校正：校正后 α' = 0.05/2 = 0.025。报告校正前后的 p 值。

---

## 6. Go/No-Go 判定阈值

### 6.1 主判定（基于 Spearman ρ）

以下阈值基于 IDEA Section 7 G2 的通过条件和 Section 3.3 的新颖性成立条件设定：

| Spearman ρ_s (R vs D) | 判定 | 含义 | 动作 |
|---|---|---|---|
| **ρ_s < 0.4** | **GO** | R 与 D 显著错位，联合控制有潜力 | 进入完整 G2 实验（实现 joint controller，与解耦组合直接比较） |
| **ρ_s > 0.7** | **NO-GO** | R 与 D 高度一致，联合控制无净收益 | 触发 G2 失败动作，转路线 B |
| **0.4 ≤ ρ_s ≤ 0.7** | **灰区** | 部分错位，需进一步分析 | 执行 6.3 灰区分析 |

### 6.2 必须同时报告的统计量

- Spearman ρ_s 点估计
- p 值（Bonferroni 校正前后）
- 95% CI（Fisher z-transform + workflow-level bootstrap）
- Kendall τ 点估计与 p 值
- 样本量 N（block 数和 workflow 数）

### 6.3 灰区分析：四类块分布

当 0.4 ≤ ρ_s ≤ 0.7 时，执行四象限分析：

**分类阈值**（中位数分割，稳健且无需先验）：
- 高复用：R_b > median(R)
- 低复用：R_b ≤ median(R)
- 高敏感：D_b > median(D)
- 低敏感：D_b ≤ median(D)

**四象限表**：

| 象限 | R | D | 含义 | 联合控制价值 |
|---|---|---|---|---|
| HH | 高 | 高 | 高复用且高敏感 | 应保留 BF16（GPU/CPU 无损），不可量化 |
| HL | 高 | 低 | 高复用但低敏感 | **理想量化候选**：保留但可量化，释放容量 |
| LH | 低 | 高 | 低复用但高敏感 | 可驱逐（若需保留则必须 BF16） |
| LL | 低 | 低 | 低复用且低敏感 | 可驱逐或量化，价值低 |

**灰区判定规则**：

| HL + LH 占比 | 含义 | 判定 |
|---|---|---|
| > 30% | 错位显著，双象限非平凡 | **GO**（联合控制有足够操作空间） |
| 15%-30% | 部分错位 | **条件 GO**：进入完整 G2，但需在 joint controller 中验证净收益是否超过自身开销 |
| < 15% | 错位不足，大部分块集中在 HH/LL | **NO-GO**（联合控制无足够操作空间） |

### 6.4 辅助判定

- 若 Q8 的 ρ_s < 0.4 但 Q4 的 ρ_s > 0.7：说明温和量化下错位成立但激进量化下一致 → **条件 GO**，限定 joint controller 主要使用 Q8。
- 若 per-workflow 聚合后 ρ_s 与 per-block ρ_s 方向相反 → 报告 Simpson's paradox，以 per-workflow 为准（统计单位为 workflow）。

---

## 7. 实验执行步骤

### Phase 0：准备（对应项目 W1-W2，依赖 G0）

| 步骤 | 内容 | 前置条件 | 产物 |
|---|---|---|---|
| 0.1 | 冻结主模型和后端（G0 通过） | G0 passed | 模型 revision、后端 commit 记录 |
| 0.2 | 克隆 τ-bench 仓库，验证 165 任务可加载 | τ-bench MIT license | `experiments/tau-bench/` 本地副本 |
| 0.3 | 按 2.3 规则选取 80 workflow，冻结 task ID 列表 | 0.2 | `experiments/g2-pilot-subset.json` |
| 0.4 | 验证后端支持 BF16 prefix cache 和 Q8/Q4 KV codec | G0 codec spike | codec 可用性记录 |
| 0.5 | 实现 block identity 哈希和 exact-prefix 索引 | G0 exactness | block index 模块 |

### Phase 1：BF16 轨迹录制与 R 标签采集（对应 W3-W4）

| 步骤 | 内容 | 前置条件 | 产物 |
|---|---|---|---|
| 1.1 | 逐个运行 80 workflow（BF16），录制完整 token 级轨迹 | Phase 0 | `experiments/g2-pilot-traces/bf16/` |
| 1.2 | 记录每次工具调用的 block 分配日志和 prefill 时间 | 1.1 | block access log |
| 1.3 | 模拟交错到达执行（Poisson, λ=4），记录每个 block 的 T_b^next、reused、saved_tokens、share_count | 1.2 | `experiments/g2-pilot-r-labels.csv` |
| 1.4 | 计算 R_b = exp(-β T_b^next) * 1(T_b^next ≤ H) * saved_tokens_b | 1.3 | R 标签表 |
| 1.5 | R 标签质量检查：unique block 数 ≥ 50，R_b 方差 > 0 | 1.4 | R 标签统计摘要 |

### Phase 2：D 标签采集（对应 W5-W6）

| 步骤 | 内容 | 前置条件 | 产物 |
|---|---|---|---|
| 2.1 | 对每个 unique block b，以 Q8 编码→解码 KV，回放冻结 continuation（前 64 token），记录 logit KL | Phase 1 | `experiments/g2-pilot-d-labels-q8.csv` |
| 2.2 | 对每个 unique block b，以 Q4 重复 2.1 | 2.1 | `experiments/g2-pilot-d-labels-q4.csv` |
| 2.3 | 对包含工具调用的 continuation，检查 tool_name 和 tool_params 是否变化 | 2.1, 2.2 | tool consistency 表 |
| 2.4 | 对 tool call 变化的 block，执行后端获取 Δsuccess | 2.3 | task success delta 表 |
| 2.5 | D 标签质量检查：D_{b,q} 方差 > 0（无退化为常数） | 2.4 | D 标签统计摘要 |

### Phase 3：统计分析与判定（对应 W7-W8）

| 步骤 | 内容 | 前置条件 | 产物 |
|---|---|---|---|
| 3.1 | 合并 R-D 配对表，过滤缺失值 | Phase 1, 2 | `experiments/g2-pilot-rd-pairs.csv` |
| 3.2 | 计算 Spearman ρ_s（Q8 和 Q4 分别），p 值，95% CI | 3.1 | 相关系数表 |
| 3.3 | 计算 Kendall τ | 3.1 | 辅助相关系数表 |
| 3.4 | per-workflow 聚合后重算 Spearman ρ_s | 3.1 | workflow 级相关系数 |
| 3.5 | workflow-level paired bootstrap 95% CI（1000 次） | 3.1 | bootstrap CI |
| 3.6 | 四象限分析（中位数分割） | 3.1 | 四类块分布表 |
| 3.7 | 按 Section 6 判定规则给出 GO / NO-GO / 灰区 | 3.2-3.6 | `experiments/g2-pilot-verdict.md` |

### Phase 4：结果冻结与汇报（对应 W8 末）

| 步骤 | 内容 | 产物 |
|---|---|---|
| 4.1 | 冻结所有数据文件和分析脚本 | `experiments/g2-pilot-freeze/` |
| 4.2 | 生成 R-D 散点图、四象限分布图 | `figures/g2-pilot-*.png` |
| 4.3 | 撰写判定报告（含原始数字，不发明结果） | `experiments/g2-pilot-verdict.md` |
| 4.4 | 若 NO-GO，触发 IDEA Section 7 G2 失败动作 | 更新 ccfa.yaml G2 status |

---

## 8. 硬件与资源约束

### 8.1 显存预算（RTX 4090D 24GB）

| 组件 | 预估占用 | 说明 |
|---|---|---|
| 模型权重（BF16） | ~15 GB | Qwen2.5-7B-Instruct（G0 冻结） |
| KV cache pool（BF16） | ~5-6 GB | 限制并发为 4-8 workflow，每个 ~4-8K context |
| Q8/Q4 codec staging | ~1 GB | 量化/反量化临时空间 |
| active decode + activation | ~1-2 GB | 当前 forward pass 的 activation |
| 安全水位 | ~1 GB | allocator reserved + 防 OOM |
| **合计** | **~22-24 GB** | 接近 24GB 上限，需监控 |

**注意**：若显存不足，降低并发到 2-4 workflow。

### 8.2 时间预估

| 阶段 | 操作 | 预估时间 |
|---|---|---|
| Phase 1.1 | 80 workflow × BF16 录制 | ~2-3 小时（每 workflow ~1.5 min，含工具调用延迟） |
| Phase 1.3 | 交错执行模拟（纯计算，无推理） | ~10 min |
| Phase 2.1 | ~400 block × Q8 干预回放（每 block 1 次 forward） | ~1.5 小时（每 block ~15s） |
| Phase 2.2 | ~400 block × Q4 干预回放 | ~1.5 小时 |
| Phase 2.3-2.4 | 工具调用一致性 + 任务成功率 | ~1 小时（仅对 tool call 变化的 block） |
| Phase 3 | 统计分析 | ~30 min |
| **合计** | | **~7-8 小时**（可在 1-2 天内完成） |

### 8.3 CPU 与主机要求（按 IDEA Section 5.3）

必须记录并报告：
- CPU 型号 / 核心数
- 可用 RAM
- pinned-memory 上限
- PCIe 代际与链路宽度
- CUDA / driver 版本
- 是否有竞争 CPU/PCIe 负载

### 8.4 模型候选

| 模型 | 参数量 | BF16 权重 | 架构 | tool calling | 候选状态 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7.62B | ~15 GB | GQA + RoPE | 原生支持 | **主候选** |
| Llama-3.1-8B-Instruct | 8B | ~16 GB | GQA + RoPE | 原生支持 | 备选（仅当 Qwen2.5-7B 兼容性问题时切换） |

Qwen3.5/3.6 系列因 Gated DeltaNet hybrid attention 与 KV cache 工具链不兼容被排除；Gemma 4 12B 因 BF16 权重 ~24GB 超出 4090D 显存被排除。2026-07-25 用户决定从原 Qwen3-8B-Instruct 变更为 Qwen2.5-7B-Instruct，变更记录见 `experiments/experiment-designs.md` Part 0.3。

**最终模型在 G0 冻结，Pilot 使用 G0 冻结的模型。**

---

## 9. 预期产出

### 9.1 数据文件

| 文件 | 内容 |
|---|---|
| `experiments/g2-pilot-subset.json` | 80 个 workflow 的 task ID 列表 |
| `experiments/g2-pilot-traces/bf16/` | 80 个 BF16 完整轨迹 |
| `experiments/g2-pilot-r-labels.csv` | R 标签表（每 block 一行） |
| `experiments/g2-pilot-d-labels-q8.csv` | Q8 D 标签表 |
| `experiments/g2-pilot-d-labels-q4.csv` | Q4 D 标签表 |
| `experiments/g2-pilot-rd-pairs.csv` | R-D 配对表（用于统计检验） |

### 9.2 图表

| 图表 | 内容 | 文件 |
|---|---|---|
| R-D 散点图（Q8） | x=R_b, y=D_{b,Q8}，含 Spearman ρ 和 95% CI | `figures/g2-pilot-scatter-q8.png` |
| R-D 散点图（Q4） | x=R_b, y=D_{b,Q4}，含 Spearman ρ 和 95% CI | `figures/g2-pilot-scatter-q4.png` |
| 四象限分布图（Q8 和 Q4） | 中位数分割后的 HH/HL/LH/LL 块数和占比 | `figures/g2-pilot-quadrant.png` |
| R 直方图 | R_b 的分布 | `figures/g2-pilot-r-hist.png` |
| D 直方图（Q8/Q4） | D_{b,q} 的分布 | `figures/g2-pilot-d-hist.png` |

### 9.3 表格

**表 1：相关系数汇总**

| 精度 | N (blocks) | N (workflows) | Spearman ρ_s | 95% CI | p 值 | Kendall τ | p 值 |
|---|---|---|---|---|---|---|---|
| Q8 | TBD | 80 | TBD | TBD | TBD | TBD | TBD |
| Q4 | TBD | 80 | TBD | TBD | TBD | TBD | TBD |
| Q8 (workflow-level) | — | 80 | TBD | TBD | TBD | TBD | TBD |
| Q4 (workflow-level) | — | 80 | TBD | TBD | TBD | TBD | TBD |

**表 2：四象限块分布**

| 精度 | HH (高R高D) | HL (高R低D) | LH (低R高D) | LL (低R低D) | HL+LH 占比 |
|---|---|---|---|---|---|
| Q8 | TBD | TBD | TBD | TBD | TBD |
| Q4 | TBD | TBD | TBD | TBD | TBD |

**表 3：工具调用一致性**

| 精度 | 检查的 block 数 | tool_name 变化数 | tool_params 变化数 | Δsuccess ≠ 0 数 |
|---|---|---|---|---|
| Q8 | TBD | TBD | TBD | TBD |
| Q4 | TBD | TBD | TBD | TBD |

> 所有 "TBD" 在实验完成后填充真实数字，不发明任何结果。

### 9.4 判定报告

`experiments/g2-pilot-verdict.md` 包含：
- 实验执行摘要（实际使用的 workflow 数、block 数、耗时）
- 统计检验结果（相关系数、p 值、CI）
- 四象限分析
- GO / NO-GO / 灰区 判定及理由
- 若灰区：条件 GO 的具体条件
- 若 NO-GO：触发失败动作的说明

---

## 10. 失败动作

### 10.1 NO-GO 触发条件

满足以下任一条件，判定 NO-GO：

1. Q4 的 Spearman ρ_s > 0.7 且 p < 0.025（Bonferroni 校正后显著）；
2. 灰区分析中 HL + LH 占比 < 15%；
3. per-block 和 per-workflow 分析方向一致且均显示 ρ_s > 0.7。

### 10.2 NO-GO 失败动作（遵循 IDEA Section 7 G2）

按 IDEA Section 7 G2 失败动作执行：

> 若 joint policy 无净收益，reuse–fidelity 主线不成立，转路线 B；不把低相关分析单独包装成方法贡献。

**具体动作**：

1. **更新 ccfa.yaml**：将 G2 status 设为 `failed`，触发 route switch 到路线 B；
2. **停止路线 A 投入**：不再开发 joint controller、Reuse-Value Estimator 和 Fidelity-Risk Estimator 的联合版本；
3. **转路线 B**：按 IDEA Section 11 路线 B 执行 —
   - 题目：*When Does Workflow Structure Create Physical KV Reuse?*
   - 保留 cache-compatible workflow compiler、trace benchmark、oracle 和简单 cost-aware policy；
   - 方法新颖性降低，但语义扎实、单卡可完成，适合 ICWS/EDGE 的 workload/system characterization；
4. **不包装**：不将"R-D 高相关"这一发现单独包装成方法贡献；可在路线 B 论文中作为 characterization 结果报告，但不作为核心贡献；
5. **保留 pilot 数据**：G2 Pilot 的轨迹和标签数据保留，供路线 B 的 workload characterization 使用。

### 10.3 灰区条件 GO 的后续动作

若判定为灰区且 HL+LH 占比在 15%-30%：

1. 进入完整 G2 实验（W9-W10），实现 joint controller；
2. 在 G2 中必须验证：joint policy 在计入自身开销后，p95 TTFT 改善是否超过最强解耦组合；
3. 若 G2 中 joint policy 无净收益 → 仍触发 10.2 NO-GO 动作；
4. 设定更严格的净收益阈值（如 ≥ 5% p95 TTFT 改善），避免边际收益被噪音淹没。

---

## 附录 A：τ-bench 数据可用性核验

| 项目 | 状态 | 来源 |
|---|---|---|
| 代码仓库 | 公开（MIT license） | https://github.com/sierra-research/tau-bench |
| 任务数据 | 公开（仓库内 `tau_bench/` 目录） | retail 115 + airline 50 = 165 任务 |
| 论文 | ICLR 2025 正式发表 | https://arxiv.org/abs/2406.12045 |
| 历史轨迹 | 仓库内 `historical_trajectories/` | 可用于参考但不依赖（Pilot 自行录制 BF16 轨迹） |
| 后端模拟器 | 仓库内 `tau_bench/` | 确定性数据库，工具结果可冻结重放 |

**注意**：τ-bench 原仓库 README 提示任务已迁移到 τ³-bench。Pilot 使用原版 τ-bench（165 任务），因为：
1. 原版已被 ICLR 2025 发表，引用明确；
2. 任务规模适中（165 vs τ³-bench 的更大规模）；
3. 原版任务定义稳定，不受 τ³-bench 后续修改影响。

---

## 附录 B：与 IDEA 各节的对应关系

| Pilot 设计要素 | IDEA 来源 |
|---|---|
| R 标签定义（T_b^next, saved-prefill） | Section 2.1（S_{b,a}(t) 定义） |
| D 标签定义（logit KL, task success, tool call） | Section 2.2（D_{b,q} 定义） |
| R 特征列表 | Section 4.3（Reuse-Value Estimator 决策时可用特征） |
| D 特征列表 | Section 4.4（Fidelity-Risk Estimator 在线特征） |
| open-loop replay 协议 | Section 6.4（Open-loop replay 定义） |
| 数据切分（workflow 为单位） | Section 6.3（数据切分规则） |
| 统计单位（workflow-level bootstrap CI） | Section 6.3 |
| 四类块分析 | Section 7 G2（Two-Axis Necessity） |
| Go/No-Go 阈值 | Section 3.3（新颖性成立条件） + Section 7 G2 |
| 失败动作 | Section 7 G2 失败动作 + Section 11 路线 B |
| 硬件约束 | Section 5（实现与硬件可行性） |
| 模型选择 | Section 5.2（3B-8B 开源模型） |
| 显存预算 | Section 5.3（KV pool budget 定义） |
