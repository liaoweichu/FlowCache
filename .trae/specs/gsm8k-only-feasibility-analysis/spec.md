# FlowCache 仅用 GSM8K 数据集可行性分析 Spec

## Why

用户在调研 QKVShare（arXiv:2605.03884）后提出疑问：QKVShare 仅用 GSM8K 150 problems 就发表了 arXiv 论文，FlowCache 是否也能只用 GSM8K 这一个数据集完成全部实验。此问题直接影响 FlowCache 的数据集组合设计与实验体系可行性，需系统分析后给出明确结论。

## 调研事实

### QKVShare 发布期刊

- **Venue**：**arXiv 预印本**（submitted on 5 May 2026）
- **正式会议/期刊接收状态**：**未被任何会议或期刊正式接收**
- **arXiv 分类**：cs.AI, cs.MA
- **页数**：12 pages, 1 figure, 3 tables
- **License**：CC BY 4.0
- **来源**：[arXiv:2605.03884](https://arxiv.org/abs/2605.03884) abs 页面，Comments 字段无任何 venue 标注

### GSM8K 数据集总量

| 配置 | 训练集 | 测试集 | 合计 |
|---|---|---|---|
| main | 7,473 | 1,319 | **8,792** |
| socratic | 7,473 | 1,319 | **8,792** |

- **来源**：OpenAI 官方 GitHub [openai/grade-school-math](https://github.com/openai/grade-school-math) + HuggingFace [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- **问题特征**：2-8 步初等算术推理（+ - * /），单轮数学应用题
- **创建方**：OpenAI（2021，论文 *Training Verifiers to Solve Math Word Problems*）

### QKVShare 的 GSM8K 用法

QKVShare 用 GSM8K 测试**多 agent 间 KV cache 交接**的精度退化：
- 150 problems × 2-5 hops（agent 链式交接）
- 每个 hop 是一个独立 agent，KV cache 在 agent 间传递
- 数学推理只是验证 KV handoff 后精度是否保留的"载体任务"
- **不是多轮工具调用，不是单 agent 多 turn 会话**

## What Changes

本 spec 为**分析性文档**，不改动代码与实验配置。结论写入 [survey-2025-2026-kv-cache-agent-papers.md](file:///d:/00MyProject/Prefix%20Caching/reviews/survey-2025-2026-kv-cache-agent-papers.md) 与 [experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.4 数据集论证段落。

## 可行性分析

### 维度 1：GSM8K 的任务结构与 FlowCache 核心主张的匹配度

| FlowCache 核心主张 | 所需工作负载特征 | GSM8K 是否具备 | 说明 |
|---|---|:---:|---|
| **C1：trace 协议** | 多轮 agent 工具调用，产生可追踪的 block-level KV 复用结构 | ❌ | GSM8K 是单轮输入→单轮输出，无工具调用、无多轮会话、无 KV block 复用模式 |
| **C2：联合 precision+residency 控制器** | 跨 tool-call 边界的 KV cache 驻留/驱逐决策（暂停/恢复语义） | ❌ | GSM8K 无 tool-call 边界，无 pause/resume，整个 prefill 一次完成 |
| **C3：reuse-fidelity 错位实证** | 前缀复用机会（R）与保真风险（D）的错位可被利用 | ❌ | GSM8K 每题独立，无共享前缀（除 few-shot prompt），无 reuse 机会即无错位可利用 |

**结论**：GSM8K **无法支撑 FlowCache 任何一条核心主张**。

### 维度 2：QKVShare 能用 GSM8K 而 FlowCache 不能的根本原因

| 维度 | QKVShare | FlowCache |
|---|---|---|
| **评估场景** | 多 agent 间 KV handoff（agent 链） | 单 agent 多轮工具调用（tool-call 链） |
| **KV 操作位置** | agent 之间（inter-agent） | agent 内部（intra-agent，跨 turn） |
| **数学任务的角色** | 载体任务，验证 handoff 后精度 | 不适用——FlowCache 不评估数学推理本身 |
| **多轮结构来源** | hop 数（2-5 个 agent 串联） | tool-call 轮次（τ-bench 平均 10+ 轮） |
| **前缀复用模式** | agent 间共享上下文（system prompt + 历史） | tool-call 前后共享 system prompt + 历史对话前缀 |
| **所需的 KV 管理决策** | 量化 bit 分配（per-token） | 驻留/驱逐 + 精度联合（per-block） |

**关键差异**：QKVShare 的"多 hop"是 agent 间串联，每个 agent 完整消费 KV 后传给下一个；FlowCache 的"多轮"是单 agent 内 tool-call 暂停/恢复，KV 必须在 tool 执行期间驻留或被驱逐。两者所需的工作负载结构完全不同。

### 维度 3：若强行只用 GSM8K 会发生什么

| 实验 | 用 GSM8K 替代后的后果 |
|---|---|
| **Ch.1 工作负载画像** | 无 multi-turn 结构可画像；overlap/LCP/next-use/working-set 全部退化为 0 或无意义（每题独立，无共享前缀） |
| **Ch.2 R-D 错位 Pilot** | 无 reuse 机会 → R 维度恒为 0 → 无法构造 R-D 错位四象限 → G2 判定 NO-GO → 路线 A 直接终止 |
| **Ch.3 估计器有效性** | reuse 侧无数据可训练；fidelity 侧可做（量化精度 vs accuracy），但仅剩单维度，无法支撑 C2 的"联合"主张 |
| **Ch.4 端到端主结果** | 无多轮 workload → 无 cache hit → 所有策略退化为 No-Cache → FlowCache-Joint 与 baseline 无差异 |
| **Ch.5 鲁棒性** | 无 family-out 可做（GSM8K 只有一个 domain） |

**结论**：强行只用 GSM8K 会导致 FlowCache 的 C1/C2/C3 三条主张全部无法验证，实验体系崩塌。

### 维度 4：GSM8K 在 FlowCache 中的合理角色

GSM8K 当前在 FlowCache 中的角色是**且仅是** Ch.3 fidelity 质量面的 accuracy sanity（100 samples）：

- **用途**：验证 Q8/Q4 量化后模型基础推理能力未崩（accuracy 非劣界检验）
- **样本量**：100（QKVShare 用 150，FlowCache 用 100 已足够）
- **不可替代性**：低——LongBench 的 QA 子任务可覆盖类似功能，但 GSM8K 是领域最通用的 accuracy sanity benchmark，保留成本低（100 samples 录制 < 0.5 GPU 小时）

**建议**：维持 GSM8K 当前角色（Ch.3 accuracy sanity, 100 samples），不扩大其角色，不将其作为主表或画像数据集。

## Impact

- **Affected specs**：
  - [trim-dataset-portfolio/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/trim-dataset-portfolio/spec.md)：5 数据集组合保持不变，GSM8K 角色不变
  - [experiment-scope-redesign/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/experiment-scope-redesign/spec.md) §5：数据集体系保持不变
  - [reconsider-g1-sample-size/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/reconsider-g1-sample-size/spec.md)：2,120 主表封顶保持不变
- **Affected documents**：
  - [reviews/survey-2025-2026-kv-cache-agent-papers.md](file:///d:/00MyProject/Prefix%20Caching/reviews/survey-2025-2026-kv-cache-agent-papers.md)：更新 QKVShare 条目，补充 venue 状态和 GSM8K 用法对比
  - [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.4.7：补充"为何不能只用 GSM8K"的论证段落
- **Affected code**：无（本 spec 为分析性文档，不改动任何代码或配置）

## ADDED Requirements

### Requirement: QKVShare venue 信息更新

survey-2025-2026-kv-cache-agent-papers.md 中 QKVShare 条目应明确标注"arXiv 预印本，未被任何会议/期刊正式接收"，避免在论文中误称为"已发表"。

#### Scenario: 引用 QKVShare 时的 venue 声明
- **WHEN** 在 FlowCache 论文 Related Work 中引用 QKVShare
- **THEN** 必须标注"arXiv preprint, May 2026, not peer-reviewed"
- **AND** 不得称为"published in"或"accepted to"任何 venue

### Requirement: GSM8K 角色边界声明

experiment-designs.md §0.4 应明确声明 GSM8K 在 FlowCache 中的角色边界：仅作 Ch.3 fidelity 质量面的 accuracy sanity（100 samples），不作为主表、画像、鲁棒性数据集。

#### Scenario: 数据集角色查询
- **WHEN** 审稿人质疑"为何不只用 GSM8K（像 QKVShare 那样）"
- **THEN** 论文应能引用 §0.4 的论证：GSM8K 无多轮工具调用结构，无法支撑 C1/C2/C3 任一主张
- **AND** 明确区分 QKVShare 的 inter-agent handoff 场景与 FlowCache 的 intra-agent multi-turn 场景

### Requirement: GSM8K 样本量保持 100

GSM8K 在 FlowCache 中的样本量保持 100，不因 QKVShare 用 150 而调整。

#### Scenario: 样本量核对
- **WHEN** 检查 experiment-designs.md §0.4 数据集组合表
- **THEN** GSM8K 行显示 100 samples（不是 150、不是 300）
- **AND** 角色列显示"accuracy sanity（Ch.3）"

## MODIFIED Requirements

### Requirement: QKVShare 条目信息

**原**：QKVShare 条目仅含基本信息（标题、作者、arXiv ID、研究方向、150 GSM8K problems）。

**现**：补充以下信息：
1. Venue 状态：arXiv 预印本，未被任何会议/期刊正式接收
2. GSM8K 用法：150 problems × 2-5 hops，测试多 agent 间 KV handoff 精度退化
3. 与 FlowCache 的场景差异：inter-agent handoff vs intra-agent multi-turn
4. 作者自承认局限：拓扑感知控制器未显优势

## REMOVED Requirements

无。本 spec 不移除任何现有需求。
