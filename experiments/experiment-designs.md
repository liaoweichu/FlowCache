# FlowCache 实验设计总册：Gates G0–G5 与 Experiments E1–E7

> **项目**：FlowCache — 复用价值–保真风险解耦的前缀缓存（memory-constrained agent workflows）
> **文档类型**：实验设计（design），**非结果报告**；所有结果数字一律为 TBD，实验完成后另行填充，不发明任何结果
> **覆盖范围**：门槛 G0、G1、G3、G4、G5 + 实验 E1、E2、E3、E4、E5、E6、E7（共 12 章）
> **G2 说明**：G2（Two-Axis Necessity）已有独立 Pilot 设计文档 `experiments/g2-pilot-design.md`，本册仅交叉引用，不重复撰写
> **上游文档**：`IDEA.rewritten.md`（§5 硬件、§6 工作负载、§7 门槛、§8 实验、§11 风险、§12 计划）、`execution-plan.md`、`ccfa.yaml`
> **创建日期**：2026-07-25
> **修订记录**：
> - v0.1（2026-07-25）：初版
> - **v0.2（2026-07-25）：数据集体系重构**——应用户要求：①全册禁止自建/合成数据集（移除合成 prompt 集与合成可控 DAG，G0 改用真实数据天然结构）；②数据集种类与样本总量大幅扩充（~880 → ~8,800 workflow 级样本）；③引入 BFCL、SWE 轨迹、Toolathlon、CATraces、LongBench、GSM8K、BurstGPT、Mooncake，选择依据对齐同类论文（详见 0.4）；**v0.5（2026-07-26）注：BFCL 全面移除，不再作为数据集**
> - **v0.3（2026-07-25）：实验体系精简**——按 `.trae/specs/experiment-scope-redesign/spec.md`（状态：approved-pending-implementation）落地：13 章 → 5 章 + 2 个小判定（G0、G3 冒烟）；核心数据集 12+ → 7；workflow 样本 ~8,800 → ~4,120；E4 replay ~702 → ~100。G1/G2/G4 不再独立运行，复用正式实验数据判定。**v0.3 重构分批进行**：本次仅对齐 G1/Ch.1 与 E1 章节顶部；其余章节（G3/G4/E2–E7）保留 v0.2 形式，待对应周次前再重构
> - **v0.3.1（2026-07-25）：τ-bench seeds 数升级**——基于 τ-bench 原论文（arXiv 2406.12045, ICLR 2025）调研，pass^k 指标用 k∈{1,2,4,8}，3 seeds 只能算 pass^3 不足以区分 consistency。τ-bench 主表样本量从 495（165 × 3 seeds）升级为 **1,320（165 × 8 seeds）**，与原论文 pass^8 完全对齐。本次仅同步更新 G1.2/G1.3/G1.11.1 与 spec v0.3 §5/§8；**G3/G4/E1/E4/E5/E6 等章节中的 "495 episodes" 与 "3 seeds" 引用暂保留 v0.2 形式**，待对应章节 v0.3 重构时统一更新为 1,320 / 8 seeds
> - **v0.5（2026-07-26）：BFCL 全面移除**——应用户决定（IDEA.rewritten.md v0.4 已将 BFCL 标记为"已删除 + rebuttal 可补"，config.yaml 已是 τ-bench only），BFCL 不再作为数据集：①删除 G1.2/G1.11.1/G3.2/G5.2/E1.2/E2.2/E4.2/E5.2/E6.2 等数据集表中的 BFCL 行；②删除 G1.2 中 BFCL 集成方式与 8 decode seeds 依据整段；③"τ-bench + BFCL" → "τ-bench"，"1,320 + 6,400 = 7,720" → "1,320"，"τ-bench / BFCL / STB" → "τ-bench / STB"；④保留 v0.2/v0.3/v0.4 历史变更记录中 BFCL 的引用作为决策记录；⑤rebuttal 时若需补做 BFCL，按 IDEA.rewritten.md v0.4 的 migration 规则执行
> - **v0.6（2026-07-27）：G3 协议修复**——现有 G3/G3′/G3″ 结果统一标记为 `PROTOCOL-INCOMPLETE`，不得触发路线切换；增加 G3-P0 单 cell 修复实验，区分 offline replay wall time、modeled cache delay、offered load 与 closed-loop TTFT/throughput；always-migrate 降级为诊断 baseline，并要求所有主比较使用相同 GPU+CPU 容量。
> - **v0.7（2026-07-28）：G3-P1 选择性迁移**——实现 bounded likelihood proxy admission、因果 share、跨驱逐频率、独立 always-migrate 消融与 O(1)/O(log N) victim；增加 task-grouped validation/test、18 组 admission 参数扫描和近似 always/never 的 fail-closed 门槛。
> - **v0.8（2026-07-28）：G3-P1 因果 GPU admission**——新增
>   Oracle-Cost 启发但不读取未来的 incoming-vs-incumbent GPU
>   admission/bypass；增加 selective-migrate-only 消融、future-index
>   物理隔离、前缀因果不变性测试、TinyLFU-style 因果 doorkeeper，以及
>   54 组 validation-only 联合参数扫描。
> **状态**：designed/in-progress — G1/Ch.1 已按 v0.3 对齐；v0.5 已全面移除 BFCL；G3 已按 v0.8 接入因果 GPU admission + 选择性迁移 validation/test 协议；其余章节按 v0.2 形式待 v0.3 重构

---

## 目录

- **Part 0：公共实验协议**（所有章节共享，各章只引用不重复）
  - 0.1 文档目的与使用方式
  - 0.2 硬件与显存预算
  - 0.3 模型与后端冻结（含 2026-07-25 模型变更记录）
  - 0.4 数据集组合与统一样本量（v0.2 重构）
  - 0.5 Cache-Compatible 序列化规则
  - 0.6 数据切分与无泄漏协议
  - 0.7 Open-loop 与 Closed-loop 使用规则
  - 0.8 统计约定与功效分析基准
  - 0.9 通用指标词汇表
  - 0.10 通用 baseline 名录
- **Part 1：Gate 验证设计**
  - G0：Exactness 与 Loadability
  - G1：Opportunity（缓存机会）
  - G3：Lossless Residency（无损驻留）
  - G4：Quantization（量化）
  - G5：Learning（学习式预测）
- **Part 2：正式实验设计**
  - E1：缓存机会与工作负载画像
  - E2：复用价值预测
  - E3：保真风险估计
  - E4：端到端主结果
  - E5：机制消融
  - E6：泛化与鲁棒性
  - E7：失败与开销

---

# Part 0：公共实验协议

## 0.1 文档目的与使用方式

本册为 FlowCache 全部可行性门槛（Gate）与正式实验（Experiment）提供**可执行的、预注册式**的设计说明。每章固定包含以下要素：

1. 实验目标与 Gate 关系
2. 数据集与子集定义（来源、选取规则、冻结文件）
3. 样本数与功效分析
4. Baseline / 对照变体
5. 测试指标（主指标 / 辅助指标 / 开销指标）
6. 运行协议（open-loop / closed-loop、到达模型、预算档位）
7. 统计检验方法
8. 判定阈值（Go/No-Go 或成功标准）
9. 执行步骤
10. 硬件与时间预算
11. 预期产物（含结果表格模板，TBD 占位）
12. 失败动作
13. 与 IDEA 各节的对应关系

**使用规则**：

- 各章引用的公共设置（硬件、模型、数据集、统计约定、baseline 名录）只在 Part 0 定义一次；章节内不再重复定义，仅给出本章特有的增量设置。
- 任何与本册冲突的临时决定，必须回写本册对应章节并注明日期，不允许只改代码不改设计。
- 所有"预注册"字段（阈值、样本量、非劣界）在对应实验的 pilot 完成后、正式运行前冻结，冻结值填入本章 TBD 处并标注冻结日期。
- **数据集禁令（2026-07-25 用户规定）**：全册任何实验的数据集不得为自建/合成数据。真实数据上的 replay 时扰动（删边、时间缩放、burst 注入等）属于实验操作而非数据集，允许使用但必须在章节中显式标注。

## 0.2 硬件与显存预算

### 0.2.1 硬件配置（所有实验统一）

| 组件 | 配置 | 说明 |
|---|---|---|
| GPU | 单卡 NVIDIA RTX 4090D 24GB | 全部实验的唯一 GPU |
| 结论边界 | memory-constrained GPU emulation | 不外推真实移动/边缘设备能耗、云端排队、跨节点重叠、大规模集群调度 |
| CPU / RAM / PCIe | TBD（正式实验前冻结并报告） | 按 IDEA §5.3：CPU 型号/核心数、可用 RAM、pinned-memory 上限、PCIe 代际与链路宽度、NUMA 位置、CUDA/driver 版本、竞争负载 |

CPU offload 结果依赖主机。若比较不同机器，PCIe 与主机差异必须单独校准。

### 0.2.2 显存预算（按 Qwen2.5-7B-Instruct 重算）

| 组件 | 预估占用 | 说明 |
|---|---|---|
| 模型权重（BF16） | ~15 GB | Qwen2.5-7B-Instruct（7.62B 参数 × 2 bytes） |
| KV cache pool（可控变量） | ~5–7 GB | 以 trace 峰值 working set 的 10%/25%/50%/100% 设档（见 0.4.5） |
| Q8/Q4 codec staging | ~1 GB | 量化/反量化临时空间（仅 G4 及之后启用） |
| active decode + activation | ~1–2 GB | 当前 forward pass 的 activation |
| 安全水位 | ~1 GB | allocator reserved + 防 OOM |
| **合计** | **~22–24 GB** | 接近 24GB 上限，全程监控 allocated/reserved |

**预算定义规则**（IDEA §5.3）：先测模型加载后真实 `allocated/reserved` → 预留引擎/activation/安全水位 → 可控变量定义为 **KV pool budget** → 按 trace 峰值 working set 比例设档 → 压力来自真实并发、长会话和工具暂停，不使用不自然上下文制造 OOM。

**必须报告**：GPU allocated、GPU reserved、KV pool 字节数、CPU pinned bytes、模型权重字节数，不接受只报告理论位宽。

## 0.3 模型与后端冻结

### 0.3.1 主模型（2026-07-25 变更）

| 项 | 值 |
|---|---|
| **主模型** | **Qwen2.5-7B-Instruct**（BF16 权重 ~15GB，GQA + RoPE，原生 tool calling） |
| 备选模型 | Llama-3.1-8B-Instruct（仅当 Qwen2.5-7B 在后端实测中出现兼容性问题时切换，仅允许切换一次） |
| 冻结时点 | G0 阶段冻结确切 revision/commit；冻结后不再切换（除非触发 G0/G4 失败动作允许的一次切换） |

> **变更记录（2026-07-25）**：主模型由 IDEA §5.2 原定的 Qwen3-8B-Instruct 变更为 **Qwen2.5-7B-Instruct**（用户决定）。本册全文统一使用 Qwen2.5-7B-Instruct。`IDEA.rewritten.md`、`experiments/g2-pilot-design.md` 与 `experiments/e1/config.yaml` 均已同步回改为 Qwen2.5-7B-Instruct。排除规则不变：Qwen3.5/3.6（Gated DeltaNet hybrid attention 与 KV cache 工具链不兼容）、Gemma 4 12B（BF16 ~24GB 超显存）均被排除。

### 0.3.2 候选排除表（沿用 IDEA §5.2）

| 模型 | 排除原因 |
|---|---|
| Qwen3.5 系列 | Gated DeltaNet hybrid attention，75% 层无传统 KV cache，block identity/父链哈希/prefix cache/KV 量化工具链不兼容 |
| Qwen3.6 系列 | 同上；27B BF16 ~56GB 远超 4090D |
| Gemma 4 12B | BF16 ~24GB，无法为 KV cache 保留空间；sliding/global attention 后端支持不成熟 |
| Gemma 4 E4B | 尺寸可行，但 sliding/global attention 的 KV cache 操作后端支持需实测；Qwen2.5-7B 通过 G0 则不切换 |
| Llama-3.1-8B-Instruct | 保留为唯一备选（GQA + RoPE） |

### 0.3.3 后端原则（IDEA §5.1）

- 所有主基线使用相同模型、引擎、dtype、请求顺序和缓存预算；
- exact-prefix 命中由引擎真实 block index 产生；
- 忠实复现不了的论文基线必须标为 `*-inspired heuristic`，不能沿用原论文名称；
- 外部引擎结果只作独立 reference，不与本引擎延迟直接混比；
- 后端必须支持：可观测的 block prefix cache、CPU offload 或可扩展 cache manager；具体选择在 G0 后冻结；
- W1–W2 先完成最小 Q-storage codec/lineage spike；正式 cache manager 先按 GPU BF16 ↔ pinned CPU BF16 ↔ evicted 无损路径实现，再在该正确性基线上接入量化。

**冻结清单**（G0 产物）：模型 revision、tokenizer、chat template、后端及 commit、Hugging Face config、CUDA/driver 版本。

## 0.4 数据集组合与统一样本量（v0.2 重构）

### 0.4.1 四层数据集组合

**Tier 1：主工具 Agent benchmark（需 rollout，模拟环境确定性可重放）**

| 数据集 | 样本数 | 选取规则 | 角色 | 同类论文依据 | 冻结状态 |
|---|---|---|---|---|---|
| **τ-bench** | 165 全量 × 3 user-simulator seeds = **495 episodes** | 全量不抽样；3 个用户模拟器种子（pass^k 惯例） | 主 A：多轮工具、任务成功率、同域共享前缀 | 项目原有（ICLR 2025） | W2 冻结 seeds |
| **StableToolBench** | **500**（I1/I2/I3 分层） | 按测试子集分层抽样；须通过 0.4.3 核验 | 主 C：工具 family 泛化、API 故障注入事件 | 项目原有（80 → 500） | W2 冻结 |

Rollout 预算：495×~1.5 min + 500×~1.5 min ≈ **~25 GPU 小时**（W3–W5 录制窗，见 execution-plan.md 修订）。

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn（800 episodes）行已删除——BFCL 不再作为数据集（IDEA.rewritten.md v0.4 + config.yaml 单数据集 τ-bench only）。rebuttal 时若需补做，按 IDEA.rewritten.md v0.4 的 migration 规则执行。

**Tier 2：真实 agent 轨迹（零 rollout，直接 replay）**

| 数据集 | 样本数 | 选取规则 | 角色 | 许可 |
|---|---|---|---|---|
| **CATraces**（CacheWise 开源真实 Claude Code 会话，~10M tokens） | 全部可用会话（~100–200，TBD 实测） | 全量 | 真实 coding-agent 长会话；与最近工作（CacheWise, arXiv:2606.16824）直接可比 | 见其仓库 |

> **v0.4 注**：SWE-rebench-openhands-trajectories 与 Toolathlon-Trajectories 已从核心数据集组合中移除（详见 §0.4.4 排除清单与 trim-dataset-portfolio spec）。Branch 结构角色由 τ-bench 内部 replay 扰动覆盖（删边/错标后继），tool-wait 角色由 τ-bench/STB 工具调用实测支撑。

**Tier 3：质量与长上下文**

| 数据集 | 样本数 | 选取规则 | 角色 | 同类论文依据 |
|---|---|---|---|---|
| **LongBench** | **1,000**（21 子任务分层） | 按子任务分层 | KV 量化质量主战场（长上下文） | ARKV（arXiv:2603.08727） |
| **GSM8K** | **100**（v0.4 从 300 降） | 随机抽样，种子冻结 | 量化 accuracy sanity（Ch.3） | QKVShare / GraphFlow / ARKV |
| **MuSiQue** | **300**（原 100） | 按底层问题分组去重后抽样 | 多跳 QA 质量 sanity | 项目原有 |
| **2WikiMultihopQA** | **300**（原 100） | 同上 | 多跳 QA 质量 sanity | 项目原有 |

**Tier 4：服务 trace 与负对照**

| 数据集 | 样本数 | 选取规则 | 角色 |
|---|---|---|---|
| **LMSYS-Chat-1M** | **2,000 会话**（原 200） | 顺序式多轮会话抽样（仅追加历史） | 多轮聊天**负对照**：高 exact-prefix reuse 但无需复杂预测 |
| **BurstGPT** | **2,000 会话窗口** | 按真实 session ID + 时间戳抽取连续窗口 | 真实到达 / 并发 / 会话结构——**替代纯 Poisson 假设的主证据** |
| **Mooncake trace** | 抽样分析（窗口 TBD） | 按前缀块 hash 抽样 | 大规模前缀块共享结构（E1 画像 + E6 压力面） |

**总量**：**~7,000 workflow 级样本**（495 + 500 + ~150 + 1,000 + 300 + 300 + 300 + 2,000 + 2,000；CATraces 以实测为准）。v0.4 移除 SWE 轨迹 500 + Toolathlon 500 后较 v0.2 的 ~8,800 减少约 1,000；v0.5 移除 BFCL 800 后再减 800。

> **v0.4 核心样本总量**：**~2,920**（τ-bench 1,320 + LongBench 1,000 + GSM8K 100 + StableToolBench 500；trim-dataset-portfolio spec）。核心数据集数封顶 4（v0.5 移除 BFCL 后）；辅助数据集（CATraces、MuSiQue、2WikiMultihopQA、LMSYS、BurstGPT、Mooncake）不计入核心总量。
>
> **v0.5（2026-07-26）注**：BFCL 800 已从核心样本总量中移除（1,320 + 1,000 + 100 + 500 = 2,920，非 v0.4 的 3,720）。核心数据集数从 5 降为 4。

**规模与边界说明**：

- 主系统结论（E4）由 Tier 1 承担；Tier 2 提供真实结构证据，Tier 3 承担质量面，Tier 4 承担到达结构与负对照——分层引用，不把任何单一数据集当万能证据。
- 第二模型、跨模型泛化移至后续版本（IDEA §6.1 约束不变）。
- ShareGPT mirror 不作为主 workload；HotpotQA 只作可选短链对照。

### 0.4.2 τ-bench 全量与多 seed 规则（与原 80 子集的关系）

- 主实验使用**全部 165 任务 × 3 个 user-simulator seeds = 495 episodes**，不再限于 80 子集。
- `experiments/g2-pilot-subset.json`（80 任务）**保留有效**：仅作为 G2-Pilot 的既有设计输入（g2-pilot-design.md 的功效分析独立成立）；主实验可在全量 trace 中定位该子集做交叉核对。
- 3 seeds 用途：用户模拟器随机性的方差估计与任务成功率重复测量（τ-bench 官方 pass^k 惯例）。
- 不修改任何任务内容（user instruction、policy、tool set、initial DB state、success criteria）；空轨迹替换规则沿用（同域替换 + 记录替换日志）。
- 预期去重后 unique exact-prefix block 数随 episode 量上升（原 80 任务估计 ~300–600，全量 ×3 后 TBD），block 级统计样本量充足。

### 0.4.3 真实事件核验要求（冻结前必做）

| 数据集 | 核验项 | 不满足时的降级 |
|---|---|---|
| StableToolBench 500 | 每 workflow 工具调用次数、API 故障注入事件计数（虚拟 API 服务器缓存命中时确定性重放）、缓存命中率 | 降级为"family 泛化 + 任务成功率"；等待/重试结论由 τ-bench 支撑（v0.4 移除 Toolathlon 后） |
| BurstGPT 2,000 | session ID 连续性与时间戳完整性 | 降级为次要到达证据，Poisson/MMPP 假设保留为主并标注 |

> **v0.4 注**：原 Toolathlon 500 与 SWE 轨迹 500 核验行已随数据集移除而删除（trim-dataset-portfolio spec）。

冻结清单：`experiments/subsets/` 下每个数据集一个冻结文件（task / trajectory / session ID 列表 + 选取种子）。

### 0.4.4 排除数据集清单（含理由）

| 数据集 | 排除理由 |
|---|---|
| 合成 prompt 集 / 合成可控 DAG（本项目原有） | **用户禁令（2026-07-25）：不许自建数据集** |
| ToolBench（DFSDT 轨迹） | CC BY-NC 4.0 非商业许可，许可不洁 |
| GAIA | test 答案私有 + 依赖实时网页，不可复现 |
| WebArena / OSWorld | 自托管 Docker/VM 基建过重；OSWorld 的 WAIT/FAIL/DONE 语义列为可选扩展，不进 14 周主证据包 |
| Azure LLM Inference Trace | 公开可得性与字段未核实，暂不使用 |
| 任何"会话暂停/恢复"原生数据集 | 公开调研确认**不存在**；暂停/恢复语义由 replay 框架的到达模拟产生（实验操作，非数据集） |
| **SWE 轨迹**（v0.4 移除，原 Ch.5 压力面 200） | 200 样本不足以单独成章；同领域论文鲁棒性章通常 0–1 个数据集；SWE 解决率低（Qwen2.5-7B 小模型能力不足）；与 FlowCache C1–C3 主线（trace 协议 / 联合控制器 / reuse-fidelity 错位）关联弱。Migration：rebuttal 时可扩展到 500 episodes 补做 |
| **Toolathlon**（v0.4 移除，原 Ch.5 压力面 200） | 200 样本证据力弱；多 agent 协作场景与 FlowCache 单 agent + 工具 workload 主线偏离；适配器集成成本高。Migration：rebuttal 时可引用 τ-bench retail/airline 两域作为 workload 多样性证据 |

### 0.4.5 统一运行参数

| 参数 | 值 | 来源 |
|---|---|---|
| block_size | 16 tokens | `experiments/e1/config.yaml` |
| KV 预算档位 | 峰值 working set 的 10% / 25% / 50% / 100%（100% 仅作上界参照，E4 主表用前 3 档） | `experiments/e1/config.yaml`、IDEA §5.3 |
| 折扣率 β | 0.005 / step | g2-pilot §3.1 |
| 调度窗口 H | 1000 step | g2-pilot §3.1 |
| 到达过程 | **主证据：BurstGPT 真实会话到达窗口**；建模参照：Poisson（λ=4，需报告与 BurstGPT 的拟合优度对比）；E6 另行扰动 | 本册 v0.2 |
| 并发档位 | 4 / 8 / 16 workflow（显存不足时降为 2–4，如实记录） | 本册统一 |

### 0.4.6 原合成 DAG 角色替代映射

| 原合成 DAG 角色（涉及章节） | 替代方案（全部为真实数据或 replay 时操作） |
|---|---|
| 结构因果敏感性：分支率/深度/next-use 与 locality 关系（E1、E2） | **τ-bench / STB 真实轨迹**按深度 / 宽度分层做剂量–反应分析；v0.4 移除 SWE 轨迹后改用 τ-bench 内部结构差异（真实结构分层，非生成） |
| DAG 边缺失/噪声、branch 误预测（E6 轴 3/4） | **replay 时特征扰动**：对预测器输入的已声明后继做删边/错标——作用于特征的实验操作，不是数据集 |
| tool-wait 受控扫描（G3、E7-F4） | **τ-bench / STB 工具调用实测等待分布** + replay 时时间缩放（×0.5 / ×2 / 重尾化，参数扰动）；v0.4 移除 Toolathlon 后改用 τ-bench/STB 支撑 |
| burst arrival（E6 轴 5） | BurstGPT 真实突发窗口为主证据；MMPP 合成模型降为次要参照 |
| 深链 parent 缺失（E7-F5） | **τ-bench 长会话压力 cell**（预算 10%）；v0.4 移除 SWE 长轨迹后改用 τ-bench 内部长会话覆盖 |
| 取消语义（原 cancel 族） | **τ-bench / STB 工具失败与重试状态**；WebArena 不可达任务（可选扩展）；v0.4 移除 Toolathlon 后改用 τ-bench/STB 的工具失败语义 |

### 0.4.7 数据集数与同领域论文对比（v0.4 新增）

trim-dataset-portfolio spec 要求 FlowCache 核心数据集数封顶 5，下表对照同领域 KV cache 管理 / 前缀缓存论文的数据集规模：

| # | 论文 | Venue/Year | 数据集数 | 总样本量 | 来源 |
|---|---|---|---|---|---|
| 1 | **CacheGen** | SIGCOMM 2024 (arXiv 2310.07240) | **4** | **662 contexts** | 论文 §1 摘要："four datasets (662 contexts in total)" |
| 2 | **EvicPress** | arXiv 2512.14946 (2025-12) | **12** | **~600 contexts**（估算） | 论文摘要："Evaluation on 12 datasets and 5 models" |
| 3 | **KVFlow** | NeurIPS 2025 (arXiv 2507.07400) | **~2** | 未明确（合成 workflow） | 论文 §4 实验：参数化合成 |
| 4 | **vLLM / PagedAttention** | SOSP 2023 | **2** | ~1,000s requests/batch | 论文 §10 评估（ShareGPT, Alpaca） |
| 5 | **SGLang / RadixAttention** | NeurIPS 2024 (arXiv 2312.07104) | **~4–5** | ~1,000s 总样本 | 论文 §5 评估（MMLU/HellaSwag/GSM-8K/ShareGPT/MT-Bench） |
| 6 | **τ-bench 原论文** | ICLR 2025 (arXiv 2406.12045) | **1** | **1,320 episodes** | 论文主表 pass^k 评估（165 tasks × 8 seeds） |
| 7 | **FlowCache v0.5** | 本册 | **4**（核心，v0.5 移除 BFCL） | **~2,920 samples**（核心：τ-bench 1,320 + LongBench 1,000 + GSM8K 100 + STB 500） | trim-dataset-portfolio spec |

**对比结论**：

- 同领域论文数据集数中位数 **2 个**（排除生产 trace 论文），范围 1–4 个（排除 EvicPress 12 和 Ada-KV 29 子任务）。
- 同领域论文总样本量中位数 **~660 contexts**，范围 150–1,320 episodes。
- FlowCache v0.5 的 4 个核心数据集是中位数的 **2×**（v0.4 的 5 个为 2.5×，v0.3 的 7 个为 3.5×），~2,920 样本是中位数的 **4.4×**（v0.4 的 ~3,720 为 5.6×，v0.3 的 ~4,120 为 6.2×）。
- v0.5 精简后与同领域论文的差距进一步缩小，且 4 个核心数据集按 3 层角色（主表 1 + 质量面 2 + 鲁棒性 1）分层，每层均有明确分工，不存在冗余。

### 0.4.8 为何不能只用 GSM8K（v0.4 新增）

**背景**：QKVShare（arXiv:2605.03884，2026-05 预印本，未被任何会议/期刊正式接收）仅用 GSM8K 150 problems 就完成了多 agent KV handoff 论文。审稿人可能质疑：FlowCache 为何不效仿，只用 GSM8K？

**核心回答**：QKVShare 与 FlowCache 的评估场景根本不同，GSM8K 无法支撑 FlowCache 的 C1/C2/C3 任一主张。

#### 0.4.8.1 场景差异

| 维度 | QKVShare | FlowCache |
|---|---|---|
| 评估场景 | inter-agent handoff（agent 间 KV 传递） | intra-agent multi-turn（单 agent 内 tool-call 暂停/恢复） |
| KV 操作位置 | agent 之间 | agent 内部（跨 turn） |
| 数学任务角色 | 载体任务，验证 handoff 后精度 | 不适用——FlowCache 不评估数学推理 |
| 多轮结构来源 | hop 数（2-5 agent 串联） | tool-call 轮次（τ-bench 平均 10+ 轮） |
| KV 管理决策 | 量化 bit 分配（per-token） | 驻留/驱逐 + 精度联合（per-block） |

QKVShare 的"多 hop"是 agent 间串联，每个 agent 完整消费 KV 后传给下一个；FlowCache 的"多轮"是单 agent 内 tool-call 暂停/恢复，KV 必须在 tool 执行期间驻留或被驱逐。两者所需的工作负载结构完全不同。

#### 0.4.8.2 GSM8K 与 FlowCache 三条核心主张的匹配度

| FlowCache 核心主张 | 所需工作负载特征 | GSM8K 是否具备 | 说明 |
|---|---|:---:|---|
| **C1：trace 协议** | 多轮 agent 工具调用，产生可追踪的 block-level KV 复用结构 | ❌ | GSM8K 是单轮输入→单轮输出，无工具调用、无多轮会话、无 KV block 复用模式 |
| **C2：联合 precision+residency 控制器** | 跨 tool-call 边界的 KV cache 驻留/驱逐决策（暂停/恢复语义） | ❌ | GSM8K 无 tool-call 边界，无 pause/resume，整个 prefill 一次完成 |
| **C3：reuse-fidelity 错位实证** | 前缀复用机会（R）与保真风险（D）的错位可被利用 | ❌ | GSM8K 每题独立，无共享前缀（除 few-shot prompt），无 reuse 机会即无错位可利用 |

#### 0.4.8.3 若强行只用 GSM8K 的后果

| 实验 | 用 GSM8K 替代后的后果 |
|---|---|
| **Ch.1 工作负载画像** | 无 multi-turn 结构可画像；overlap/LCP/next-use/working-set 全部退化为 0 或无意义 |
| **Ch.2 R-D 错位 Pilot** | 无 reuse 机会 → R 维度恒为 0 → 无法构造 R-D 错位四象限 → G2 判定 NO-GO → 路线 A 直接终止 |
| **Ch.3 估计器有效性** | reuse 侧无数据可训练；fidelity 侧可做但仅剩单维度，无法支撑 C2 的"联合"主张 |
| **Ch.4 端到端主结果** | 无多轮 workload → 无 cache hit → 所有策略退化为 No-Cache → FlowCache-Joint 与 baseline 无差异 |
| **Ch.5 鲁棒性** | 无 family-out 可做（GSM8K 只有一个 domain） |

**结论**：强行只用 GSM8K 会导致 FlowCache 的 C1/C2/C3 三条主张全部无法验证，实验体系崩塌。

#### 0.4.8.4 GSM8K 在 FlowCache 中的合理角色

GSM8K 在 FlowCache 中的角色**且仅是** Ch.3 fidelity 质量面的 accuracy sanity（100 samples）：

- **用途**：验证 Q8/Q4 量化后模型基础推理能力未崩（accuracy 非劣界检验）
- **样本量**：100（QKVShare 用 150，FlowCache 用 100 已足够；GSM8K 测试集总量 1,319，100 为随机抽样子集）
- **不可替代性**：低——LongBench 的 QA 子任务可覆盖类似功能，但 GSM8K 是领域最通用的 accuracy sanity benchmark，保留成本低（100 samples 录制 < 0.5 GPU 小时）
- **角色边界**：不作为主表、画像、鲁棒性数据集；不扩大样本量；不因 QKVShare 用 150 而调整

## 0.5 Cache-Compatible 序列化规则（IDEA §6.2）

每类数据集必须给出显式编译规则，写入对应 trace 的元数据：

- 哪些事件进入 prompt；
- 事件顺序和格式是否稳定；
- 分支是否重新序列化历史；
- 哪些 block 能形成共同前缀；
- workflow resume 是否保持完全相同的 prefix；
- 历史摘要、截断或模板变化如何触发 invalidation。

**硬约束**：若某数据集的 exact-prefix overlap 很低，报告这一事实，不用语义引用制造伪命中。缓存兼容性由确定性规则判断（模型、版本、tokenizer、chat template、adapter、cache 配置、位置、父块哈希、token block、compute lineage + storage encoding 可解码性），DAG 只预测未来访问，不决定兼容性（IDEA §0.2/§1.4）。

**禁止特征**（IDEA §4.3，全部实验的预测器共同遵守）：

- 未来才产生的 DAG 边或答案引用；
- 目标访问发生后的 attention；
- test workflow 的未来事件；
- 由最终标签直接计算出的"估计剩余步骤"。

**禁止在线特征**：完整 `output_attentions=True`（空间复杂度 O(BHL²)，IDEA §4.4 明确不可作为默认低开销在线特征）。

## 0.6 数据切分与无泄漏协议（IDEA §6.3）

1. 以**完整 workflow/episode/trajectory/session** 为最小切分单位，所有 step 必须在同一 split；
2. 进一步按模板、graph signature、源文档、仓库、实体或底层问题分组（group split）；
3. 对跨 split 的 token prefix 做近重复检查（prefix dedup）；
4. validation 用于选择阈值、模型和预算；**test 只运行冻结配置**；
5. 交叉验证只在 train 内执行，不与最终 test 混用；
6. 统计单位为 workflow（episode/trajectory/session 同义），采用 paired workflow-level bootstrap 95% CI（1000 次重采样）。

**默认切分比例**（除章节另有说明）：train 60% / validation 20% / test 20%，按 workflow 分组切分，切分种子冻结并记录。

## 0.7 Open-loop 与 Closed-loop 使用规则（IDEA §6.4）

| 模式 | 定义 | 用途 | 禁止用途 |
|---|---|---|---|
| **Open-loop replay** | 冻结 token IDs、DAG snapshot、工具结果和到达时间；所有策略看到完全相同未来事件 | 系统性能与 policy 比较（TTFT、hit rate、overhead） | 不能证明量化后的真实任务质量 |
| **Closed-loop live run** | 允许缓存量化影响模型输出和后续工具调用 | 最终质量、任务成功率、失败分析 | 不能保证各策略经历完全相同请求序列 |

**铁律**：两种模式不可混在同一主表中。每章必须声明其结果行属于哪种模式。

## 0.8 统计约定与功效分析基准

### 0.8.1 通用约定

- 显著性水平 α = 0.05（双侧），报告 p 值与 95% CI；
- 统计单位为 workflow；同 workflow 内 block 不完全独立时，除 per-block 主结果外，必须同时报告 per-workflow 聚合结果（防 Simpson's paradox，以 per-workflow 为准）；
- 多重检验：同一家族检验采用 Bonferroni 校正，报告校正前后 p 值；
- 所有主比较使用 paired workflow-level bootstrap 95% CI（1000 次重采样）；
- 报告效应量，不只报告显著性。

### 0.8.2 功效分析基准表（Fisher z 近似，Spearman 适用，引自 g2-pilot §5.4）

| 目标 ρ | z = atanh(ρ) | 所需 N（α=0.05, power=0.80） |
|---|---|---|
| 0.3 | 0.3095 | 82 |
| 0.4 | 0.4236 | 47 |
| 0.5 | 0.5493 | 29 |
| 0.6 | 0.6931 | 19 |

参考：N=50 → 80% 功效可检 ρ ≥ 0.38；N=100 → 可检 ρ ≥ 0.27；N=300 → 可检 ρ ≥ 0.16；N=500 → 可检 ρ ≥ 0.12。

### 0.8.3 均值/比例类比较的功效规则

- 对于"p95 TTFT 改善 X%""任务成功率非劣"这类比较，pilot（E1/G3/G4 pilot）先测量 workflow 级指标的变异系数 CV 与组内相关；
- 预注册规则：若主 cell 的 paired 单元（各章给定，Tier 1 主面 ≥ 495 episodes）的 bootstrap 95% CI 半宽大于待检效应的 50%，则按预注册规则增加 replay 种子数或样本量，并在章节中记录触发值；
- 二值指标（任务成功率）按配对比例检验估计：N=495 paired episodes 的 95% CI 半宽约 4–5 个百分点（基线成功率 0.5–0.7 区间，精确值 TBD 于 pilot 后冻结）；非劣检验所需样本量在 G4 pilot 后按预注册非劣界 ε 重新计算。

## 0.9 通用指标词汇表（定义一次，各章引用）

| 指标 | 定义 | 级别 |
|---|---|---|
| TTFT | Time To First Token，请求到首个输出 token 的延迟 | request |
| JCT | Job Completion Time，请求到完整结束的延迟 | request |
| p50/p95/p99 | 对应分位数 | 分布 |
| SLO goodput | 满足预设 SLO（如 TTFT ≤ 阈值、JCT ≤ 阈值，阈值 TBD 预注册）的请求吞吐 | service |
| max admitted concurrency | 不违反 SLO 与 OOM 约束的最大并发 | service |
| token/block/byte cache hit | exact-prefix 命中比例（三种粒度） | cache |
| saved-prefill tokens/time | 因缓存命中避免重算的 token 数 / 实测时间 | cache |
| miss cost | 未命中导致的重算代价（ms 或 tokens） | cache |
| policy regret | 策略总成本与离线 oracle 总成本之差（同 trace、同预算） | policy |
| exact-prefix overlap | 跨/内 workflow 可复用 exact-prefix block 占比 | workload |
| LCP tokens | 最长公共前缀 token 数 | workload |
| next-use distance | block 从 inactive 到下次 exact-prefix 访问的 step 数 | block |
| logit KL | 量化干预 vs BF16 的逐 token logit KL 散度（continuation 前 K=64 token 均值） | token |
| top-k change | continuation 前 K 个 token 中 top-5 集合变化比例 | token |
| 工具调用一致性 | 函数名/参数 JSON 与 BF16 基线的精确匹配 | call |
| 任务成功率 | 数据库最终状态匹配（τ-bench） | workflow |
| QA EM/F1 | exact match / token-level F1 | question |
| PR-AUC / Brier / ECE | 预测器的排序质量 / 校准指标 | predictor |
| Precision@budget | 给定保留预算下预测"将复用"的精确率 | predictor |
| codec latency | 量化/反量化编解码时间 | overhead |
| H2D/D2H | host↔device 传输时间与字节数 | overhead |
| controller overhead | 控制器单次决策耗时 × 调用次数（含预测器推理） | overhead |
| GPU allocated/reserved、CPU pinned | 实测显存与 pinned memory 字节数 | resource |

## 0.10 通用 baseline 名录（定义一次，各章引用子集）

| 简称 | 描述 | 类型 |
|---|---|---|
| No-Cache | 无可复用缓存 / cold recompute | 下界参照 |
| APC-LRU | 推理引擎实际同引擎 automatic prefix caching（LRU 驱逐） | 工程 baseline |
| LRU | 标准 LRU 驱逐（block 级） | 简单启发式 |
| LFU | 标准 LFU 驱逐 | 简单启发式 |
| LRU-K / 2Q | LRU-K 或 2Q 队列驱逐（实现其一并记录） | 简单启发式 |
| GDSF | size-aware Greedy-Dual-Size-Frequency | 强启发式 |
| SizeCost | size/recompute-cost 启发式（age + size + measured recompute cost） | 强启发式（FlowCache 第一档估计器） |
| Oracle-Belady | 离线 Belady（未来已知）驱逐 | 上界参照 |
| Oracle-Cost | 离线 cost-aware lookahead heuristic（未来已知 + 成本模型） | 诊断参照；非严格最优上界 |
| KVFlow† / PBKV† | 至少一个在公平协议下忠实运行的 closest baseline；无法忠实复现的标 `*-inspired` 并先解决可比性 | 最近工作 baseline |
| ThunderAgent-inspired† | ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；API 级代理非块级缓存，提取 program-aware + 2^{-t} time decay 核心 idea | 最近工作 baseline（inspired variant） |
| Uniform-Q8 / Uniform-Q4 | 全部 inactive block 统一 Q8 / Q4 存储（同容量规则） | 量化 baseline |
| Reuse-Only | 只用复用价值估计决定驻留（精度统一） | 消融 baseline |
| Fidelity-Only | 只用保真风险估计决定精度（驻留用强启发式） | 消融 baseline |
| Decoupled-Best | 最强解耦组合：最强 reuse policy + uniform quantization 的机械组合 | 关键对照 |
| FlowCache-Joint | 本文联合 precision/residency controller | 待验方法 |
| FlowCache-Lossless | 仅 GPU BF16 ↔ CPU BF16 ↔ evict 的 FlowCache 无损版本（G3） | 中间版本 |

†：PBKV/KVFlow/ThunderAgent 的可比性判定流程见 G1 章 G1.4.1 检查清单。PBKV 无官方代码 → inspired variant；ThunderAgent 官方代码可用但为 API 级代理非块级缓存 → inspired variant；KVFlow 官方代码可用但需 WSL2 + CUDA + Rust → faithful 待 adapter。若三者均无法忠实运行，按 IDEA §11"所有 closest baseline 均无法忠实比较"风险处理，inspired variant 只能作次要补充。

---

# Part 1：Gate 验证设计

# G0：Exactness 与 Loadability

> **周次**：W1–W2 | **依赖**：无 | **关键路径**：是
> **失败动作**：允许切换一次受支持的模型/后端；仍失败则路线 A No-Go，转路线 B，不进入预测器开发

## G0.1 实验目标与 Gate 关系

G0 是所有后续工作的正确性基础，验证两件事：

1. **Exactness**：inactive prefix KV 的缓存恢复与完整重算在预设数值容差内一致；block identity、父链、invalidation 无错误关联；
2. **Loadability**：目标后端能在 4090D 24GB 上加载主模型并拦截/恢复所需 KV，且最小 Q-storage codec/staging/precision-lineage spike 跑通。

G0 不产生任何研究结论，只产生"可以继续做研究"的资格。IDEA §7 G0 的通过条件逐条对应本章 G0.8 的判定表。

### G0.1.1 G0 不证明的内容

- 不证明任何 workload 中存在可复用 locality（那是 G1/E1）；
- 不证明量化对任务质量无害（那是 G4/E3）；
- 不评估任何缓存策略的收益（无 baseline 比较，只有真值一致性检查）。

## G0.2 数据集与子集定义（v0.2：全部真实数据，禁合成）

G0 需要**共享结构先验已知**的测试用例。v0.2 起不再使用合成 prompt 集，改用真实数据中天然存在、真值可由数据来源直接推断的结构：

| 类别 | 真实数据来源 | 数量 | 先验真值来源 |
|---|---|---|---|
| ① 共享 system prompt | **τ-bench 同域任务对**（retail 域内 15 对 + airline 域内 15 对） | 30 对 | benchmark 定义保证：同域任务共享 system + policy + tool schema 前缀；应共享且仅共享该公共前缀 |
| ② 分支历史 | **SWE 轨迹中含重试的会话对**（同一任务的重试轨迹共享重试点之前的历史，20 对）+ **LMSYS 多轮会话**（10 条） | 30 组 | 轨迹事件序列直接确定 LCP 位置；重试点之后不得共享 |
| ③ chat template 变化 | **同一真实会话（LMSYS 长会话 10 条）× Qwen2.5 chat template 不同 revision 渲染**（2 版本） | 10 条 × 2 | 模板 diff 已知 → 变化点之后的 block 必须全部失效。**说明**：数据内容为真实会话，仅渲染模板受控变换，不属于自建数据集 |
| ④ 模型/adapter 标识变化 | 真实文本（③ 的会话）+ I_b 元组中 m/r/a 字段受控变换 | 10 组 | 元数据级变换：fail-closed 必须判定全部不兼容，禁止命中 |
| ⑤ 纯追加长会话 | **LMSYS 长会话**（≥8 轮，4–8K tokens） | 10 条 | 追加语义天然：会话内前缀应逐轮递增复用 |
| ⑥ 无共享对照 | **τ-bench 跨域任务对**（retail × airline，5 对）+ **LMSYS 随机会话对**（5 对） | 10 对 | 跨域/随机对预期跨条命中 = 0 |

**codec spike block 集**：~100 个 unique block，从 G0 后续 τ-bench 试录的 BF16 trace 中抽取（覆盖 system/user/assistant/tool 各 role、不同父链深度），用于 Q8/Q4 编码→解码→staging→lineage 隔离 spike。

**端到端 loadability 子集**：τ-bench 10 workflow（retail 6 + airline 4，系统抽样）。

**冻结产物**：`experiments/g0/real-structure-cases.json`（含每个用例的类别、来源 ID、共享结构先验标注）、`experiments/g0/subset.json`、`experiments/g0/codec-blocks.json`。

## G0.3 样本数与功效分析

G0 是**确定性正确性验证**而非统计推断，样本量由覆盖性决定：

| 验证项 | 样本量 | 理由 |
|---|---|---|
| BF16 恢复 vs 重算数值一致性 | 90 组真实结构用例的全部 block（预计 ≥ 200 block） + 10 端到端 workflow 的全部 inactive block（预计 ≥ 150 block） | 覆盖全部六类结构与 role 类型；一致性要求**逐 block 100% 通过**，非抽样检验 |
| block identity/父链/invalidation | 全部 90 组用例的 block | 共享结构先验已知，允许精确判定假阳性/假阴性；要求 0 错误 |
| codec roundtrip | 100 block × 2 精度（Q8/Q4） = 200 次 roundtrip | 覆盖 role × 深度分层；误差为连续量，N=100/精度 足以估计误差分布 95% CI（半宽 ≈ 0.2σ） |
| 显存峰值测量 | 每配置 5 次重复加载 | 取 max，报告波动 |

## G0.4 Baseline / 对照

| 对照 | 角色 |
|---|---|
| **完整重算（cold recompute）** | 数值真值：每条 prompt 从头 prefill，不使用任何缓存 |
| **GPU 常驻不重放** | 引擎自身 APC 原生路径（若后端自带），用于交叉比对自定义 cache manager 的命中行为 |

注意：G0 不比较任何驱逐/驻留策略；LRU 等 baseline 从 G1 才开始出现。

## G0.5 测试指标

| 类别 | 指标 | 定义/测量方式 |
|---|---|---|
| 数值一致性 | max abs diff、mean abs diff、cosine similarity | 恢复路径 vs 重算路径：①KV 张量逐元素对比；②同一 continuation 的 logits 对比；③greedy decode top-1 token 一致率 |
| 身份正确性 | block 哈希假阳性数、假阴性数 | 与真实结构的先验共享标注比对；父链连续性校验（IDEA §1.2 的 I_b 全字段） |
| invalidation 正确性 | 应失效 block 的失效比例、不应失效 block 的保留比例 | 模板/标识变化场景（类别③④）逐 block 检查 |
| codec | roundtrip 误差（MSE、max abs err、logit KL）、编码/解码延迟、staging 峰值字节数 | Q8/Q4 各 100 block；codec 必须来自后端实际支持的实现，不用离线 numpy 模拟 |
| lineage | approximate/canonical lineage 隔离正确性 | 从量化祖先生成的 child 不与 canonical BF16 lineage 错误别名；lineage 缺失触发 fail-closed 重算 |
| 资源 | 模型加载后 allocated/reserved 峰值；可承载的最大并发与 context 长度上限 | 每配置 5 次重复，取 max |

## G0.6 运行协议

1. 后端候选按 Part 0.3.3 原则评估，G0 内冻结其一；
2. 全部测试在单卡 4090D 上执行，BF16 dtype；
3. 数值一致性测试使用固定 seed、确定性 decode（temperature=0）；
4. codec spike 只验证"inactive Q-storage → active BF16 材料化 + staging + lineage 追踪"链路，**不**要求混合精度 attention kernel（IDEA §2.3）；
5. 记录全部冻结项：模型 revision、tokenizer、chat template、后端 commit、HF config、CUDA/driver（产物 `experiments/g0/freeze-record.json`）；
6. 真实结构用例的选取 ID 与先验标注全部落盘（`real-structure-cases.json`），保证第三方可复现同一测试集。

## G0.7 统计检验

- 一致性判定为**逐样本通过/失败**，不做显著性检验；
- codec roundtrip 误差报告分布（median、p95、max）与 95% CI；
- 显存峰值报告 max 与波动范围。

## G0.8 判定阈值（Go/No-Go，对应 IDEA §7 G0 逐条）

| # | 通过条件 | 阈值 |
|---|---|---|
| 1 | BF16 缓存恢复与完整重算一致 | 无损路径 KV 逐元素 bit-identical；logits max abs diff ≤ 1e-3（预注册容差）；greedy top-1 token 一致率 100% |
| 2 | block identity、父链、invalidation 无错误关联 | 假阳性 = 0 且假阴性 = 0；invalidation 场景 100% 正确触发 |
| 3 | 冻结记录完整 | freeze-record.json 字段齐全 |
| 4 | Q-storage codec/staging/lineage spike 跑通 | ≥1 个 block 全链路成功；100 block roundtrip 无 crash；误差分布已记录；approximate lineage 隔离 100% 正确 |
| 5 | 后端能拦截和恢复所需 KV | 实测通过，或 W1–W2 内完成最小扩展点并复测通过 |
| 6 | 显存可承载 | 权重 + active cache + staging + 安全水位 ≤ 24GB；无法容纳的预算档位直接删除并记录 |

**任一条件失败**：执行失败动作（允许切换一次模型/后端并重测；仍失败则路线 A No-Go 转路线 B）。

## G0.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 0.1 | 后端候选评估与冻结（Part 0.3.3 清单） | 后端评估记录 |
| 0.2 | 加载 Qwen2.5-7B-Instruct，测量 allocated/reserved 峰值（5 次重复） | 显存测量表 |
| 0.3 | 按 G0.2 六类从真实数据集选取结构用例并标注先验 | `g0/real-structure-cases.json` |
| 0.4 | 实现 block identity 哈希、父链、invalidation | block index 模块 |
| 0.5 | BF16 恢复 vs 重算一致性测试（90 组用例 + 10 workflow） | exactness 报告 |
| 0.6 | Q-storage codec/staging/lineage spike（1 block → 100 block） | codec spike 报告 |
| 0.7 | 汇总判定表，更新 ccfa.yaml G0 status | `g0/g0-verdict.md` |

## G0.10 硬件与时间预算

| 阶段 | 预估时间 |
|---|---|
| 后端评估 + 模型加载测量 | 0.5–1 天 |
| 真实用例选取标注 + block index 实现 | 1–2 天 |
| exactness 测试 | 0.5 天 |
| codec spike | 1–2 天 |
| **合计** | **W1–W2 两周内** |

## G0.11 预期产物

| 产物 | 路径 |
|---|---|
| 冻结记录 | `experiments/g0/freeze-record.json` |
| 真实结构用例集 | `experiments/g0/real-structure-cases.json` |
| exactness 报告 | `experiments/g0/exactness-report.md` |
| codec spike 报告 | `experiments/g0/codec-spike-report.md` |
| 判定报告 | `experiments/g0/g0-verdict.md` |

### G0.11.1 结果表格模板（完成后填充，不发明数字）

**表 G0-1：BF16 恢复 vs 完整重算数值一致性**

| 数据面 | block 数 | KV bit-identical 比例 | logits max abs diff | logits mean abs diff | cosine sim | top-1 token 一致率 |
|---|---|---|---|---|---|---|
| 真实结构用例（90 组） | TBD | TBD | TBD | TBD | TBD | TBD |
| τ-bench 10 workflow | TBD | TBD | TBD | TBD | TBD | TBD |

**表 G0-2：identity / 父链 / invalidation 正确性（按 G0.2 六类分列）**

| 类别 | block 数 | 哈希假阳性 | 哈希假阴性 | 父链断裂 | invalidation 误触发 | invalidation 漏触发 |
|---|---|---|---|---|---|---|
| ① τ-bench 同域共享 | TBD | TBD | TBD | TBD | TBD | TBD |
| ② SWE/LMSYS 分支 | TBD | TBD | TBD | TBD | TBD | TBD |
| ③ template 变化 | TBD | TBD | TBD | TBD | TBD | TBD |
| ④ 标识变化 | TBD | TBD | TBD | TBD | TBD | TBD |
| ⑤ LMSYS 纯追加 | TBD | TBD | TBD | TBD | TBD | TBD |
| ⑥ 无共享对照 | TBD | TBD | TBD | TBD | TBD | TBD |

**表 G0-3：codec roundtrip（Q8/Q4 × 100 block）**

| 精度 | MSE (median / p95 / max) | max abs err | logit KL (median / p95) | 编码 ms | 解码 ms | staging 峰值 bytes | lineage 隔离错误 |
|---|---|---|---|---|---|---|---|
| Q8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Q4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 G0-4：显存峰值测量（5 次重复，取 max）**

| 配置 | 权重 allocated | 权重 reserved | KV pool 上限 | active + staging | 安全水位 | 合计 | 是否 ≤ 24GB |
|---|---|---|---|---|---|---|---|
| 仅加载模型 | TBD | TBD | — | — | — | TBD | TBD |
| 并发 4 × 4K ctx | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 并发 8 × 8K ctx | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 并发 16 × 8K ctx | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## G0.12 失败动作

按 IDEA §7 G0：允许切换一次受支持的模型/后端；仍失败则路线 A No-Go，转路线 B，不进入预测器开发。触发时更新 ccfa.yaml（G0 → failed，route → B）。

## G0.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| exactness/loadability 判定条件 | §7 G0 |
| block identity 字段 I_b / lineage C_b、E_b | §1.2 |
| inactive/active 状态分离 | §1.3 |
| codec spike 先行、无损路径先行 | §5.1 |
| 显存预算定义与报告规则 | §5.3 |
| 模型选择 | §5.2（2026-07-25 用户变更为 Qwen2.5-7B） |
| 真实数据用例（禁自建数据集） | 用户规定 2026-07-25（v0.2） |

---

# G1：Opportunity（缓存机会）

> **周次**：W6（v0.2 起，原 W5 顺延） | **依赖**：G0、E1 | **关键路径**：是
> **失败动作**：转向"何时工作流结构产生物理 KV 复用"的 benchmark/characterization 论文（路线 B）
>
> **v0.3 对齐说明（2026-07-25）**：按 spec v0.3，G1 不再独立运行，**复用 Ch.1（即 E1+G1 合并）画像数据**做判定。判定逻辑（headroom ≥ 10% + closest baseline 可比性）与阈值不变；数据来源从"独立 Gate 实验"改为"E1 画像 + oracle headroom 表"。本章数据集体系同步精简：StableToolBench/SWE/Toolathlon 移出（仅在 Ch.5 出现），主表仅保留 τ-bench 495 + BFCL 800（v0.5 移除 BFCL，仅 τ-bench 1,320）。原 v0.2 的 G1.11.1 表 G1-3（StableToolBench 确认表）降级为附录，不参与 Go/No-Go 判定。

## G1.1 实验目标与 Gate 关系

G1 验证 FlowCache 的**第一核心假设**（IDEA §0.3-1）：

> 真实 Agent workload 中存在非平凡的 exact-prefix 再访问，且离线 oracle 明显优于 LRU/简单启发式。

同时，G1 承担 **closest baseline 可比性判定**：至少保证 PBKV 或 KVFlow 中一个能在公平协议下忠实运行；若只能实现 inspired variant，必须先解决可比性（IDEA §7 G1）。

### G1.1.1 G1 不证明的内容

- 不证明复用价值与保真风险的错位（G2）；
- 不证明任何控制器端到端收益（G3/E4）；
- 不使用任何量化（纯无损路径）。

## G1.2 数据集与子集定义

**v0.3 精简后主表数据集**（参与 Go/No-Go 判定）：

| 数据集 | 样本数 | 说明 |
|---|---|---|
| τ-bench | **1,320 episodes（165 任务 × 8 seeds，retail + airline 两域）** | 主判定 workload ①；LLM 用户模拟器（`llm_user`）+ 真实工具 backend；8 seeds 与原论文 pass^k（k≤8）对齐 |

**τ-bench seeds 数冻结依据（2026-07-25 调研后冻结）**：τ-bench 原论文（arXiv 2406.12045, ICLR 2025）主表用 165 任务全量，pass^k 指标用 k∈{1,2,4,8}。3 seeds 只能算 pass^3，统计上不足以区分 consistency；8 seeds 与原论文 pass^8 完全对齐，最稳健。相比 3 seeds 增量 660 episodes，按 4090D ~30s/episode 估算约 5.5 GPU 小时。原 v0.3 的 495（3 seeds）已升级为 1320（8 seeds）。

> **v0.5（2026-07-26）注**：原 G1.2 表中 BFCL v3 multi-turn 行（6,400 episodes，4 子集 × 200 × 8 decode seeds）已删除——BFCL 不再作为数据集。原"BFCL 8 decode seeds 依据"段落与"BFCL v3 multi-turn 集成方式"段落（含可用性、样本结构、多轮真实性、工具 backend、用户模拟器、集成命令、样本量对齐 7 项）整段删除。rebuttal 时若需补做 BFCL，按 IDEA.rewritten.md v0.4 的 migration 规则执行。

**v0.3 移出主表的数据集**（仅在 Ch.5 鲁棒性章节使用，不参与 G1 判定）：

| 数据集 | 样本数（Ch.5 用量） | v0.2 原计划用量 | 说明 |
|---|---|---|---|
| StableToolBench | 500 | 500 | Ch.5 family-out 轴 |

> **v0.4 注（trim-dataset-portfolio spec）**：SWE-rebench-openhands-trajectories（原 200）与 Toolathlon-Trajectories（原 200）已从 Ch.5 移除。Ch.5 鲁棒性压力面仅保留 StableToolBench 500 作 family-out 主证据；branch 噪声轴改由 τ-bench 内部 replay 扰动（删边/错标后继）覆盖，不另设数据集。Migration：rebuttal 时可扩展 STB 到更大样本或补做 SWE 500。

trace 来源：E1 录制的可重放 trace（τ-bench 为 rollout 录制；到达结构主证据 BurstGPT，Poisson λ=4 为建模参照，H=1000，block_size=16）。

## G1.3 样本数与功效分析

| 项 | 值 | 依据 |
|---|---|---|
| 判定单元 | 1,320 episodes（τ-bench，165 任务 × 8 seeds） | workflow/episode 为统计单位 |
| unique block 数（估计） | 随 episode 量上升（原 80 任务 ~300–600；全量 ×8 后 TBD） | g2-pilot §2.4 外推 |
| 判定效应量 | oracle vs 最佳简单策略的 miss-cost 或 p95 TTFT 差距 ≥ 10% | IDEA §7 G1 内部参考阈值 |
| 功效 | 1,320 paired episodes 下 CI 半宽显著小于 80 单元情形（0.8.3）；8 seeds 支持 pass^k（k≤8）分析 | 0.8.3 预注册规则 + τ-bench 原论文 pass^k 对齐 |
| 样本量封顶 | τ-bench 1,320 episodes（v0.5 移除 BFCL 后单数据集；不再追加 STB/SWE/Toolathlon） | spec v0.3 §5 主表原 τ-bench + BFCL，v0.5 移除 BFCL；STB/SWE/Toolathlon 仅 Ch.5 |
| 报告补充指标 | total LLM calls（预计 5K–15K）、平均 turn/episode、pass^k（k∈{1,2,4,8}） | pass^k 与 τ-bench 原论文对齐 |
| seeds 数冻结 | 8 seeds（2026-07-25 调研后冻结） | τ-bench 原论文（arXiv 2406.12045, ICLR 2025）用 k≤8；3 seeds 只能算 pass^3 不足以区分 consistency |

## G1.4 Baseline / 对照

| Baseline | 角色 | 实现要求 |
|---|---|---|
| APC-LRU | 工程 baseline | 同引擎真实 block index |
| LRU | 简单启发式 | block 级 |
| GDSF | 强启发式 | size-aware |
| SizeCost | 强启发式（age + size + measured recompute cost） | IDEA §4.3 第一档 |
| Oracle-Belady | 上界（未来已知） | 离线计算 |
| Oracle-Cost | 离线 lookahead 诊断（未来已知 + 成本模型；非严格最优上界） | 驱逐时选 `saved_prefill_ms / next_use_distance` 最小的块；`block_cost` 来自 step 级 `prefill_ms` 按 token 范围比例分摊（见 `experiments/e1/compare_oracle.py:build_access_trace`） |
| **KVFlow 或 PBKV（≥1 个）** | closest baseline 可比性判定 | 公平协议（同引擎/模型/trace/预算）下忠实运行；不可忠实复现 → 标 `*-inspired` 并记录不兼容原因清单。KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 上激活（`config.yaml: kvflow_faithful.enabled: true`），adapter 实现中 |
| **ThunderAgent-inspired** | 补充 closest baseline（workflow-aware time decay） | ICML 2026 Spotlight，纯 Python 100%，原生 Windows 可跑；ThunderAgent 是 API 级代理非块级缓存策略，提取 time decay (2^{-t}) + program-aware 调度核心 idea 实现 `baselines/thunderagent_inspired.py` |

**可比性判定流程**：①获取官方代码/论文协议 → ②接口适配评估 → ③在同一 open-loop replay 上运行 → ④若引擎/语义不兼容，记录具体不兼容点（5 项以内）并给出 inspired variant 的忠实度说明 → ⑤两个均失败则触发 IDEA §11"所有 closest baseline 均无法忠实比较"风险条目。

### G1.4.1 Closest baseline 可比性检查清单（逐项留痕）

| 检查项 | KVFlow | PBKV | ThunderAgent |
|---|---|---|---|
| 官方代码/协议可获得性 | AVAILABLE（[github.com/PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow)，NeurIPS 2025，Apache-2.0，末次 commit 2026-03-13） | UNAVAILABLE（arXiv 2605.06472 无代码链接，多轮搜索无官方/第三方 repo） | AVAILABLE（[github.com/ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent)，ICML 2026 Spotlight，MIT，末次 commit 2026-06-06，144 commits） |
| 所需引擎钩子（block index / eviction hook / prefetch hook）本后端是否具备 | NEEDS_ADAPTER：仓库含魔改 SGLang + SScheduler PFEngine，需将 τ-bench `block_assignments` 翻译为 SGLang prefix-tree 请求 + `PlanManager.update_agent_timestep(...)`；AutoDL Linux + CUDA + Rust toolchain 可用（2026-07-26 升级，原 WSL2 约束不再适用） | N/A：无代码 | NEEDS_ADAPTER（inspired variant）：ThunderAgent 是 FastAPI 代理（OpenAI 兼容 API + `program_id`），非块级缓存策略；需将 API 级 time decay 适配为 block 级优先级评分，并丢弃 `--gpu-memory-pressure` 在线反馈（open-loop replay 无活动后端） |
| 其缓存语义是否与本研究 exact-prefix 语义一致 | NEEDS_ADAPTER：ASG（Agent Step Graph）抽象与 `block_hash`/`parent_hash` DAG 天然契合，但需适配层 | N/A | NEEDS_ADAPTER：program-aware 调度与 `workflow_id` 自然契合；prefix-chain eviction 与 APC-LRU/PBKV-Inspired 语义一致 |
| 其特征是否违反本研究禁止特征清单（未来信息泄漏检查） | TBD（待 adapter 实现后验证） | N/A | 通过：inspired variant 仅使用决策时可见的 `workflow_id` 与历史访问时间，无未来信息泄漏 |
| 在本 replay 协议下可忠实运行的 trace 覆盖率 | TBD（待 adapter 实现后测量） | N/A | 100%（inspired variant 在所有 τ-bench trace 上可运行；trace 需携带 `workflow_id` 字段，缺失时退化为默认 workflow ""） |
| 判定（faithful / inspired / 不可比 + 原因 ≤ 5 项） | **faithful（AutoDL Linux adapter 实现中）**：官方代码可用，AutoDL Linux + CUDA + Rust 可编译（2026-07-26 升级），τ-bench trace 需 adapter 但语义兼容；`config.yaml` 中 `kvflow_faithful.enabled: true` | **inspired variant**：无官方代码，按论文 GraphSAGE + workflow-history attention + 多步预测核心 idea 实现 `pbkv_inspired.py`，明确标注差异 | **inspired variant**：官方代码可用但为 API 级代理非块级缓存；按论文 program-aware + 2^{-t} time decay + cross-workflow capacity scheduling 核心 idea 实现 `thunderagent_inspired.py`，明确标注差异（无 GPU pressure 反馈、无在线 decay rate 调优、hand-tuned decay_rate=0.05） |

## G1.5 测试指标

| 类别 | 指标 |
|---|---|
| 机会画像（与 E1 共享） | exact-prefix overlap、LCP tokens 分布、next-use distance 分布、share_count 分布、block working-set size、KV/总显存占比 |
| 策略差距 | per-workflow miss cost（重算 ms）、p95 TTFT、token/block/byte hit rate |
| lookahead gap | Oracle-Cost vs 最佳简单策略的 miss-cost 相对差、p95 cache-delay 相对差；该值是未来信息参照差距，不宣称严格可达 headroom |
| 可比性 | closest baseline 的运行覆盖率（多少比例 trace 可忠实重放）、与论文报告行为的定性一致性 |

预算档位：10% / 25% / 50% / 100%（100% 作上界参照）。

## G1.6 运行协议

- 全部 open-loop replay（Part 0.7）；冻结 token IDs、工具结果、到达时间；
- 所有策略看到完全相同未来事件（oracle 类例外：仅 oracle 允许看未来，这正是其上界含义）；
- 无损路径（GPU BF16 ↔ CPU BF16 ↔ evicted），不启用量化；
- 每个（策略 × 预算档位 × 数据集）单元运行该数据集全部样本；3 个 replay 种子（到达时间扰动）估计方差；
- 到达结构：BurstGPT 真实窗口为主，Poisson λ=4 参照（报告拟合优度）。

## G1.7 统计检验

- 主判定量：oracle vs 最佳简单策略的 per-workflow miss-cost 相对差与 p95 TTFT 相对差；
- paired workflow-level bootstrap 95% CI（1000 次）；
- 多预算档位视为同一家族检验，Bonferroni 校正；
- 报告各策略完整分布（不只均值）。

## G1.8 判定阈值（Go/No-Go）

| 条件 | 阈值 | 判定 |
|---|---|---|
| headroom | oracle 相对最佳简单策略存在 ≥ 10% 的 miss-cost 或 p95 TTFT 改进空间（IDEA §7 G1 内部参考） | 达到 → 通过第一项 |
| 可比性 | ≥1 个 PBKV/KVFlow 在公平协议下忠实运行，或不兼容原因已清楚解释且 inspired variant 仅作次要补充 | 满足 → 通过第二项 |
| overlap 非平凡 | exact-prefix overlap 显著大于 0 且 next-use distance 存在非平凡方差（描述性，不设硬阈值，报告事实） | 报告 |

**两项同时通过 → G1 passed。** 否则执行失败动作。

## G1.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 1.1 | 复用 E1 trace 与画像报告，确认输入完整 | 输入清单 |
| 1.2 | 实现/接入 LRU、GDSF、SizeCost、APC-LRU | 策略代码 |
| 1.3 | 实现离线 Oracle-Belady / Oracle-Cost | oracle 模块（可复用 `experiments/e1/compare_oracle.py`） |
| 1.4 | PBKV/KVFlow/ThunderAgent 可比性评估与适配 | 可比性记录（[RESEARCH_NOTES.md](e1/baselines/RESEARCH_NOTES.md) + G1.4.1 检查清单）；PBKV-inspired 与 ThunderAgent-inspired 已实现并集成到 `compare_oracle.py`，KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活，adapter 实现中 |
| 1.5 | 全网格运行（策略 × 预算 × 数据集 × 3 种子） | 原始结果表（`experiments/g1/run_grid.py` → `results/raw_results.csv`） |
| 1.6 | 统计分析与判定 | `experiments/g1/verdict.py` → `experiments/g1/g1-verdict.md` + `g1-verdict.json` |

## G1.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 策略 + oracle 实现 | 2 天 |
| closest baseline 适配 | 2–3 天（不含不可兼容时的记录工作） |
| 运行（open-loop replay 为主） | ~6–8 小时（数据量上升） |
| **合计** | **W6 一周内** |

## G1.11 预期产物

| 产物 | 路径 |
|---|---|
| 策略对比原始结果 | `experiments/g1/run_grid.py` → `experiments/g1/results/raw_results.csv` |
| headroom 图（策略 × 预算） | `experiments/g1/plot_headroom.py` → `experiments/g1/figures/g1-headroom.png` |
| closest baseline 可比性记录 | `experiments/g1/baseline-comparability.md` |
| 判定报告 | `experiments/g1/verdict.py` → `experiments/g1/g1-verdict.md` + `g1-verdict.json` |

### G1.11.1 结果表格模板（完成后填充，不发明数字）

**表 G1-1：headroom 主表（τ-bench 1,320 episodes = 165 任务 × 8 seeds，pass^k k∈{1,2,4,8}）**

| 策略 | 预算 10% miss cost (ms) | 预算 25% miss cost | 预算 50% miss cost | 预算 10% p95 TTFT (ms) | 预算 25% p95 TTFT | 预算 50% p95 TTFT |
|---|---|---|---|---|---|---|
| APC-LRU | TBD | TBD | TBD | TBD | TBD | TBD |
| LRU | TBD | TBD | TBD | TBD | TBD | TBD |
| GDSF | TBD | TBD | TBD | TBD | TBD | TBD |
| SizeCost | TBD | TBD | TBD | TBD | TBD | TBD |
| PBKV-inspired† | TBD | TBD | TBD | TBD | TBD | TBD |
| ThunderAgent-inspired† | TBD | TBD | TBD | TBD | TBD | TBD |
| KVFlow（faithful，AutoDL Linux adapter 实现中） | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle-Belady | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle-Cost | TBD | TBD | TBD | TBD | TBD | TBD |
| **oracle vs 最佳简单策略相对差** | TBD | TBD | TBD | TBD | TBD | TBD |

†：G1.4.1 判定后填入实际可用的 closest baseline 名称。PBKV-inspired 与 ThunderAgent-inspired 均为 inspired variant（无官方代码 / API 级代理非块级缓存），KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活（`config.yaml: kvflow_faithful.enabled: true`），adapter 实现中。

**表 G1-2：headroom 第二主表（v0.5 移除 BFCL 800，本表已删除；rebuttal 时若补做 BFCL 可恢复同构主表，结构同表 G1-1）**。

**表 G1-3：locality 画像摘要（判定输入，与 E1/Ch.1 共享数据）**

| 数据集 | exact-prefix overlap | LCP tokens (median / p95) | next-use distance (median / p95) | share_count ≥ 2 的 block 占比 | KV/总显存占比 |
|---|---|---|---|---|---|
| τ-bench | TBD | TBD | TBD | TBD | TBD |

> **v0.3 注**：原 v0.2 的 G1-3 确认表（StableToolBench 500）与 G1-4 locality 摘要中的 STB/SWE/Toolathlon 行已删除，这些数据集移至 Ch.5 鲁棒性章节。如需 STB 确认性参照，在 Ch.5 family-out 轴中给出。

## G1.12 失败动作

按 IDEA §7 G1：转向"何时工作流结构产生物理 KV 复用"的 benchmark/characterization 论文（路线 B）。E1 画像数据与本章 trace 保留，直接作为路线 B 的核心资产。触发时更新 ccfa.yaml（G1 → failed，route → B）。

## G1.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 核心假设 1（oracle 优于 LRU） | §0.3 |
| G1 通过条件与失败动作 | §7 G1 |
| 成本模型 C^res | §2.1 |
| baseline 公平性规则 | §5.1 |
| 数据集组合 | §6.1（v0.2 用户扩展） |
| open-loop replay | §6.4 |
| closest baseline 不可比风险 | §11 风险表 |

---

# G3：Lossless Residency（无损驻留）

> **周次**：W7–W8 | **依赖**：G0、G1 | **关键路径**：是
> **当前状态（2026-07-28）**：
> `causal GPU admission + selective migration implemented /
> PROTOCOL-INCOMPLETE`。28 个回归测试通过；受控 hot/cold 回归在相同
> 9,999 hits 下把 modeled movement 从 1,100.943 ms 降至 0 ms，但完整
> workload 尚未验证；等成本循环负对照已消除激进旁路退化。
> **失败动作**：只有协议完整且 closed-loop 指标有效后失败，才判路线 A No-Go；协议问题先修复，不切换路线。

## G3.1 实验目标与 Gate 关系

G3 验证：在**只用无损动作**（GPU BF16 ↔ CPU BF16 ↔ Evict，IDEA §2.3 动作空间 A₀）时，一个简单的价值感知 controller 是否已经优于最强简单驱逐策略。这是联合 controller 的"地基"——如果连无损驻留都没有净收益，加入精度维度（G4/G2）也没有意义。

### G3.1.1 G3 不证明的内容

- 不涉及任何量化精度决策（G4）；
- 不证明复用价值与保真风险错位（G2）；
- 不要求学习式预测器（G5 在 G3 的 controller 框架内单独判定；G3 使用 heuristic/survival 级别的 reuse 估计）。

### G3.1.2 G3-P1 因果 GPU 准入与选择性迁移前置门

在任何全网格或 Go/No-Go 前，必须先在 2 GiB、concurrency=4 单 cell 完成：

- GPU LRU victim 与 CPU value victim 不再对整个缓存逐次线性扫描；
- H2D/D2H 采用相应 block size 的实测中位数或非负分段插值；
- miss、迁移、恢复和 controller 建模成本逐请求计入 `modeled cache delay`；
- per-task 移动成本之和与 global 计数/成本一致，`fallback_count=0`；
- open-loop 只输出 offered load，不能生成吞吐非劣结论；
- always-migrate 只作容量扩展/压力 baseline，不以 FlowCache 方法结果命名；
- share_count 只由截至决策时刻的 trailing window 生成，task-grouped validation/test 无交叉；
- 只有离线 oracle 能接收 `future_accesses`；在线 baseline 接收该参数立即
  fail-closed，并通过“相同历史、不同未来”的前缀决策不变性测试；
- 首次/低证据 incoming 只有在成本明显低于 incumbent，或 incumbent 已有
  更强历史复用证据时才使用低先验；等成本且等证据时保守准入；
- test CPU-migration selection rate 与 GPU bypass rate 均须处于
  1%–99%，migration 较 always-migrate 至少下降 10%，transfer 较
  selective-migrate-only 至少下降 5%，且 modeled p95/总 service cost
  相对两项消融的增幅均不超过 5%。

详细命令、检查项与晋级条件见 `experiments/g3-next-experiment.md`。

## G3.2 数据集与子集定义

| 数据集 | 样本数 | 说明 |
|---|---|---|
| τ-bench | 495 episodes | 主判定 workload |
| StableToolBench | 500（核验通过后） | 确认性 workload |
| Toolathlon 轨迹 | 500（取含工具等待时间戳的子集） | 真实 tool-wait 分布对驻留收益的影响（接管原合成 tool-wait 族角色；replay 时时间缩放 ×0.5/×2 作敏感性，属参数扰动非数据集） |

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn 800 行已删除——BFCL 不再作为数据集。

trace 与到达模型同 G1（BurstGPT 主证据 + Poisson 参照，H=1000，3 个 replay 种子）。

## G3.3 样本数与功效分析

| 项 | 值 |
|---|---|
| 析因单元 | 3 KV 预算（10%/25%/50%）× 3 并发（4/8/16）= 9 cell / 数据集 |
| 每 cell | 495 episodes（τ-bench）× 3 replay 种子 |
| 判定效应量 | p95 TTFT 改善 ≥ ~15%（固定质量下）；吞吐下降 ≤ ~5%（IDEA §7 G3 内部参考） |
| 功效规则 | 495 paired episodes 下 CI 半宽 ~2× 窄于原 80 单元设计（0.8.3）；首跑测 CV 后按规则决定是否增加种子 |

## G3.4 Baseline / 对照

| 对照 | 说明 |
|---|---|
| No-Cache | 下界参照 |
| APC-LRU | 工程 baseline |
| GDSF | 强启发式驱逐（IDEA §7 G3 明确要求的比较对象） |
| SizeCost-LRU | size-aware LRU（IDEA §7 G3 明确要求的比较对象） |
| Two-tier LRU / GDSF | 与方法使用相同 GPU+CPU 容量的公平分层启发式 |
| Always-Migrate Tiered-LRU | G3-P1 诊断 baseline；每个 GPU victim 都进 CPU，不得作为 FlowCache 主结果 |
| Always-Admit + Selective-Migrate | 隔离 CPU selective migration 收益，所有 incoming miss 强制进入 GPU |
| **FlowCache-Lossless-Causal-Admission** | 已实现、待 held-out 验证：先以因果 cost value 比较 incoming 与 GPU incumbent，再对 displaced victim 做 selective migrate/evict |
| Two-tier Lookahead Reference | 同 GPU+CPU 资源、bypass-aware 的离线未来参照；GPU-only Oracle-Cost 只是 cost/distance heuristic，不是严格最优上界 |

## G3.5 测试指标

| 类别 | 指标 |
|---|---|
| P1 诊断 | offline replay wall time / µs per access、GPU admission selected/bypassed、CPU migration selected/rejected、movement reduction、restore yield、额外 miss cost、modeled cache-delay p50/p95、offered load（非吞吐）、成本守恒、fallback、future-index-used=false |
| 正式主指标 | closed-loop p95 TTFT（vs two-tier GDSF/SizeCost-LRU 的相对改善）、achieved throughput、SLO goodput |
| 辅助 | TTFT/JCT p50/p99、token/block/byte hit rate、saved-prefill tokens/time |
| 开销 | 恢复时间（CPU→GPU）、迁移时间（GPU→CPU）、H2D/D2H 字节数、controller 单次决策耗时与总开销 |
| 约束 | 任务成功率 = BF16 基线（无损路径应**零质量差异**，逐 workflow 核验）、GPU allocated/reserved、CPU pinned bytes |
| 可行性 | 恢复 + 迁移开销 < 所节省 prefill（逐 block 判定，报告违反比例） |

## G3.6 运行协议

- open-loop replay 只用于策略语义、hit/miss、movement、modeled cache delay 与 policy regret；不得把到达窗口反推值称为 throughput；
- closed-loop serving 用真实 request start/first-token/completion timestamps 测 TTFT、JCT、queueing、throughput 与 SLO goodput；另做 BF16 抽检确认无损路径零质量差异；
- 所有策略同引擎、同模型、同 dtype、同预算、同请求顺序（IDEA §8 E4 主结论约束，本章同样遵守）；
- controller 触发时机：请求到达、暂停、恢复、完成、显存压力变化（IDEA §4.5）；
- 保留安全水位，避免 allocator reserved 导致临界 OOM；
- 预测器失效回退：controller 内部异常时回退 SizeCost-LRU 并记录回退次数。

### G3.6.1 迁移/恢复成本标定（进入策略前的实测协议）

IDEA §2.1 要求所有成本按父前缀长度、block 大小、batch/concurrency、PCIe 状态和引擎状态**实测建模**：

| 成本项 | 标定方法 | 拟合形式 |
|---|---|---|
| prefill 成本 C^res_evict | 对 block_size=16 在不同父前缀长度（0.5K/1K/2K/4K/8K）× 并发（1/4/8/16）下实测 prefill ms | 分段线性或查表（记录 R²） |
| GPU→CPU 迁移 C^place | pinned buffer 上不同字节数 × 并发负载的 D2H 时间 | 线性（截距 + 斜率/byte） |
| CPU→GPU 恢复 C^res_CPU | 同上，H2D 方向 | 线性 |
| hold 机会成本 C^hold | 以"同预算下被挤占 block 的期望 miss cost"近似（oracle 辅助估计，记录方法） | 标量/byte·step |
| controller 决策成本 C^policy | 单次决策耗时 × 调用频率实测 | 标量/decision |

标定结果冻结于 `experiments/g3/cost-model.json`，G3/G5/E4 共用同一成本模型，禁止各章各自标定。

回放查值规则：优先 exact-size 样本中位数；区间内做分段插值；区间外的拟合值必须 clamp 到非负。任何负 transfer cost 直接判 P0 失败。

## G3.7 统计检验

- 主比较：FlowCache-Lossless vs GDSF、vs SizeCost-LRU 的 per-workflow TTFT 配对差；
- paired workflow-level bootstrap 95% CI（1000 次）；
- 9 cell × 2 主比较 = 同一家族检验，Bonferroni 校正；
- 吞吐变化报告 CI 并检验是否超过 −5% 下界（非劣式判定）。

## G3.8 判定阈值（Go/No-Go，对应 IDEA §7 G3）

**协议前置条件**：真实 TTFT 与 achieved throughput/goodput 已由 closed-loop 测得；强基线齐全；所有主比较共享相同 GPU/CPU 容量；当前策略不是 always-migrate 诊断变体，也未在 test 上调参。任一前置条件缺失时状态为 `PROTOCOL-INCOMPLETE`，不得输出 GO/NO-GO。

| 条件 | 阈值 |
|---|---|
| 开销可行性 | 恢复和迁移开销 < 所节省 prefill（聚合层面成立；逐 block 违反比例如实报告） |
| 选择性工程门槛 | held-out test 上 CPU selection 与 GPU bypass rate 均 ∈ [1%,99%]；migration 较 always-migrate 至少减少 10%；transfer 较 selective-migrate-only 至少减少 5%；modeled p95/总 service cost 相对两项消融增幅均 ≤ 5% |
| 主收益 | 固定质量下 p95 TTFT 改善 ≥ ~15%（内部参考阈值；主 cell：预算 25%、并发 8 必须达标，其余 cell 报告趋势） |
| 吞吐非劣 | 吞吐下降 ≤ ~5% |
| 优于强启发式 | controller 显著优于 SizeCost-LRU/GDSF（bootstrap CI 不含 0） |

**全部满足 → G3 passed。** 任一关键条件失败 → 失败动作。

## G3.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 3.0 | 已完成：O(1)/O(log N) victim、transfer cost 缓存、逐请求计费、future-index 隔离、因果 doorkeeper 与 fail-closed verdict | 28 个回归测试 |
| 3.1 | 已实现因果 GPU admission、selective migration 及两项独立消融；在 task-grouped validation 冻结四个 admission 参数；固定基线/CPU 参数消融结果复用，将 54 组扫描由 324 降为 76 次 baseline replay | `tune_selective_migration.py` + selection report |
| 3.2 | held-out test 跑 2 GiB/c=4 全 trace；只出诊断报告，不作 GO/NO-GO | G3-P1 单 cell open-loop 结果 |
| 3.2b | 补齐公平 two-tier LRU/GDSF/SizeCost 与 two-tier oracle | 无损公平 baseline 组 |
| 3.3 | 主 cell closed-loop serving 与 20-episode 无损质量核验 | 真实 TTFT/throughput/goodput |
| 3.4 | 单 cell 通过后运行 9 cell | 原始结果 |
| 3.5 | paired workflow bootstrap 与正式 Gate 判定 | `experiments/g3/g3-verdict.md` |

## G3.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| cache manager 无损版 + controller | 3–4 天 |
| G3-P1 单 cell | validation 参数冻结后仅跑一次 held-out test；目标 replay wall ≤ 3× Oracle-Cost（工程阈值，非论文结论） |
| 全网格运行 | 仅在 P0 与 closed-loop 主 cell 通过后估时 |
| **合计** | **W7–W8 两周内**（与 E2/G5 并行） |

## G3.11 预期产物

| 产物 | 路径 |
|---|---|
| 无损 cache manager + controller v1 | `experiments/g3/` |
| 成本模型 | `experiments/g3/cost-model.json` |
| 结果表与图（p95 TTFT × 预算 × 并发） | `experiments/g3/results/`、`figures/g3-*.png` |
| 判定报告 | `experiments/g3/g3-verdict.md` |

### G3.11.1 结果表格模板（完成后填充，不发明数字）

**表 G3-1：主 cell（预算 25%、并发 8，τ-bench 495 episodes × 3 种子）**

| 策略 | p95 TTFT (ms) | vs FlowCache-Lossless 相对差 | 吞吐 (req/s) | 吞吐相对变化 | block hit | saved-prefill ms | controller 开销 ms |
|---|---|---|---|---|---|---|---|
| No-Cache | TBD | TBD | TBD | TBD | TBD | TBD | — |
| APC-LRU | TBD | TBD | TBD | TBD | TBD | TBD | — |
| GDSF | TBD | TBD | TBD | TBD | TBD | TBD | — |
| SizeCost-LRU | TBD | TBD | TBD | TBD | TBD | TBD | — |
| FlowCache-Lossless | TBD | — | TBD | — | TBD | TBD | TBD |
| Oracle-Cost | TBD | TBD | TBD | TBD | TBD | TBD | — |

**表 G3-2：全 9 cell 摘要（τ-bench；v0.5 移除 BFCL 800 同构主表；StableToolBench 500 出确认表）**

| 预算 | 并发 | FlowCache p95 TTFT | 最佳简单策略 p95 TTFT | 相对改善 [95% CI] | 吞吐变化 [95% CI] | 判定达标? |
|---|---|---|---|---|---|---|
| 10% | 4 | TBD | TBD | TBD | TBD | TBD |
| 10% | 8 | TBD | TBD | TBD | TBD | TBD |
| 10% | 16 | TBD | TBD | TBD | TBD | TBD |
| 25% | 4 | TBD | TBD | TBD | TBD | TBD |
| 25% | 8（主 cell） | TBD | TBD | TBD | TBD | TBD |
| 25% | 16 | TBD | TBD | TBD | TBD | TBD |
| 50% | 4 | TBD | TBD | TBD | TBD | TBD |
| 50% | 8 | TBD | TBD | TBD | TBD | TBD |
| 50% | 16 | TBD | TBD | TBD | TBD | TBD |

**表 G3-3：Toolathlon 真实 tool-wait 敏感性（附表）**

| 等待缩放 | FlowCache p95 TTFT | 最佳简单策略 p95 TTFT | 相对改善 | 结论 |
|---|---|---|---|---|
| ×0.5 | TBD | TBD | TBD | TBD |
| ×1.0（真实） | TBD | TBD | TBD | TBD |
| ×2.0 | TBD | TBD | TBD | TBD |

**表 G3-4：无损质量抽检（closed-loop，20 episodes）**

| 检查项 | 结果 |
|---|---|
| BF16 缓存路径 vs 无缓存的任务成功率差 | TBD（要求 = 0） |
| 输出文本逐 token 一致率 | TBD |
| 工具调用序列一致率 | TBD |

## G3.12 失败动作

协议或公平性缺失：标记 `PROTOCOL-INCOMPLETE`，修复后重跑，不切换路线。只有在协议完整、强基线齐全、closed-loop 指标有效且仍未满足 §G3.8 时，才按 IDEA §7 G3 判路线 A No-Go、转路线 B；实现可保留为工程基线，但不以无损 residency 单独投稿。

## G3.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 动作空间 A₀ | §2.3 |
| G3 通过条件与失败动作 | §7 G3 |
| controller 触发时机与安全水位 | §4.5 |
| 优化目标（无损情形） | §2.4 |
| 预算档位 | §5.3 |

---

# G4：Quantization（量化）

> **周次**：W9–W10 | **依赖**：G0 | **关键路径**：是
> **失败动作**：在 G0 已允许的一次模型/后端切换后仍失败，则路线 A No-Go 并转路线 B；**不能删除量化后继续使用 reuse–fidelity 主标题投稿**

## G4.1 实验目标与 Gate 关系

G4 验证量化路径的**端到端可行性**（IDEA §7 G4）：

1. 真实后端支持目标模型的 KV quantization 与恢复；
2. active runtime 统一材料化为 BF16，staging 峰值和 tainted lineage 均被正确追踪；
3. 量化/反量化不破坏 end-to-end latency；
4. pilot 后预注册绝对质量非劣界、δ 和样本量，且 95% CI 窄到足以检验该界；
5. 至少在一个真实工具 workload 上验证，而不只看 logit KL。

### G4.1.1 G4 不证明的内容

- 不证明复用价值–保真风险错位（G2，G4 是 G2 的输入之一）；
- 不证明 joint controller 的端到端收益（E4）；
- G4 的 D 标签采集协议与 G2-Pilot §4 一致，本章聚焦于**系统可行性与质量非劣**，G2-Pilot 聚焦于相关性判定。

## G4.2 数据集与子集定义

| 数据集 | 样本数 | 用途 |
|---|---|---|
| τ-bench | 495 episodes（全量） | 主验证 workload（满足"至少一个真实工具 workload"；多 seed 提供任务成功率重复测量） |
| LongBench | 1,000 | 长上下文量化质量主战场（ARKV 同类用法） |
| GSM8K | 300 | 量化 accuracy（QKVShare/GraphFlow/ARKV 同类用法） |
| MuSiQue + 2WikiMultihopQA | 各 300 | 多跳 QA 质量 sanity |
| codec block 集 | τ-bench 全量 trace 去重（预计 ≥ 1,000 unique block，TBD） | codec 开销与误差分布（含 G0 的 100 block 作为子集复用） |

## G4.3 样本数与功效分析

G4 的核心统计任务是**非劣检验**：任务成功率变化 Δsuccess 的 95% CI 上界 ≤ 预注册非劣界 ε。

| 项 | 值/规则 |
|---|---|
| 预注册字段 | 非劣界 ε（绝对任务成功率差）、δ = 0.05、样本量 N —— **pilot 后冻结**（IDEA §7 G4 明确要求） |
| ε 建议起点 | Δsuccess ≤ 0.02（2 个百分点）——具体值由 τ-bench BF16 基线成功率与 pilot 方差决定，pilot 后冻结并记录理由 |
| 样本量初估 | τ-bench 495 episodes（165 × 3 seeds）配对二值非劣：基线成功率 0.5–0.7 时，N=495 的 95% CI 半宽 ≈ 4–5 个百分点——检验 ε=0.02 仍需 step 级细化（工具调用步骤级样本预计 ≥ 3,000）或更高 seed 数；pilot 后按实测方差计算最终 N 并预注册 |
| 功效目标 | 1 − β ≥ 0.80；CI 半宽 ≤ ε/2 |
| conformal（如启用） | 独立 calibration split 上经验覆盖率 ≥ 1 − δ；样本量/exchangeability 不足时只能称"经验风险预算"，不宣称硬保证（IDEA §2.2） |

## G4.4 Baseline / 对照

| 对照 | 说明 |
|---|---|
| BF16 无损路径（G3 产物） | 质量与延迟的双重基线 |
| Uniform-Q8 | 全部 inactive block 统一 Q8 |
| Uniform-Q4 | 全部 inactive block 统一 Q4 |
| （仅指标参考）G2-Pilot 的逐 block D 标签 | 一致性交叉核对 |

## G4.5 测试指标

| 类别 | 指标 |
|---|---|
| 系统可行性 | codec 编码/解码延迟（per block）、staging 峰值字节数、量化后 KV pool 容量增益（同预算下可驻留 block 数倍数）、E2E TTFT/JCT 影响（vs BF16 无损路径） |
| lineage 正确性 | tainted lineage 追踪正确率（=100% 要求）、approximate child 隔离正确率、fail-closed 重算触发正确性 |
| 质量（token 级） | logit KL（前 K=64 token 均值）、top-k change |
| 质量（任务级，主判定） | 任务成功率变化 Δsuccess（τ-bench 数据库状态匹配）、LongBench 准确率变化、GSM8K accuracy、QA EM/F1 变化（MuSiQue/2Wiki）、工具调用函数名/参数一致率 |
| 校准（如启用） | 风险上界 D̂^UCB 的经验覆盖率（目标 ≥ 1 − δ） |

## G4.6 运行协议

1. 量化精度档：Q8、Q4，per-block 全层统一精度（IDEA §1.2/§2.3）；
2. codec 必须使用后端实际支持的实现，记录 codecVersion、scaleLayout、checksum（IDEA §1.2 的 E_b 字段）；
3. 恢复时先材料化为 active BF16，预留 staging 与 active-cache 空间（IDEA §2.3）；
4. 质量判定用 **closed-loop**（允许量化影响输出与后续工具调用）；系统开销测量用 open-loop；两者分表（Part 0.7）；
5. 从 approximate lineage 继续生成的 child 保持隔离，默认只供同一 workflow 使用（IDEA §1.2 保守规则）；
6. pilot 先行（~50 episodes），产出方差估计 → 冻结 ε、δ、N → 正式运行。

### G4.6.1 预注册模板（`experiments/g4/preregistration.md` 字段）

| 字段 | 说明 | 冻结值 |
|---|---|---|
| 非劣界 ε | 绝对任务成功率差（τ-bench） | TBD（pilot 后冻结） |
| ε_QA | QA/LongBench/GSM8K 非劣界 | TBD |
| δ | 风险容忍（CI 与覆盖率共用） | 0.05（默认） |
| 评估单位 | workflow 级 / step 级（二选一，含理由） | TBD |
| 样本量 N | 按 pilot 方差计算，满足 CI 半宽 ≤ ε/2 | TBD |
| 重复次数 K | 每 workflow 的量化 replay 次数（若用重复均值；τ-bench 已有 3 seeds） | TBD |
| CI 方法 | paired bootstrap（workflow 级，1000 次）或 Newcombe 配对比例 CI | TBD |
| 延迟噪声界 | Q-storage vs BF16 的 p95 TTFT 允许劣化上限（建议 ≤ 2%） | TBD |
| conformal 是否启用 | 启用/不启用 + 理由 | TBD |
| 冻结日期与签字 | — | TBD |

预注册冻结后：正式运行的分析代码只读冻结值；任何修改触发"方案偏离记录"，并在论文中披露。

## G4.7 统计检验

- 主判定：paired（同 episode BF16 vs 量化）Δsuccess 的 bootstrap 95% CI 上界 vs ε；
- LongBench/GSM8K/QA：paired bootstrap CI；
- token 级：KL/top-k 分布报告（median、p95、max），与 G2-Pilot D 标签交叉核对一致性；
- 覆盖率：calibration split 上的经验覆盖率与二值 CI。

## G4.8 判定阈值（Go/No-Go，对应 IDEA §7 G4）

| # | 条件 | 阈值 |
|---|---|---|
| 1 | 后端支持 | Q8/Q4 KV 量化与恢复在真实后端跑通（G0 spike 的规模化复测，100% block 无 crash） |
| 2 | 材料化与追踪 | active 统一 BF16；staging 峰值 ≤ 预算水位；tainted lineage 追踪 100% 正确 |
| 3 | 延迟不破 | 量化/反量化不破坏 E2E latency：同预算下 Q-storage 路径的 p95 TTFT 不劣于 BF16 无损路径超过噪声界（预注册：pilot 后冻结，建议 ≤ 2%） |
| 4 | 质量非劣 | Δsuccess 95% CI 上界 ≤ ε（预注册）；LongBench/GSM8K/QA 指标下降 CI 上界 ≤ ε_QA（预注册） |
| 5 | CI 宽度 | 95% CI 半宽 ≤ ε/2（否则增大 N 或声明无法检验，不提出非劣 claim） |
| 6 | 真实 workload | 以上在 τ-bench 全量达成（不只 logit KL） |

**全部满足 → G4 passed。** 失败 → 失败动作。

## G4.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 4.1 | Q-storage 集成到 G3 无损 cache manager（动作空间扩展至 A₁） | cache manager 量化版 |
| 4.2 | 全量 block codec 误差/延迟/容量测量（≥1,000 block × Q8/Q4） | codec 测量表 |
| 4.3 | pilot（50 episodes）→ 方差估计 → 预注册 ε、δ、N | 预注册记录 `g4/preregistration.md` |
| 4.4 | 正式 closed-loop 质量运行（τ-bench 495 × Q8/Q4） | 质量结果表 |
| 4.5 | 长上下文/QA 质量运行（LongBench 1,000 + GSM8K 300 + QA 各 300 × Q8/Q4） | 质量结果表 |
| 4.6 | lineage 隔离与 fail-closed 专项测试 | 正确性记录 |
| 4.7 | 统计判定 | `experiments/g4/g4-verdict.md` |

## G4.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| Q-storage 集成 | 2–3 天 |
| codec 全量测量 | ~4–6 小时 |
| closed-loop 质量运行（495 × 2 精度 + 1,900 × 2 精度静态集） | ~12–18 小时 |
| **合计** | **W9–W10 内（与 G2/E3 交错）** |

## G4.11 预期产物

| 产物 | 路径 |
|---|---|
| 预注册记录 | `experiments/g4/preregistration.md` |
| codec 测量表 | `experiments/g4/codec-measurements.csv` |
| 质量非劣结果 | `experiments/g4/quality-results.csv` |
| 判定报告 | `experiments/g4/g4-verdict.md` |

### G4.11.1 结果表格模板（完成后填充，不发明数字）

**表 G4-1：系统可行性（open-loop，τ-bench 全量 trace）**

| 路径 | codec 编码 ms/block (median / p95) | codec 解码 ms/block | staging 峰值 bytes | 同预算可驻留 block 数（vs BF16 倍数） | p95 TTFT vs BF16 无损（相对差 [95% CI]） |
|---|---|---|---|---|---|
| Q8-storage | TBD | TBD | TBD | TBD | TBD |
| Q4-storage | TBD | TBD | TBD | TBD | TBD |

**表 G4-2：任务级质量非劣（closed-loop，主判定表）**

| 数据面 | 精度 | N | BF16 成功率/准确率 | 量化成功率/准确率 | Δ [95% CI] | CI 上界 ≤ ε? |
|---|---|---|---|---|---|---|
| τ-bench | Q8 | 495 | TBD | TBD | TBD | TBD |
| τ-bench | Q4 | 495 | TBD | TBD | TBD | TBD |
| LongBench | Q8/Q4 | 1,000 | TBD | TBD | TBD | TBD |
| GSM8K | Q8/Q4 | 300 | TBD | TBD | TBD | TBD |
| MuSiQue | Q8/Q4 | 300 | TBD (EM/F1) | TBD | TBD | TBD |
| 2WikiMultihopQA | Q8/Q4 | 300 | TBD (EM/F1) | TBD | TBD | TBD |

**表 G4-3：lineage 正确性专项**

| 检查项 | 要求 | 结果 |
|---|---|---|
| tainted lineage 追踪正确率 | 100% | TBD |
| approximate child 跨 workflow 误共享数 | 0 | TBD |
| lineage 缺失时 fail-closed 重算触发率 | 100% | TBD |

## G4.12 失败动作

按 IDEA §7 G4：在 G0 已允许的一次模型/后端切换后仍失败，则路线 A No-Go 并转路线 B；**不能删除量化后继续使用 reuse–fidelity 主标题投稿**。同时按 IDEA §11：量化收益被 codec/PCIe 抵消（端到端无正收益）同样触发 No-Go。更新 ccfa.yaml（G4 → failed，route → B）。

## G4.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| G4 通过条件与失败动作 | §7 G4 |
| 动作空间 A₁、Q-storage 语义 | §2.3 |
| storage encoding E_b / lineage 规则 | §1.2 |
| 风险上界与 conformal 约束 | §2.2 |
| 量化收益被 codec/PCIe 抵消的风险 | §11 风险表 |
| 离线干预回放协议 | §4.4 |

---

# G5：Learning（学习式预测）

> **周次**：W7–W8 | **依赖**：G1、G3 | **关键路径**：否（可选，失败不触发路线切换）
> **失败动作**：保留简单、可解释的 controller，不为论文形式强行加入 GNN

## G5.1 实验目标与 Gate 关系

G5 验证：**学习模型**（survival/hazard、GNN）在**未见 workflow family** 上，相对最佳确定性启发式是否降低 policy regret（IDEA §7 G5）。G5 与 E2 共享预测器实现与标签管线；G5 是判定（gate），E2 是完整评估（experiment）。

### G5.1.1 模型选择顺序（IDEA §4.3，强制）

1. `age + size + measured recompute cost` 启发式（SizeCost）；
2. 校准的 survival/hazard 模型；
3. **只有前两者与 oracle 仍存在明显差距时**，才使用 GNN 编码 partial DAG。

GNN 是可选实现，不是论文贡献本身。

## G5.2 数据集与子集定义

| 数据集 | 角色 | 样本数 |
|---|---|---|
| τ-bench | train / validation / in-family test（episode 级 group split） | 495 episodes：train 297 / val 99 / test 99（按 task 分组，同 task 的 3 seeds 必同 split；prefix dedup；种子冻结） |
| StableToolBench | **workflow-family-out test 通道 ①**（G5 主判定面） | 500（全量只作 test，绝不参与训练/调参） |
| SWE 轨迹 | **workflow-family-out test 通道 ②**（结构差异最大的 family） | 500（全量只作 test） |
| LMSYS-Chat-1M | 负对照（仅追加式会话，next-use 接近确定） | 2,000（仅描述性报告：学习预测器不应优于简单规则） |

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn 行（train 480 / val 160 / test 160）已删除——BFCL 不再作为数据集。

标签与特征：与 E2 共享（来自 open-loop replay 的真实 exact-prefix block access；特征集见 E2 章，遵守 Part 0.5 禁止特征清单）。

## G5.3 样本数与功效分析

| 项 | 值 |
|---|---|
| 训练样本 | τ-bench train 297 episodes → 预计 unique block 的正/负 next-use 标签数 TBD（v0.5 移除 BFCL train 480 后样本量下降，必要时 rebuttal 补 BFCL train 扩充） |
| 判定效应量 | 净端到端收益 ≥ ~5%，或 policy regret 改善 ≥ ~10%（IDEA §7 G5 内部参考，均含模型推理开销） |
| 功效规则 | family-out test 500（通道①）/ 500（通道②）paired 比较；按 0.8.3 规则评估 CI 宽度，不足时增加 replay 种子压缩方差 |
| 负对照期望 | LMSYS 上学习模型相对简单规则的 regret 差 ≈ 0（若显著为负，说明学习器过拟合工具型分布） |

## G5.4 Baseline / 对照

| 对照 | 说明 |
|---|---|
| SizeCost（最佳确定性启发式） | G5 的比较基准 |
| Survival/Hazard（校准） | 学习档 1：离散时间 hazard 模型，输出 P(T_b^next ≤ h) 多窗口概率 + 校准区间 |
| GNN（条件启用） | 学习档 2：partial-DAG 编码；仅当 survival 与 oracle 仍有明显差距时启用（IDEA §4.3） |
| Oracle-Cost | regret 的离线 lookahead 参照；非严格最优上界 |

### G5.4.1 Survival/Hazard 模型规格（预注册式）

| 项 | 规格 |
|---|---|
| 形式 | 离散时间 hazard：逻辑斯蒂输出头，hazard bin 边界 h ∈ {1, 4, 16, 64, 256, 1000} step（覆盖 H 窗口） |
| 输入 | E2 章 18 项决策时可见特征（禁用特征清单见 Part 0.5） |
| 训练目标 | 逐 bin 二值交叉熵 + 成本加权（byte-weighted） |
| 校准 | validation 上 isotonic 回归逐 bin 校准；报告校准前后 ECE |
| 推理开销预算 | ≤ 1 ms/block-decision（CPU 推理；超出则在 G5.8 净收益中如实扣减） |
| 超参搜索 | 仅 train/val 上网格搜索，搜索空间与选中值记录到 `experiments/g5/hparams.json` |

## G5.5 测试指标

| 类别 | 指标 |
|---|---|
| 主判定 | policy regret（vs Oracle-Cost，同 trace 同预算）相对 SizeCost 的改善比例；净端到端收益（saved-prefill ms − 全部开销，含预测器推理耗时） |
| 预测质量 | PR-AUC、Brier、ECE、byte/cost-weighted recall、Precision@budget |
| 开销 | 预测器单次推理耗时（ms/decision）、训练时间、特征提取开销 |
| 负对照 | LMSYS 上学习 vs SizeCost 的 regret 差 |

## G5.6 运行协议

- 训练/校准只用 train/val（τ-bench）；阈值与超参在 val 上选择；test（in-family + 两条 family-out 通道）只跑一次冻结配置（Part 0.6）；
- 系统面评估在 open-loop replay 中把各预测器接入 G3 的 controller（动作空间 A₀），同 trace 同预算；
- 预测器推理开销计入 controller overhead（IDEA §4.5：论文必须同时报告收益和自身成本）。

## G5.7 统计检验

- 主比较：Survival（或 GNN）vs SizeCost 的 per-workflow regret 配对差，paired bootstrap 95% CI；
- 两条 family-out 通道与 in-family 分表报告；
- 负对照报告点估计 + CI，不设显著性门槛。

## G5.8 判定阈值（Go/No-Go，对应 IDEA §7 G5）

| 条件 | 阈值 |
|---|---|
| family-out regret | 学习模型在**至少一条** family-out 通道（StableToolBench 或 SWE）上相对 SizeCost 降低 policy regret（目标 ~10%，CI 不含 0），且另一条通道不显著劣化 |
| 净收益 | 净端到端收益 ≥ ~5%（含推理开销） |

**满足 → G5 passed，保留学习预测器（并按 §4.3 顺序决定是否需要 GNN）。** 失败 → 仅删除学习档/GNN，保留 SizeCost + controller，路线 A 继续（不切换路线）。

## G5.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 5.1 | 标签/特征管线（与 E2 共建） | 标签表 |
| 5.2 | survival/hazard 训练 + 校准（train/val） | 预测器 v1 |
| 5.3 | 接入 controller，in-family + 双通道 family-out 系统评估 | 结果表 |
| 5.4 | （条件）GNN 启用判定与实现 | 预测器 v2（可选） |
| 5.5 | LMSYS 负对照 | 负对照记录 |
| 5.6 | 判定 | `experiments/g5/g5-verdict.md` |

## G5.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 管线 + survival | 2 天 |
| 系统评估（复用 G3 网格子集） | ~6 小时（数据量上升） |
| （条件）GNN | +2–3 天 |
| **合计** | **W7–W8 内（与 G3/E2 并行）** |

## G5.11 预期产物

| 产物 | 路径 |
|---|---|
| 预测器与训练记录 | `experiments/g5/` |
| family-out 结果表 | `experiments/g5/family-out-results.csv` |
| 判定报告 | `experiments/g5/g5-verdict.md` |

### G5.11.1 结果表格模板（完成后填充，不发明数字）

**表 G5-1：family-out 主判定表（通道①：train τ-bench → test StableToolBench 500；v0.5 移除 BFCL train 扩充）**

| 预测器 | policy regret (ms) | vs SizeCost regret 改善 [95% CI] | 净端到端收益（含推理开销）[95% CI] | 推理耗时 ms/decision | 达标? |
|---|---|---|---|---|---|
| SizeCost | TBD | — | TBD | ~0 | — |
| Survival（校准） | TBD | TBD | TBD | TBD | TBD |
| GNN（如启用） | TBD | TBD | TBD | TBD | TBD |
| Oracle-Cost | 0（参照） | — | — | — | — |

**表 G5-2：family-out 通道②（test SWE 500，同构表）**。

**表 G5-3：in-family test（τ-bench test 99；v0.5 移除 BFCL test 160 同构附表）**。

**表 G5-4：LMSYS 负对照（2,000 会话）**

| 预测器 | regret (ms) | vs SizeCost 差 [95% CI] | 期望 |
|---|---|---|---|
| SizeCost | TBD | — | — |
| Survival | TBD | TBD | ≈ 0（无伪收益） |
| GNN（如启用） | TBD | TBD | ≈ 0 |

## G5.12 失败动作

按 IDEA §7 G5：保留简单、可解释的 controller，不为论文形式强行加入 GNN。ccfa.yaml 中 G5 → failed，但 `triggers_route_switch: false`，路线 A 继续。

## G5.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| G5 通过条件与失败动作 | §7 G5 |
| 模型选择顺序与禁止特征 | §4.3 |
| 负对照（纯追加会话） | §1.1、§6.1 |
| 训练指标要求（不只 ROC-AUC） | §4.3 |

---

# Part 2：正式实验设计

# E1：缓存机会与工作负载画像

> **周次**：W7（v0.2 起，原 W6 顺延） | **对应 Gate**：G1 | **实验状态**：τ-bench 管线代码已有基础（`experiments/e1/`），需扩展至全数据集组合
> **定位**：中心 claim 的前提，**不放在附录**（IDEA §8 E1）
>
> **v0.3 对齐说明（2026-07-25）**：按 spec v0.3，**E1 已并入 Ch.1（与 G1 合并）**。本章保留作为画像方法的详细参考（指标定义、序列化规则、执行步骤模板）。**数据集范围以 G1.2 为准**（仅 τ-bench 495 + BFCL 800；v0.5 移除 BFCL，仅 τ-bench）；本章 E1.2 表格列出的 12 数据集 v0.2 计划已被 spec v0.3 §5 精简为 7 核心 + 2 辅助，主表用量见 G1.2，Ch.5 用量见 spec v0.3 §5。原 E1.10 的 ~33–40 GPU 小时 Tier-1 录制预算，v0.3 缩减为 ~20 GPU 小时（spec v0.3 §8）。

## E1.1 实验目标

**问题**：真实 workload 中是否存在值得管理的 exact-prefix locality？（IDEA §8 E1）

E1 是描述性 + 测量性实验，为 G1 判定提供中心证据，同时为后续所有实验提供 trace 基础与 workload 画像。E1 不评价任何 FlowCache 组件。

## E1.2 数据集与子集定义

使用 Part 0.4.1 全部四层组合：

| 数据集 | 样本数 | E1 中的角色 |
|---|---|---|
| τ-bench | 495 episodes（rollout 录制） | 主画像对象 ① |
| StableToolBench | 500（rollout 录制 + 核验） | 主画像对象 ③ |
| SWE-rebench-openhands-trajectories | 500（真实轨迹整理） | 结构画像：真实重试/分支（接管原合成 DAG 角色） |
| Toolathlon-Trajectories | 500（真实轨迹整理） | 工具等待时间画像（真实时间戳） |
| CATraces | ~100–200（TBD 实测） | 真实 coding-agent 长会话画像 |
| LongBench | 1,000 | 长上下文画像（静态，无 rollout） |
| MuSiQue + 2WikiMultihopQA | 各 300 | 多跳 QA 画像 |
| GSM8K | 300 | 短链 QA 画像 |
| LMSYS-Chat-1M | 2,000 | 负对照画像（仅追加式） |
| BurstGPT | 2,000 会话窗口 | 真实到达/并发结构画像 |
| Mooncake trace | 抽样（TBD 窗口） | 大规模前缀块 hash 共享画像 |

## E1.3 样本数与功效分析

- E1 以**描述统计**为主，无单一假设检验；功效分析服务于其支撑的 G1 判定（见 G1.3）；
- 画像指标报告分布（median、p5/p95）而非仅均值；
- 每数据集的 unique block 数、overlap 等若显示 locality 退化为 ~0，如实报告（IDEA §6.2 硬约束），不人为挑选高 locality 子集；
- 总量 ~8,800 workflow 级样本，各层 N 见 E1.2 表。

## E1.4 Baseline / 对照

E1 的"对照"是机会空间的上下界（与 G1 共享实现）：

| 对照 | 角色 |
|---|---|
| APC-LRU / LRU / GDSF / SizeCost | 简单策略参照点 |
| Oracle-Belady / Oracle-Cost | 前者为统一 miss-cost 下的离线参照；后者为 cost-aware lookahead heuristic，单独报告且不称严格上界 |

## E1.5 测试指标（IDEA §8 E1 全量）

| 类别 | 指标 |
|---|---|
| 结构画像 | workflow 长度、深度、宽度、分支率、工具等待时间分布 |
| locality 画像 | exact-prefix overlap、LCP tokens、next-use distance 分布、share_count 分布 |
| 资源画像 | block working-set size、KV/总显存占比 |
| 机会画像 | offline oracle 与 LRU/简单 heuristic 的 headroom（miss cost、p95 TTFT 差距 × 预算档位 10/25/50/100%） |

### E1.5.1 画像指标形式化定义（统一口径，防歧义）

| 指标 | 定义 |
|---|---|
| workflow 长度 | 该 workflow 的总 step 数（user/assistant/tool_call/tool_result 事件计数） |
| 深度 | DAG 最长路径上的节点数（真实轨迹按工具调用链近似并说明近似方法） |
| 宽度 | 同一层可并行分支的最大数（真实轨迹按同时刻并行分支近似，记为 TBD-approx） |
| 分支率 | 含 ≥ 2 个已声明后继的节点占比（SWE 轨迹按重试/分叉事件计） |
| 工具等待 | 工具调用发出到结果返回的 wall-clock（Toolathlon 时间戳实测；rollout 数据集为录制实测） |
| exact-prefix overlap | （共享 block 的 token 总数）/（全部 block 的 token 总数）；共享 = share_count ≥ 2 |
| LCP tokens | 每对（workflow_i, workflow_j）最长公共前缀 token 数的分布（同域/跨域分别报告） |
| next-use distance | block 从变 inactive 到下次 exact-prefix 访问的 step 数；未再访问记 ∞ 并单独计数 |
| working-set size | 滑动窗口（H=1000 step）内被访问过的 unique block KV 字节数峰值 |
| KV/总显存占比 | KV pool 峰值 / GPU reserved 峰值 |

## E1.6 运行协议与现有代码对齐

E1 复用并扩展已有管线（`experiments/e1/`：`record_trajectories.py`、`trace_utils.py`、`characterize_workload.py`、`compare_oracle.py`、`plot_characterization.py`，配置 `config.yaml`：block_size 16、预算 [0.10, 0.25, 0.50, 1.00]、H=1000、β=0.005、输出 traces/bf16 + outputs + figures）：

| 步骤 | 脚本 | 产物 |
|---|---|---|
| BF16 轨迹录制（τ-bench/StableToolBench rollout） | `record_trajectories.py`（扩展多数据集适配） | `experiments/e1/traces/bf16/<dataset>/*.json` |
| 真实轨迹整理（SWE/Toolathlon/CATraces → 统一 trace 格式） | 新增 `import_real_traces.py` | `experiments/e1/traces/real/<dataset>/` |
| 静态文本整理（LongBench/QA/GSM8K/LMSYS/BurstGPT/Mooncake） | 新增 `import_static.py` | `experiments/e1/traces/static/<dataset>/` |
| block identity/父链/去重 | `trace_utils.py` | block 索引 |
| 画像统计 | `characterize_workload.py` | `experiments/e1/outputs/`（json + markdown） |
| oracle vs 简单策略 headroom | `compare_oracle.py` | headroom 表 |
| 图表生成 | `plot_characterization.py` | `experiments/e1/figures/` |

**扩展要求**（W7 执行时落地）：

1. 全部 12 个数据集按相同 block_size=16、H=1000、β=0.005 处理；到达结构主证据 BurstGPT，Poisson λ=4 为建模参照（报告拟合优度）；
2. 每个数据集输出 cache-compatible 序列化规则文档（Part 0.5 清单，模板见 E1.6.1）；
3. 无泄漏 split（Part 0.6）在画像阶段同步产出并冻结；
4. 0.4.3 的核验报告与本章画像一并产出。

### E1.6.1 序列化规则模板（τ-bench 示例；其余 11 个数据集照此模板补齐）

| 规则项 | τ-bench 约定 |
|---|---|
| 进入 prompt 的事件 | system（domain policy + tool schema）、user 消息、assistant 回复、tool_call、tool_result，按时间序 |
| 顺序/格式稳定性 | 固定 role 顺序与分隔符；JSON key order 固定；tool schema 序列化固定 |
| 分支重序列化 | 用户模拟器分支不发生时历史不变；重试时以"追加新事件"方式而非重写历史（若原 benchmark 行为不同，如实记录） |
| 共同前缀 | 同域任务共享 system + policy + tool schema 前缀；跨域不共享 |
| resume 一致性 | 工具结果返回后续写完全相同的 prefix（录制 trace 逐字节核验） |
| invalidation 触发 | policy 文档/模板/tool schema 任何变化 → 变化点后全链失效 |

任何数据集无法满足"resume 一致性"的，该数据集只用于画像，不用于策略评估。

## E1.7 统计检验

- 描述性分布 + workflow-level bootstrap 95% CI；
- 跨数据集对比只作描述，不做因果推断（SWE 分层内部可做结构参数–locality 的剂量–反应描述）。

## E1.8 成功标准（非 Go/No-Go，判定在 G1）

| 标准 | 说明 |
|---|---|
| 画像完整性 | 12 个数据集的 E1.5 全指标齐备 |
| trace 可重放 | 全部 trace 冻结 token IDs/工具结果/到达时间，满足 open-loop 要求；不可重放则阻塞 G1 |
| 诚实报告 | locality 低的数据集如实报告，不制造伪命中 |

## E1.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 1.1 | τ-bench 495 / StableToolBench 500 rollout 录制（W3–W5 窗） | traces/bf16/ |
| 1.2 | SWE/Toolathlon/CATraces 真实轨迹整理 + 核验（0.4.3） | traces/real/ + 核验报告 |
| 1.3 | 静态集整理（LongBench/QA/GSM8K/LMSYS/BurstGPT/Mooncake） | traces/static/ |
| 1.4 | 画像 + oracle headroom（复用管线） | outputs/figures |
| 1.5 | 各数据集序列化规则文档 + 无泄漏 split 冻结 | split 记录 |
| 1.6 | 汇总画像报告（作为 G1 输入） | `experiments/e1/e1-report.md` |

## E1.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| Tier 1 rollout 录制（995 episodes，v0.5 移除 BFCL 800 后：τ-bench 495 + STB 500） | ~20–27 GPU 小时（W3–W5） |
| Tier 2/4 下载整理与核验 | ~2 天（CPU/IO 为主） |
| 画像与图表 | ~0.5 天 |
| **合计** | **W7 前完成画像汇总** |

## E1.11 预期产物

| 产物 | 路径 |
|---|---|
| 12 个数据集可重放 trace | `experiments/e1/traces/{bf16,real,static}/` |
| 核验报告 | `experiments/e1/verification-report.md` |
| 画像报告 | `experiments/e1/e1-report.md`（中心证据，不放附录） |
| 图表 | `experiments/e1/figures/` |
| split 冻结记录 | `experiments/e1/splits.json` |

### E1.11.1 结果表格模板（完成后填充，不发明数字）

**表 E1-1：结构画像（12 数据集）**

| 数据集 | N | 长度 (median / p95) | 深度 | 宽度 | 分支率 | 工具等待 ms (median / p95) |
|---|---|---|---|---|---|---|
| τ-bench | 495 | TBD | TBD | TBD | TBD | TBD |
| StableToolBench | 500 | TBD | TBD | TBD | TBD | TBD |
| SWE 轨迹 | 500 | TBD | TBD | TBD | TBD | TBD |
| Toolathlon | 500 | TBD | TBD | TBD | TBD | TBD（真实时间戳） |
| CATraces | ~150 | TBD | TBD | TBD | TBD | TBD |
| LongBench | 1,000 | TBD | — | — | — | — |
| MuSiQue | 300 | TBD | TBD | — | — | — |
| 2WikiMultihopQA | 300 | TBD | TBD | — | — | — |
| GSM8K | 300 | TBD | — | — | — | — |
| LMSYS-Chat-1M | 2,000 | TBD | — | — | — | — |
| BurstGPT | 2,000 窗口 | TBD | — | — | — | — |

**表 E1-2：locality 与机会画像（12 数据集）**

| 数据集 | exact-prefix overlap | LCP tokens (median / p95) | next-use distance (median / p95 / ∞占比) | working-set 峰值 (MB) | KV/显存占比 | oracle vs LRU headroom @ 预算 25% |
|---|---|---|---|---|---|---|
| τ-bench | TBD | TBD | TBD | TBD | TBD | TBD |
| StableToolBench | TBD | TBD | TBD | TBD | TBD | TBD |
| SWE 轨迹 | TBD | TBD | TBD | TBD | TBD | TBD |
| Toolathlon | TBD | TBD | TBD | TBD | TBD | TBD |
| CATraces | TBD | TBD | TBD | TBD | TBD | TBD |
| LongBench | TBD | TBD | TBD | TBD | TBD | TBD |
| MuSiQue | TBD | TBD | TBD | TBD | TBD | TBD |
| 2WikiMultihopQA | TBD | TBD | TBD | TBD | TBD | TBD |
| GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| LMSYS-Chat-1M | TBD | TBD | TBD | TBD | TBD | TBD（负对照期望：高 overlap、低预测价值） |
| BurstGPT | TBD | TBD | TBD | TBD | TBD | TBD |

**表 E1-3：到达结构（BurstGPT vs Poisson 拟合优度）**

| 指标 | BurstGPT 实测 | Poisson λ=4 拟合 | 拟合优度 |
|---|---|---|---|
| 到达间隔分布 | TBD | TBD | TBD |
| 并发度分布 | TBD | TBD | TBD |
| burst 系数（variance/mean） | TBD | TBD | TBD |

## E1.12 失败/异常处理

- 若画像显示 exact-prefix overlap 过低或 oracle headroom 很小 → 反映 G1 未真正通过，回溯 G1 判定；
- 若某数据集核验（0.4.3）不通过 → 按降级规则处理并如实标注。

## E1.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| E1 指标清单 | §8 E1 |
| 数据集组合与角色 | §6.1（v0.2 用户扩展：四层组合） |
| 序列化规则 | §6.2 |
| 数据切分 | §6.3 |
| open-loop replay | §6.4 |

---

# E2：复用价值预测

> **周次**：W7–W8 | **对应 Gate**：G5
> **定位**：Reuse-Value Estimator 的完整评估；准确率提升若不能转换为系统收益，不构成贡献（IDEA §8 E2）

## E2.1 实验目标

评估 IDEA §4.3 的复用价值预测目标：

- 多时间窗 next-use hazard：P(T_b^next ≤ h)；
- 预期 saved-prefill ms；
- 预测置信度或校准区间。

G5 已做 Go/No-Go 判定；E2 提供完整的预测质量画像、成本加权指标与系统转换（regret/saved-prefill），回答"预测质量沿决策相关维度如何分布、在何处失败"。

## E2.2 数据集与子集定义

| 数据集 | 角色 | 样本数 |
|---|---|---|
| τ-bench | train 297 / val 99 / test 99（episode 级 group split，同 task 的 3 seeds 同 split，种子冻结，与 G5 共用） | 495 |
| StableToolBench | family-out test 通道① | 500 |
| SWE 轨迹 | family-out test 通道② + **结构分层敏感性分析**（接管原合成 DAG 角色：按深度/宽度/重试次数分层做剂量–反应） | 500 |
| Toolathlon 轨迹 | 工具等待维度的补充 test 面 | 500 |
| LMSYS-Chat-1M | 负对照（描述性） | 2,000 |

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn 行（train 480 / val 160 / test 160）已删除——BFCL 不再作为数据集。

**标签**（来自 open-loop replay 的真实 exact-prefix block access，IDEA §4.3）：T_b^next、reused_b = 1(T_b^next ≤ H)、saved_tokens_b、saved_ms_b、share_count_b（定义同 g2-pilot §3.1）。

**特征**（决策时刻 t 可见，IDEA §4.3 + g2-pilot §3.4 的 18 项特征全集；禁止特征见 Part 0.5）：

| 类别 | 特征 | 说明 |
|---|---|---|
| block | block_size | token 数（默认 16） |
| block | ancestor_depth | 父链深度（从根到该 block 的 block 数） |
| block | recency_last_access | 上一次被访问到现在的 step 数 |
| block | historical_access_count | 截至时刻 t 的历史访问次数 |
| block | measured_prefill_ms | 实测的该 block prefill 时间 |
| workflow | completed_nodes | 当前 workflow 已完成节点数 |
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
| （派生） | size × recency、size × cost 交互项 | 启发式基线的向量化形式，记录派生规则 |

## E2.3 样本数与功效分析

| 项 | 值 |
|---|---|
| block 级标签 | train 297 episodes（τ-bench；v0.5 移除 BFCL 480 后样本量下降）→ 预计 unique block 数 TBD 实测；test 99 + family-out 1,000 |
| 功效 | 按 0.8.2：test N ≥ 250 block 可检 ρ ≥ 0.20（power=0.80）；系统面判定沿用 G5 功效规则（0.8.3） |
| 聚类处理 | block 级指标同时报告 per-workflow 聚合（0.8.1） |

## E2.4 Baseline / 对照变体（IDEA §8 E2 全量）

| 变体 | 说明 |
|---|---|
| age/LRU | 最低档 |
| size/recompute-cost heuristic（SizeCost） | 确定性启发式 |
| survival/hazard（校准） | 学习档 1 |
| partial-DAG GNN | 学习档 2，仅 G1/G5 证明必要时启用 |
| Oracle（未来已知） | 上界 |

## E2.5 测试指标（IDEA §8 E2 全量）

| 类别 | 指标 |
|---|---|
| 排序/分类质量 | PR-AUC（报告，**不只 ROC-AUC**，IDEA §4.3） |
| 校准 | Brier、ECE（可靠性图数据一并产出） |
| 成本加权 | byte-weighted recall、recompute-cost-weighted recall |
| 预算相关 | Precision@budget（budget = 各 KV 预算档位对应的保留容量） |
| 系统转换 | policy regret（vs Oracle-Cost）、saved-prefill ms（接入 G3 controller 实测） |
| 开销 | 单次推理耗时、特征提取耗时、训练时长 |

## E2.6 运行协议

- 标签采集：open-loop replay（与 G1/E1 同 trace，BurstGPT 主到达 + Poisson 参照，3 种子）；
- 训练/调参：train/val only；test 冻结后单次运行；
- 系统转换评估：各变体接入 G3 controller（动作空间 A₀），在 budget 25%、concurrency 8 主 cell + 全预算档位扫描；
- SWE 分层敏感性：按深度/宽度/重试次数分层，报告各预测器的性能–结构参数关系（真实结构分层分析，非合成生成）。

## E2.7 统计检验

- block 级指标：bootstrap 95% CI（workflow 为重采样单位）；
- 变体间比较：paired workflow-level bootstrap；
- family-out 双通道单独成表；
- 报告全部种子方差。

## E2.8 成功标准

| 标准 | 说明 |
|---|---|
| 完整性 | E2.4 全部已启用变体 × E2.5 全指标齐备 |
| 系统转换 | 报告每个变体的 regret 与 saved-prefill ms（不允许只报预测指标） |
| 校准 | survival/GNN 必须报告 ECE；校准失败则结论限定为"排序可用、概率不可用" |

## E2.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 2.1 | 标签/特征管线（与 G5 共建，复用 E1 trace） | `experiments/e2/labels.csv`、`features.csv` |
| 2.2 | 四个变体实现与训练 | predictors |
| 2.3 | 预测质量评估（in-family + 双通道 family-out） | 指标表 |
| 2.4 | 系统转换评估（接入 controller） | regret/saved-prefill 表 |
| 2.5 | SWE 分层敏感性 + Toolathlon 补充面 + LMSYS 负对照 | 敏感性曲线 |
| 2.6 | 汇总报告 | `experiments/e2/e2-report.md` |

## E2.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 管线 + 启发式 + survival | 2 天（与 G5 并行共建） |
| 评估 + 系统转换 | ~6 小时（数据量上升） |
| （条件）GNN | +2–3 天 |
| **合计** | **W7–W8 内** |

## E2.11 预期产物

| 产物 | 路径 |
|---|---|
| 标签/特征表 | `experiments/e2/labels.csv`、`experiments/e2/features.csv` |
| 指标与系统转换表 | `experiments/e2/results/*.csv` |
| 图（PR 曲线、可靠性图、敏感性曲线） | `figures/e2-*.png` |
| 报告 | `experiments/e2/e2-report.md` |

### E2.11.1 结果表格模板（完成后填充，不发明数字）

**表 E2-1：预测质量主表（in-family test / family-out 通道① StableToolBench / 通道② SWE 三列）**

| 变体 | PR-AUC (in / ① / ②) | Brier (in / ① / ②) | ECE (in / ① / ②) | byte-weighted recall | cost-weighted recall | Precision@budget (10%/25%/50%) |
|---|---|---|---|---|---|---|
| age/LRU | TBD | TBD | TBD | TBD | TBD | TBD |
| SizeCost | TBD | TBD | TBD | TBD | TBD | TBD |
| Survival（校准） | TBD | TBD | TBD | TBD | TBD | TBD |
| GNN（如启用） | TBD | TBD | TBD | TBD | TBD | TBD |

**表 E2-2：系统转换表（接入 G3 controller，预算 25%、并发 8）**

| 变体 | policy regret (ms) | saved-prefill ms | 预测器推理耗时 | 特征提取耗时 | 净收益 [95% CI] |
|---|---|---|---|---|---|
| age/LRU | TBD | TBD | ~0 | ~0 | TBD |
| SizeCost | TBD | TBD | ~0 | ~0 | TBD |
| Survival | TBD | TBD | TBD | TBD | TBD |
| GNN（如启用） | TBD | TBD | TBD | TBD | TBD |
| Oracle | 0（参照） | TBD（上界） | — | — | — |

**表 E2-3：SWE 真实结构分层敏感性（剂量–反应）**

| 分层维度 | 分层档位 | SizeCost regret | Survival regret | 结论 |
|---|---|---|---|---|
| 轨迹深度（轮数） | TBD 分桶 | TBD | TBD | TBD |
| 重试次数 | TBD 分桶 | TBD | TBD | TBD |
| 分支宽度 | TBD 分桶 | TBD | TBD | TBD |

## E2.12 失败/异常处理

- 预测指标高但系统转换无收益 → 如实报告，结论限定为"准确率不构成贡献"（IDEA §8 E2 末句）；
- GNN 无净收益 → 按 G5 失败动作删除 GNN，报告保留在 E2 作为负结果。

## E2.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 预测目标与特征/禁止特征 | §4.3 |
| E2 变体与指标 | §8 E2 |
| 标签定义（R 分量） | §2.1（经 g2-pilot §3.1 形式化） |
| 负对照 | §1.1、§6.1 |

---

# E3：保真风险估计

> **周次**：W9、W11 | **对应 Gate**：G4
> **定位**：Fidelity-Risk Estimator 的完整评估（IDEA §4.4）；没有干预回放时，不把 attention 权重直接当作质量真值

## E3.1 实验目标

评估保真风险估计的四个变体（IDEA §8 E3），回答：

1. 哪种风险信号（uniform / 静态规则 / norm-range proxy / 学习式估计器）最准确地预测量化损伤？
2. token 级风险是否传导到任务级（工具调用、最终状态、任务成功率）？
3. 风险估计的校准质量如何（后续 controller 的 D̂^UCB 是否可用）？

G4 判定量化路径的系统可行性与任务级非劣；E3 提供**风险估计方法学**的完整对比。D 标签采集协议与 G2-Pilot §4 一致并扩展。

## E3.2 数据集与子集定义

| 数据集 | 样本数 | 角色 |
|---|---|---|
| τ-bench | 495 episodes（全量，unique block 预计 ≥ 1,000 × Q8/Q4） | 主评估（干预回放 + closed-loop 任务质量） |
| LongBench | 1,000 | 长上下文质量面（量化对长文档 KV 的损伤） |
| GSM8K | 300 | accuracy 质量面 |
| MuSiQue + 2WikiMultihopQA | 各 300 | 多跳 QA 质量 sanity |
| 校准 split | train/val 内独立划分的 calibration 子集（比例 TBD，预注册） | D̂^UCB / conformal 标定（IDEA §2.2） |

**D 标签分量**（g2-pilot §4.1 全量）：KL_{b,q}、topk_change_{b,q}、tool_name_changed_{b,q}、tool_params_changed_{b,q}、Δsuccess_{b,q}；K=64。

**在线特征**（IDEA §4.4 + g2-pilot §4.5）：block 位置/长度/role-type、K/V 范数、range、方差、outlier 比例、跨层 max/quantile/直方图摘要、量化尺度统计、离线标定的模型级敏感度先验；可选 kernel 内流式 sketch。**禁止** output_attentions=True。

## E3.3 样本数与功效分析

| 项 | 值 |
|---|---|
| block 级干预 | τ-bench 全量 unique block（预计 ≥ 1,000）× 2 精度 = ≥ 2,000 次干预回放 |
| 任务级 | τ-bench 495 × 2 精度 closed-loop；LongBench 1,000 + GSM8K 300 + QA 600 × 2 精度 |
| 功效 | token 级（连续 KL）：N ≥ 1,000，功效充足（0.8.2）；任务级（二值）：按 G4.3 的预注册 N 与 seed 重复 |
| 聚类处理 | per-block 与 per-workflow 聚合双报告（0.8.1） |

## E3.4 Baseline / 对照变体（IDEA §8 E3 全量）

| 变体 | 说明 |
|---|---|
| uniform precision | 不估计风险，统一 Q8 或 Q4（对照下界/容量上界） |
| 静态 layer/position rule | 按 block 位置/role 的固定规则选择精度 |
| norm/range proxy | 用 K/V 范数、range、outlier 比例的阈值规则 |
| FlowCache fidelity estimator | 学习式（在线特征 → D̂_{b,q}，含校准/UCB） |
| （真值）离线干预回放标签 | 非策略，作为各变体的评估基准 |

### E3.4.1 变体实现规格（预注册式）

| 变体 | 规则/训练细节 |
|---|---|
| 静态 rule | 规则集（预注册）：system 前缀块 → BF16；assistant 推理块 → Q8；tool_result 块 → Q4；其余按 ancestor_depth 阈值（深度 > d* → Q4，d* 在 val 上选）。规则在执行前写入 `experiments/e3/static-rules.json` |
| norm/range proxy | 特征：K/V 范数均值、range、outlier 比例（g2-pilot §4.5）；决策：单特征阈值或线性打分，阈值在 train/val 上以"预测 KL 的 Spearman ρ 最大化"选择，搜索网格记录 |
| FlowCache estimator | 学习式回归（梯度提升或小型 MLP，train/val only），输出 D̂_{b,q}（连续 KL 预测）+ 校准（isotonic）；推理开销预算 ≤ 1 ms/block |

## E3.5 测试指标（IDEA §8 E3 全量）

| 类别 | 指标 |
|---|---|
| token 级 | logit KL、top-k change（各变体预测风险 vs 真值风险的相关与误差：Spearman ρ、MAE） |
| 任务级 | τ-bench 成功率、LongBench 准确率、GSM8K accuracy、QA EM/F1、工具调用正确率、最终状态一致率（各变体精度分配下的 closed-loop 质量） |
| 校准 | 风险校准（ECE / reliability）；D̂^UCB 经验覆盖率（目标 ≥ 1−δ，IDEA §2.2） |
| 系统 | codec latency、实际容量收益（同预算可驻留 block 数倍数）、风险估计自身开销（ms/block） |

## E3.6 运行协议

1. **干预回放（open-loop）**：与 g2-pilot §4.2 相同协议（编码→解码→冻结 continuation 单次 forward→KL；工具调用一致性允许 decode 首个 tool call；tool 未变化则 Δsuccess=0 不重放后端）；
2. **closed-loop 质量**：各变体的精度分配策略接入完整系统，跑 τ-bench 495 与静态质量集（LongBench/GSM8K/QA），记录任务级指标（与干预回放分表）；
3. 校准：calibration split 上标定 D̂^UCB（split-conformal 或明确报告覆盖率的方法，IDEA §2.2）；calibration 与 test 严格分离；
4. 保守回退验证：风险不确定时选择 BF16/驱逐的回退路径被正确触发（IDEA §4.4）。

## E3.7 统计检验

- 变体排序：Spearman ρ（预测风险 vs 真值 KL）+ Kendall τ，Bonferroni 校正（Q8/Q4 两族）；
- 任务级：paired bootstrap 95% CI；
- 覆盖率：经验覆盖率 + 二值 CI；不足时只能称"经验风险预算"（IDEA §2.2 措辞约束）；
- per-workflow 聚合复核（防 Simpson's paradox）。

## E3.8 成功标准

| 标准 | 说明 |
|---|---|
| 方法学结论 | 明确哪个变体在 token 级与任务级均最优，或如实报告"proxy 与学习式无显著差异" |
| 传导性 | 报告 token 级风险 → 任务级质量的传导率（KL 高但 tool call 不变的良性比例） |
| 校准可用性 | D̂^UCB 覆盖率达到 ≥ 1−δ 或如实降级表述 |

## E3.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 3.1 | D 标签全量采集（复用 G2-Pilot 管线，扩至 τ-bench 495 全量 × Q8/Q4） | `experiments/e3/d-labels-*.csv` |
| 3.2 | 在线特征提取 | 特征表 |
| 3.3 | 四个变体实现（含学习式估计器训练，train/val only） | estimators |
| 3.4 | token 级评估（预测 vs 真值） | 相关/误差表 |
| 3.5 | 校准与 D̂^UCB 标定 | 校准报告 |
| 3.6 | closed-loop 任务级评估（τ-bench + LongBench + GSM8K + QA） | 任务级结果表 |
| 3.7 | 汇总报告 | `experiments/e3/e3-report.md` |

## E3.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| D 标签采集（≥1,000 block × 2 精度） | ~6–8 小时 |
| 变体实现与训练 | 2 天 |
| closed-loop 任务级运行 | ~10–14 小时 |
| **合计** | **W9 + W11 内** |

## E3.11 预期产物

| 产物 | 路径 |
|---|---|
| D 标签表 | `experiments/e3/d-labels-q8.csv`、`d-labels-q4.csv` |
| 变体评估表 | `experiments/e3/results/*.csv` |
| 校准报告 | `experiments/e3/calibration-report.md` |
| 图（风险散点、可靠性图） | `figures/e3-*.png` |
| 报告 | `experiments/e3/e3-report.md` |

### E3.11.1 结果表格模板（完成后填充，不发明数字）

**表 E3-1：token 级风险预测质量（预测 vs 干预回放真值）**

| 变体 | Spearman ρ (Q8 / Q4) | Kendall τ (Q8 / Q4) | MAE (Q8 / Q4) | 估计开销 ms/block |
|---|---|---|---|---|
| uniform（无预测） | — | — | — | ~0 |
| 静态 rule | TBD | TBD | TBD | ~0 |
| norm/range proxy | TBD | TBD | TBD | TBD |
| FlowCache estimator | TBD | TBD | TBD | TBD |

**表 E3-2：任务级传导（closed-loop，各变体精度分配下）**

| 变体 | τ-bench 成功率 [95% CI] | LongBench 准确率 | GSM8K accuracy | QA EM/F1（MuSiQue / 2Wiki） | 容量收益（可驻留 block 倍数） |
|---|---|---|---|---|---|
| uniform Q8 | TBD | TBD | TBD | TBD | TBD |
| uniform Q4 | TBD | TBD | TBD | TBD | TBD |
| 静态 rule | TBD | TBD | TBD | TBD | TBD |
| norm/range proxy | TBD | TBD | TBD | TBD | TBD |
| FlowCache estimator | TBD | TBD | TBD | TBD | TBD |

**表 E3-3：传导率解剖**

| 项 | Q8 | Q4 |
|---|---|---|
| KL > 中位数但 tool call 不变的 block 占比（良性） | TBD | TBD |
| KL ≤ 中位数但 tool call 变化的 block 占比（危险漏判） | TBD | TBD |
| token 级风险 → 任务级变化的传导率 | TBD | TBD |

**表 E3-4：校准与覆盖率**

| 项 | 目标 | 结果 |
|---|---|---|
| D̂_{b,q} 校准 ECE | 报告 | TBD |
| D̂^UCB 经验覆盖率（calibration split） | ≥ 1−δ | TBD |
| 覆盖率不足时的措辞降级（"经验风险预算"） | — | TBD |

## E3.12 失败/异常处理

- 所有 proxy 与 uniform 无显著差异 → 如实报告；controller 退化为静态规则（不影响 G4，但削弱 E5 中 fidelity 轴的贡献，在 E5/E4 结论中如实反映）；
- 覆盖率不足 → 按 IDEA §2.2 降级为"经验风险预算"措辞；
- token 级风险不传导到任务级 → 报告该事实，风险模型的主指标切换为任务级。

## E3.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| E3 变体与指标 | §8 E3 |
| 离线干预回放与在线特征 | §4.4 |
| D 定义与多 block 非线性叠加约束 | §2.2 |
| 保守回退 | §4.4 |

---

# E4：端到端主结果

> **周次**：W11 | **对应 Gate**：G2、G3、G4（端到端汇总验证）
> **定位**：论文主表/主图的来源。**主结论必须来自相同引擎、模型、dtype、预算和请求顺序**（IDEA §8 E4）

## E4.1 实验目标

在统一协议下比较 FlowCache joint policy 与全部对照的端到端系统表现与任务质量，直接验证 IDEA §0.3 的核心假设 3 与 4：

> 联合控制在扣除预测、迁移和量化开销后，优于最强的同后端解耦基线；且质量损失被预先设定的非劣区间约束。

E4 同时是 G2 双轴必要性的端到端证据（joint vs Decoupled-Best 的直接比较）、G3/G4 结论在完整系统中的复核。

### E4.1.1 前置条件（W11 进入 E4 前必须全部满足）

- G0–G4 全部 passed（任一 failed 则已触发路线切换，E4 不再执行）；
- G2-Pilot 判定为 GO 或条件 GO；
- 质量非劣界 ε、δ、样本量已按 G4.3 预注册并冻结；
- 全部对照实现就绪并通过 G1 可比性检查。

## E4.2 数据集与子集定义

| 数据集 | 样本数 | 角色 |
|---|---|---|
| τ-bench | 495 episodes | 主 workload ① |
| StableToolBench | 500（核验通过后） | 确认 workload（核验降级时列为泛化附表并标注） |
| SWE 轨迹 | 500 | 真实结构确认面（附表） |
| BurstGPT | 2,000 会话窗口 | 到达/并发结构来源（不直接作为策略评估对象，提供真实请求流） |

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn 800 行（主 workload ②）已删除——BFCL 不再作为数据集。析因设计从 18 cell 降为 9 cell（单主 workload）。

**test 专用性**：E4 系统评估使用各数据集全量样本的 replay（系统策略无需训练）；任何**含学习组件的策略**（FlowCache-Joint、Reuse-Only 的学习档）其预测器只在 train/val 上训练（τ-bench 297/99），test 与 family-out 数据对这些组件而言是 held-out。

## E4.3 样本数与功效分析

| 项 | 值 |
|---|---|
| 析因设计 | 3 KV 预算（10% / 25% / 50%）× 3 并发（4 / 8 / 16）× 1 主 workload = **9 cell**（v0.5 移除 BFCL 后单主 workload；确认面另计） |
| 每 cell | 495 episodes（τ-bench）× 3 replay 种子 |
| 总运行单元 | 9 cell × 13 对照 × 3 种子 = 351 次 replay（open-loop，主面；v0.5 较 v0.2 的 702 减半）+ 确认面附表 + closed-loop 质量子集（E4.6） |
| 主判定效应量 | joint vs Decoupled-Best 的 p95 TTFT / SLO goodput 差异；**净收益阈值**：条件 GO 情形下按 g2-pilot §10.3 设定更严格阈值（如 ≥ 5% p95 TTFT 改善，预注册） |
| 功效 | 主 cell 495 paired episodes：按 0.8.3，CI 半宽 ~2× 窄于原 80 单元设计；pilot（G3/E3 数据）估计 CV 后冻结最终重复数 |

## E4.4 Baseline / 对照（13 项，IDEA §8 E4 全量展开）

| # | 对照 | 说明 |
|---|---|---|
| 1 | No-Cache | cold recompute 下界 |
| 2 | APC-LRU | 同引擎实际 APC |
| 3 | LFU | 简单启发式 |
| 4 | LRU-K / 2Q | 简单启发式（实现其一，记录选择） |
| 5 | GDSF | 强启发式 |
| 6 | KVFlow† 或 PBKV† | ≥1 个可公平运行的 closest baseline；另一项不兼容才用标注的 inspired variant |
| 7 | Uniform-Q8 | 统一 Q8（容量规则与 joint 相同） |
| 8 | Uniform-Q4 | 统一 Q4 |
| 9 | Reuse-Only | 复用价值驱动驻留 + 统一精度 |
| 10 | Fidelity-Only | 保真风险驱动精度 + 强启发式驻留 |
| 11 | **Decoupled-Best** | 最强"reuse policy + uniform quantization"解耦组合（**关键对照**） |
| 12 | **FlowCache-Joint** | 待验联合 policy |
| 13 | Oracle-Cost | 读取未来的离线 lookahead 诊断（非严格最优上界） |

**公平性规则**（IDEA §5.1/§8）：全部 13 项同引擎、同模型（Qwen2.5-7B-Instruct）、同 dtype、同请求顺序、同预算定义；量化类对照与 joint 使用同一 codec；外部引擎结果只作独立 reference 不混比。

### E4.4.1 对照实现检查清单（W11 前逐项验收）

| # | 对照 | 实现位置 | 单元测试 | 公平性核验（同 trace 冒烟） |
|---|---|---|---|---|
| 1 | No-Cache | TBD | TBD | TBD |
| 2 | APC-LRU | TBD | TBD | TBD |
| 3 | LFU | TBD | TBD | TBD |
| 4 | LRU-K / 2Q | TBD | TBD | TBD |
| 5 | GDSF | TBD | TBD | TBD |
| 6 | KVFlow/PBKV† | TBD | TBD | TBD |
| 7 | Uniform-Q8 | TBD | TBD | TBD |
| 8 | Uniform-Q4 | TBD | TBD | TBD |
| 9 | Reuse-Only | TBD | TBD | TBD |
| 10 | Fidelity-Only | TBD | TBD | TBD |
| 11 | Decoupled-Best | TBD | TBD | TBD |
| 12 | FlowCache-Joint | TBD | TBD | TBD |
| 13 | Oracle-Cost | TBD | TBD | TBD |

### E4.4.2 SLO 定义（预注册字段）

| 字段 | 冻结值 |
|---|---|
| SLO-TTFT（ms） | TBD（依据 E4 前 BF16 基线 p50 TTFT 的倍数设定，如 2×，W11 前冻结） |
| SLO-JCT（ms） | TBD |
| SLO goodput 定义 | 单位时间内同时满足 SLO-TTFT 与 SLO-JCT 的完成请求数 |
| max admitted concurrency 定义 | goodput 不低于峰值 95% 且无 OOM 的最大并发 |
| 净收益阈值（joint vs Decoupled-Best） | TBD（条件 GO 时 ≥ 5% p95 TTFT，g2-pilot §10.3） |

## E4.5 测试指标（IDEA §8 E4 全量）

| 类别 | 指标 |
|---|---|
| 延迟 | TTFT p50/p95/p99、JCT p50/p95/p99 |
| 吞吐 | throughput、**SLO goodput**（SLO 阈值 TBD 预注册）、max admitted concurrency |
| 缓存 | token/block/byte cache hit、saved-prefill tokens/time |
| 资源 | GPU allocated/reserved、CPU pinned bytes |
| 开销 | H2D/D2H 时间与字节、codec 时间、controller 时间（含预测器推理）、queueing 时间 |
| 质量 | 任务成功率 + 预注册非劣区间（closed-loop 子集） |
| 成本分解 | 每个策略的 saved-prefill − (place + hold + policy + SLO) 净收益（IDEA §2.4 成本项） |

## E4.6 运行协议

1. **open-loop 主表**：9 cell × 13 对照 × 3 种子（v0.5 移除 BFCL 后单主 workload，原 18 cell）；全部系统指标；到达结构主证据 BurstGPT（Poisson λ=4 参照并报告拟合优度），H=1000；
2. **closed-loop 质量子表**：主 cell（预算 25%、并发 8）+ 边界 cell（预算 10%、并发 16）× 质量相关对照（No-Cache、APC-LRU、Uniform-Q8、Uniform-Q4、Decoupled-Best、FlowCache-Joint）× τ-bench 495，记录任务成功率与失败分析；**与 open-loop 分表**（Part 0.7 铁律）；
3. controller 配置：滚动时域 greedy/index policy（IDEA §4.5），更新频率冻结值来自 E5 扫描；
4. 安全水位与 OOM 处理：保留水位（Part 0.2.2）；任何 OOM 事件如实记录（进 E7 数据）；
5. 主图：相同后端的**延迟–容量 Pareto 主图**（IDEA §14 写作就绪条件 5）。

## E4.7 统计检验

- 主比较：FlowCache-Joint vs Decoupled-Best（首要）、vs 各单一对照（次要）的 per-workflow 配对差；
- paired workflow-level bootstrap 95% CI（1000 次）；
- 多重性：9 cell × 主要比较族（v0.5 移除 BFCL 后单主 workload，原 18 cell），Bonferroni 校正；主 cell 单独报告未校正与校正值；
- 质量：Δsuccess 95% CI 上界 vs 预注册 ε（非劣判定，同 G4.7 口径）；
- 报告全部 cell 结果（不只报喜 cell）；负结果 cell 进 E7 分析。

## E4.8 成功标准（对应写作就绪条件，IDEA §14）

| # | 标准 |
|---|---|
| 1 | joint 在主 cell 计入自身开销后显著优于 Decoupled-Best（CI 不含 0，且 ≥ 预注册净收益阈值） |
| 2 | 优势在多数预算/并发 cell 上方向一致（允许边界 cell 无收益，如实报告） |
| 3 | 质量非劣：Δsuccess CI 上界 ≤ ε |
| 4 | 主图：同后端 Pareto 图 + workflow-level CI（写作就绪条件 5） |
| 5 | 无收益区域与失败条件被识别并移交 E7（贡献 3 的"无收益区域"证据） |

## E4.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 4.1 | 冻结全部配置（预算、并发、SLO 阈值、controller 参数、种子） | `experiments/e4/frozen-config.yaml` |
| 4.2 | 13 对照全网格 open-loop 运行 | 原始结果 |
| 4.3 | closed-loop 质量子集运行 | 质量结果 |
| 4.4 | 统计分析 + Pareto 主图 | 主表/主图 |
| 4.5 | 成本分解与开销归因 | 成本分解表 |
| 4.6 | 汇总报告（移交 E5 消融输入、E7 失败输入） | `experiments/e4/e4-report.md` |

## E4.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 351 次 open-loop replay（v0.5 移除 BFCL 后单主 workload，原 702 次；引擎批处理） | ~18–24 小时 GPU |
| closed-loop 子集（2 cell × 6 对照 × 495） | ~6–8 小时 GPU |
| 分析与制图 | 1 天 |
| **合计** | **W11 一周内（需提前冻结配置，禁止边跑边改；如超时，确认面附表移至 W12）** |

## E4.11 预期产物

| 产物 | 路径 |
|---|---|
| 冻结配置 | `experiments/e4/frozen-config.yaml` |
| 主结果表（open-loop） | `experiments/e4/results/main-openloop.csv` |
| 质量结果表（closed-loop） | `experiments/e4/results/quality-closedloop.csv` |
| Pareto 主图 | `figures/e4-pareto-main.png` |
| 成本分解表 | `experiments/e4/results/cost-breakdown.csv` |
| 报告 | `experiments/e4/e4-report.md` |

### E4.11.1 结果表格模板（完成后填充，不发明数字）

**表 E4-1：主表（open-loop，主 cell：预算 25%、并发 8，τ-bench 495 × 3 种子）**

| # | 对照 | TTFT p50 / p95 / p99 (ms) | JCT p95 (ms) | SLO goodput | block hit | saved-prefill ms | GPU reserved (GB) | CPU pinned (GB) | 开销合计 ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | No-Cache | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |
| 2 | APC-LRU | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | LFU | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | LRU-K/2Q | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | GDSF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | KVFlow/PBKV† | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 | Uniform-Q8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | Uniform-Q4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | Reuse-Only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | Fidelity-Only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 11 | Decoupled-Best | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 12 | FlowCache-Joint | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 13 | Oracle-Cost | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |

（v0.5 移除 BFCL 800 同构第二主表；全 9 cell 同构输出，其余进附录 CSV。）

**表 E4-2：质量子表（closed-loop，与 open-loop 分表）**

| 对照 | cell | τ-bench 成功率 [95% CI] | vs BF16 基线 Δsuccess [95% CI] | 非劣（CI 上界 ≤ ε）? |
|---|---|---|---|---|
| No-Cache（BF16 参照） | 主 cell | TBD | — | — |
| APC-LRU | 主 cell | TBD | TBD | TBD |
| Uniform-Q8 | 主 cell + 边界 cell | TBD | TBD | TBD |
| Uniform-Q4 | 主 cell + 边界 cell | TBD | TBD | TBD |
| Decoupled-Best | 主 cell + 边界 cell | TBD | TBD | TBD |
| FlowCache-Joint | 主 cell + 边界 cell | TBD | TBD | TBD |

**表 E4-3：成本分解（IDEA §2.4 成本项，主 cell，ms/workflow）**

| 对照 | C^res（恢复/重算） | C^place（迁移/编码） | C^hold（机会成本） | C^policy（控制器） | C^SLO（违约代价） | 总成本 | vs Oracle regret |
|---|---|---|---|---|---|---|---|
| Decoupled-Best | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlowCache-Joint | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 E4-4：joint vs Decoupled-Best 逐 cell 配对差（主判定）**

| 预算 | 并发 | Δp95 TTFT [95% CI]（τ-bench） | Δgoodput [95% CI] | ≥ 净收益阈值? |
|---|---|---|---|---|
| 10% | 4 / 8 / 16 | TBD | TBD | TBD |
| 25% | 4 / 8 / 16 | TBD | TBD | TBD |
| 50% | 4 / 8 / 16 | TBD | TBD | TBD |

## E4.12 失败/异常处理

- joint 未显著优于 Decoupled-Best → 回溯 G2 判定（G2 实际上未通过），按 IDEA §7 G2 失败动作转路线 B；不把中间结果包装成方法贡献；
- 质量越界（CI 上界 > ε）→ 不提出质量非劣 claim（IDEA §2.4）；按 G4 口径回溯；
- 个别 cell OOM/异常 → 如实记录，进 E7；不删除失败 cell。

## E4.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| E4 对照与指标全量 | §8 E4 |
| 优化目标与成本项 | §2.4 |
| controller 语义 | §4.5 |
| 公平性规则 | §5.1 |
| open/closed-loop 分表 | §6.4 |
| 写作就绪条件 | §14 |

---

# E5：机制消融

> **周次**：W11 | **对应 Gate**：G2
> **定位**：关键问题**不是**"删掉模块性能是否下降"，而是**联合建模是否解决了单一分数的错误分配**（IDEA §8 E5）

## E5.1 实验目标

逐项拆解 FlowCache-Joint 的设计决策，验证：

1. 双轴（reuse × fidelity）联合效用相对单轴/串联的收益；
2. 各结构性组件（partial DAG、成本校准、parent-closure、CPU tier、动态预算）的必要性；
3. 用**四象限错误分配分析**（g2-pilot §6.3 框架）直接展示：单分数策略在 HL/LH 块上犯了什么错，joint 如何修正。

## E5.2 数据集与子集定义

| 数据集 | 样本数 | 角色 |
|---|---|---|
| τ-bench | 495 episodes | 主消融面 |
| StableToolBench | 500 | 确认面（核验降级时仅 τ-bench） |

> **v0.5（2026-07-26）注**：原 BFCL v3 multi-turn 800 行（第二消融面）已删除——BFCL 不再作为数据集。

运行面：E4 的主 cell（预算 25%、并发 8）为消融核心；关键消融（双轴四项）扩至全 3 预算档位；3 种子。

## E5.3 样本数与功效分析

- 每消融变体：主 cell 495 episodes × 3 种子（与 E4 完全配对，共用 trace）；
- 双轴四项对比扩展至 3 预算 × 1 主 workload（v0.5 移除 BFCL 后单 workload）；
- 功效：与 E4 共用配对框架；消融项间差异预期小于主对照，按 0.8.3 规则报告 CI（CI 过宽时结论限定为"无显著差异"，不硬判无效）。

## E5.4 消融变体（IDEA §8 E5 全量 10 项）

| # | 变体 | 回答的问题 |
|---|---|---|
| 1 | Reuse-Only | 只看复用价值会怎样（E4 已有，此处做错误分配解剖） |
| 2 | Fidelity-Only | 只看保真风险会怎样（同上） |
| 3 | 两者独立串联 | 先 reuse 选驻留、再 fidelity 选精度（无联合效用） |
| 4 | **Joint utility（完整 FlowCache）** | 参照 |
| 5 | 无 partial DAG | 只用 block/service 特征，不用 workflow 结构特征 |
| 6 | 无成本校准 | 只用预测概率，不乘实测成本 |
| 7 | 无 parent-closure | 允许不完整父链被当作可恢复 |
| 8 | 无 CPU tier | 只有 GPU 与 Evict 两层 |
| 9 | 静态阈值 vs 动态预算 | 固定保留阈值 vs 随压力动态调整 |
| 10 | controller 更新频率 | 每事件 / 定期（如每 10 step）/ 压力触发三档扫描 |

### E5.4.1 变体开关规格（配置化，禁止改代码逻辑）

| 变体 | 开关（frozen-config 字段） | 备注 |
|---|---|---|
| 3 串联 | `controller.mode = "sequential"` | 先跑 reuse 分配驻留，再在其结果上跑 fidelity 分配精度 |
| 4 joint | `controller.mode = "joint"` | IDEA §4.5 的 V_{b,a} 联合效用 |
| 5 无 partial DAG | `features.workflow_struct = false` | 预测器重训（仅用剩余特征） |
| 6 无成本校准 | `controller.utility = "prob_only"` | V = P(reuse)，不乘成本 |
| 7 无 parent-closure | `controller.parent_closure = false` | 预期产生 F5 类失败，记录恢复失败事件 |
| 8 无 CPU tier | `tiers = ["GPU", "Evict"]` | — |
| 9 静态阈值 | `budget.dynamic = false` + 固定保留比例 | 保留比例取动态规则的均值 |
| 10 更新频率 | `controller.trigger = {"event", "periodic:10", "pressure"}` | 三档 |

## E5.5 测试指标

| 类别 | 指标 |
|---|---|
| 系统（E4 子集） | p95 TTFT、SLO goodput、saved-prefill ms、总开销 |
| 质量 | 任务成功率（closed-loop 子集：变体 1–4 + 7） |
| **错误分配解剖** | 四象限误判率：在变体的实际分配决策中，HL 块被错误保持 BF16（浪费容量）/ LH 块被错误量化（引入风险）/ HH 块被量化的比例；逐变体对比 |
| 开销归因 | 每变体的 controller 耗时、codec 触发次数、迁移次数 |

## E5.6 运行协议

- 与 E4 共用冻结 trace 与配置，仅切换机制开关；open-loop 为主；
- 质量相关变体（1–4、7）加跑 closed-loop 子集（同 E4.6 子集）；
- 错误分配解剖使用 G2-Pilot 的 R/D 标签作为象限真值参照（中位数分割）。

## E5.7 统计检验

- 变体 vs Joint 的 paired bootstrap 95% CI；
- 错误分配比例的变化用配对比例检验；
- 10 变体族 Bonferroni 校正（主比较为变体 3 串联 vs 变体 4 joint）。

## E5.8 成功标准

| 标准 | 说明 |
|---|---|
| 机制归因清晰 | 每个组件的贡献（或无贡献）有数字支撑 |
| 错误分配故事成立 | joint 在 HL/LH 块上的误判率显著低于单分数/串联变体，或其收益来源被如实归因于其他机制 |
| 负结果诚实 | 无贡献组件（如 partial DAG 特征）如实报告并默认移除 |

## E5.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 5.1 | 变体开关实现与配置冻结 | ablation configs |
| 5.2 | 主 cell × 10 变体 × 3 种子运行 | 结果表 |
| 5.3 | 双轴四项 × 3 预算 × 1 workload 扩展（v0.5 移除 BFCL 后单 workload） | 扩展表 |
| 5.4 | closed-loop 质量子集（变体 1–4、7） | 质量表 |
| 5.5 | 四象限错误分配解剖 | 解剖图 |
| 5.6 | 汇总 | `experiments/e5/e5-report.md` |

## E5.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 主 cell 消融（10 × 495 × 3） | ~10–12 小时 |
| 扩展面 + 质量子集 | ~8–10 小时 |
| **合计** | **W11 内（与 E4 同一周，错峰 GPU）** |

## E5.11 预期产物

| 产物 | 路径 |
|---|---|
| 消融结果表 | `experiments/e5/results/*.csv` |
| 错误分配解剖图 | `figures/e5-misallocation.png` |
| 报告 | `experiments/e5/e5-report.md` |

### E5.11.1 结果表格模板（完成后填充，不发明数字）

**表 E5-1：消融主表（主 cell：预算 25%、并发 8，τ-bench 495 × 3 种子）**

| # | 变体 | p95 TTFT (ms) | vs Joint 相对差 [95% CI] | SLO goodput | saved-prefill ms | 任务成功率（closed-loop 子集） |
|---|---|---|---|---|---|---|
| 1 | Reuse-Only | TBD | TBD | TBD | TBD | TBD |
| 2 | Fidelity-Only | TBD | TBD | TBD | TBD | TBD |
| 3 | 独立串联 | TBD | TBD | TBD | TBD | TBD |
| 4 | Joint（参照） | TBD | — | TBD | TBD | TBD |
| 5 | 无 partial DAG | TBD | TBD | TBD | TBD | — |
| 6 | 无成本校准 | TBD | TBD | TBD | TBD | — |
| 7 | 无 parent-closure | TBD | TBD | TBD | TBD | TBD |
| 8 | 无 CPU tier | TBD | TBD | TBD | TBD | — |
| 9 | 静态阈值 | TBD | TBD | TBD | TBD | — |
| 10 | 更新频率（3 档） | TBD | TBD | TBD | TBD | — |

**表 E5-2：四象限错误分配解剖（变体 1/2/3/4，R/D 真值标签中位数分割）**

| 变体 | HL 块误保持 BF16 占比（容量浪费） | HL 块误量化占比 | LH 块误量化占比（风险引入） | HH 块误量化占比（严重错误） | 误判率合计 |
|---|---|---|---|---|---|
| Reuse-Only | TBD | TBD | TBD | TBD | TBD |
| Fidelity-Only | TBD | TBD | TBD | TBD | TBD |
| 独立串联 | TBD | TBD | TBD | TBD | TBD |
| Joint | TBD | TBD | TBD | TBD | TBD |

## E5.12 失败/异常处理

- joint 收益主要来自某一单组件（如 CPU tier）而非双轴交互 → 如实归因；若双轴交互无独立贡献，回溯 G2 判定口径；
- 变体间无显著差异 → 报告 CI，不夸大。

## E5.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 消融清单 | §8 E5 |
| 四象限框架 | §7 G2（经 g2-pilot §6.3 形式化） |
| controller 组件 | §4.5 |
| parent-closure 约束 | §2.4 |

---

# E6：泛化与鲁棒性

> **周次**：W12（精简版） | **对应 Gate**：G5
> **定位**：第二模型和额外 dataset-out 只作为资源允许时的扩展，不是 14 周主证据包的前置条件（IDEA §8 E6）

## E6.1 实验目标

在分布偏移与扰动下检验 FlowCache 的稳健性，给出主结论的适用边界：

1. **workflow-family-out**：真实 family 之间互泛化（IDEA §8 E6 首项，与 G5 主判定面一致；v0.4 精简为单通道，仅 τ-bench ↔ StableToolBench）；
2. **9 个扰动轴**上的性能保持率与失效模式（v0.2：扰动全部由真实数据驱动或 replay 时操作实现，无合成数据集）。

> **v0.4 注（trim-dataset-portfolio spec）**：鲁棒性章节从 3 轴（family-out / 到达扰动 / branch 噪声）精简为 2 轴（family-out / 到达扰动）。branch 噪声轴用 τ-bench 内部 replay 扰动覆盖（删边/错标后继，轴 3/4），不另设数据集。原 family-out 通道②（SWE ↔ Toolathlon）已移除。

## E6.2 数据集与子集定义

| 数据 | 样本数 | 用途 |
|---|---|---|
| τ-bench ↔ StableToolBench | 495 ↔ 500 | family-out 通道①（工具型互泛化，v0.4 唯一通道） |
| LMSYS-Chat-1M | 2,000 | 负对照（burst 与纯追加场景） |
| BurstGPT | 2,000 窗口 | 真实 burst arrival 证据（轴 5 主证据） |
| Mooncake trace | 抽样（TBD） | 大规模前缀共享压力面（新增轴） |

> **v0.5（2026-07-26）注**：原 BFCL 800 行（长度维度的补充面）已删除——BFCL 不再作为数据集。轴 2"不同上下文长度"改由 LongBench 长度分层单独覆盖。

## E6.3 样本数与功效分析

- 每个扰动轴：主 cell（预算 25%、并发 8）× 495 episodes × 3 种子；
- family-out：双向各 500 test 单元（v0.4 单通道：τ-bench ↔ STB）；
- 功效：描述性 + CI；鲁棒性结论以"退化幅度是否在预注册可接受带内"判定（带：TBD，建议 p95 TTFT 退化 ≤ 20%、goodput 退化 ≤ 15%，W12 前预注册）。

## E6.4 对照与扰动轴（IDEA §8 E6 全量，v0.2 真实数据化）

对照：FlowCache-Joint（完整）vs Decoupled-Best vs APC-LRU（参照）。

| # | 扰动轴 | 操作化（v0.2：真实数据 / replay 时操作） |
|---|---|---|
| 1 | workflow-family-out | 单通道交叉（E6.2；v0.4 移除 SWE↔Toolathlon 通道②，仅 τ-bench↔STB） |
| 2 | 不同上下文长度 | LongBench 长度分层（真实长度分布；v0.5 移除 BFCL long_context 类） |
| 3 | DAG 边缺失/噪声 | **replay 时特征扰动**：对预测器输入的已声明后继随机删边 p ∈ {5%, 15%, 30%}（实验操作，非数据集；缓存兼容性判定不受影响，IDEA §1.4 铁律） |
| 4 | branch misprediction | **replay 时特征扰动**：分支标签错标（错误率档位 TBD）；v0.4 起由 τ-bench 内部 replay 扰动覆盖（删边/错标后继），不另设数据集 |
| 5 | burst arrival | **BurstGPT 真实突发窗口**（主证据）；MMPP 合成模型次要参照（×2/×4 强度） |
| 6 | 工具等待分布变化 | **τ-bench / STB 工具调用实测等待分布** + replay 时时间缩放（×0.5 / ×2 / 重尾化）；v0.4 移除 Toolathlon 后改用 τ-bench/STB 支撑 |
| 7 | GPU budget 突变 | 运行中预算阶跃（如 50% → 10%），观察恢复 |
| 8 | CPU 带宽竞争 | 注入并发 PCIe 负载（受控背景传输） |
| 9 | predictor calibration drift | 用旧校准参数跑新分布（时间错位模拟） |
| 10 | 大规模前缀共享压力（新增） | Mooncake trace 前缀块 hash 共享结构驱动的压力面 |

## E6.5 测试指标

| 类别 | 指标 |
|---|---|
| 保持率 | p95 TTFT 保持率（扰动/基线）、SLO goodput 退化幅度 |
| 预测稳健性 | policy regret 变化、ECE 漂移 |
| 质量稳健性 | 任务成功率退化（closed-loop 子集：轴 1、5、6） |
| 失效行为 | 回退触发次数（预测器失效 → SizeCost-LRU，IDEA §4.5）、OOM/拒绝事件（移交 E7） |

## E6.6 运行协议

- open-loop 为主；质量相关轴（1、5、6）加 closed-loop 子集；
- 扰动参数逐轴冻结并记录；每轴与无扰动基线严格配对（同 trace 底子、同种子）；
- replay 时特征扰动只影响**预测器可见的 partial 信息**，不影响缓存兼容性判定（IDEA §1.4 铁律）。

## E6.7 统计检验

- 扰动 vs 基线的 paired bootstrap 95% CI；
- 多轴报告效应量森林图（不逐轴做显著性门槛，报告 CI 宽度）。

## E6.8 成功标准

| 标准 | 说明 |
|---|---|
| 边界明确 | 每个扰动轴给出"适用/退化/失效"的定性 + 定量结论 |
| 失效可解释 | 显著退化轴能归因到具体机制（如 drift → 校准失效 → 回退） |
| 负对照干净 | LMSYS 上无伪收益（IDEA §1.1） |

## E6.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 6.1 | 扰动轴参数冻结（预注册可接受带） | `experiments/e6/perturbation-config.yaml` |
| 6.2 | family-out 单通道运行（v0.4：τ-bench↔STB） | 结果表 |
| 6.3 | 各扰动轴运行（主 cell） | 结果表 |
| 6.4 | 质量子集 + 失效归因 | 归因记录 |
| 6.5 | 汇总 | `experiments/e6/e6-report.md` |

## E6.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| family-out + 10 轴 × 3 种子 | ~14–18 小时 |
| 分析 | 0.5 天 |
| **合计** | **W12 精简版内**（第二模型/GNN 后移，不视为失败；时间不足时裁剪轴 2/8/10 并记录） |

## E6.11 预期产物

| 产物 | 路径 |
|---|---|
| 扰动结果表 | `experiments/e6/results/*.csv` |
| 效应量森林图 | `figures/e6-robustness-forest.png` |
| 报告 | `experiments/e6/e6-report.md` |

### E6.11.1 结果表格模板（完成后填充，不发明数字）

**表 E6-1：扰动轴总表（主 cell；对照 = Joint / Decoupled-Best / APC-LRU）**

| 轴 | 档位 | Joint p95 TTFT 保持率 [95% CI] | Joint goodput 退化 | Joint regret 变化 | 是否在可接受带内 | 失效归因 |
|---|---|---|---|---|---|---|
| 1 family-out ① | τ→STB | TBD | TBD | TBD | TBD | TBD |
| 1 family-out ① | STB→τ | TBD | TBD | TBD | TBD | TBD |
| 2 上下文长度 | 分层 | TBD | TBD | TBD | TBD | TBD |
| 3 特征删边 | 5% / 15% / 30% | TBD | TBD | TBD | TBD | TBD |
| 4 branch 错标 | 5% / 15% / 30% | TBD | TBD | TBD | TBD | TBD |
| 5 burst（BurstGPT） | 真实窗口 | TBD | TBD | TBD | TBD | TBD |
| 6 工具等待 | ×0.5 / ×2 / 重尾 | TBD | TBD | TBD | TBD | TBD |
| 7 预算突变 | →25% / →10% | TBD | TBD | TBD | TBD | TBD |
| 8 CPU 竞争 | 25% / 50% | TBD | TBD | TBD | TBD | TBD |
| 9 calibration drift | val→test | TBD | TBD | TBD | TBD | TBD |
| 10 前缀共享压力（Mooncake） | 抽样窗口 | TBD | TBD | TBD | TBD | TBD |

> **v0.4 注**：原"1 family-out ②（SWE→Toolathlon / Toolathlon→SWE）"两行已随 SWE/Toolathlon 数据集移除而删除。

**表 E6-2：LMSYS 负对照（2,000 会话）**

| 对照 | p95 TTFT vs SizeCost 差 | goodput 差 | 结论（无伪收益 / 无退化） |
|---|---|---|---|
| FlowCache-Joint | TBD | TBD | TBD |
| Decoupled-Best | TBD | TBD | TBD |

## E6.12 失败/异常处理

- 某轴失效 → 如实报告失效条件（这本身是贡献 3 的"失败条件"证据，IDEA §9-3）；
- 时间不足 → 裁剪轴 2/8/10 并记录，保留轴 1/3/5/6/9 为核心。
- **v0.4 鲁棒性压力面 rebuttal 策略（trim-dataset-portfolio spec）**：v0.4 将 Ch.5 压力面从 3 数据集（STB 500 + SWE 200 + Toolathlon 200）精简为 1 数据集（仅 STB 500）。若 rebuttal 时审稿人要求补充压力面证据：
  - **首选**：扩展 StableToolBench 到更大样本（如 500 → 1,000）增强 family-out 统计功效；
  - **次选**：补做 SWE 轨迹 500 episodes（当前 200 样本不足，扩到 500 后可单独成章）；
  - **多 agent 场景**：引用 τ-bench retail/airline 两域作为 workload 多样性证据，替代 Toolathlon 的多 agent 协作场景；
  - 当前 14 周主证据包不做上述扩展，仅在 rebuttal 阶段按需启动。

## E6.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 扰动轴清单 | §8 E6 |
| family-out 与 G5 关系 | §7 G5 |
| DAG 不决定兼容性 | §1.4 |
| 预测器失效回退 | §4.5 |

---

# E7：失败与开销

> **周次**：W12（精简版） | **对应 Gate**：—（独立，不绑定特定 gate）
> **定位**：失败模式与开销**必须单独报告**（IDEA §8 E7）；是贡献 3（无收益区域与失败条件）的直接证据

## E7.1 实验目标

系统性记录 FlowCache（及对照）在何处、为何、以多大代价失败，以及全部自身开销。E7 不追求"通过"，追求**完整诚实的负空间地图**。

## E7.2 数据集与子集定义（v0.2：全部真实数据）

| 失败模式 | 主要数据面 | 样本数 |
|---|---|---|
| F1 exact-prefix overlap 过低 | E1 画像中 overlap 最低的数据集/子集（由 E1 实测确定，候选：GSM8K 短链、跨域对） | 由 E1 实测确定（TBD） |
| F2 纯追加式、预测无价值 | LMSYS-Chat-1M | 2,000 |
| F3 量化敏感块误判 | τ-bench（E3 D 标签中 D 最高的 10% block） | ~100+ block（Top-10%） |
| F4 高频 GPU↔CPU 抖动 | **Toolathlon 短等待轨迹子集**（真实短等待；可叠加 replay 时时间缩放 ×0.5 加密，参数扰动） | ~150 轨迹（TBD 按等待分布筛选） |
| F5 parent 缺失导致后继不可用 | **SWE 长轨迹（≥60 轮）** + τ-bench 压力 cell（预算 10%） | ~100 轨迹 + 495 |
| F6 模板/adapter 变化大面积 invalidation | **真实会话 × chat template 不同 revision 渲染**（G0 类别③扩展：LMSYS 30 条 × 2 模板版本；真实数据 + 渲染变换） | 30 × 2 |
| F7 controller 开销超过 saved-prefill | E4 全网格的开销日志（复用） | 702 replay 全量 |
| F8 overload 下退化/拒绝/OOM | τ-bench 495，并发爬升扫描（4→8→16→24→32，超出设计档位直至 SLO 崩溃） | 495 × 5 档位 |

## E7.3 样本数与功效分析

- E7 以**计量与归因**为主，不做 Go/No-Go 检验；
- 每失败模式报告发生率（分母明确）、幅度分布、触发条件；
- F3/F5 等 block 级模式报告 per-block 计数与 per-workflow 聚合（0.8.1）。

## E7.4 对照

- FlowCache-Joint 为主体；APC-LRU、GDSF、Decoupled-Best 为对照（失败模式是否 FlowCache 特有）；
- F2 负对照期望：FlowCache 在 LMSYS 上不退化（回退机制生效）且无伪收益。

## E7.5 测试指标（按失败模式）

| 模式 | 指标 |
|---|---|
| F1 | overlap 分布、oracle headroom 残余量、策略间差异塌缩度 |
| F2 | 学习/复杂策略 vs SizeCost 的 regret 差（期望 ≈ 0）、开销差 |
| F3 | 误判率（被量化的 Top-D 块比例）、误判块的 Δsuccess 贡献、风险回退触发率 |
| F4 | 单位时间迁移次数、抖动块的迁移成本/收益比、thrash 期间的 TTFT 尖峰 |
| F5 | parent-missing 事件率、受影响 block 数、由此产生的重算量占比 |
| F6 | invalidation 波及率（一次模板变化失效的 block 比例）、恢复稳态所需时间 |
| F7 | controller 总开销 / saved-prefill 比（逐 cell；>1 的 cell 清单）、单决策耗时分布 |
| F8 | 拒绝率、SLO 违约率、OOM 计数、graceful degradation 曲线（goodput vs offered load） |

### E7.5.1 开销透明账（`overhead-ledger.csv` 列定义，IDEA §4.5 末句的落地）

| 列 | 定义 |
|---|---|
| cell_id | 预算 × 并发 × workload × 种子 |
| policy | 策略名 |
| saved_prefill_ms | 命中节省的 prefill 总量 |
| h2d_ms / d2h_ms | 传输时间与次数 |
| codec_ms | 编解码总耗时与次数 |
| controller_ms | 决策总耗时（含预测器推理、特征提取） |
| queueing_ms | 排队附加延迟 |
| net_benefit_ms | saved_prefill − 全部开销 |
| overhead_ratio | 全部开销 / saved_prefill（>1 即 F7 失败 cell） |

### E7.5.2 结果表格模板（完成后填充，不发明数字）

**表 E7-1：失败模式登记簿**

| 模式 | 发生率（分母明确） | 幅度 (median / p95) | 触发条件 | 是否 FlowCache 特有 | 归因 |
|---|---|---|---|---|---|
| F1 overlap 过低 | TBD | TBD | TBD | TBD | TBD |
| F2 预测无价值 | TBD | TBD | TBD | TBD | TBD |
| F3 量化误判 | TBD | TBD | TBD | TBD | TBD |
| F4 GPU↔CPU 抖动 | TBD | TBD | TBD | TBD | TBD |
| F5 parent 缺失 | TBD | TBD | TBD | TBD | TBD |
| F6 大面积 invalidation | TBD | TBD | TBD | TBD | TBD |
| F7 开销倒挂 | TBD | TBD | TBD | TBD | TBD |
| F8 overload 崩溃 | TBD | TBD | TBD | TBD | TBD |

**表 E7-2：overload 压力扫描（F8，τ-bench 495）**

| 并发档位 | Joint goodput | Joint SLO 违约率 | Joint 拒绝率 | OOM 计数 | APC-LRU goodput（对照） |
|---|---|---|---|---|---|
| 4 | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD |
| 24 | TBD | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD | TBD |

## E7.6 运行协议

- F1/F2/F4/F5/F6 用 open-loop；F3 与 F8 的质量维度用 closed-loop 子集；
- F7 完全复用 E4 日志（不新增运行）；
- F8 的并发爬升超出设计档位属**压力测试**，明确标注不可与主表混比；
- 所有失败事件保留原始日志（block id、时刻、上下文摘要）供审计。

## E7.7 统计检验

- 发生率 + 幅度的描述统计与 CI；
- 模式间不做显著性比较；与对照的差异报告效应量。

## E7.8 成功标准

| 标准 | 说明 |
|---|---|
| 8 模式全覆盖 | 每模式有数据、有归因、有触发条件描述（IDEA §8 E7 清单逐项落实） |
| 开销透明 | controller 收益与自身成本同表报告（IDEA §4.5 末句） |
| 可复现审计 | 失败事件日志完整可查 |

## E7.9 执行步骤

| 步骤 | 内容 | 产物 |
|---|---|---|
| 7.1 | E4/E6 日志汇聚（F7、F8 素材） | 日志库 |
| 7.2 | F1–F6 专项运行 | 各模式数据 |
| 7.3 | F8 压力扫描 | 退化曲线 |
| 7.4 | 归因分析 | 归因记录 |
| 7.5 | 汇总 | `experiments/e7/e7-report.md` |

## E7.10 硬件与时间预算

| 项 | 预估 |
|---|---|
| 专项运行（F1–F6、F8） | ~8–10 小时 |
| 归因与报告 | 1 天 |
| **合计** | **W12 精简版内** |

## E7.11 预期产物

| 产物 | 路径 |
|---|---|
| 失败模式数据与日志 | `experiments/e7/` |
| 开销透明表 | `experiments/e7/overhead-ledger.csv` |
| graceful degradation 图 | `figures/e7-overload.png` |
| 报告 | `experiments/e7/e7-report.md` |

## E7.12 失败/异常处理

E7 本身无失败动作；其发现若为**致命**（如 F7 普遍 >1、F8 无 graceful 退化），回溯对应 gate（G3/G5）的判定口径，必要时在论文中收缩 claim（IDEA §9 贡献声明的约束）。

## E7.13 与 IDEA 各节的对应关系

| 本章要素 | IDEA 来源 |
|---|---|
| 失败模式清单 | §8 E7 |
| 开销报告义务 | §4.5 |
| 贡献 3（无收益区域与失败条件） | §9 |
| 负对照语义 | §1.1、§6.1 |
| graceful degradation / OOM | §8 E7 |

---

## 附：全册章节–Gate–周次–产物速查表

| 章 | Gate | 周次 | 主数据 | 主判定/交付 |
|---|---|---|---|---|
| G0 | G0 | W1–W2 | 真实结构用例 90 组 + τ-bench 10 + 100 block | exactness/loadability 判定 |
| G1 | G1 | W6（v0.2 顺延） | τ-bench 1,320（v0.5 移除 BFCL 800；+STB 500 确认） | oracle headroom ≥10% + baseline 可比性 |
| G2 | G2 | W9–W10 | 见 `g2-pilot-design.md`（τ-bench 80 子集，设计独立成立） | R–D 错位判定（已有独立文档） |
| G3 | G3 | W7–W8 | τ-bench 495（v0.5 移除 BFCL 800；9 cell） | p95 TTFT ~15%、吞吐 ≥ −5% |
| G4 | G4 | W9–W10 | τ-bench 495 + LongBench 1,000 + GSM8K 300 + QA 600 | 非劣检验（预注册 ε/δ/N） |
| G5 | G5 | W7–W8 | τ-bench 297/99/99（v0.5 移除 BFCL 480/160/160）→ STB 500 / SWE 500 | family-out regret −10% |
| E1 | G1 | W7（v0.2 顺延） | 12 个数据集全量（~8,800 样本） | workload 画像 + trace 资产 |
| E2 | G5 | W7–W8 | τ-bench（train/val/test，v0.5 移除 BFCL）+ STB/SWE family-out + Toolathlon + LMSYS 2,000 | 预测质量 + 系统转换 |
| E3 | G4 | W9, W11 | τ-bench 495 + LongBench 1,000 + GSM8K 300 + QA 600 | 风险估计方法学对比 |
| E4 | G2/G3/G4 | W11 | τ-bench 495（v0.5 移除 BFCL 800；9 cell × 13 对照；BurstGPT 到达） | 主表/主图/Pareto |
| E5 | G2 | W11 | τ-bench 495（v0.5 移除 BFCL 800） | 机制归因 + 错误分配解剖 |
| E6 | G5 | W12 | family-out 双通道 + BurstGPT/Mooncake + LMSYS 2,000 | 鲁棒性边界 |
| E7 | — | W12 | 按模式（F1–F8，全部真实数据） | 失败空间地图 + 开销透明 |

> 本册全部结果数字为 TBD；任何章节的预注册字段冻结后，回写对应章节并标注冻结日期。
> v0.2（2026-07-25）：数据集体系重构——禁自建数据集、样本总量 ~880 → ~8,800、新增 BFCL/SWE 轨迹/Toolathlon/CATraces/LongBench/GSM8K/BurstGPT/Mooncake（详见 0.4）。**v0.5（2026-07-26）注：BFCL 全面移除，不再作为数据集**。
