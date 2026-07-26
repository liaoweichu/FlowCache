# 精简数据集组合 Spec

## Why

FlowCache v0.3 当前规划 **7 个核心数据集 + 2 个辅助数据集**，~4,120 workflow 级样本。对比同领域 KV cache 管理 / 前缀缓存论文（详见 §研究对比表），多数论文使用 **1–4 个数据集**，样本量在数百到千级。FlowCache 的 7 个数据集虽按 3 层角色（主表 / 质量面 / 鲁棒性压力面）分层，但：

1. **数据集数维度偏高**：7 个核心数据集是 CacheGen（4）、vLLM/PagedAttention（2）、τ-bench 原论文（1）、KVFlow（~2）的 1.75–7 倍；仅 EvicPress（12）和 Ada-KV（29 子任务，实为 2 个 benchmark family）超过 FlowCache。
2. **总样本量偏高**：4,120 是 CacheGen（662）的 6.2×、EvicPress（~600）的 6.9×、vLLM×Mooncake blog（610）的 6.8×、τ-bench 原论文（1,320）的 3.1×。
3. **Ch.5 鲁棒性 3 个压力数据集（STB 500 + SWE 200 + Toolathlon 200 = 900）边际价值存疑**：同领域论文鲁棒性章节通常 0–1 个数据集；FlowCache 已有 τ-bench retail/airline 两域作为天然 family-out，STB/SWE/Toolathlon 三者是否都必要需重新审视。
4. **算力约束**：14 周硬约束下，每多 200 episodes 录制约增加 ~1.5 GPU 小时；3 个压力数据集合计 ~900 episodes 占用 ~7 GPU 小时，若砍掉 2 个可省 ~4.5 GPU 小时用于主线实验复跑。

## 研究对比表

### A. KV Cache 管理 / 前缀缓存论文（核心可比对象）

| # | 论文 | Venue/Year | 数据集数 | 数据集名称 | 单数据集样本量 | 总样本量 | 来源 |
|---|---|---|---|---|---|---|---|
| 1 | **CacheGen** | SIGCOMM 2024 (arXiv 2310.07240) | **4** | LongChat, TriviaQA, NarrativeQA, 第4个 TBD | ~150–200 contexts/数据集 | **662 contexts** | 论文 §1 摘要原文："four datasets (662 contexts in total)" |
| 2 | **EvicPress** | arXiv 2512.14946 (2025-12) | **12** | 12 个数据集名称 TBD（论文未在摘要列出） | ~50/数据集（估算） | **~600 contexts**（估算） | 论文摘要："Evaluation on 12 datasets and 5 models" |
| 3 | **KVFlow** | NeurIPS 2025 (arXiv 2507.07400) | **~2** | synthetic workflow (configurable branches) + PEER | 未明确 | 未明确 | 论文 §4 实验："4096/32/32 with branches=1" 等参数化合成 |
| 4 | **vLLM / PagedAttention** | SOSP 2023 | **2** | ShareGPT (LMSYS-Chat-1M), Alpaca | ~500–1000s requests/batch | ~1,000s requests/batch | 论文 §10 评估 |
| 5 | **SGLang / RadixAttention** | NeurIPS 2024 (arXiv 2312.07104) | **~4–5** | MMLU, HellaSwag, GSM-8K, ShareGPT, MT-Bench | ~100–1000s/数据集 | ~1,000s 总样本 | 论文 §5 评估：few-shot / agent / chat / RAG 多场景 |
| 6 | **Mooncake** | FAST 2025 (arXiv 2407.00079) | **1 (生产 trace)** | Kimi 真实生产 trace | 百万级 requests/天 | 生产部署（不可比） | 论文 §5：真实 workload，日处理 100B+ tokens |
| 7 | **vLLM × Mooncake agentic** | vLLM blog 2026-05 | **1** | SWE-bench Pro | 610 traces | **610 traces** | blog 工作负载画像 |
| 8 | **τ-bench 原论文** | ICLR 2025 (arXiv 2406.12045) | **1** | τ-bench (165 tasks) | 165 × 8 seeds = 1,320 | **1,320 episodes** | 论文主表 pass^k 评估 |
| 9 | **FlowKV** | arXiv 2504.03775 (2025-04) | **1** | LongBench | 未明确 | 未明确 | 论文 §4：LongBench 数据集 |
| 10 | **Ada-KV** | NeurIPS 2025 | **2 benchmark families (29 子任务)** | Ruler (13 tasks) + LongBench (16 tasks) | 各子任务 ~100–500 | ~2,900 总样本（估算） | 论文摘要：13+16 datasets，实为子任务 |

### B. 长上下文 / KV 压缩论文（补充参考）

| # | 论文 | Venue/Year | 数据集数 | 总样本量 |
|---|---|---|---|---|
| 11 | **MInference** | arXiv 2407.02490 (2024-07) | ~10+ | ~1,000s（LongBench/InfiniteBench/RULER 等） |
| 12 | **Lossless KV Cache Compression to 2%** | arXiv 2410.15252 (2024-10) | ~5–10 | 未明确 |
| 13 | **LaCache** | ICML 2025 | 多 benchmark | 未明确 |
| 14 | **KVP (KV Policy)** | ICLR 2026 投稿（拒） | 2 | RULER + OASST2-4k |

### C. 关键统计

| 统计维度 | 数值 |
|---|---|
| 同领域论文数据集数中位数 | **2 个**（排除生产 trace 论文） |
| 同领域论文数据集数范围 | 1–4 个（排除 EvicPress 12 和 Ada-KV 29 子任务） |
| 同领域论文总样本量中位数 | **~660 contexts** |
| 同领域论文总样本量范围 | 150–1,320 episodes |
| **FlowCache v0.3** | **7 个数据集 / ~4,120 样本** |
| FlowCache / 中位数倍数 | 数据集 3.5× / 样本 6.2× |

### D. 核心结论

- **主表（τ-bench 1,320 + BFCL 800 = 2,120）合理**：与 τ-bench 原论文（1,320）和 vLLM×Mooncake（610）同量级或略高，且需覆盖两个工具 workload family。
- **质量面（LongBench 1,000 + GSM8K 100 = 1,100）合理**：LongBench 是 KV 压缩论文（FlowKV、Ada-KV、MInference）的标准 benchmark；GSM8K 100 仅作 accuracy sanity，样本量小成本可忽略。
- **鲁棒性压力面（STB 500 + SWE 200 + Toolathlon 200 = 900）偏多**：
  - 同领域论文鲁棒性章节通常 0–1 个数据集（CacheGen 无鲁棒性章；τ-bench 原论文仅 retail/airline 两域 family-out）
  - FlowCache 已有 τ-bench retail/airline 两域作为天然 family-out
  - SWE 200 和 Toolathlon 200 各自样本量偏小（200），单独成章证据力弱
  - 三者合计 900 episodes 占用 ~7 GPU 小时，是 v0.3 Tier-1 录制预算的 ~27%

## What Changes

### 方案：Ch.5 鲁棒性压力面从 3 个数据集精简为 1 个

- **保留 StableToolBench 500**（family-out 主证据，工具家族外推）
- **删除 SWE 轨迹 200**（压力面证据力弱，200 样本不足以单独成章；若 rebuttal 需要可补）
- **删除 Toolathlon 200**（同上；多 agent 协作场景与 FlowCache 主线 C1–C3 关联弱）
- **核心数据集数：7 → 5**（主表 2 + 质量面 2 + 鲁棒性 1）
- **核心样本总量：~4,120 → ~3,720**（降幅 10%；1,320+800+1,000+100+500=3,720）
- **节省 ~4.5 GPU 小时**（SWE 1.5h + Toolathlon 3h）
- 鲁棒性章节 3 轴变 2 轴：family-out（STB 500）+ 到达扰动（BurstGPT 窗口）；branch 噪声用 τ-bench 内部 replay 扰动覆盖，不另设数据集
- **BREAKING**：撤销 experiment-scope-redesign §5 的 7 数据集封顶，改为 5 数据集

### 不变的部分

- 主表 τ-bench 1,320 + BFCL 800 = 2,120（与 reconsider-g1-sample-size spec 一致）
- 质量面 LongBench 1,000 + GSM8K 100 = 1,100
- 辅助数据集 BurstGPT 窗口（到达结构 replay 参数，不产生 workflow 样本）
- 辅助数据集 LMSYS-Chat-1M 500（Ch.1 画像附注，负对照）

## Impact

- Affected specs:
  - [experiment-scope-redesign/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/experiment-scope-redesign/spec.md) §5 数据集体系、§3 Ch.5 表格、§8 规模对比表
  - [experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) §0.4 数据集组合、Ch.5 章节设计
  - [reconsider-g1-sample-size/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/reconsider-g1-sample-size/spec.md) 总样本量从 2,120 调整描述（主表仍 2,120，但全册总量从 ~4,120 降到 ~3,720）
- Affected code:
  - [experiments/e1/config.yaml](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/config.yaml) workload.datasets（移除 swe_trajectory、toolathlon）
  - [experiments/e1/record_trajectories.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/record_trajectories.py) 移除 SWE/Toolathlon adapter 调用（若有）
- 受益：
  - 节省 ~4.5 GPU 小时
  - 数据集数从 7 降到 5，与同领域论文（中位数 2）的差距从 3.5× 缩小到 2.5×
  - 鲁棒性章节聚焦 STB 500 单一压力面，证据更集中
  - 减少 2 个数据集的集成/适配工作（SWE 轨迹收集、Toolathlon 适配器）

## ADDED Requirements

### Requirement: Ch.5 鲁棒性压力面单一数据集

Ch.5 鲁棒性章节仅使用 1 个压力数据集（StableToolBench 500）做 family-out 评估，不再引入 SWE 轨迹和 Toolathlon。branch 噪声轴用 τ-bench 内部 replay 扰动覆盖（删边/错标后继），不另设数据集。

#### Scenario: Ch.5 数据集使用
- **WHEN** 运行 Ch.5 鲁棒性实验
- **THEN** 系统仅使用 StableToolBench 500 作为 family-out 压力数据集
- **AND** 不录制 SWE 轨迹和 Toolathlon episodes
- **AND** branch 噪声轴通过 τ-bench trace 的 replay 时特征扰动产生（属实验操作，非数据集）

#### Scenario: 到达扰动轴
- **WHEN** 运行到达扰动实验
- **THEN** 使用 BurstGPT 窗口作为到达结构 replay 参数（辅助数据集，不产生 workflow 样本）

### Requirement: 核心数据集数封顶 5

FlowCache 全册核心数据集封顶为 5 个：τ-bench（主表）、BFCL v3 multi-turn（主表）、LongBench（质量面）、GSM8K（质量面 sanity）、StableToolBench（Ch.5 family-out）。辅助数据集（BurstGPT 窗口、LMSYS-Chat-1M 500）不计入核心样本总量。

#### Scenario: 数据集数检查
- **WHEN** 审视 FlowCache 全册实验设计
- **THEN** 核心数据集数 = 5（不超过封顶）
- **AND** 核心样本总量 ≈ 3,720（τ-bench 1,320 + BFCL 800 + LongBench 1,000 + GSM8K 100 + STB 500）
- **AND** 辅助数据集 2 个（BurstGPT、LMSYS-Chat-1M）不计入核心总量

## MODIFIED Requirements

### Requirement: 全册数据集体系

**原（experiment-scope-redesign v0.3 §5）**：7 个核心数据集（τ-bench、BFCL、LongBench、GSM8K、STB、SWE、Toolathlon），~4,120 样本。

**现（本 spec）**：5 个核心数据集（τ-bench、BFCL、LongBench、GSM8K、STB），~3,720 样本。SWE 和 Toolathlon 删除，理由：① 同领域论文鲁棒性章通常 0–1 个数据集；② 200 样本单独成章证据力弱；③ 与 C1–C3 主线关联弱；④ 节省 ~4.5 GPU 小时。

## REMOVED Requirements

### Requirement: SWE 轨迹数据集

**Reason**: 200 样本不足以单独成章；同领域论文鲁棒性章通常 0–1 个数据集；SWE 解决率低（Qwen2.5-7B 小模型能力不足）；与 FlowCache C1–C3 主线（trace 协议 / 联合控制器 / reuse-fidelity 错位）关联弱。

**Migration**: 若 rebuttal 时审稿人要求补 SWE 压力面，可扩展到 500 episodes 补做；当前不做。

### Requirement: Toolathlon 数据集

**Reason**: 200 样本证据力弱；多 agent 协作场景与 FlowCache 单 agent + 工具 workload 主线偏离；适配器集成成本高。

**Migration**: 若 rebuttal 时需要多 agent 场景证据，可引用 τ-bench retail/airline 两域作为 workload 多样性证据；当前不做。

## 与已有 spec 的关系

- **experiment-scope-redesign**：本 spec 修改其 §5 数据集体系（7→5）和 §8 规模对比表
- **reconsider-g1-sample-size**：本 spec 不修改 BFCL seeds 决定（BFCL 仍单 seed 800），但全册总量描述从 ~4,120 调整为 ~3,220
- **g0-exactness-loadability**：不受影响（G0 用真实数据集结构用例，不涉及 SWE/Toolathlon）
