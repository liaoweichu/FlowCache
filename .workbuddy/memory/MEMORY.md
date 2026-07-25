# FlowCache 项目长期记忆

## 项目约定（跨会话有效）

- **主模型：Qwen2.5-7B-Instruct**（2026-07-25 用户指定，取代 IDEA §5.2 的 Qwen3-8B；备选 Llama-3.1-8B）。注意：IDEA.rewritten.md 与 g2-pilot-design.md 原文仍写 Qwen3-8B，属历史记录，以本约定与 `experiments/experiment-designs.md` Part 0.3 为准。
- **数据集禁令（2026-07-25 用户规定）**：实验中**禁止自建/合成数据集**；数据集种类与样本量须充足；选择须对齐同类论文实践。真实数据上的 replay 时扰动（删边/时间缩放/burst）属实验操作而非数据集，允许但须显式标注。
- 硬件：单卡 RTX 4090D 24GB；结论边界 = memory-constrained GPU emulation。
- 实验设计单一事实来源：`experiments/experiment-designs.md`（v0.2，G0/G1/G3/G4/G5 + E1–E7）；G2 见 `experiments/g2-pilot-design.md`（τ-bench 80 子集，独立成立）。ccfa.yaml 各条目有 design_doc 字段。
- 统一参数：block_size=16；KV 预算档 10/25/50/100%；β=0.005/step；H=1000 step；到达主证据 BurstGPT（Poisson λ=4 为参照）；统计单位=workflow（paired bootstrap 95% CI，1000 次）；open-loop 与 closed-loop 结果绝不混表。
- 数据集组合（v0.2，~8,800 样本）：τ-bench 165×3=495 / BFCL v3 800 / StableToolBench 500 / SWE-rebench 轨迹 500 / Toolathlon 500 / CATraces ~150 / LongBench 1000 / GSM8K 300 / MuSiQue/2Wiki 各 300 / LMSYS 2000（负对照）/ BurstGPT 2000 / Mooncake 抽样。排除：合成 DAG、ToolBench（CC BY-NC）、GAIA、WebArena/OSWorld（基建重）。
- 时间线（v0.2）：G0 W1–W2；录制 W3–W5（rollout ~33-40 GPUh）；G1 W6；E1 W7；G3/G5/E2 W7–W8；G4/G2/E3 W9–W10；E4/E5 W11；E6/E7 W12；W13 复跑；W14 稿件。
- 红线：设计文档中所有结果数字为 TBD，严禁发明实验结果；预注册字段（ε/δ/N、SLO 阈值、净收益阈值）在对应 pilot 后冻结并回写文档。
