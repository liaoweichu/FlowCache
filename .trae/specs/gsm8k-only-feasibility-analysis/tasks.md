# Tasks

- [x] Task 1: 更新 survey-2025-2026-kv-cache-agent-papers.md 中 QKVShare 条目
  - [x] SubTask 1.1: 在 QKVShare 条目的 Venue/Year 字段标注"arXiv 预印本，未被任何会议/期刊正式接收"
  - [x] SubTask 1.2: 在 QKVShare 详细信息中补充 GSM8K 用法（150 problems × 2-5 hops，inter-agent handoff）
  - [x] SubTask 1.3: 在 QKVShare 条目中补充"与 FlowCache 的场景差异"段落（inter-agent vs intra-agent）
  - [x] SubTask 1.4: 补充作者自承认局限（拓扑感知控制器未显优势）

- [x] Task 2: 更新 experiment-designs.md §0.4 数据集论证
  - [x] SubTask 2.1: 在 §0.4.7（或新增 §0.4.8）补充"为何不能只用 GSM8K"论证段落
  - [x] SubTask 2.2: 论证段落引用 QKVShare 的 inter-agent handoff 场景与 FlowCache 的 intra-agent multi-turn 场景的根本差异
  - [x] SubTask 2.3: 论证段落包含 C1/C2/C3 三条主张与 GSM8K 任务结构的匹配度分析表

- [x] Task 3: 验证 GSM8K 样本量与角色在所有文档中一致
  - [x] SubTask 3.1: 检查 experiment-designs.md §0.4 数据集组合表 GSM8K 行：100 samples, accuracy sanity (Ch.3)
  - [x] SubTask 3.2: 检查 trim-dataset-portfolio/spec.md 中 GSM8K 行：100 samples
  - [x] SubTask 3.3: 检查 experiment-scope-redesign/spec.md §5 中 GSM8K 行：100 samples

# Task Dependencies

- Task 1, 2, 3 相互独立，可并行
