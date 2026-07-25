# Tasks

- [x] Task 1: 修改 IDEA.rewritten.md §6.1 最小 workload 组合
  - [x] SubTask 1.1: 主表 workload 严格限定 τ-bench 495 + BFCL 800；删除合成 DAG、MuSiQue、2WikiMultihopQA、LMSYS 作为主 workload 的描述
  - [x] SubTask 1.2: 新增 Ch.5 压力面 workload 描述（STB 500 / SWE 200 / Toolathlon 200）和 Ch.3 质量面（LongBench 1000 / GSM8K 100）
  - [x] SubTask 1.3: 更新末段"14 周版本只保留..."为反映 v0.3 的 7 核心 + 2 辅助数据集体系

- [x] Task 2: 修改 IDEA.rewritten.md §7 可行性门槛
  - [x] SubTask 2.1: G1/G2/G4 各增加一句"数据来源改为复用正式实验数据（Ch.x）"，更新 design_doc 引用
  - [x] SubTask 2.2: G3 增加 W8 冒烟前置描述（主 cell × 4 无损对照 × 100 子集）与主表最终确认两时点
  - [x] SubTask 2.3: 删除 G5（Learning）整节；在 §7 开头或末尾加一句说明"G5 已删除，GNN 不启用是设计选择"

- [x] Task 3: 修改 IDEA.rewritten.md §8 正式实验计划
  - [x] SubTask 3.1: 将 E1–E7 七个独立小节重构为 Ch.1–Ch.5 五个小节，保留各实验的指标和成功标准
  - [x] SubTask 3.2: Ch.4 主表部分按 v0.3 写明 10 对照 × 6 cell（对照 13→10、cell 18→6）；核心 4 变体 + 2 设计消融同表
  - [x] SubTask 3.3: Ch.3 写明 reuse 侧 2 变体 + fidelity 侧 2 变体，GNN 删除
  - [x] SubTask 3.4: Ch.5 写明 3 轴（family-out、到达扰动、branch 噪声）+ 失败模式从 Ch.4 负结果提取

- [x] Task 4: 修改 IDEA.rewritten.md §11 路线切换
  - [x] SubTask 4.1: route A 描述中去掉 G5 相关；确认切换条件为 G0/G1/G2/G3/G4 任一失败
  - [x] SubTask 4.2: 删除风险表中"G5 失败"相关行（若有）；GNN 相关风险合并到"简单 heuristic 接近 oracle"行

- [x] Task 5: 修改 IDEA.rewritten.md §12 14 周执行计划
  - [x] SubTask 5.1: 周次表替换为 v0.3 的 10 行版本（W1–W2 G0、W3–W5 录制、W6–W7 画像、W7–W8 Pilot、W8 G3 冒烟、W9 fidelity、W9 末标定、W10–W11 主表、W12 鲁棒性、W13–W14 冻结写作）
  - [x] SubTask 5.2: 更新表后说明：G5 已删除；GNN 和第二模型仅在主结果稳定且仍有时间时加入（保留原句，但 G5 不再作为 gate）

- [x] Task 6: 修改 IDEA.rewritten.md §14 写作就绪条件
  - [x] SubTask 6.1: 移除 G5 相关条件；确认 G0–G3 通过、G2/G4 通过条件不变

- [x] Task 7: 修改 ccfa.yaml gates 段
  - [x] SubTask 7.1: 删除 G5 条目
  - [x] SubTask 7.2: G1 的 design_doc 改为 `experiments/experiment-designs.md#ch1`，week 保持 `W6-W7`
  - [x] SubTask 7.3: G2 的 design_doc 改为 `experiments/g2-pilot-design.md; experiments/experiment-designs.md#ch4`，week 改为 `W7-W8`
  - [x] SubTask 7.4: G3 增加 `smoke_check` 子字段（W8 冒烟描述）；design_doc 改为 `experiments/experiment-designs.md#ch4`，week 改为 `W8, W10-W11`
  - [x] SubTask 7.5: G4 的 design_doc 改为 `experiments/experiment-designs.md#ch3`，week 保持 `W9`

- [x] Task 8: 修改 ccfa.yaml experiments 段
  - [x] SubTask 8.1: 将 E1–E7 重构为 Ch.1–Ch.5 五个条目，更新 description/week/gates/design_doc
  - [x] SubTask 8.2: Ch.1 保留 E1 的 scripts 列表；其余 Ch.x 无 scripts 字段（待 experiment-designs.md 重构时补）

- [x] Task 9: 修改 ccfa.yaml stage 段
  - [x] SubTask 9.1: updated_at 更新为 `2026-07-25`
  - [x] SubTask 9.2: 新增 `scope_spec: ".trae/specs/experiment-scope-redesign/spec.md"` 字段

# Task Dependencies

- Task 2 依赖 Task 1（§7 引用 §6.1 的 workload）
- Task 3 依赖 Task 2（§8 引用 §7 的 Gate）
- Task 5 依赖 Task 3（§12 周次表引用 §8 的实验）
- Task 7/8/9 相互独立，可与 Task 1–6 并行（yaml 与 md 是不同文件）
