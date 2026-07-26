# Tasks

- [x] Task 1: 修改 IDEA.rewritten.md §6.1 最小 workload 组合
  - [x] SubTask 1.1: 修改 §6.1 主表 workload 表格，删除 BFCL v3 行，τ-bench 样本从 495 改为 1,320
  - [x] SubTask 1.2: 删除 §6.1 Ch.5 鲁棒性压力面表（StableToolBench/SWE/Toolathlon 三行）
  - [x] SubTask 1.3: 删除 §6.1 Ch.3 fidelity 质量面表（LongBench/GSM8K 两行）
  - [x] SubTask 1.4: 删除 §6.1 辅助角色表中的 LMSYS-Chat-1M 行，保留 BurstGPT 行
  - [x] SubTask 1.5: 修改 §6.1 "已删除的 workload" 段落，新增 BFCL/LongBench/GSM8K/STB/SWE/Toolathlon/LMSYS-Chat-1M 为本次删除
  - [x] SubTask 1.6: 修改 §6.1 末尾 "14 周版本按 v0.3 的 7 核心 + 2 辅助数据集体系" 段落，改为 "1 核心数据集（τ-bench 1,320）+ 1 辅助（BurstGPT 窗口）"

- [x] Task 2: 修改 IDEA.rewritten.md §7-§8 各章数据来源
  - [x] SubTask 2.1: 修改 §7 G1 运行方式（line 570），数据来源从 "τ-bench 495 + BFCL 800" 改为 "τ-bench 1,320"
  - [x] SubTask 2.2: 修改 §8 Ch.1 数据来源（line 623），从 "τ-bench 495 + BFCL v3 multi-turn 800" 改为 "τ-bench 1,320 episodes"
  - [x] SubTask 2.3: 修改 §8 Ch.2 数据来源（line 640），从 "τ-bench 80 workflow 子集" 改为 "τ-bench 80 workflow 子集（从 1,320 episodes 中抽样）"
  - [x] SubTask 2.4: 修改 §8 Ch.3 fidelity 侧（line 661），从 "数据：LongBench 1000 + GSM8K 100" 改为 "数据：τ-bench 1,320 episodes（复用 Ch.1 trace）"
  - [x] SubTask 2.5: 修改 §8 Ch.4 cell 表（line 700-705），删除 BFCL 行（主-2/主-4/边界-2），保留 τ-bench 行并调整 seeds
  - [x] SubTask 2.6: 修改 §8 Ch.4 "运行量" 段落（line 707），重新计算总 replay 数
  - [x] SubTask 2.7: 修改 §8 Ch.5 鲁棒性轴（line 740-742），删除 family-out 和 branch 噪声轴，保留到达扰动轴
  - [x] SubTask 2.8: 修改 §8 Ch.5 降级附录段落（line 744），删除 SWE/Toolathlon 余量补做引用

- [x] Task 3: 修改 IDEA.rewritten.md §12 14 周执行计划
  - [x] SubTask 3.1: 修改 §12 周次表 W3–W5 行（line 839），从 "τ-bench 495 + BFCL 800 轨迹录制" 改为 "τ-bench 1,320 episodes 轨迹录制"，窗口从 W3–W5 压缩为 W3–W4
  - [x] SubTask 3.2: 修改 §12 W9 末行（line 844），封顶从 "495/800" 改为 "1,320"
  - [x] SubTask 3.3: 修改 §12 W12 行（line 846），从 "Ch.5 鲁棒性（STB 500 录制在此窗口）" 改为 "Ch.5 鲁棒性（到达扰动 replay）"
  - [x] SubTask 3.4: 修改 §12 引言段落（line 834），更新周次安排说明

- [x] Task 4: 修改 experiments/e1/config.yaml
  - [x] SubTask 4.1: 修改 workload.datasets 从 ["tau-bench", "bfcl_v3"] 改为 ["tau-bench"]
  - [x] SubTask 4.2: 删除 workload.bfcl_v3 整个配置块（subsets/per_subset/decode_mode）
  - [x] SubTask 4.3: 修改文件头部注释，从 "Multi-dataset (tau-bench + BFCL v3) × 8 seeds = 7720 episodes" 改为 "Single dataset (tau-bench) × 8 seeds = 1320 episodes"
  - [x] SubTask 4.4: 修改 output.trace_subdirs 从 ["tau_bench", "bfcl_v3"] 改为 ["tau_bench"]
  - [x] SubTask 4.5: 保留 seeds: [42, 123, 456, 789, 101112, 131415, 161718, 192021] 8 seeds 不变

- [x] Task 5: 修改 IDEA.rewritten.md §10 参考文献链接（可选）
  - [x] SubTask 5.1: 删除 §10 中 BFCL/StableToolBench/MuSiQue/2WikiMultihopQA 参考链接（line 932-937 中已删除数据集对应的链接），保留 τ-bench 链接

- [x] Task 6: 验证文档一致性
  - [x] SubTask 6.1: 在 IDEA.rewritten.md 中 grep "BFCL|LongBench|GSM8K|StableToolBench|Toolathlon|SWE 轨迹|LMSYS" 确认所有引用已更新或标注为"已删除"
  - [x] SubTask 6.2: 在 experiments/e1/config.yaml 中确认无 bfcl_v3 残留
  - [x] SubTask 6.3: 确认核心样本总量在 IDEA.rewritten.md 和 config.yaml 中一致为 1,320

# Task Dependencies

- Task 1, 2, 3 可并行（同一文件不同章节）
- Task 4 独立（不同文件）
- Task 5 依赖 Task 1-3 完成
- Task 6 依赖 Task 1-5 完成
