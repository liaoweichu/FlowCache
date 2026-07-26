# Tasks

- [x] Task 1: 更新 experiment-designs.md §0.4 数据集组合
  - [x] SubTask 1.1: 修改 §0.4.1 四层数据集组合表，删除 SWE 轨迹行和 Toolathlon 行
  - [x] SubTask 1.2: 修改 §0.4 末尾总量描述：~4,120 → ~3,720（修正 spec 算术笔误 3,220→3,720）
  - [x] SubTask 1.3: 修改 §0.4.4 排除数据集清单，新增 SWE 和 Toolathlon 行及排除理由
  - [x] SubTask 1.4: 在 §0.4 新增"数据集数与同领域论文对比"段落（§0.4.7），引用 spec 中的对比表

- [x] Task 2: 更新 experiment-designs.md Ch.5 鲁棒性章节
  - [x] SubTask 2.1: 修改 Ch.5 表格，删除 SWE 200 和 Toolathlon 200 行
  - [x] SubTask 2.2: 修改 Ch.5 鲁棒性轴从 3 轴调整为 2 轴（family-out / 到达扰动），branch 噪声用 τ-bench 内部 replay 扰动覆盖
  - [x] SubTask 2.3: 修改 Ch.5 §7.3 鲁棒性篇幅风险段落（E6.12 rebuttal 策略），更新应对策略

- [x] Task 3: 更新 experiment-scope-redesign/spec.md
  - [x] SubTask 3.1: 修改 §3 Ch.5 表格数据来源列，移除 SWE/Toolathlon
  - [x] SubTask 3.2: 修改 §5 数据集体系表，删除 SWE 200 和 Toolathlon 200 行，总量 ~4,120 → ~3,720
  - [x] SubTask 3.3: 修改 §5 "核心样本总量"行：~4,120 → ~3,720，降幅 53% → 58%
  - [x] SubTask 3.4: 修改 §8 规模对比表"核心数据集"行：7 → 5，降幅 42% → 58%
  - [x] SubTask 3.5: 修改 §7.3 鲁棒性篇幅风险段落
  - [x] 额外: §5 标题、表格标签、§9 IDEA 对应关系表一致性修复

- [x] Task 4: 更新 reconsider-g1-sample-size/spec.md 总样本量描述
  - [x] SubTask 4.1: 修改"总样本量封顶 2,120"段落，补充说明这是主表封顶；全册核心样本总量为 ~3,720
  - [x] SubTask 4.2: 在 Impact 部分添加对本 spec 的引用

- [x] Task 5: 更新 config.yaml
  - [x] SubTask 5.1: 检查 experiments/e1/config.yaml 是否有 swe_trajectory / toolathlon 配置（结果：无，无需修改）
  - [x] SubTask 5.2: 更新 workload.datasets 字段（结果：已是 ["tau-bench", "bfcl_v3"]，无需修改）

- [x] Task 6: 更新 g1-experiment-implementation.md 算力预算
  - [x] SubTask 6.1: 修改 §2.1 算力预算表（结果：原表无 SWE/Toolathlon 行，无需修改）
  - [x] SubTask 6.2: 更新 Tier-1 总录制预算（结果：原预算已只覆盖 τ-bench + BFCL，无需修改）
  - [x] SubTask 6.3: 在 §2 决策表新增"Ch.5 压力数据集精简"行（3→1，保留 STB 500）

- [x] Task 7: 运行测试验证无回归
  - [x] SubTask 7.1: py -m pytest experiments/e1/tests/ -v —— 43 passed, 3 skipped
  - [x] SubTask 7.2: 检查 config.yaml 加载正确，无 swe_trajectory / toolathlon 引用 —— 确认无引用

# Task Dependencies

- Task 1, 2 可并行（同文档不同章节）
- Task 3, 4, 6 可并行（独立文档）
- Task 5 依赖 Task 1（确认 config 字段范围）
- Task 7 依赖 Task 5
