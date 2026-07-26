# Tasks

- [ ] Task 1: 回退 experiment-designs.md G1.2/G1.3 表格到 BFCL 800（单 seed）
  - [ ] SubTask 1.1: 修改 G1.2 表格 BFCL 行：6,400 → 800，移除 "8 decode seeds" 描述
  - [ ] SubTask 1.2: 修改 G1.3 表格样本量行：7,720 → 2,120
  - [ ] SubTask 1.3: 移除 "BFCL 8 decode seeds 依据（2026-07-26 用户确认）" 段落
  - [ ] SubTask 1.4: 添加 "BFCL 单 seed 依据" 段落，引用同领域论文样本量对比

- [ ] Task 2: 更新 config.yaml BFCL seeds 配置
  - [ ] SubTask 2.1: 修改 workload.seeds 字段：移除 BFCL decode seeds 的 8 个值
  - [ ] SubTask 2.2: 新增 workload.bfcl_v3.seeds: [0]（单 seed）字段
  - [ ] SubTask 2.3: 修改 workload.bfcl_v3.decode_mode: "sampling" → "greedy"
  - [ ] SubTask 2.4: 更新 workload.seeds 注释，明确仅用于 τ-bench

- [ ] Task 3: 更新 record_trajectories.py BFCL seed 循环逻辑
  - [ ] SubTask 3.1: 修改 `_record_bfcl_g1` 方法：BFCL seeds 从全局 seeds 改为 bfcl_v3.seeds（单 seed）
  - [ ] SubTask 3.2: 修改 `_init_adapter` 调用：BFCL seed=0, do_sample=False
  - [ ] SubTask 3.3: 修改 `_run_episode_bfcl` 调用 `_generate_response`：seed=None（greedy decode）
  - [ ] SubTask 3.4: 添加防回归测试：BFCL trace 文件数 = 800（不是 6400）

- [ ] Task 4: 更新 g1-experiment-implementation.md §2.1 算力预算表
  - [ ] SubTask 4.1: BFCL 录制预算：36 GPU 小时 → 4.5 GPU 小时
  - [ ] SubTask 4.2: Tier-1 总录制预算：~50 GPU 小时 → ~15.5 GPU 小时
  - [ ] SubTask 4.3: 更新 §2 决策表 BFCL 多 seed 行

- [ ] Task 5: 运行全部测试验证无回归
  - [ ] SubTask 5.1: py -m pytest experiments/e1/tests/ -v，确保所有测试 PASS
  - [ ] SubTask 5.2: 检查 config.yaml 加载正确

# Task Dependencies

- Task 1, 2 可并行（独立文档/配置修改）
- Task 3 依赖 Task 2（config 字段先定义）
- Task 4 可并行（独立文档）
- Task 5 依赖 Task 2, 3
