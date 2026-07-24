# Tasks

## Task 1: 创建 E1 实验目录结构与配置
- [x] Task 1.1: 创建 `experiments/e1/` 目录结构
  - [x] SubTask 1.1.1: 创建 `experiments/e1/traces/bf16/` 轨迹存储目录
  - [x] SubTask 1.1.2: 创建 `experiments/e1/outputs/` 报告输出目录
  - [x] SubTask 1.1.3: 创建 `experiments/e1/figures/` 图表输出目录
- [x] Task 1.2: 创建 `experiments/e1/config.yaml` 配置文件
  - 定义模型（Qwen3-8B-Instruct）、τ-bench 子集（80 workflows）、block_size、KV budget、到达率等参数

## Task 2: 实现轨迹录制脚本
- [x] Task 2.1: 实现 `experiments/e1/record_trajectories.py`
  - [x] SubTask 2.1.1: 加载 Qwen3-8B-Instruct 模型和后端（transformers）
  - [x] SubTask 2.1.2: 加载 τ-bench 80 workflow 子集（从 g2-pilot-subset.json 或 tau-bench data 目录）
  - [x] SubTask 2.1.3: 逐个运行 workflow，记录每步 token IDs、工具调用参数、工具返回结果
  - [x] SubTask 2.1.4: 记录 block 分配日志（token 范围 → block hash 映射，SHA-256 16 hex）
  - [x] SubTask 2.1.5: 记录每步 prefill/decode 时间（perf_counter 毫秒级）
  - [x] SubTask 2.1.6: 将每条轨迹保存为 JSON（含 meta、steps、global_block_index）
- [x] Task 2.2: 实现 `experiments/e1/trace_utils.py` 公共工具函数
  - [x] SubTask 2.2.1: block identity 哈希函数（compute_block_hash，含 token_ids + parent_hash）
  - [x] SubTask 2.2.2: 父链计算函数（compute_parent_chain，验证链完整性）
  - [x] SubTask 2.2.3: 跨 workflow prefix 去重函数（deduplicate_blocks，统计 share_count）

## Task 3: 实现工作负载画像脚本
- [x] Task 3.1: 实现 `experiments/e1/characterize_workload.py`
  - [x] SubTask 3.1.1: 计算 workflow 结构统计（长度、深度、宽度、分支率、工具等待时长）
  - [x] SubTask 3.1.2: 计算 exact-prefix overlap ratio（跨 workflow 共享 prefix 占比）
  - [x] SubTask 3.1.3: 计算 LCP tokens 分布（最长公共前缀统计）
  - [x] SubTask 3.1.4: 计算 next-use distance 分布（block 首次出现到再次访问的 step 数）
  - [x] SubTask 3.1.5: 计算 block working-set size 和 KV/总显存占比（Qwen3-8B: 36层/GQA）
  - [x] SubTask 3.1.6: 输出 JSON 格式画像报告和 Markdown 表格

## Task 4: 实现 Oracle vs Heuristic 对比脚本
- [x] Task 4.1: 实现 `experiments/e1/compare_oracle.py`
  - [x] SubTask 4.1.1: 实现 LRU 驱逐策略（OrderedDict，O(1) 访问）
  - [x] SubTask 4.1.2: 实现 GDSF 策略（min-heap + lazy deletion，priority = clock + freq）
  - [x] SubTask 4.1.3: 实现离线 Belady oracle（bisect_right 二分查找，驱逐最远未来访问）
  - [x] SubTask 4.1.4: 实现 open-loop replay 模拟器（build_access_trace，跨轨迹展平）
  - [x] SubTask 4.1.5: 计算 saved-prefill ms、miss-cost、cache hit rate（4 个 budget 级别）
  - [x] SubTask 4.1.6: 输出对比 JSON 和终端摘要表格（含 oracle headroom）

## Task 5: 实现可视化脚本
- [x] Task 5.1: 实现 `experiments/e1/plot_characterization.py`
  - [x] SubTask 5.1.1: exact-prefix overlap 分布直方图（含均值垂直线）
  - [x] SubTask 5.1.2: next-use distance CDF 曲线（含 log-scale 自适应，标注 median/p95/p99）
  - [x] SubTask 5.1.3: KV working-set size 图（含 KV budget 阈值线）
  - [x] SubTask 5.1.4: oracle vs LRU/GDSF 分组柱状图（含 headroom 标注）
  - [x] SubTask 5.1.5: 所有图表保存到 `experiments/e1/figures/`（300 DPI）

## Task 6: 同步 ccfa.yaml
- [x] Task 6.1: 更新 `ccfa.yaml` 的 E1 experiment 字段
  - [x] SubTask 6.1.1: 添加 E1 条目的 scripts 字段（5 个脚本路径）
  - [x] SubTask 6.1.2: 更新 G1 gate 的 depends_on 从 ["G0"] 为 ["G0", "E1"]

# Task Dependencies
- Task 2 依赖 Task 1（目录结构和配置）
- Task 3 依赖 Task 2（需要录制的轨迹作为输入）
- Task 4 依赖 Task 2（需要轨迹和 block 信息）
- Task 5 依赖 Task 3 和 Task 4（需要画像数据和对比结果）
- Task 6 可与其他 Task 并行，最后提交时整合
