# Checklist

## 目录结构
- [x] `experiments/e1/` 目录存在
- [x] `experiments/e1/traces/bf16/` 子目录存在
- [x] `experiments/e1/outputs/` 子目录存在
- [x] `experiments/e1/figures/` 子目录存在
- [x] `experiments/e1/config.yaml` 配置文件存在且参数可修改

## 轨迹录制（Task 2）
- [x] `experiments/e1/record_trajectories.py` 已实现（含完整 TrajectoryRecorder 类）
- [x] 使用 Qwen3-8B-Instruct 模型（非 Qwen2.5-7B）
- [x] τ-bench 子集加载：支持 g2-pilot-subset.json / tau-bench data 目录 / 内置合成 fallback
- [x] 每条轨迹 JSON 包含 meta（workflow_id, domain, model）、steps（token_ids, role, tool_call, tool_result）、global_block_index
- [x] prefill/decode 时间记录在轨迹 JSON 中（perf_counter 毫秒级）
- [x] `experiments/e1/trace_utils.py` 提供 block identity 哈希（SHA-256 16 hex）、父链计算、去重函数
- [x] 零工具调用时记录错误日志并继续下一个 workflow

## 工作负载画像（Task 3）
- [x] `experiments/e1/characterize_workload.py` 已实现（4 个指标函数）
- [x] 输出 E1 画像 JSON：workflow 长度/深度/宽度/分支率/工具等待时长
- [x] 输出 exact-prefix overlap ratio 和 LCP tokens 分布
- [x] 输出 next-use distance 分布（统计量：mean, median, p95, p99）
- [x] 输出 block working-set size 和 KV/总显存占比（Qwen3-8B: 36层/GQA 8 KV heads）
- [x] 输出 Markdown 格式表格到 `experiments/e1/outputs/e1-report.md`
- [x] 使用真实轨迹数据计算（读取 JSON 轨迹文件），不伪造数值

## Oracle vs Heuristic 对比（Task 4）
- [x] `experiments/e1/compare_oracle.py` 已实现（3 个策略类）
- [x] LRU 策略正确实现（OrderedDict，popitem O(1) 驱逐最旧）
- [x] GDSF 策略正确实现（min-heap + lazy deletion，priority = clock + freq）
- [x] Belady oracle 正确实现（bisect_right 二分查找最远未来访问，无未来访问 → sys.maxsize 优先驱逐）
- [x] open-loop replay 模拟器支持可配置的 KV budget（10%/25%/50%/100%）
- [x] 输出 saved-prefill ms、miss-cost、cache hit rate 对比表
- [x] oracle headroom 数值来自真实轨迹 block 访问序列计算

## 可视化（Task 5）
- [x] `experiments/e1/plot_characterization.py` 已实现（4 个绘图函数）
- [x] 生成 exact-prefix overlap 分布直方图
- [x] 生成 next-use distance CDF 图（含 log-scale 自适应）
- [x] 生成 KV working-set size 图（含 budget 阈值线）
- [x] 生成 oracle vs LRU/GDSF 分组柱状图（含 headroom 标注）
- [x] 图表使用中英文标签，300 DPI 分辨率适合论文

## ccfa.yaml 同步（Task 6）
- [x] ccfa.yaml 中 E1 experiment 字段完整（id, description, status, week, gates, scripts）
- [x] E1 scripts 字段列出 `experiments/e1/` 下的 5 个脚本路径
- [x] G1 gate 的 depends_on 更新为 ["G0", "E1"]
