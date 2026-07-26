# Checklist

## IDEA.rewritten.md §6.1 数据集组合
- [x] §6.1 主表 workload 表格仅含 τ-bench 1 行，样本量 1,320，删除 BFCL v3 行
- [x] §6.1 Ch.5 鲁棒性压力面表（StableToolBench/SWE/Toolathlon）已删除
- [x] §6.1 Ch.3 fidelity 质量面表（LongBench/GSM8K）已删除
- [x] §6.1 辅助角色表删除 LMSYS-Chat-1M 行，保留 BurstGPT 行
- [x] §6.1 "已删除的 workload" 段落新增 BFCL/LongBench/GSM8K/STB/SWE/Toolathlon/LMSYS-Chat-1M
- [x] §6.1 末尾 "14 周版本" 段落改为 "1 核心数据集（τ-bench 1,320）+ 1 辅助（BurstGPT 窗口）"

## IDEA.rewritten.md §7-§8 各章数据来源
- [x] §7 G1 运行方式数据来源改为 "τ-bench 1,320"
- [x] §7 G4 运行方式数据来源改为 "τ-bench 1,320 episodes，复用 Ch.1 trace"（额外修复 line 590）
- [x] §8 Ch.1 数据来源改为 "τ-bench 1,320 episodes"
- [x] §8 Ch.2 数据来源标注 "从 1,320 episodes 中抽样 80 workflow 子集"
- [x] §8 Ch.3 fidelity 侧数据改为 "τ-bench 1,320 episodes（复用 Ch.1 trace）"
- [x] §8 Ch.4 cell 表删除 BFCL 行（主-2/主-4/边界-2）
- [x] §8 Ch.4 运行量段落重新计算总 replay 数（50 replay）
- [x] §8 Ch.5 鲁棒性轴仅保留到达扰动，删除 family-out 和 branch 噪声
- [x] §8 Ch.5 降级附录段落删除 SWE/Toolathlon 引用

## IDEA.rewritten.md §12 14 周执行计划
- [x] §12 W3–W5 行改为 "τ-bench 1,320 episodes 轨迹录制"，窗口压缩为 W3–W4
- [x] §12 W9 末行封顶改为 "1,320"
- [x] §12 W12 行改为 "Ch.5 鲁棒性（到达扰动 replay）"
- [x] §12 引言段落更新周次安排说明

## experiments/e1/config.yaml
- [x] workload.datasets = ["tau-bench"]（删除 "bfcl_v3"）
- [x] workload.bfcl_v3 配置块已删除
- [x] 文件头部注释改为 "Single dataset (tau-bench) × 8 seeds = 1320 episodes"
- [x] output.trace_subdirs = ["tau_bench"]
- [x] seeds 保持 8 个不变

## IDEA.rewritten.md §10 参考文献（可选）
- [x] §10 删除 BFCL/StableToolBench/MuSiQue/2WikiMultihopQA/LMSYS-Chat-1M 链接，保留 τ-bench 链接

## 一致性验证
- [x] IDEA.rewritten.md 中 grep "BFCL|LongBench|GSM8K|StableToolBench|Toolathlon|SWE 轨迹|LMSYS" 无未更新残留（5 处匹配均为"已删除"段落或 rebuttal 风险声明，符合预期）
- [x] experiments/e1/config.yaml 中无 bfcl_v3 残留
- [x] 核心样本总量在 IDEA.rewritten.md 和 config.yaml 中一致为 1,320
