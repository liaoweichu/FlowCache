# G3 分析、必改项与快速实验方案

> 更新：2026-07-30  
> 当前结论：**G3-P1 held-out 工程检查 PASS；正式 G3 仍为
> PROTOCOL-INCOMPLETE**。尚无修复后真实 GPU closed-loop 数值，不能声称
> TTFT/throughput 已通过。

## 1. 项目与 G3 主张

FlowCache 面向 agent workflow 的前缀 KV 复用。G3 只研究无损驻留动作：
`GPU BF16 ↔ CPU BF16 ↔ Evict`。核心主张是：相同 GPU/CPU KV 容量下，
价值感知的选择性 CPU 迁移应比 always-migrate 的公平双层缓存减少无效
搬运，并改善真实 p95 TTFT，同时吞吐下降不超过 5%。

| 主张 | 当前证据 | 状态 |
|---|---|---|
| 选择性迁移减少搬运 | held-out P1：171,292 vs 502,746 次迁移，减少 65.93% | 工程证据 PASS |
| 搬运/成本统计守恒 | D2H/H2D task 汇总与全局值一致，负成本=0，fallback=0 | PASS |
| 真实 p95 TTFT 改善 ≥15% | 旧 20-sample pilot 口径无效；修复后尚未跑 GPU | 缺失 |
| 吞吐下降 ≤5% | open-loop 只有 offered load，不能判定 | 缺失 |
| 优于公平双层强基线 | open-loop LRU/GDSF/SizeCost runner 已修；closed-loop 目前只有 LRU | 部分缺失 |

held-out P1 的 modeled p95 为 8,553.09 ms，always-migrate 为
8,413.86 ms，selective-migration-only 为 8,422.44 ms；这说明搬运已经
明显减少，但 modeled cache delay 没有证明端到端收益。因此下一步必须测
真实 serving，而不是继续从 open-loop 推断 TTFT。

## 2. 已完成的必改项

| 问题 | 风险 | 修复 |
|---|---|---|
| 全量 grid 漏掉 `two_tier_fair` | 声称跑了强基线，实际未运行 | 纳入 LRU/GDSF/SizeCost 全量组 |
| two-tier 无 `get_stats()` | 全局迁移次数/时间错误显示为 0 | 统一输出迁移、恢复、命中和成本统计 |
| GDSF/SizeCost 的 CPU hit 不增频次 | 恢复块被系统性低估 | CPU restore 也计一次访问 |
| connector 使用 `kv_producer` | 只存不取，CPU hit/恢复可能失效 | 两策略均改为 `kv_both` |
| TTFT 用 `scheduled→first` | 排队时间被排除，无法代表用户延迟 | 改为同一 monotonic clock 的 `queued→first`，另报 queue/service TTFT |
| 增量提交的缺失输出被压缩 | 后续请求与 task 错配 | 结果按原索引保留空位 |
| 缺失/无效时间戳仍进入指标 | 0 ms 污染分位数，可能误判 | 直接标记失败并输出 `INCOMPLETE` |
| bootstrap 检验 task mean | 与主指标 p95 不一致 | 改为 task-cluster bootstrap 的全局 p95 差 |
| 过滤迁移后未释放 GPU ref | GPU block 被隐式 pin，容量不公平 | 同时释放 CPU allocation 与 GPU touch ref |
| ratio=0 仍强制迁移 1 block | 无法得到真正 never-migrate 消融 | ratio=0 现在迁移 0 block |
| 用 utilization 猜 2 GiB KV | 策略容量无法严格相同 | 显式传 `kv_cache_memory_bytes=2 GiB` |
| vLLM 使用宽松版本范围 | private API 变化可静默破坏实验 | 固定 `vllm==0.26.0` 并做启动预检 |
| P1 checker 把诊断项并入总门槛 | 已冻结结果被误报 FAIL | required/diagnostic 分离并重算为 PASS |

当前回归测试：**44/44 PASS**。

## 3. 推荐的快速实验

快速模式只跑主 cell：GPU KV 2 GiB、CPU KV 2 GiB、并发 4，只比较
`twotier_lru` 与 `flowcache_lossless`。

样本规则在运行前固定：

- airline / retail 各 4 个 task；
- 每个 task 选择 2 条最接近该 task 中位请求数的完整 workflow；
- task 先按 domain 中位 workload footprint 集中选择，再用固定
  SHA-256 seed 打破并列；
- 两策略使用完全相同的 8 task / 16 workflow；
- 不截断单个 workflow，不按性能结果挑样本。

在当前 1,320 条 trace 上，该规则得到 **294 个请求**，workflow 请求数
`min=11 / median=18 / max=27`；筛选同时考虑请求数和 prompt 字符量，
不会按任何性能结果选样本。这比旧的“取前 20 episodes、只覆盖
3 个 task”更集中且可配对。它只占完整双策略请求量约 1.2%，适合快速判断
方向，但因只含 8 个 task 且刻意集中 workload，必须标为 pilot。

```bash
python experiments/g3/closed_loop/run_closed_loop.py \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master \
  --trace-dir experiments/e1/traces/bf16/tau_bench \
  --output-dir experiments/g3/results/closed-loop-quick \
  --quick-pilot \
  --kv-cache-memory-gib 2.0 \
  --cpu-capacity-gib 2.0 \
  --gpu-memory-utilization 0.76 \
  --max-model-len 16384 \
  -v
```

快速模式自动维持固定 4 个 in-flight 请求（完成即补请求），使用 16 个输出
token，并写出 `quick-pilot-selection.json`。它不等待 trace 的空闲间隔，
因此比 arrival replay 更快，吞吐也代表饱和服务能力。结果只能是
`PILOT_PASS` /
`PILOT_FAIL` / `INCOMPLETE`，不能被 `g3_verdict.py` 转成正式 GO。

## 4. Pilot 后的决策

1. 若为 `INCOMPLETE`：只修协议/运行错误，不解释性能。
2. 若 FlowCache p95 与吞吐都明显退化：先检查真实 D2H/H2D 标定和迁移率，
   不直接投入全量。
3. 若方向为正：冻结全部参数，扩到 165 task × 8 seeds 的正式 paired run。
4. 正式论文主张若要覆盖 “优于强双层策略”，仍需实现或接入真实
   closed-loop GDSF/SizeCost CPU eviction；在此之前只可主张相对
   vLLM two-tier LRU 的结果。

## 5. 未完成项

- 修复后 quick/full GPU closed-loop 尚未实际运行；
- connector 的迁移/恢复审计计数还需从 engine process 导出到结果文件；
- closed-loop GDSF/SizeCost 尚未实现；
- 因此目前不能填写真实 TTFT、throughput、goodput 或显著性结论。

没有生成或补写任何实验性能数值；所有 closed-loop 结果必须来自修复后的
实际 GPU 输出。
