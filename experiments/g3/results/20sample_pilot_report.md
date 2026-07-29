# G3 小样本（20）试运行报告

**目的**：在正式实验前，用最小样本量检查每个实验的输出是否合理、是否可用于最终结论。  
**运行时间**：2026-07-29  
**环境**：Qwen2.5-7B-Instruct BF16, 2 GiB KV, c=4, max_model_len=16384, gpu_memory_utilization=0.76

---

## 一、Open-loop Smoke Test（20 episodes）

命令：
```bash
python run_g3_grid.py --config config.yaml --smoke-test --max-episodes 20 \
  --output results/smoke-20ep/raw_results.csv
```

| 指标 | 结果 |
|---|---|
| **Tasks covered** | 3 / 165（airline-0, airline-10, airline-11） |
| **Total accesses** | 131,874 |
| **apc_lru p95 delay** | 176.3 s（异常高） |
| **twotier_lru p95 delay** | 9.8 s |
| **flowcache_lossless p95 delay** | 10.4 s |
| **flowcache selection_rate** | 95.9% |

### 结论
- **不合理**：apc_lru p95 异常高，说明 20 episodes 只随机命中了某个长 prompt task，样本偏差严重。
- **不能用于正式结果**：仅覆盖 3 个 task，无法代表 165 task 的总体分布，也无法满足 bootstrap CI 的统计要求。

---

## 二、Open-loop Protocol Test（20 episodes）

命令：
```bash
python run_g3_grid.py --config config.yaml --protocol-test --max-episodes 20 \
  --output results/protocol-20ep/raw_results.csv
```

| Baseline | p95 cache delay (s) | hit_rate | selection_rate |
|---|---:|---:|---:|
| twotier_lru | 9.71 | 0.937 | — |
| twotier_gdsf | 9.70 | 0.937 | — |
| twotier_sizecost | 9.83 | 0.936 | — |
| flowcache_lossless | 10.62 | 0.923 | 56.3% |

### 结论
- **与完整 trace 结论相反**：在 20 episodes 下 flowcache 的 p95 反而比 twotier_lru 高约 9%。
- 原因仍是样本太少，只覆盖 3 个 task，随机波动淹没了真实差异。
- **不能用于正式结果**。

---

## 三、Closed-loop Serving（20 requests，3 策略）

命令：
```bash
python run_closed_loop.py --model ... --trace-dir ... --output-dir /tmp/g3-closed-loop-20req-all4 \
  --max-requests 20 --max-new-tokens 64 --gpu-memory-utilization 0.76 --max-model-len 16384 \
  --strategies apc_lru,twotier_lru,flowcache_lossless
```

| Strategy | Requests | TTFT p50 (ms) | TTFT p95 (ms) | JCT p95 (ms) | Throughput (req/s) | Goodput |
|---|---:|---:|---:|---:|---:|---:|
| apc_lru | 20/20 | 41.5 | 384.5 | 1491.3 | 3.24 | 100% |
| twotier_lru | 20/20 | 43.7 | 217.2 | 1316.9 | 3.88 | 100% |
| flowcache_lossless | 20/20 | 42.3 | 260.2 | 1267.3 | 3.88 | 100% |

Verdict（20 样本）：
- p95 TTFT improvement vs twotier_lru: **-19.8%** → FAIL（需要 ≥ +15%）
- Throughput drop: **0.0%** → PASS（需要 ≤ 5%）
- Bootstrap CI: **[-5.4, 5.9] ms**（包含 0）→ FAIL

### 结论
- **各策略均能成功运行**，说明 closed-loop pipeline（模型加载、推理、指标采集）基本可用。
- **flowcache_lossless 比 twotier_lru 差**，这与完整 trace 的预期相反。
- 但 20 requests 仅覆盖约 15 个 tasks，且每个 task 仅 1-2 个样本，
  p95 估计极不稳定（apc_lru p95 在 232-384 ms 之间波动）。
- **不能用于正式 verdict**。

---

## 四、Pilot 中发现并修复的代码问题

### 1. FlowCacheConnector 代理类缺少 vLLM v0.26.0 类方法
- **现象**：`type object 'FlowCacheConnector' has no attribute 'requires_piecewise_for_cudagraph'`
- **修复**：在 `flowcache_connector.py` 代理类上转发 `requires_piecewise_for_cudagraph`、`build_prom_metrics`、`build_kv_connector_stats`、`get_required_kvcache_layout`。

### 2. BlockPool.blocks 数据结构从 dict 变为 list
- **现象**：`'list' object has no attribute 'get'`
- **修复**：`_selective_store_specs` 中改为 `blocks[gpu_id]` 列表索引访问。

### 3. Closed-loop 默认参数在当前环境会 OOM
- **现象**：`max_model_len=24576, gpu_memory_utilization=0.70` 导致 KV cache 不足。
- **修复**：`config.yaml` 中 `closed_loop.vllm.max_model_len=16384`，`gpu_memory_utilization=0.76`。

---

## 五、总体评估：20 样本能否用于正式结果？

| 实验 | 输出是否合理 | 能否用于正式结果 | 原因 |
|---|---|---|---|
| Open-loop smoke (20 ep) | ⚠️ 部分合理 | ❌ 不能 | 仅 3 tasks，apc_lru p95 异常 |
| Open-loop protocol (20 ep) | ⚠️ 部分合理 | ❌ 不能 | 仅 3 tasks，结论与完整 trace 相反 |
| Closed-loop (20 req) | ✅ 可运行 | ❌ 不能 | 样本太少，p95/bootstrap 不稳定 |

### 建议的最小样本量
- **Open-loop**：至少覆盖全部 165 tasks 的 held-out test（config 中约 100k episodes）。20 episodes 仅作代码连通性检查。
- **Closed-loop**：至少每个 task 有 3-5 个样本（约 500-1000 requests），才能稳定估计 per-task p95 和 bootstrap CI。20 requests 仅作 smoke test。

---

## 六、下一步

1. 用修复后的代码跑 **完整 closed-loop**（3 策略，所有 trace requests）。
2. 如需快速验证，可跑 **closed-loop 100-200 requests**（覆盖更多 tasks），但仍不作为 verdict。
3. Open-loop 如仅用于机制诊断，可跑 **完整 held-out test**；如已冻结（FROZEN.md PASS），可直接跳过。
