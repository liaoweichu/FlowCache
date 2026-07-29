# §4.2 Closed-Loop Serving — Cloud Execution Manual

## 概述

本手册指导在 AutoDL 云端服务器上运行 §4.2 closed-loop serving 实验。
实验使用 vLLM 真实 GPU serving，测量 4 个策略的 TTFT/JCT/throughput/goodput。

## 1. 环境

### 1.1 硬件要求
- GPU: ≥24GB VRAM（RTX 4090 / A100 / A10）
- 磁盘: ≥50GB
- RAM: ≥32GB

### 1.2 软件依赖

```bash
# 创建 venv
python3 -m venv .venv-closed-loop
source .venv-closed-loop/bin/activate

# 安装 vLLM (v0.7+，需要 KVConnectorBase_V1 接口)
pip install vllm>=0.7.0

# 安装其他依赖
pip install pyyaml

# 验证
python -c "from vllm import LLM; print('vLLM OK')"
python -c "from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import SimpleCPUOffloadConnector; print('Connector OK')"
```

## 2. 数据准备

### 2.1 模型

模型应位于 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master`。

### 2.2 τ-bench traces

e1 trace 文件位于 `experiments/e1/traces/bf16/tau_bench/*.json`。
每个 JSON 文件是一个 τ-bench episode（165 tasks × 8 seeds = 1320 files）。

如果 trace 文件不在云端，从本地上传：

```bash
# 本地打包
cd "D:\00MyProject\Prefix Caching"
tar -czf e1-traces.tgz experiments/e1/traces/bf16/tau_bench/

# 上传
scp e1-traces.tgz root@<server-ip>:/root/

# 云端解压
cd /root/autodl-tmp/Prefix\ Caching
tar -xzf /root/e1-traces.tgz
```

## 3. 上传 closed-loop 代码

```bash
# 本地打包 closed-loop 目录
cd "D:\00MyProject\Prefix Caching"
tar -czf closed-loop.tgz experiments/g3/closed_loop/

# 上传
scp closed-loop.tgz root@<server-ip>:/root/

# 云端解压
cd /root/autodl-tmp/Prefix\ Caching
tar -xzf /root/closed-loop.tgz
```

## 4. 运行实验

### 4.1 Smoke test（50 请求，2 策略）

```bash
cd /root/autodl-tmp/Prefix\ Caching
source .venv-closed-loop/bin/activate

export PYTHONPATH="experiments/g3:$PYTHONPATH"

python experiments/g3/closed_loop/run_closed_loop.py \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master \
  --trace-dir experiments/e1/traces/bf16/tau_bench \
  --output-dir experiments/g3/results/closed-loop-smoke \
  --smoke \
  --max-new-tokens 32 \
  -v
```

### 4.2 完整实验（4 策略，全部 episodes）

```bash
# 在 tmux 中运行（防止 SSH 断线）
tmux new -s closed-loop

cd /root/autodl-tmp/Prefix\ Caching
source .venv-closed-loop/bin/activate

export PYTHONPATH="experiments/g3:$PYTHONPATH"

python experiments/g3/closed_loop/run_closed_loop.py \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master \
  --trace-dir experiments/e1/traces/bf16/tau_bench \
  --output-dir experiments/g3/results/closed-loop \
  --strategies no_cache,apc_lru,twotier_lru,flowcache_lossless \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 4 \
  --cpu-capacity-gib 2.0 \
  --block-size 16 \
  --max-model-len 24576 \
  --max-new-tokens 64 \
  --slo-threshold-ms 2000 \
  --bootstrap-samples 1000 \
  --flowcache-min-benefit-ms 0.0 \
  --flowcache-d2h-ms 0.10 \
  --flowcache-h2d-ms 0.15 \
  --flowcache-prefill-ms-per-block 5.0 \
  --flowcache-share-window 1000 \
  --flowcache-share-cap 8 \
  -v
```

### 4.3 自定义参数

```bash
# 只跑 FlowCache vs Two-tier LRU（核心对比）
python experiments/g3/closed_loop/run_closed_loop.py \
  --model ... \
  --trace-dir ... \
  --strategies twotier_lru,flowcache_lossless \
  --flowcache-min-benefit-ms 0.5 \
  --flowcache-migrate-ratio 0.3

# 限制 episode 数（快速验证）
python ... --max-episodes 100 --max-requests 500

# 调整 GPU 内存（KV cache 容量）
python ... --gpu-memory-utilization 0.65  # ~1.5 GiB KV cache
python ... --gpu-memory-utilization 0.80  # ~4.5 GiB KV cache
```

## 5. 输出文件

```
results/closed-loop/
├── closed-loop-no_cache.csv              # per-request metrics
├── closed-loop-apc_lru.csv
├── closed-loop-twotier_lru.csv
├── closed-loop-flowcache_lossless.csv
├── closed-loop-summary-no_cache.csv      # per-strategy summary
├── closed-loop-summary-apc_lru.csv
├── closed-loop-summary-twotier_lru.csv
├── closed-loop-summary-flowcache_lossless.csv
└── closed-loop-verdict.json              # G3 verdict
```

### verdict.json 结构

```json
{
  "experiment": "g3_closed_loop",
  "ttft_metric_valid": true,
  "throughput_metric_valid": true,
  "verdict": "PASS|FAIL|INCOMPLETE",
  "checks": {
    "p95_ttft_improvement": {
      "control_p95_ms": 1234.5,
      "treatment_p95_ms": 987.6,
      "improvement_pct": 19.99,
      "threshold_pct": 15.0,
      "pass": true
    },
    "throughput_drop": {
      "control_throughput": 3.21,
      "treatment_throughput": 3.15,
      "drop_pct": 1.87,
      "threshold_pct": 5.0,
      "pass": true
    },
    "bootstrap_ci_ttft": {
      "mean_diff_ms": 246.9,
      "ci_low_ms": 89.2,
      "ci_high_ms": 412.3,
      "n_tasks": 165,
      "significant": true,
      "ci_excludes_zero": true,
      "pass": true
    }
  }
}
```

## 6. G3 Pass 条件

| 检查 | 条件 | 说明 |
|---|---|---|
| p95 TTFT 改善 | flowcache vs twotier_lru ≥ 15% | 核心收益 |
| Throughput 非劣 | flowcache vs twotier_lru ≤ 5% drop | 不牺牲吞吐 |
| Bootstrap CI | per-task CI 排除 0 | 统计显著 |
| 真实测量 | ttft_metric_valid=true | 非建模值 |

三项全部 PASS → G3 PASS → 进入 G4 Quantization

## 7. 故障排查

### 7.1 vLLM OOM
- 降低 `--gpu-memory-utilization`（如 0.60）
- 降低 `--max-model-len`（如 4096）
- 减少 `--max-num-seqs`（如 2）

### 7.2 FlowCacheConnector 加载失败
- 确认 `PYTHONPATH` 包含 `experiments/g3`
- 确认 vLLM 版本 ≥ 0.7.0
- 查看 `experiments/g3/closed_loop/flowcache_connector.py` 的 import 路径

### 7.3 策略间 GPU 内存泄漏
- 每个策略运行后 vLLM 实例会被 `del` + `torch.cuda.empty_cache()`
- 如果仍有泄漏，在策略间手动 `nvidia-smi` 检查

### 7.4 TTFT 异常高
- 检查是否启用了 prefix caching（`--strategies` 包含 apc_lru 而非 no_cache）
- 确认 GPU 没有其他进程占用（`nvidia-smi`）
- 降低 `--max-new-tokens`（如 32）以聚焦 TTFT

## 8. 注意事项

1. **max_new_tokens=64**：closed-loop 实验聚焦 TTFT（prefill），不需要长生成。
   64 tokens 足以触发 first token 并测量 TTFT。

2. **gpu_memory_utilization**：RTX 4090 (24GB) 上，0.70 ≈ 2 GiB KV cache。
   根据实际 nvidia-smi 输出微调。

3. **CPU tier 容量**：`--cpu-capacity-gib 2.0` 对应 G3-P1 的 2 GiB 配置。
   FlowCache 和 Two-tier LRU 使用相同 CPU 容量，确保公平比较。

4. **策略顺序**：no_cache → apc_lru → twotier_lru → flowcache_lossless。
   每个策略创建新 vLLM 实例，策略间无状态泄漏。

5. **统计单元**：bootstrap CI 以 task_id（165 组）聚类，与 G3-P1 一致。
   每个 task 的 8 个 seed 的请求作为一个聚类单元。
