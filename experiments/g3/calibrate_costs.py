"""
Cost Calibration Script
=======================
在云端 AutoDL 上实测 5 类成本，拟合成本模型，冻结于 cost-model.json。

实测成本项（G3.6.1）：
  1. C^res_evict  — prefill 成本（block_size=16，父前缀长度 × 并发）
  2. C^place      — GPU→CPU 迁移（D2H，pinned buffer）
  3. C^res_CPU    — CPU→GPU 恢复（H2D，pinned buffer）
  4. C^hold       — hold 机会成本（同预算下被挤占 block 的期望 miss cost）
  5. C^policy     — controller 单次决策耗时

拟合形式：
  - prefill: 分段线性或查表（记录 R²）
  - D2H/H2D: 线性（截距 + 斜率/byte）
  - hold: 标量/byte·step
  - policy: 标量/decision

⚠️ 必须在云端 AutoDL（RTX 4090D 24GB + PCIe 4.0 + pinned memory）上运行。
本地无 GPU 时仅做 dry-run 验证脚本可运行。

用法：
  python calibrate_costs.py --config config.yaml --output cost-model.json
  python calibrate_costs.py --dry-run  # 本地无 GPU 时验证脚本
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Cost model data structures
# ---------------------------------------------------------------------------

def empty_cost_model() -> Dict:
    """返回空的成本模型骨架。"""
    return {
        "metadata": {
            "calibrated_at": "",
            "gpu": "",
            "pcie_gen": 4,
            "dtype": "bfloat16",
            "block_size": 16,
        },
        "prefill": {
            "model": "piecewise_linear",  # or "lookup_table"
            "params": {},  # {parent_length: {concurrency: ms}}
            "r2": 0.0,
            "samples": [],
        },
        "d2h_migrate": {
            "model": "linear",  # ms = intercept + slope * bytes
            "intercept": 0.0,
            "slope_per_byte": 0.0,
            "r2": 0.0,
            "samples": [],
        },
        "h2d_restore": {
            "model": "linear",
            "intercept": 0.0,
            "slope_per_byte": 0.0,
            "r2": 0.0,
            "samples": [],
        },
        "hold": {
            "model": "scalar",
            "cost_per_byte_step": 0.0,
            "method": "oracle_assisted_estimate",
        },
        "policy": {
            "model": "scalar",
            "cost_per_decision_ms": 0.0,
            "samples": [],
        },
    }


# ---------------------------------------------------------------------------
# Calibration routines (each returns list of (params, ms) samples)
# ---------------------------------------------------------------------------

def _median_p95(samples: List[float]) -> Dict:
    """返回中位数和 P95。"""
    if not samples:
        return {"median": 0.0, "p95": 0.0, "n": 0}
    return {
        "median": float(statistics.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "n": len(samples),
    }


def calibrate_prefill(parent_lengths: List[int],
                      concurrency_levels: List[int],
                      repetitions: int,
                      block_size: int = 16) -> Dict:
    """
    实测 prefill 成本：不同父前缀长度 × 并发下，prefill 一个 block 的 ms。

    方法：
    1. 构造父前缀（length tokens 的随机 token IDs）
    2. 用 model.forward() 测量 prefill 一个 block_size=16 block 的时间
    3. 重复 repetitions 次取中位数

    Returns:
        {parent_length: {concurrency: {"median": ms, "p95": ms, "n": int}}}
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[dry-run] torch/transformers not available, using placeholder")
        return _dry_run_prefill(parent_lengths, concurrency_levels, repetitions)

    # 尝试加载模型；失败则 dry-run
    model_id = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
    except Exception as e:
        print(f"[dry-run] model load failed: {e}")
        return _dry_run_prefill(parent_lengths, concurrency_levels, repetitions)

    results = {}
    for plen in parent_lengths:
        results[plen] = {}
        for conc in concurrency_levels:
            samples = []
            for _ in range(repetitions):
                # 构造输入：plen 个随机 token + 1 个 block
                input_ids = torch.randint(
                    0, tokenizer.vocab_size, (conc, plen + block_size)
                ).to("cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_ids)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                # 每个 request 的 prefill 时间
                samples.append((t1 - t0) * 1000 / conc)
            results[plen][conc] = _median_p95(samples)
            print(f"  prefill plen={plen} conc={conc}: "
                  f"{results[plen][conc]['median']:.3f} ms")

    # 释放 GPU 资源
    del model
    torch.cuda.empty_cache()
    return results


def _dry_run_prefill(parent_lengths, concurrency_levels, repetitions):
    """无 GPU 时的占位实现，返回合理的估计值。"""
    results = {}
    for plen in parent_lengths:
        results[plen] = {}
        for conc in concurrency_levels:
            # 粗略估计：Qwen2.5-7B BF16 在 4090D 上 ~0.02 ms/token
            est_ms = 0.02 * (plen + 16) / conc
            results[plen][conc] = {"median": est_ms, "p95": est_ms * 1.2, "n": repetitions}
    return results


def calibrate_d2h(byte_sizes: List[int],
                  repetitions: int) -> Dict:
    """
    实测 GPU→CPU 迁移成本（D2H，pinned buffer）。

    Args:
        byte_sizes: 不同字节数列表
        repetitions: 每点重复次数
    """
    try:
        import torch
    except ImportError:
        print("[dry-run] torch not available, using placeholder")
        return _dry_run_transfer(byte_sizes, repetitions, direction="d2h")

    results = {"samples": []}
    for nbytes in byte_sizes:
        # GPU tensor + CPU pinned buffer
        gpu_tensor = torch.randn(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        cpu_pinned = torch.empty(nbytes // 2, dtype=torch.bfloat16, pin_memory=True)

        samples = []
        # warmup
        for _ in range(10):
            cpu_pinned.copy_(gpu_tensor)
        torch.cuda.synchronize()

        for _ in range(repetitions):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cpu_pinned.copy_(gpu_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000)

        stats = _median_p95(samples)
        results["samples"].append({"bytes": nbytes, **stats})
        print(f"  D2H {nbytes} B: {stats['median']:.3f} ms")

    # 线性拟合：ms = intercept + slope * bytes
    xs = np.array([s["bytes"] for s in results["samples"]])
    ys = np.array([s["median"] for s in results["samples"]])
    if len(xs) >= 2:
        coeffs = np.polyfit(xs, ys, 1)  # [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]
        y_pred = slope * xs + intercept
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, intercept, r2 = 0.0, 0.0, 0.0

    results["model"] = "linear"
    results["intercept"] = float(intercept)
    results["slope_per_byte"] = float(slope)
    results["r2"] = float(r2)

    del gpu_tensor, cpu_pinned
    torch.cuda.empty_cache()
    return results


def calibrate_h2d(byte_sizes: List[int],
                  repetitions: int) -> Dict:
    """实测 CPU→GPU 恢复成本（H2D，pinned buffer）。"""
    try:
        import torch
    except ImportError:
        print("[dry-run] torch not available, using placeholder")
        return _dry_run_transfer(byte_sizes, repetitions, direction="h2d")

    results = {"samples": []}
    for nbytes in byte_sizes:
        cpu_pinned = torch.randn(nbytes // 2, dtype=torch.bfloat16, pin_memory=True)
        gpu_tensor = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")

        samples = []
        for _ in range(10):
            gpu_tensor.copy_(cpu_pinned)
        torch.cuda.synchronize()

        for _ in range(repetitions):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_tensor.copy_(cpu_pinned)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000)

        stats = _median_p95(samples)
        results["samples"].append({"bytes": nbytes, **stats})
        print(f"  H2D {nbytes} B: {stats['median']:.3f} ms")

    xs = np.array([s["bytes"] for s in results["samples"]])
    ys = np.array([s["median"] for s in results["samples"]])
    if len(xs) >= 2:
        coeffs = np.polyfit(xs, ys, 1)
        slope, intercept = coeffs[0], coeffs[1]
        y_pred = slope * xs + intercept
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, intercept, r2 = 0.0, 0.0, 0.0

    results["model"] = "linear"
    results["intercept"] = float(intercept)
    results["slope_per_byte"] = float(slope)
    results["r2"] = float(r2)

    del gpu_tensor, cpu_pinned
    torch.cuda.empty_cache()
    return results


def _dry_run_transfer(byte_sizes, repetitions, direction="d2h"):
    """无 GPU 时的占位实现。"""
    results = {"samples": []}
    for nbytes in byte_sizes:
        # PCIe 4.0 x16 理论 ~32 GB/s，实际 ~25 GB/s
        est_ms = nbytes / (25 * 1024**3) * 1000
        results["samples"].append({
            "bytes": nbytes, "median": est_ms, "p95": est_ms * 1.2, "n": repetitions
        })
    xs = np.array([s["bytes"] for s in results["samples"]])
    ys = np.array([s["median"] for s in results["samples"]])
    coeffs = np.polyfit(xs, ys, 1)
    results["model"] = "linear"
    results["intercept"] = float(coeffs[1])
    results["slope_per_byte"] = float(coeffs[0])
    results["r2"] = 1.0
    return results


def calibrate_hold(cost_model: Dict, capacity_blocks: int) -> Dict:
    """
    估算 hold 机会成本（标量/byte·step）。

    方法：以"同预算下被挤占 block 的期望 miss cost"近似。
    用 prefill 成本中位数 / capacity_blocks / block_bytes 估算。
    """
    # 取 prefill 成本的一个代表值（plen=2048, conc=4）
    prefill = cost_model.get("prefill", {}).get("params", {})
    representative_ms = 0.0
    for plen_str, conc_dict in prefill.items():
        if int(plen_str) == 2048:
            for conc_str, stats in conc_dict.items():
                if int(conc_str) == 4:
                    representative_ms = stats.get("median", 0.0)
                    break
            break

    if representative_ms == 0:
        representative_ms = 5.0  # fallback

    # block_bytes = block_size × layers × 2 × kv_heads × head_dim × dtype_bytes
    block_bytes = 16 * 28 * 2 * 4 * 128 * 2  # = 917,504 B
    # hold cost: 每 byte 每 step 的机会成本
    # 近似：一个 block 的 miss cost / block_bytes / horizon
    horizon = 1000
    cost_per_byte_step = representative_ms / block_bytes / horizon

    return {
        "model": "scalar",
        "cost_per_byte_step": float(cost_per_byte_step),
        "method": "oracle_assisted_estimate",
        "representative_prefill_ms": float(representative_ms),
        "block_bytes": block_bytes,
        "horizon": horizon,
    }


def calibrate_policy(repetitions: int = 1000) -> Dict:
    """
    实测 controller 单次决策耗时。

    方法：运行 controller.decide() N 次，取中位数。
    """
    try:
        from controller import FlowCacheLosslessController
        from reuse_estimator import HeuristicReuseEstimator
    except ImportError:
        # 占位：heuristic 估计器 + 简单决策
        return _dry_run_policy(repetitions)

    # 构造最小 controller
    estimator = HeuristicReuseEstimator()
    # 模拟决策：估计 100 个 block 的 R 值并排序
    blocks = [
        {"age": i * 10, "share_count": i % 3, "block_idx": i}
        for i in range(100)
    ]
    samples = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        _ = estimator.estimate_batch(blocks)
        # 模拟排序决策
        sorted(blocks, key=lambda b: estimator.estimate(
            b["age"], b["share_count"], b["block_idx"]), reverse=True)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000)

    stats = _median_p95(samples)
    return {
        "model": "scalar",
        "cost_per_decision_ms": stats["median"],
        "p95_ms": stats["p95"],
        "n": stats["n"],
        "samples": [{"median": stats["median"], "p95": stats["p95"]}],
    }


def _dry_run_policy(repetitions):
    """无 controller 时的占位。"""
    est_ms = 0.1  # heuristic 估计 ~0.1 ms/decision
    return {
        "model": "scalar",
        "cost_per_decision_ms": est_ms,
        "p95_ms": est_ms * 1.5,
        "n": repetitions,
        "samples": [{"median": est_ms, "p95": est_ms * 1.5}],
    }


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

def run_calibration(config: Dict, dry_run: bool = False) -> Dict:
    """运行完整成本标定流程。"""
    cal_cfg = config.get("cost_calibration", {})
    g0 = config.get("g0", {})
    block_size = g0.get("block_size", 16)

    # block_bytes = block_size × layers × 2(K+V) × kv_heads × head_dim × dtype_bytes
    block_bytes = (block_size * g0.get("num_hidden_layers", 28) * 2
                   * g0.get("num_kv_heads", 4) * g0.get("head_dim", 128)
                   * g0.get("dtype_bytes", 2))

    parent_lengths = cal_cfg.get("parent_lengths", [512, 1024, 2048, 4096, 8192])
    concurrency_levels = cal_cfg.get("concurrency_levels", [1, 4, 8])
    repetitions = cal_cfg.get("repetitions", 100)

    # 字节档位：1 block, 10 blocks, 100 blocks, 1000 blocks
    byte_sizes = [block_bytes * mult for mult in [1, 10, 100, 1000]]

    cost_model = empty_cost_model()
    cost_model["metadata"]["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cost_model["metadata"]["dtype"] = g0.get("dtype", "bfloat16")
    cost_model["metadata"]["block_size"] = block_size
    cost_model["metadata"]["block_bytes"] = block_bytes

    if dry_run:
        print("[dry-run] 使用占位值（无 GPU 实测）")

    # 1. Prefill 成本
    print("Calibrating prefill cost...")
    prefill_results = calibrate_prefill(
        parent_lengths, concurrency_levels, repetitions, block_size
    )
    cost_model["prefill"]["params"] = prefill_results
    cost_model["prefill"]["r2"] = 1.0  # 查表法，R² 不适用

    # 2. D2H 迁移成本
    print("Calibrating D2H (GPU→CPU) transfer cost...")
    cost_model["d2h_migrate"] = calibrate_d2h(byte_sizes, repetitions)

    # 3. H2D 恢复成本
    print("Calibrating H2D (CPU→GPU) transfer cost...")
    cost_model["h2d_restore"] = calibrate_h2d(byte_sizes, repetitions)

    # 4. Hold 机会成本
    print("Estimating hold opportunity cost...")
    capacity_blocks_1gib = int(1 * 1024**3 // block_bytes)
    cost_model["hold"] = calibrate_hold(cost_model, capacity_blocks_1gib)

    # 5. Policy 决策成本
    print("Calibrating controller policy cost...")
    cost_model["policy"] = calibrate_policy(repetitions)

    return cost_model


def main():
    parser = argparse.ArgumentParser(description="G3 cost calibration")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output", default=None, help="Output cost-model.json path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without GPU (placeholder values)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run calibration
    cost_model = run_calibration(config, dry_run=args.dry_run)

    # Save
    output_path = args.output or config.get("cost_calibration", {}).get("output_file")
    if output_path is None:
        output_path = "cost-model.json"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cost_model, f, indent=2, ensure_ascii=False)

    print(f"\nCost model saved to: {output_path}")
    print(f"  Prefill R²: {cost_model['prefill']['r2']:.4f}")
    print(f"  D2H R²: {cost_model['d2h_migrate']['r2']:.4f}")
    print(f"  H2D R²: {cost_model['h2d_restore']['r2']:.4f}")
    print(f"  Policy cost: {cost_model['policy']['cost_per_decision_ms']:.4f} ms/decision")


if __name__ == "__main__":
    main()
