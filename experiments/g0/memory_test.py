"""
显存峰值测量 (Task 8).

测量以下 3 种配置的 allocated/reserved 峰值（各 5 次重复取 max）：
1. 仅加载模型 (concurrent=0, context_len=0)
2. 并发 4 × 4K context
3. 并发 8 × 8K context

判定：权重 + active cache + staging + 安全水位 ≤ 24GB。
"""
import json
import os
from typing import Dict, List

import torch


def measure_memory_peak(
    backend, context_len: int, concurrent: int, repeats: int = 5
) -> Dict:
    """
    测量给定配置的 GPU 显存峰值。

    对于 concurrent > 0 且 context_len > 0 的配置，模拟并发请求：
    生成 concurrent 个长度为 context_len 的随机 input_ids，
    依次执行 forward 并保留 past_key_values，测量峰值。

    对于 context_len=0 或 concurrent=0 的配置，仅测量模型加载后的
    基础显存占用。

    Args:
        backend: Backend 实例。
        context_len: 每个请求的上下文长度（0 = 仅模型）。
        concurrent: 并发请求数（0 = 仅模型）。
        repeats: 测量重复次数。

    Returns:
        {
            'allocated_peak_gb': float,
            'reserved_peak_gb': float,
            'allocated_values': [float, ...],
            'reserved_values': [float, ...],
        }
    """
    allocated_values = []
    reserved_values = []

    has_cuda = torch.cuda.is_available()

    for _ in range(repeats):
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if context_len > 0 and concurrent > 0:
            # 模拟并发请求：生成 concurrent 个 KV cache
            kv_caches = []
            for _ in range(concurrent):
                input_ids = torch.randint(
                    0,
                    backend.model.config.vocab_size,
                    (1, context_len),
                    device=backend.device,
                )
                with torch.no_grad():
                    _, past_kv = backend.forward_with_kv(input_ids)
                kv_caches.append(past_kv)

            if has_cuda:
                allocated = torch.cuda.max_memory_allocated() / 1e9
                reserved = torch.cuda.max_memory_reserved() / 1e9
            else:
                allocated = 0.0
                reserved = 0.0

            # 释放 KV cache
            del kv_caches
            if has_cuda:
                torch.cuda.empty_cache()
        else:
            # 仅模型加载
            if has_cuda:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
            else:
                allocated = 0.0
                reserved = 0.0

        allocated_values.append(allocated)
        reserved_values.append(reserved)

    return {
        "allocated_peak_gb": max(allocated_values) if allocated_values else 0.0,
        "reserved_peak_gb": max(reserved_values) if reserved_values else 0.0,
        "allocated_values": allocated_values,
        "reserved_values": reserved_values,
    }


def run_memory_test(backend, config: Dict, output_path: str) -> Dict:
    """
    对所有配置运行显存测量并生成报告。

    Args:
        backend: Backend 实例。
        config: 配置字典。
        output_path: memory-report.md 输出路径。

    Returns:
        {
            'configs': [...],
            'verdict': 'PASS'/'FAIL',
        }
    """
    repeats = config["memory"]["repeats"]
    memory_configs = config["memory"]["configs"]

    results = []
    for cfg in memory_configs:
        name = cfg["name"]
        concurrent = cfg["concurrent"]
        context_len = cfg["context_len"]

        print(f"  Measuring {name} (concurrent={concurrent}, ctx={context_len})...")
        result = measure_memory_peak(
            backend, context_len=context_len, concurrent=concurrent, repeats=repeats
        )
        result["name"] = name
        result["concurrent"] = concurrent
        result["context_len"] = context_len
        results.append(result)

        print(
            f"    allocated_peak={result['allocated_peak_gb']:.3f} GB, "
            f"reserved_peak={result['reserved_peak_gb']:.3f} GB"
        )

    # 判定：权重 + active cache + staging + 安全水位 ≤ 24GB
    # 安全水位 = reserved 的 10%
    memory_limit_gb = 24.0
    safety_margin = 0.1  # 10% 安全水位

    # 找到最大配置（并发 8x8k）的 reserved 峰值
    max_reserved = max(r["reserved_peak_gb"] for r in results)
    required_with_safety = max_reserved * (1 + safety_margin)
    verdict = "PASS" if required_with_safety <= memory_limit_gb else "FAIL"

    output = {
        "configs": results,
        "memory_limit_gb": memory_limit_gb,
        "safety_margin": safety_margin,
        "required_with_safety_gb": required_with_safety,
        "verdict": verdict,
    }

    # 生成报告
    generate_report(output, output_path)

    # 保存 JSON 结果
    json_path = output_path.replace(".md", "-results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to {json_path}")

    return output


def generate_report(results: Dict, output_path: str) -> None:
    """
    生成 memory-report.md。

    Args:
        results: run_memory_test 返回的结果字典。
        output_path: 输出文件路径。
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    configs = results["configs"]
    memory_limit = results["memory_limit_gb"]
    safety_margin = results["safety_margin"]
    required = results["required_with_safety_gb"]
    verdict = results["verdict"]

    lines = []
    lines.append("# G0 Memory Report")
    lines.append("")
    lines.append("显存峰值测量：3 种配置，每种重复 5 次取最大值。")
    lines.append("")

    # ---- 显存峰值表 ----
    lines.append("## 显存峰值测量")
    lines.append("")
    lines.append(
        "| Config | Concurrent | Context Len | Allocated Peak (GB) | "
        "Reserved Peak (GB) | Allocated Values (GB) |"
    )
    lines.append(
        "|--------|------------|-------------|---------------------|"
        "--------------------|----------------------|"
    )

    for r in configs:
        alloc_vals = ", ".join(f"{v:.3f}" for v in r["allocated_values"])
        lines.append(
            f"| {r['name']} | {r['concurrent']} | {r['context_len']} | "
            f"{r['allocated_peak_gb']:.3f} | "
            f"{r['reserved_peak_gb']:.3f} | {alloc_vals} |"
        )

    lines.append("")

    # ---- 判定 ----
    lines.append("## 判定")
    lines.append("")
    lines.append(f"- 显存上限: {memory_limit:.1f} GB")
    lines.append(f"- 安全水位: {safety_margin*100:.0f}% (reserved × {1+safety_margin})")
    lines.append(f"- 最大 reserved 峰值: {max(r['reserved_peak_gb'] for r in configs):.3f} GB")
    lines.append(f"- 含安全水位的所需显存: {required:.3f} GB")
    lines.append(f"- 判定: {'✓ PASS' if verdict == 'PASS' else '✗ FAIL'}")
    lines.append("")
    lines.append(
        f"权重 + active cache + staging + 安全水位 ≤ {memory_limit:.0f}GB: "
        f"{'✓' if verdict == 'PASS' else '✗'}"
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {output_path}")


if __name__ == "__main__":
    import sys
    import yaml

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from backend import Backend

    backend = Backend(
        model_name=config["model"]["name"],
        dtype=getattr(torch, config["model"]["dtype"]),
        device_map=config["model"].get("device_map", "auto"),
    )
    run_memory_test(
        backend, config,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config["output"]["memory_report"]),
    )
