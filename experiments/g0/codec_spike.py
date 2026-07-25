"""
Q8/Q4 codec 100 block roundtrip spike 测试 (Task 7).

核心流程：
1. 从结构用例中收集所有 unique block（KV 张量）
2. 随机抽取 100 个
3. 对每个 block 执行 Q8/Q4 roundtrip：
   - 记录 MSE、max abs err、编解码延迟
   - 用 dequant KV 组装 past_key_values，forward 一个 token，对比 logit KL
4. 验证 lineage 隔离：BF16 block hash ≠ Q8 block hash ≠ Q4 block hash
5. 生成 codec-spike-report.md（含表 G0-3）
"""
import gc
import json
import os
import random
import time
from typing import Dict, List, Tuple

import torch

from codec import (
    encode_q8,
    decode_q8,
    encode_q4,
    decode_q4,
    check_lineage_isolation,
    measure_encode_time,
    measure_decode_time,
)
from block_index import compute_block_hash


def _release_kv(*vars):
    """释放 KV cache 张量并清理 GPU 显存。"""
    for v in vars:
        del v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_block_to_device(block: Dict, device: str) -> Dict:
    """将 block 的所有 KV 张量移到指定设备（返回新 block，不修改原 block）。"""
    gpu_block = {
        "block_idx": block["block_idx"],
        "token_range": block["token_range"],
        "layer_k": [],
        "layer_v": [],
    }
    for layer_idx in range(len(block["layer_k"])):
        gpu_block["layer_k"].append(block["layer_k"][layer_idx].to(device))
        gpu_block["layer_v"].append(block["layer_v"][layer_idx].to(device))
    return gpu_block


# =============================================================================
# Block 收集
# =============================================================================

def load_cases(config: Dict) -> Dict:
    """从 outputs/real-structure-cases.json 加载结构用例。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, config["output"]["structure_cases"])
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_message_sequences(cases: Dict):
    """遍历所有用例中的消息序列，yield (case_id, messages)。"""
    for cat_key, cat_cases in cases["categories"].items():
        for case in cat_cases:
            case_id = case["case_id"]
            if "pair" in case:
                for i, item in enumerate(case["pair"]):
                    yield f"{case_id}_p{i}", item["messages"]
            elif "turns" in case:
                for i, turn in enumerate(case["turns"]):
                    yield f"{case_id}_t{i}", turn["messages"]


def collect_blocks(backend, cases: Dict, block_size: int, max_blocks: int = 500) -> List[Dict]:
    """
    从结构用例中收集 unique block。

    对每条消息序列执行 forward，切片为 block，按 KV 张量内容的指纹去重。

    Args:
        backend: Backend 实例。
        cases: 结构用例字典。
        block_size: block 大小。
        max_blocks: 最多收集的 unique block 数量。

    Returns:
        unique block 列表，每个 block 包含 layer_k/layer_v/block_idx/token_range。
    """
    blocks = []
    seen_fingerprints = set()

    for seq_idx, (seq_label, messages) in enumerate(_iter_message_sequences(cases)):
        if len(blocks) >= max_blocks:
            break

        # 每 10 条序列检查一次显存水位
        if seq_idx > 0 and seq_idx % 10 == 0:
            if not backend.assert_memory_available(
                required_gb=2.0, tag=f"collect_blocks seq={seq_idx}"
            ):
                print(f"  [mem] 显存不足，提前停止收集（已收集 {len(blocks)} blocks）")
                break

        try:
            input_ids = backend.tokenize_chat(messages)
            # 使用 safe_forward 而非 forward_with_kv，OOM 时降级跳过
            with torch.no_grad():
                logits, past_kv = backend.safe_forward(
                    input_ids, tag=f"collect {seq_label}"
                )
            if past_kv is None:
                print(f"  [SKIP] {seq_label}: OOM, 跳过该序列")
                _release_kv(input_ids)
                continue
            seq_blocks = backend.slice_kv_into_blocks(past_kv, block_size)

            # 收集完后立即释放 past_kv
            _release_kv(past_kv, input_ids, logits if logits is not None else None)

            for block in seq_blocks:
                if len(blocks) >= max_blocks:
                    break
                # 用第一层 key 张量的内容哈希作为指纹
                k_tensor = block["layer_k"][0]
                content_hash = hash(k_tensor.cpu().numpy().tobytes())
                if content_hash not in seen_fingerprints:
                    seen_fingerprints.add(content_hash)
                    # 将 block 的 KV 张量移到 CPU，避免 GPU 显存累积
                    for layer_idx in range(len(block["layer_k"])):
                        block["layer_k"][layer_idx] = block["layer_k"][layer_idx].cpu()
                        block["layer_v"][layer_idx] = block["layer_v"][layer_idx].cpu()
                    blocks.append(block)
            # 释放本轮 seq_blocks 中未入池的 block 张量
            _release_kv(seq_blocks)
        except Exception as e:
            print(f"  [WARN] 跳过 {seq_label}: {e}")
            _release_kv()
            continue

    return blocks


# =============================================================================
# 量化 Roundtrip 测试
# =============================================================================

def quantize_block_kv(block: Dict, precision: str, num_layers: int) -> Dict:
    """
    对一个 block 的所有层 KV 张量执行量化 roundtrip，返回 dequant 后的 block。

    Args:
        block: 原始 BF16 block。
        precision: "q8" 或 "q4"。
        num_layers: 模型层数。

    Returns:
        量化 roundtrip 后的 block（dequant KV），附带量化统计信息。
    """
    encode_fn = encode_q8 if precision == "q8" else encode_q4
    decode_fn = decode_q8 if precision == "q8" else decode_q4

    dequant_block = {
        "block_idx": block["block_idx"],
        "token_range": block["token_range"],
        "layer_k": [],
        "layer_v": [],
    }

    stats = {
        "mse": 0.0,
        "max_abs_err": 0.0,
        "encode_time_ms": 0.0,
        "decode_time_ms": 0.0,
        "num_layers": num_layers,
    }

    total_sq_err = 0.0
    total_elements = 0
    max_err = 0.0

    for layer_idx in range(num_layers):
        k_orig = block["layer_k"][layer_idx]
        v_orig = block["layer_v"][layer_idx]

        # 编码 + 解码
        t0 = time.time()
        k_encoded = encode_fn(k_orig)
        v_encoded = encode_fn(v_orig)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        stats["encode_time_ms"] += (time.time() - t0) * 1000

        t0 = time.time()
        k_dequant = decode_fn(k_encoded)
        v_dequant = decode_fn(v_encoded)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        stats["decode_time_ms"] += (time.time() - t0) * 1000

        # 误差统计
        k_err = (k_orig.float() - k_dequant.float())
        v_err = (v_orig.float() - v_dequant.float())
        total_sq_err += (k_err ** 2).sum().item() + (v_err ** 2).sum().item()
        total_elements += k_err.numel() + v_err.numel()
        max_err = max(max_err, k_err.abs().max().item(), v_err.abs().max().item())

        dequant_block["layer_k"].append(k_dequant)
        dequant_block["layer_v"].append(v_dequant)

    stats["mse"] = total_sq_err / max(total_elements, 1)
    stats["max_abs_err"] = max_err
    stats["encode_time_ms"] /= num_layers
    stats["decode_time_ms"] /= num_layers

    return dequant_block, stats


def compute_logit_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """
    计算两个 logits 的 KL 散度（使用 softmax 转概率分布）。

    KL(P || Q) = sum P * log(P / Q)

    Args:
        logits_a: 参考 logits（BF16 路径）。
        logits_b: 量化后 logits。

    Returns:
        KL 散度值（float）。
    """
    if logits_a.dim() == 3:
        logits_a = logits_a[:, -1, :]
        logits_b = logits_b[:, -1, :]

    log_p = torch.nn.functional.log_softmax(logits_a.float(), dim=-1)
    log_q = torch.nn.functional.log_softmax(logits_b.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
    return kl


def test_lineage_isolation_for_block(
    block: Dict, token_ids: List[int], block_size: int,
    model_id: str, revision: str, template_hash: str, config_hash: str
) -> Dict:
    """
    对单个 block 验证 lineage 隔离。

    用不同的 adapter_id（代表不同 lineage）计算 3 个哈希：
    - BF16: adapter_id = "" (canonical)
    - Q8: adapter_id = "approximate_q8"
    - Q4: adapter_id = "approximate_q4"

    Args:
        block: block 字典。
        token_ids: 该 block 的 token id 列表。
        block_size: block 大小。
        model_id, revision, template_hash, config_hash: identity 元数据。

    Returns:
        {'isolated': bool, 'bf16_hash': str, 'q8_hash': str, 'q4_hash': str}
    """
    block_idx = block["block_idx"]
    parent_hash = ""  # lineage 测试中用空父哈希即可

    bf16_hash = compute_block_hash(
        token_ids, parent_hash, block_idx, block_size,
        model_id=model_id, revision=revision,
        template_hash=template_hash, config_hash=config_hash,
        adapter_id="",
    )
    q8_hash = compute_block_hash(
        token_ids, parent_hash, block_idx, block_size,
        model_id=model_id, revision=revision,
        template_hash=template_hash, config_hash=config_hash,
        adapter_id="approximate_q8",
    )
    q4_hash = compute_block_hash(
        token_ids, parent_hash, block_idx, block_size,
        model_id=model_id, revision=revision,
        template_hash=template_hash, config_hash=config_hash,
        adapter_id="approximate_q4",
    )

    return {
        "isolated": check_lineage_isolation(bf16_hash, q8_hash, q4_hash),
        "bf16_hash": bf16_hash,
        "q8_hash": q8_hash,
        "q4_hash": q4_hash,
    }


# =============================================================================
# 主测试函数
# =============================================================================

def run_codec_spike(
    backend, cases: Dict, block_size: int, num_blocks: int, output_path: str
) -> Dict:
    """
    100 block Q8/Q4 roundtrip spike 测试主函数。

    Args:
        backend: Backend 实例。
        cases: 结构用例字典。
        block_size: block 大小。
        num_blocks: 要测试的 block 数量（默认 100）。
        output_path: codec-spike-report.md 输出路径。

    Returns:
        测试结果字典。
    """
    print(f"  Collecting blocks from structure cases (max {num_blocks * 3})...")

    # 收集足够的 unique block
    collected = collect_blocks(backend, cases, block_size, max_blocks=num_blocks * 3)
    print(f"  Collected {len(collected)} unique blocks")

    if len(collected) < num_blocks:
        print(f"  [WARN] 仅收集到 {len(collected)} 个 block，少于请求的 {num_blocks}")

    # 随机抽取
    random.seed(42)
    if len(collected) > num_blocks:
        sampled = random.sample(collected, num_blocks)
    else:
        sampled = collected

    # 获取 continuation token
    cont_token_ids = backend.tokenizer.encode(" test", add_special_tokens=False)
    cont_token_id = cont_token_ids[0] if cont_token_ids else backend.tokenizer.eos_token_id
    cont_token = torch.tensor([[cont_token_id]], device=backend.device)

    # 准备 lineage 测试的元数据
    from block_index import compute_template_hash, compute_config_hash
    model_info = backend.get_model_info()
    template_hash = compute_template_hash(backend.tokenizer.chat_template or "")
    config_hash = compute_config_hash(backend.model.config.to_dict())
    model_id = backend.model_name
    revision = model_info.get("model_sha", "unknown")

    # 对每个 block 执行 roundtrip 测试
    block_results = []
    lineage_results = []

    for i, block in enumerate(sampled):
        # 每 20 个 block 打印显存状态，确保不会因累积导致 OOM
        if i > 0 and i % 20 == 0:
            backend.print_memory_status(tag=f"codec block {i}/{len(sampled)}")

        # ---- 量化 roundtrip ----
        # block 的 KV 张量在 CPU 上（collect_blocks 已转移），量化在 CPU 完成
        # Q8
        dequant_q8, stats_q8 = quantize_block_kv(block, "q8", backend.num_layers)
        # Q4
        dequant_q4, stats_q4 = quantize_block_kv(block, "q4", backend.num_layers)

        # ---- logit KL（用单 block forward）----
        # 将 block 移到 GPU 做 forward，完成后立即释放
        kl_q8 = float("nan")
        kl_q4 = float("nan")
        try:
            with torch.no_grad():
                # 原始 BF16 block → GPU（先释放上一轮残留显存）
                _release_kv()
                gpu_block = _move_block_to_device(block, backend.device)
                past_kv_bf16 = backend.restore_kv_from_blocks([gpu_block])
                logits_bf16, _ = backend.safe_forward(
                    cont_token, past_kv_bf16, tag=f"block{i} bf16"
                )
                if logits_bf16 is None:
                    raise RuntimeError("BF16 forward OOM")

                # Q8 dequant block → GPU
                gpu_dequant_q8 = _move_block_to_device(dequant_q8, backend.device)
                past_kv_q8 = backend.restore_kv_from_blocks([gpu_dequant_q8])
                logits_q8, _ = backend.safe_forward(
                    cont_token, past_kv_q8, tag=f"block{i} q8"
                )
                if logits_q8 is None:
                    raise RuntimeError("Q8 forward OOM")

                # Q4 dequant block → GPU
                gpu_dequant_q4 = _move_block_to_device(dequant_q4, backend.device)
                past_kv_q4 = backend.restore_kv_from_blocks([gpu_dequant_q4])
                logits_q4, _ = backend.safe_forward(
                    cont_token, past_kv_q4, tag=f"block{i} q4"
                )
                if logits_q4 is None:
                    raise RuntimeError("Q4 forward OOM")

            kl_q8 = compute_logit_kl(logits_bf16, logits_q8)
            kl_q4 = compute_logit_kl(logits_bf16, logits_q4)

            # 释放 GPU 上的临时张量
            _release_kv(gpu_block, gpu_dequant_q8, gpu_dequant_q4,
                        past_kv_bf16, past_kv_q8, past_kv_q4,
                        logits_bf16, logits_q8, logits_q4)
        except Exception as e:
            print(f"  [WARN] Block {i} forward failed: {e}")
            _release_kv()

        # ---- staging 峰值字节（量化后数据大小）----
        # 在释放 dequant 前计算
        q8_bytes = sum(
            dequant_q8["layer_k"][l].numel() + dequant_q8["layer_v"][l].numel()
            for l in range(backend.num_layers)
        )
        q8_staging_bytes = q8_bytes * 1  # int8 = 1 byte per element
        q4_staging_bytes = q8_bytes * 1  # int4 存储为 int8，仍 1 byte
        bf16_staging_bytes = q8_bytes * 2  # BF16 = 2 bytes per element

        # 释放 CPU 上的 dequant block
        _release_kv(dequant_q8, dequant_q4)

        block_results.append({
            "block_idx": i,
            "original_block_idx": block["block_idx"],
            "q8_mse": stats_q8["mse"],
            "q8_max_abs_err": stats_q8["max_abs_err"],
            "q8_encode_ms": stats_q8["encode_time_ms"],
            "q8_decode_ms": stats_q8["decode_time_ms"],
            "q8_kl": kl_q8,
            "q4_mse": stats_q4["mse"],
            "q4_max_abs_err": stats_q4["max_abs_err"],
            "q4_encode_ms": stats_q4["encode_time_ms"],
            "q4_decode_ms": stats_q4["decode_time_ms"],
            "q4_kl": kl_q4,
            "bf16_staging_bytes": bf16_staging_bytes,
            "q8_staging_bytes": q8_staging_bytes,
            "q4_staging_bytes": q4_staging_bytes,
        })

        # ---- lineage 隔离测试 ----
        # 获取该 block 的 token_ids（从前 3 层张量形状推断 block_size）
        actual_block_size = block["token_range"][1] - block["token_range"][0]
        # 用占位 token_ids 进行 lineage 测试（token 内容不影响 lineage 隔离逻辑）
        placeholder_tokens = list(range(actual_block_size))
        lineage = test_lineage_isolation_for_block(
            block, placeholder_tokens, actual_block_size,
            model_id, revision, template_hash, config_hash,
        )
        lineage_results.append(lineage)

        if (i + 1) % 20 == 0:
            print(f"  Tested {i + 1}/{len(sampled)} blocks")

    # 汇总统计
    summary = _compute_summary(block_results, lineage_results)

    results = {
        "num_blocks_tested": len(sampled),
        "block_results": block_results,
        "lineage_results": lineage_results,
        "summary": summary,
    }

    # 生成报告
    generate_report(results, output_path)

    # 保存 JSON 结果
    json_path = output_path.replace(".md", "-results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to {json_path}")

    return results


def _compute_summary(block_results: List[Dict], lineage_results: List[Dict]) -> Dict:
    """汇总 roundtrip 和 lineage 统计。"""
    n = len(block_results)
    if n == 0:
        return {}

    q8_mses = [r["q8_mse"] for r in block_results]
    q4_mses = [r["q4_mse"] for r in block_results]
    q8_kls = [r["q8_kl"] for r in block_results if r["q8_kl"] == r["q8_kl"]]  # filter NaN
    q4_kls = [r["q4_kl"] for r in block_results if r["q4_kl"] == r["q4_kl"]]

    lineage_isolated = sum(1 for r in lineage_results if r["isolated"])

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def maximum(lst):
        return max(lst) if lst else 0.0

    return {
        "num_blocks": n,
        "q8": {
            "mse_mean": mean(q8_mses),
            "mse_max": maximum(q8_mses),
            "max_abs_err_mean": mean([r["q8_max_abs_err"] for r in block_results]),
            "max_abs_err_max": maximum([r["q8_max_abs_err"] for r in block_results]),
            "kl_mean": mean(q8_kls),
            "kl_max": maximum(q8_kls),
            "encode_ms_mean": mean([r["q8_encode_ms"] for r in block_results]),
            "decode_ms_mean": mean([r["q8_decode_ms"] for r in block_results]),
        },
        "q4": {
            "mse_mean": mean(q4_mses),
            "mse_max": maximum(q4_mses),
            "max_abs_err_mean": mean([r["q4_max_abs_err"] for r in block_results]),
            "max_abs_err_max": maximum([r["q4_max_abs_err"] for r in block_results]),
            "kl_mean": mean(q4_kls),
            "kl_max": maximum(q4_kls),
            "encode_ms_mean": mean([r["q4_encode_ms"] for r in block_results]),
            "decode_ms_mean": mean([r["q4_decode_ms"] for r in block_results]),
        },
        "lineage": {
            "isolated": lineage_isolated,
            "total": len(lineage_results),
            "all_isolated": lineage_isolated == len(lineage_results),
        },
    }


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(results: Dict, output_path: str) -> None:
    """
    生成 codec-spike-report.md（含表 G0-3）。

    Args:
        results: run_codec_spike 返回的结果字典。
        output_path: 输出文件路径。
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    block_results = results["block_results"]
    summary = results["summary"]

    lines = []
    lines.append("# G0 Codec Spike Report")
    lines.append("")
    lines.append("Q8/Q4 KV cache 量化 codec 的 100 block roundtrip spike 测试。")
    lines.append(f"测试 block 数量: {results['num_blocks_tested']}")
    lines.append("")

    # ---- 表 G0-3: codec roundtrip 结果 ----
    lines.append("## 表 G0-3: Q8/Q4 Codec Roundtrip 结果")
    lines.append("")
    lines.append(
        "| Precision | MSE (mean) | MSE (max) | Max Abs Err (mean) | "
        "Max Abs Err (max) | Logit KL (mean) | Logit KL (max) | "
        "Encode (ms) | Decode (ms) |"
    )
    lines.append(
        "|-----------|------------|-----------|--------------------|"
        "-------------------|-----------------|----------------|"
        "-------------|-------------|"
    )

    for precision in ["q8", "q4"]:
        s = summary[precision]
        lines.append(
            f"| {precision.upper()} | "
            f"{s['mse_mean']:.6e} | {s['mse_max']:.6e} | "
            f"{s['max_abs_err_mean']:.6e} | {s['max_abs_err_max']:.6e} | "
            f"{s['kl_mean']:.6e} | {s['kl_max']:.6e} | "
            f"{s['encode_ms_mean']:.4f} | {s['decode_ms_mean']:.4f} |"
        )

    lines.append("")

    # ---- 逐 block 明细表 ----
    lines.append("## 逐 Block 明细")
    lines.append("")
    lines.append(
        "| Block # | Q8 MSE | Q8 Max Err | Q8 KL | Q4 MSE | Q4 Max Err | "
        "Q4 KL | Q8 Encode (ms) | Q4 Encode (ms) |"
    )
    lines.append(
        "|---------|--------|------------|--------|--------|------------|"
        "--------|-----------------|-----------------|"
    )

    for r in block_results[:20]:  # 只显示前 20 行
        lines.append(
            f"| {r['block_idx']} | "
            f"{r['q8_mse']:.6e} | {r['q8_max_abs_err']:.6e} | "
            f"{r['q8_kl']:.6e} | "
            f"{r['q4_mse']:.6e} | {r['q4_max_abs_err']:.6e} | "
            f"{r['q4_kl']:.6e} | "
            f"{r['q8_encode_ms']:.4f} | {r['q4_encode_ms']:.4f} |"
        )

    if len(block_results) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... |")
        lines.append(f"| (共 {len(block_results)} 行) | | | | | | | | |")

    lines.append("")

    # ---- Lineage 隔离 ----
    lines.append("## Lineage 隔离验证")
    lines.append("")
    lineage = summary["lineage"]
    lines.append(
        f"- 测试 block 数: {lineage['total']}"
    )
    lines.append(
        f"- Lineage 隔离通过: {lineage['isolated']}/{lineage['total']}"
    )
    lines.append(
        f"- 全部通过: {'✓' if lineage['all_isolated'] else '✗'}"
    )
    lines.append("")
    lines.append(
        "验证方式：对每个 block 用不同 adapter_id（canonical=空, "
        "Q8=approximate_q8, Q4=approximate_q4）计算 identity 哈希，"
        "三者应两两不同。"
    )
    lines.append("")

    # ---- Staging 峰值字节 ----
    lines.append("## Staging 峰值字节")
    lines.append("")
    lines.append(
        "| Precision | 单 block 存储大小 (bytes) | 相对 BF16 压缩比 |"
    )
    lines.append(
        "|-----------|--------------------------|-------------------|"
    )
    if block_results:
        bf16_bytes = block_results[0]["bf16_staging_bytes"]
        q8_bytes = block_results[0]["q8_staging_bytes"]
        q4_bytes = block_results[0]["q4_staging_bytes"]
        lines.append(f"| BF16 | {bf16_bytes} | 1.00x |")
        lines.append(f"| Q8 | {q8_bytes} | {bf16_bytes/max(q8_bytes,1):.2f}x |")
        lines.append(f"| Q4 | {q4_bytes} | {bf16_bytes/max(q4_bytes,1):.2f}x |")
    lines.append("")

    # ---- 判定 ----
    lines.append("## 判定")
    lines.append("")
    codec_ok = summary["q8"]["mse_mean"] > 0 and summary["q4"]["mse_mean"] > 0
    lineage_ok = lineage["all_isolated"]
    overall_pass = codec_ok and lineage_ok
    verdict = "PASS" if overall_pass else "FAIL"

    lines.append(f"- Q8/Q4 roundtrip 执行成功: {'✓' if codec_ok else '✗'}")
    lines.append(f"- Lineage 隔离正确: {'✓' if lineage_ok else '✗'}")
    lines.append("")
    lines.append(f"**Overall: {verdict}**")
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
    cases = load_cases(config)
    run_codec_spike(
        backend, cases, config["cache"]["block_size"],
        config["codec"]["num_blocks"],
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config["output"]["codec_report"]),
    )
