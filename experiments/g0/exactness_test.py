"""
BF16 缓存恢复 vs 重算一致性测试 (Task 5).

验证三个核心指标：
1. KV 张量逐元素 bit-identical（缓存恢复路径 vs 重算路径）
2. logits max abs diff ≤ 1e-3
3. greedy decode top-1 token 一致率 100%

同时验证 block identity / 父链 / invalidation 正确性（6 类结构用例）。

技术路径：
- 重算路径：forward_with_kv(input_ids) → past_kv_recompute + logits
- 缓存路径：slice → restore → forward(continuation) → logits_cached
- 对比两条路径的 KV 张量和 logits
"""
import json
import os
from typing import Dict, List, Tuple

import torch

from block_index import (
    compute_block_hash,
    verify_parent_chain,
    check_invalidation,
    compute_template_hash,
    compute_config_hash,
)


# =============================================================================
# 辅助函数
# =============================================================================

def load_cases(config: Dict) -> Dict:
    """从 outputs/real-structure-cases.json 加载结构用例。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, config["output"]["structure_cases"])
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_metadata(backend) -> Dict:
    """获取默认 block identity 元数据（model_id, revision, template_hash, config_hash）。"""
    model_info = backend.get_model_info()
    template_str = backend.tokenizer.chat_template or ""
    return {
        "model_id": backend.model_name,
        "revision": model_info.get("model_sha", "unknown"),
        "template_hash": compute_template_hash(template_str),
        "config_hash": compute_config_hash(backend.model.config.to_dict()),
        "adapter_id": "",
    }


def compare_kv_tensors(kv_a, kv_b) -> Dict:
    """
    对比两个 past_key_values 的 KV 张量。

    检查每层 key_cache 和 value_cache 是否 bit-identical，
    同时记录最大绝对误差。

    Args:
        kv_a: 第一个 past_key_values（DynamicCache 或 tuple）。
        kv_b: 第二个 past_key_values。

    Returns:
        {
            'bit_identical': bool,        # 全部层是否 bit-identical
            'max_abs_diff': float,        # 最大绝对误差
            'num_layers': int,            # 对比的层数
            'layer_details': [...],       # 每层对比详情
        }
    """
    # 兼容 DynamicCache 和 legacy tuple
    if hasattr(kv_a, "key_cache") and hasattr(kv_a, "value_cache"):
        a_keys, a_vals = kv_a.key_cache, kv_a.value_cache
        b_keys, b_vals = kv_b.key_cache, kv_b.value_cache
    else:
        a_keys = [layer[0] for layer in kv_a]
        a_vals = [layer[1] for layer in kv_a]
        b_keys = [layer[0] for layer in kv_b]
        b_vals = [layer[1] for layer in kv_b]

    num_layers = min(len(a_keys), len(b_keys))
    all_identical = True
    max_abs_diff = 0.0
    layer_details = []

    for i in range(num_layers):
        k_identical = torch.equal(a_keys[i], b_keys[i])
        v_identical = torch.equal(a_vals[i], b_vals[i])
        layer_identical = k_identical and v_identical
        if not layer_identical:
            all_identical = False
            # 计算 float 路径的绝对误差
            k_diff = (a_keys[i].float() - b_keys[i].float()).abs().max().item()
            v_diff = (a_vals[i].float() - b_vals[i].float()).abs().max().item()
            layer_max = max(k_diff, v_diff)
            max_abs_diff = max(max_abs_diff, layer_max)
        else:
            layer_max = 0.0
        layer_details.append({
            "layer": i,
            "k_identical": k_identical,
            "v_identical": v_identical,
            "max_abs_diff": layer_max,
        })

    return {
        "bit_identical": all_identical,
        "max_abs_diff": max_abs_diff,
        "num_layers": num_layers,
        "layer_details": layer_details,
    }


def compare_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> Dict:
    """
    对比两个 logits 张量。

    提取最后一个 token 的 logits 进行比较。

    Args:
        logits_a: 第一个 logits 张量 [batch, seq, vocab] 或 [batch, vocab]。
        logits_b: 第二个 logits 张量。

    Returns:
        {
            'max_abs_diff': float,
            'mean_abs_diff': float,
            'cosine_sim': float,
            'top1_match': bool,       # argmax 是否一致
            'top1_a': int,
            'top1_b': int,
        }
    """
    # 取最后一个 token 的 logits
    if logits_a.dim() == 3:
        logits_a_last = logits_a[:, -1, :].float()
        logits_b_last = logits_b[:, -1, :].float()
    else:
        logits_a_last = logits_a.float()
        logits_b_last = logits_b.float()

    diff = (logits_a_last - logits_b_last).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    # cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        logits_a_last.flatten().unsqueeze(0),
        logits_b_last.flatten().unsqueeze(0),
        dim=0,
    ).item()

    # top-1 token
    top1_a = logits_a_last.argmax(dim=-1).item()
    top1_b = logits_b_last.argmax(dim=-1).item()
    top1_match = top1_a == top1_b

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_sim": cosine_sim,
        "top1_match": top1_match,
        "top1_a": top1_a,
        "top1_b": top1_b,
    }


def compute_sequence_block_hashes(
    backend, messages: List[Dict], block_size: int, metadata: Dict
) -> Tuple[List[Dict], List[str]]:
    """
    对一条消息序列执行 forward，切片为 block，计算每个 block 的 identity 哈希。

    Args:
        backend: Backend 实例。
        messages: chat 消息列表。
        block_size: block 大小。
        metadata: block identity 元数据。

    Returns:
        (blocks, hashes)：blocks 为带 hash/parent_hash 字段的 block 列表，
                         hashes 为哈希字符串列表。
    """
    input_ids = backend.tokenize_chat(messages)
    token_ids = input_ids[0].tolist()

    with torch.no_grad():
        _, past_kv = backend.forward_with_kv(input_ids)

    blocks = backend.slice_kv_into_blocks(past_kv, block_size)

    hashes = []
    parent_hash = ""
    for block in blocks:
        start, end = block["token_range"]
        block_token_ids = token_ids[start:end]
        h = compute_block_hash(
            block_token_ids,
            parent_hash,
            block["block_idx"],
            block_size,
            model_id=metadata.get("model_id", ""),
            revision=metadata.get("revision", ""),
            template_hash=metadata.get("template_hash", ""),
            config_hash=metadata.get("config_hash", ""),
            adapter_id=metadata.get("adapter_id", ""),
        )
        block["hash"] = h
        block["parent_hash"] = parent_hash
        parent_hash = h
        hashes.append(h)

    return blocks, hashes


# =============================================================================
# Block Identity 测试
# =============================================================================

def test_block_identity(backend, cases: Dict, block_size: int) -> Dict:
    """
    测试 6 类结构用例的 block identity / 父链 / invalidation 正确性。

    对每类用例验证：
    - ① 同域任务对：system prompt block hash 一致
    - ② 分支历史：分支点前 block hash 一致，分支点后不同
    - ③ template 变化：所有 block hash 不同
    - ④ 标识变化：所有 block hash 不同
    - ⑤ 纯追加：前缀 block hash 一致（增量复用）
    - ⑥ 无共享：block hash 全部不同

    Args:
        backend: Backend 实例。
        cases: 结构用例字典。
        block_size: block 大小。

    Returns:
        {'cases': [...], 'summary': {...}}
    """
    default_metadata = get_default_metadata(backend)
    results = []

    for cat_key, cat_cases in cases["categories"].items():
        for case in cat_cases:
            case_id = case["case_id"]
            category = case["category"]

            # ---- cat5: 纯追加，增量共享 ----
            if category == 5:
                prev_hashes = None
                incremental_ok = True
                chain_ok = True
                num_turns = len(case["turns"])
                for turn in case["turns"]:
                    blocks, hashes = compute_sequence_block_hashes(
                        backend, turn["messages"], block_size, default_metadata
                    )
                    chain_valid, _ = verify_parent_chain(blocks)
                    if not chain_valid:
                        chain_ok = False
                    if prev_hashes is not None:
                        # 前一轮的 hashes 应是当前 hashes 的前缀
                        if len(prev_hashes) > len(hashes):
                            incremental_ok = False
                        else:
                            for i, h in enumerate(prev_hashes):
                                if hashes[i] != h:
                                    incremental_ok = False
                                    break
                    prev_hashes = hashes

                results.append({
                    "case_id": case_id,
                    "category": category,
                    "identity_check": incremental_ok,
                    "parent_chain": chain_ok,
                    "invalidation": None,
                    "detail": f"turns={num_turns}, incremental_sharing={incremental_ok}",
                })
                continue

            # ---- pair-based 类别 (1, 2, 3, 4, 6) ----
            pair_results = []
            for item in case["pair"]:
                metadata = default_metadata.copy()
                if category == 3:
                    # template 变化：使用用例提供的 template_hash
                    metadata["template_hash"] = item.get("template_hash", "")
                elif category == 4:
                    # 标识变化：使用用例提供的 metadata 字段
                    item_meta = item.get("metadata", {})
                    if "model_id" in item_meta:
                        metadata["model_id"] = item_meta["model_id"]
                    if "revision" in item_meta:
                        metadata["revision"] = item_meta["revision"]
                    if "adapter_id" in item_meta:
                        metadata["adapter_id"] = item_meta["adapter_id"]

                blocks, hashes = compute_sequence_block_hashes(
                    backend, item["messages"], block_size, metadata
                )
                pair_results.append((blocks, hashes))

            hashes_a = pair_results[0][1]
            hashes_b = pair_results[1][1]

            # 计算公共前缀长度
            common_len = 0
            min_len = min(len(hashes_a), len(hashes_b))
            for i in range(min_len):
                if hashes_a[i] == hashes_b[i]:
                    common_len += 1
                else:
                    break

            # 检查公共前缀之后是否全部不同
            rest_differ = all(
                hashes_a[i] != hashes_b[i] for i in range(common_len, min_len)
            )

            # 父链校验
            chain_valid_a, _ = verify_parent_chain(pair_results[0][0])
            chain_valid_b, _ = verify_parent_chain(pair_results[1][0])
            chain_ok = chain_valid_a and chain_valid_b

            # invalidation 检查（仅对 cat3/cat4 有意义）
            invalidation_result = None
            if category in [3, 4]:
                inv = check_invalidation(
                    pair_results[0][0], pair_results[1][0], change_point=0
                )
                invalidation_result = inv["post_change_differ"]

            # 判定
            if category in [1, 2]:
                identity_ok = common_len > 0 and rest_differ
            elif category in [3, 4, 6]:
                identity_ok = common_len == 0
            else:
                identity_ok = False

            results.append({
                "case_id": case_id,
                "category": category,
                "identity_check": identity_ok,
                "parent_chain": chain_ok,
                "invalidation": invalidation_result,
                "detail": (
                    f"common_prefix={common_len}, "
                    f"blocks_a={len(hashes_a)}, blocks_b={len(hashes_b)}, "
                    f"rest_differ={rest_differ}"
                ),
            })

    # 汇总
    total = len(results)
    identity_pass = sum(1 for r in results if r["identity_check"])
    chain_pass = sum(1 for r in results if r["parent_chain"])
    inv_pass = sum(
        1 for r in results if r["invalidation"] is not None and r["invalidation"]
    )
    inv_total = sum(1 for r in results if r["invalidation"] is not None)

    return {
        "cases": results,
        "summary": {
            "total": total,
            "identity_pass": identity_pass,
            "parent_chain_pass": chain_pass,
            "invalidation_pass": inv_pass,
            "invalidation_total": inv_total,
        },
    }


# =============================================================================
# 数值一致性测试
# =============================================================================

def run_exactness_test(
    backend, cases: Dict, block_size: int, output_path: str
) -> Dict:
    """
    主测试函数：BF16 缓存恢复 vs 重算一致性 + block identity。

    对每个结构用例的每条消息序列：
    1. 重算路径：forward → past_kv_recompute
    2. 缓存路径：slice → restore → past_kv_restored
    3. 对比 KV 张量（bit-identical）
    4. 对比 logits（continuation token forward）
    5. 对比 greedy top-1 token

    Args:
        backend: Backend 实例。
        cases: 结构用例字典。
        block_size: block 大小。
        output_path: exactness-report.md 输出路径。

    Returns:
        {'numerical': [...], 'identity': {...}}
    """
    # 准备 continuation token（用于续算对比）
    cont_token_ids = backend.tokenizer.encode(" test", add_special_tokens=False)
    cont_token_id = cont_token_ids[0] if cont_token_ids else backend.tokenizer.eos_token_id
    cont_token = torch.tensor([[cont_token_id]], device=backend.device)

    numerical_results = []

    for cat_key, cat_cases in cases["categories"].items():
        for case in cat_cases:
            case_id = case["case_id"]
            category = case["category"]

            # 收集要测试的消息序列
            sequences = []
            if "pair" in case:
                for i, item in enumerate(case["pair"]):
                    sequences.append((f"{case_id}_p{i}", item["messages"]))
            elif "turns" in case:
                for i, turn in enumerate(case["turns"]):
                    sequences.append((f"{case_id}_t{i}", turn["messages"]))

            for seq_label, messages in sequences:
                result = _test_single_sequence(
                    backend, messages, block_size, cont_token, seq_label, category
                )
                numerical_results.append(result)

                if result["kv_bit_identical"] and result["top1_match"]:
                    status = "OK"
                else:
                    status = "FAIL"
                print(
                    f"  [{status}] {seq_label}: "
                    f"kv_bit_identical={result['kv_bit_identical']}, "
                    f"logits_diff={result['logits_max_abs_diff']:.2e}, "
                    f"top1={result['top1_match']}"
                )

    # 运行 block identity 测试
    print("\n  Running block identity test...")
    identity_results = test_block_identity(backend, cases, block_size)
    id_summary = identity_results["summary"]
    print(
        f"  Identity: {id_summary['identity_pass']}/{id_summary['total']} passed, "
        f"Parent chain: {id_summary['parent_chain_pass']}/{id_summary['total']} passed"
    )

    # 汇总结果
    results = {
        "numerical": numerical_results,
        "identity": identity_results,
    }

    # 生成报告
    generate_report(results, output_path)

    # 保存 JSON 结果（供 verdict.py 读取）
    json_path = output_path.replace(".md", "-results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to {json_path}")

    return results


def _test_single_sequence(
    backend, messages: List[Dict], block_size: int,
    cont_token: torch.Tensor, seq_label: str, category: int
) -> Dict:
    """对单条消息序列执行数值一致性测试。"""
    input_ids = backend.tokenize_chat(messages)
    seq_len = input_ids.shape[1]

    # 重算路径：完整 forward
    with torch.no_grad():
        logits_recompute, past_kv_recompute = backend.forward_with_kv(input_ids)

    # 缓存路径：slice → restore
    blocks = backend.slice_kv_into_blocks(past_kv_recompute, block_size)
    past_kv_restored = backend.restore_kv_from_blocks(blocks)

    # 对比 KV 张量
    kv_comparison = compare_kv_tensors(past_kv_recompute, past_kv_restored)

    # 续算对比：用同一个 continuation token 分别在两条路径上 forward
    with torch.no_grad():
        logits_recompute_cont, _ = backend.forward_with_kv(
            cont_token, past_kv_recompute
        )
        logits_cached, _ = backend.forward_with_kv(
            cont_token, past_kv_restored
        )

    # 对比 logits
    logits_comparison = compare_logits(logits_recompute_cont, logits_cached)

    return {
        "case_id": seq_label,
        "category": category,
        "seq_len": seq_len,
        "num_blocks": len(blocks),
        "kv_bit_identical": kv_comparison["bit_identical"],
        "kv_max_abs_diff": kv_comparison["max_abs_diff"],
        "logits_max_abs_diff": logits_comparison["max_abs_diff"],
        "logits_mean_abs_diff": logits_comparison["mean_abs_diff"],
        "logits_cosine_sim": logits_comparison["cosine_sim"],
        "top1_match": logits_comparison["top1_match"],
        "top1_a": logits_comparison["top1_a"],
        "top1_b": logits_comparison["top1_b"],
    }


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(results: Dict, output_path: str) -> None:
    """
    生成 exactness-report.md，包含表 G0-1（数值一致性）和表 G0-2（identity 正确性）。

    Args:
        results: run_exactness_test 返回的结果字典。
        output_path: 输出文件路径。
    """
    numerical = results["numerical"]
    identity = results["identity"]

    # 确保输出目录存在
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines = []
    lines.append("# G0 Exactness Report")
    lines.append("")
    lines.append("BF16 缓存恢复 vs 重算一致性测试 + block identity 正确性验证。")
    lines.append("")

    # ---- 表 G0-1: 数值一致性 ----
    lines.append("## 表 G0-1: BF16 缓存恢复 vs 重算数值一致性")
    lines.append("")
    lines.append(
        "| Case ID | Category | Seq Len | Num Blocks | KV Bit-Identical | "
        "Logits Max Abs Diff | Logits Mean Abs Diff | Cosine Sim | Top-1 Match |"
    )
    lines.append(
        "|---------|----------|---------|------------|------------------|"
        "---------------------|----------------------|------------|-------------|"
    )

    for r in numerical:
        kv_status = "✓" if r["kv_bit_identical"] else "✗"
        top1_status = "✓" if r["top1_match"] else "✗"
        lines.append(
            f"| {r['case_id']} | {r['category']} | {r['seq_len']} | "
            f"{r['num_blocks']} | {kv_status} | "
            f"{r['logits_max_abs_diff']:.2e} | "
            f"{r['logits_mean_abs_diff']:.2e} | "
            f"{r['logits_cosine_sim']:.6f} | {top1_status} |"
        )

    lines.append("")

    # 汇总统计
    total = len(numerical)
    kv_ok = sum(1 for r in numerical if r["kv_bit_identical"])
    logits_ok = sum(1 for r in numerical if r["logits_max_abs_diff"] <= 1e-3)
    top1_ok = sum(1 for r in numerical if r["top1_match"])
    lines.append("**汇总统计：**")
    lines.append("")
    lines.append(f"- 测试序列总数: {total}")
    lines.append(f"- KV bit-identical: {kv_ok}/{total}")
    lines.append(f"- Logits max abs diff ≤ 1e-3: {logits_ok}/{total}")
    lines.append(f"- Top-1 token 一致: {top1_ok}/{total}")
    lines.append("")

    # ---- 表 G0-2: identity/父链/invalidation ----
    lines.append("## 表 G0-2: Block Identity / 父链 / Invalidation 正确性")
    lines.append("")
    lines.append(
        "| Case ID | Category | Identity Check | Parent Chain | "
        "Invalidation | Detail |"
    )
    lines.append(
        "|---------|----------|----------------|--------------|"
        "--------------|--------|"
    )

    for r in identity["cases"]:
        id_status = "✓ PASS" if r["identity_check"] else "✗ FAIL"
        chain_status = "✓ PASS" if r["parent_chain"] else "✗ FAIL"
        if r["invalidation"] is None:
            inv_status = "N/A"
        else:
            inv_status = "✓ PASS" if r["invalidation"] else "✗ FAIL"
        lines.append(
            f"| {r['case_id']} | {r['category']} | {id_status} | "
            f"{chain_status} | {inv_status} | {r['detail']} |"
        )

    lines.append("")

    # 汇总
    id_summary = identity["summary"]
    lines.append("**汇总统计：**")
    lines.append("")
    lines.append(f"- 用例总数: {id_summary['total']}")
    lines.append(f"- Identity check 通过: {id_summary['identity_pass']}/{id_summary['total']}")
    lines.append(f"- 父链校验通过: {id_summary['parent_chain_pass']}/{id_summary['total']}")
    lines.append(
        f"- Invalidation 通过: {id_summary['invalidation_pass']}/{id_summary['invalidation_total']}"
        if id_summary["invalidation_total"] > 0
        else "- Invalidation: N/A"
    )
    lines.append("")

    # ---- 判定 ----
    lines.append("## 判定")
    lines.append("")
    all_kv_ok = kv_ok == total
    all_logits_ok = logits_ok == total
    all_top1_ok = top1_ok == total
    all_id_ok = id_summary["identity_pass"] == id_summary["total"]
    all_chain_ok = id_summary["parent_chain_pass"] == id_summary["total"]

    overall_pass = all([all_kv_ok, all_logits_ok, all_top1_ok, all_id_ok, all_chain_ok])
    verdict = "PASS" if overall_pass else "FAIL"

    lines.append(f"- BF16 缓存恢复 KV bit-identical: {'✓' if all_kv_ok else '✗'}")
    lines.append(f"- Logits max abs diff ≤ 1e-3: {'✓' if all_logits_ok else '✗'}")
    lines.append(f"- Top-1 token 100% 一致: {'✓' if all_top1_ok else '✗'}")
    lines.append(f"- Block identity 正确: {'✓' if all_id_ok else '✗'}")
    lines.append(f"- 父链连续性正确: {'✓' if all_chain_ok else '✗'}")
    lines.append("")
    lines.append(f"**Overall: {verdict}**")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {output_path}")


if __name__ == "__main__":
    # 独立运行：需要先加载 backend 和 cases
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
    run_exactness_test(
        backend, cases, config["cache"]["block_size"],
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config["output"]["exactness_report"]),
    )
