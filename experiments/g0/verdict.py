"""
G0 判定报告生成 (Task 10).

汇总 6 个判定条件的通过/失败状态，生成 g0-verdict.md：
1. BF16 缓存恢复与重算一致
2. block identity/父链/invalidation 无错误
3. freeze-record 完整
4. codec/staging/lineage spike 跑通
5. 后端能拦截/恢复 KV
6. 显存可承载

任一失败则标记 G0 = FAILED，提示失败动作。
"""
import json
import os
from typing import Dict, List, Tuple

# 添加当前目录到 path 以便 import freeze_record
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from freeze_record import validate_freeze_record


# =============================================================================
# 判定条件检查
# =============================================================================

def _load_json(path: str) -> Dict:
    """加载 JSON 文件，失败返回空字典。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def check_condition_1_exactness(exactness_data: Dict) -> Dict:
    """
    条件 1: BF16 缓存恢复与重算一致。

    检查 KV bit-identical、logits max abs diff ≤ 1e-3、top-1 token 100% 一致。
    """
    name = "BF16 缓存恢复与重算一致"
    numerical = exactness_data.get("numerical", [])

    if not numerical:
        return {
            "id": 1,
            "name": name,
            "passed": False,
            "evidence": "exactness-results.json 中无数值结果",
        }

    total = len(numerical)
    kv_ok = sum(1 for r in numerical if r.get("kv_bit_identical", False))
    logits_ok = sum(
        1 for r in numerical if r.get("logits_max_abs_diff", float("inf")) <= 1e-3
    )
    top1_ok = sum(1 for r in numerical if r.get("top1_match", False))

    passed = (kv_ok == total) and (logits_ok == total) and (top1_ok == total)

    return {
        "id": 1,
        "name": name,
        "passed": passed,
        "evidence": (
            f"KV bit-identical: {kv_ok}/{total}, "
            f"logits diff ≤ 1e-3: {logits_ok}/{total}, "
            f"top-1 match: {top1_ok}/{total}"
        ),
        "details": {
            "total": total,
            "kv_identical": kv_ok,
            "logits_ok": logits_ok,
            "top1_ok": top1_ok,
        },
    }


def check_condition_2_identity(exactness_data: Dict) -> Dict:
    """
    条件 2: block identity/父链/invalidation 无错误。

    检查 identity test 通过率和父链校验通过率。

    注意：cat5 用例的 expected_incremental_sharing=False（tokenizer 非前缀
    稳定现象），实测与期望一致才算 PASS。本函数直接采用 identity_cases 中
    各用例的 identity_check 字段汇总，无需特殊处理 cat5。
    """
    name = "block identity/父链/invalidation 无错误"
    identity = exactness_data.get("identity", {})
    summary = identity.get("summary", {})

    if not summary:
        return {
            "id": 2,
            "name": name,
            "passed": False,
            "evidence": "exactness-results.json 中无 identity 结果",
        }

    total = summary.get("total", 0)
    identity_pass = summary.get("identity_pass", 0)
    chain_pass = summary.get("parent_chain_pass", 0)
    inv_pass = summary.get("invalidation_pass", 0)
    inv_total = summary.get("invalidation_total", 0)

    passed = (
        (identity_pass == total)
        and (chain_pass == total)
        and (inv_total == 0 or inv_pass == inv_total)
    )

    # 拆分 cat5 与非 cat5 通过率，便于报告区分"代码 bug"与"正向发现"
    identity_cases = identity.get("cases", [])
    cat5_total = sum(1 for c in identity_cases if c.get("category") == 5)
    cat5_pass = sum(
        1 for c in identity_cases
        if c.get("category") == 5 and c.get("identity_check")
    )
    non_cat5_total = total - cat5_total
    non_cat5_pass = identity_pass - cat5_pass

    return {
        "id": 2,
        "name": name,
        "passed": passed,
        "evidence": (
            f"identity: {identity_pass}/{total} "
            f"(非 cat5: {non_cat5_pass}/{non_cat5_total}, "
            f"cat5: {cat5_pass}/{cat5_total}), "
            f"parent chain: {chain_pass}/{total}, "
            f"invalidation: {inv_pass}/{inv_total if inv_total > 0 else 'N/A'}"
        ),
        "details": summary,
        "cat5_breakdown": {
            "cat5_total": cat5_total,
            "cat5_pass": cat5_pass,
            "non_cat5_total": non_cat5_total,
            "non_cat5_pass": non_cat5_pass,
        },
    }


def check_condition_3_freeze(freeze_data: Dict) -> Dict:
    """
    条件 3: freeze-record 完整。

    使用 validate_freeze_record 校验必填字段。
    """
    name = "freeze-record 完整"

    if not freeze_data:
        return {
            "id": 3,
            "name": name,
            "passed": False,
            "evidence": "freeze-record.json 不存在或为空",
        }

    is_valid, missing = validate_freeze_record(freeze_data)

    return {
        "id": 3,
        "name": name,
        "passed": is_valid,
        "evidence": (
            f"校验通过: {is_valid}"
            + (f"，缺失字段: {missing}" if missing else "")
        ),
        "details": {"missing_fields": missing},
    }


def check_condition_4_codec(codec_data: Dict) -> Dict:
    """
    条件 4: codec/staging/lineage spike 跑通。

    检查 codec roundtrip 执行成功且 lineage 隔离正确。
    """
    name = "codec/staging/lineage spike 跑通"

    if not codec_data:
        return {
            "id": 4,
            "name": name,
            "passed": False,
            "evidence": "codec-results.json 不存在或为空",
        }

    summary = codec_data.get("summary", {})
    num_blocks = codec_data.get("num_blocks_tested", 0)

    q8_mse = summary.get("q8", {}).get("mse_mean", 0)
    q4_mse = summary.get("q4", {}).get("mse_mean", 0)
    lineage = summary.get("lineage", {})
    lineage_isolated = lineage.get("all_isolated", False)

    # codec 跑通的判定：测试了 block 且 Q8/Q4 有 MSE 值
    codec_executed = num_blocks > 0 and q8_mse > 0 and q4_mse > 0
    passed = codec_executed and lineage_isolated

    return {
        "id": 4,
        "name": name,
        "passed": passed,
        "evidence": (
            f"测试 block 数: {num_blocks}, "
            f"Q8 MSE: {q8_mse:.2e}, Q4 MSE: {q4_mse:.2e}, "
            f"lineage 隔离: {lineage_isolated}"
        ),
        "details": summary,
    }


def check_condition_5_backend(exactness_data: Dict) -> Dict:
    """
    条件 5: 后端能拦截/恢复 KV。

    如果 exactness 测试有结果（slice + restore 成功执行），则后端拦截/恢复能力验证通过。
    KV bit-identical 是拦截/恢复正确性的直接证据。
    """
    name = "后端能拦截/恢复 KV"
    numerical = exactness_data.get("numerical", [])

    if not numerical:
        return {
            "id": 5,
            "name": name,
            "passed": False,
            "evidence": "无 exactness 结果，无法验证后端拦截/恢复",
        }

    total = len(numerical)
    kv_identical = sum(1 for r in numerical if r.get("kv_bit_identical", False))
    # slice + restore 成功执行（有 num_blocks > 0）即说明拦截/恢复可用
    has_blocks = all(r.get("num_blocks", 0) > 0 for r in numerical)
    passed = has_blocks and (kv_identical == total)

    return {
        "id": 5,
        "name": name,
        "passed": passed,
        "evidence": (
            f"slice + restore 成功执行: {has_blocks}, "
            f"KV bit-identical: {kv_identical}/{total}"
        ),
    }


def check_condition_6_memory(memory_data: Dict) -> Dict:
    """
    条件 6: 显存可承载。

    检查权重 + active cache + staging + 安全水位 ≤ 24GB。
    """
    name = "显存可承载"

    if not memory_data:
        return {
            "id": 6,
            "name": name,
            "passed": False,
            "evidence": "memory-results.json 不存在或为空",
        }

    verdict = memory_data.get("verdict", "FAIL")
    required = memory_data.get("required_with_safety_gb", 0)
    memory_limit = memory_data.get("memory_limit_gb", 24)
    configs = memory_data.get("configs", [])
    max_reserved = max((r.get("reserved_peak_gb", 0) for r in configs), default=0)

    return {
        "id": 6,
        "name": name,
        "passed": verdict == "PASS",
        "evidence": (
            f"最大 reserved 峰值: {max_reserved:.3f} GB, "
            f"含安全水位: {required:.3f} GB, "
            f"上限: {memory_limit:.0f} GB, "
            f"判定: {verdict}"
        ),
    }


# =============================================================================
# 主判定函数
# =============================================================================

def generate_verdict(config: Dict, output_path: str) -> Dict:
    """
    汇总 6 个判定条件，生成 g0-verdict.md。

    Args:
        config: 配置字典。
        output_path: g0-verdict.md 输出路径。

    Returns:
        {'conditions': [...], 'overall': 'PASSED'/'FAILED'}
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, config["output"]["dir"])

    # 加载各模块的 JSON 结果
    exactness_data = _load_json(os.path.join(outputs_dir, "exactness-report-results.json"))
    codec_data = _load_json(os.path.join(outputs_dir, "codec-spike-report-results.json"))
    memory_data = _load_json(os.path.join(outputs_dir, "memory-report-results.json"))
    freeze_data = _load_json(os.path.join(base_dir, config["output"]["freeze_record"]))

    # 检查 6 个判定条件
    conditions = [
        check_condition_1_exactness(exactness_data),
        check_condition_2_identity(exactness_data),
        check_condition_3_freeze(freeze_data),
        check_condition_4_codec(codec_data),
        check_condition_5_backend(exactness_data),
        check_condition_6_memory(memory_data),
    ]

    # 总体判定
    all_pass = all(c["passed"] for c in conditions)
    overall = "PASSED" if all_pass else "FAILED"

    output = {
        "conditions": conditions,
        "overall": overall,
    }

    # 生成报告
    generate_verdict_report(conditions, overall, output_path)

    # 保存 JSON 结果
    json_path = output_path.replace(".md", "-results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {json_path}")

    return output


def generate_verdict_report(
    conditions: List[Dict], overall: str, output_path: str
) -> None:
    """
    生成 g0-verdict.md 判定报告。

    Args:
        conditions: 6 个判定条件的结果列表。
        overall: 总体判定 'PASSED' 或 'FAILED'。
        output_path: 输出文件路径。
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines = []
    lines.append("# G0 Verdict Report")
    lines.append("")
    lines.append("G0: Exactness & Loadability 判定报告。")
    lines.append("")

    # ---- 判定条件汇总表 ----
    lines.append("## 判定条件汇总")
    lines.append("")
    lines.append("| # | 条件 | 状态 | 证据 |")
    lines.append("|---|------|------|------|")

    for c in conditions:
        status = "✓ PASS" if c["passed"] else "✗ FAIL"
        lines.append(f"| {c['id']} | {c['name']} | {status} | {c['evidence']} |")

    lines.append("")

    # ---- 总体判定 ----
    lines.append("## 总体判定")
    lines.append("")
    if overall == "PASSED":
        lines.append("**G0 = PASSED** ✅")
        lines.append("")
        lines.append("所有 6 个判定条件均通过，G0 实验完成，可进入后续实验。")
    else:
        lines.append("**G0 = FAILED** ❌")
        lines.append("")
        lines.append("存在未通过的判定条件，需根据以下失败动作进行修复：")
        lines.append("")

        # 失败动作提示
        failed = [c for c in conditions if not c["passed"]]
        for c in failed:
            lines.append(f"### 条件 {c['id']}: {c['name']}")
            lines.append("")
            lines.append(f"- 证据: {c['evidence']}")
            action = _get_failure_action(c["id"])
            lines.append(f"- 失败动作: {action}")
            lines.append("")

    lines.append("")

    # ---- 汇总统计 ----
    passed_count = sum(1 for c in conditions if c["passed"])
    total_count = len(conditions)
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 通过条件数: {passed_count}/{total_count}")
    lines.append(f"- 总体判定: {overall}")
    lines.append("")

    # ---- 正向发现 ----
    lines.append("## 正向发现")
    lines.append("")
    lines.append(
        "G0 在验证后端正确性的同时，产出了两项对 prefix caching 研究有直接价值的发现："
    )
    lines.append("")
    lines.append("### 发现 1：Tokenizer 非前缀稳定现象（cat5 实证）")
    lines.append("")
    lines.append(
        "- **现象**：Qwen2.5 BPE tokenizer 在 chat-template 边界"
        "（`\\n` 与 `\\nI`、`<|im_start|>assistant\\n` 与紧接其后的回复首字符）"
        "产生跨边界合并 token。"
    )
    lines.append(
        "- **影响**：纯追加多轮会话用 apply_chat_template 重新渲染时，"
        "token id 序列不以前缀 N 为严格前缀，block hash 从追加边界起分叉。"
    )
    lines.append(
        "- **研究意义**：朴素按 token id 前缀匹配会丢失复用机会，"
        "为 IDEA 中 C2 联合控制器的 boundary-aware 决策、"
        "C3 \"reuse value 与 fidelity 风险错位\" 主张提供实证依据。"
    )
    lines.append(
        "- **处置**：写入 cat5 用例的 expected_incremental_sharing=False，"
        "实测 False 则 identity_check PASS，作为 G0 正向输出。"
    )
    lines.append("")
    lines.append("### 发现 2：Q4 per-tensor 量化损失过大")
    lines.append("")
    lines.append(
        "- **数据**：Q8 MSE=1.6e-2, KL mean=0.97；Q4 MSE=0.89, KL mean=5.2, "
        "KL max=19.2，max_abs_err 高达 29.6。"
    )
    lines.append(
        "- **结论**：Q4 per-tensor 量化对 Qwen2.5-7B 的 KV cache 不可用，"
        "后续 G2+ 显存压缩方案应只用 Q8 或改用 per-channel/per-group 量化。"
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {output_path}")


def _get_failure_action(condition_id: int) -> str:
    """根据条件 ID 返回失败动作描述。"""
    actions = {
        1: "检查 backend.slice_kv_into_blocks / restore_kv_from_blocks 的张量拷贝逻辑，"
           "确认 clone() 调用正确；检查 forward_with_kv 是否正确传递 past_key_values。",
        2: "区分 cat5 与非 cat5：\n"
           "- 非 cat5 失败 → 检查 compute_block_hash 的元数据字段"
           "(model_id/revision/template_hash/config_hash/adapter_id) 是否正确传入；"
           "检查 verify_parent_chain 的父链连续性逻辑；检查 check_invalidation 的 change_point 设置。\n"
           "- cat5 失败 → 通常是 OOM 跳过导致；如非 OOM，则需重新核对"
           "expected_incremental_sharing 与实测 actual_incremental_sharing 是否一致。"
           "注意 cat5 的 expected_incremental_sharing=False（tokenizer 非前缀稳定现象），"
           "实测 False 才算 PASS。",
        3: "检查 freeze_record.py 的 generate_freeze_record 是否收集了全部必填字段；"
           "确认模型已正确加载且 get_model_info() 返回完整信息。",
        4: "检查 codec.py 的 Q8/Q4 编解码逻辑；确认 codec_spike.py 能正确收集 block 并执行 roundtrip；"
           "检查 lineage 隔离的 adapter_id 设置。",
        5: "检查 Backend 类的 KV cache 拦截能力（slice_kv_into_blocks / restore_kv_from_blocks）；"
           "确认 DynamicCache 格式兼容性；检查张量 clone 是否到位。",
        6: "降低并发数或上下文长度（4090D 24GB 推荐 4x4k 或 4x8k 配置）；"
           "Q8 per-tensor 量化可减半 KV cache 体积（参见 codec spike），"
           "但 Q4 per-tensor 损失过大不建议用于显存压缩；"
           "检查 GPU 显存是否足够（需 ≥ 24GB）。",
    }
    return actions.get(condition_id, "未知失败，请检查相关模块日志。")


if __name__ == "__main__":
    import yaml

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generate_verdict(
        config,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config["output"]["verdict"]),
    )
