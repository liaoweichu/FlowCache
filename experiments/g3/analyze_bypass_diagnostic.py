"""
G3-P1 Bypass 机制诊断脚本
=========================
从 tune 阶段的输出中分析 GPU admission/bypass 增量不足的机制原因。

诊断目标：
  判断 bypass 增量不足是
  (A) proxy 过保守（机制问题，可改进）
  (B) workload 特性决定（主张问题，需调整）

诊断维度：
  1. incoming/incumbent 价值比值分布
     - >> 1：incoming 确实价值高，bypass 少合理 → workload 特性
     - ≈ 1：等价值时 doorkeeper 主导，incoming 总是被 admit → proxy 过保守
  2. bypass rate vs 参数组合
     - 某些参数能显著提高 bypass rate → 机制可调
     - 所有组合都类似 → workload 决定
  3. transfer reduction vs bypass rate
     - bypass rate 高但 transfer reduction 低 → 被旁路 block 的 transfer 成本小
     - bypass rate 低 → 问题是 bypass 太少
  4. displacement value 分布
     - displacement_value ≈ 0 → CPU 无法回收被驱逐 victim 的价值
     - displacement_value > 0 → CPU 能回收，但 incoming 仍优于 incumbent

输入：results/selective-tuning/ 目录下的 summary.csv 和 {candidate_id}.csv
输出：bypass-diagnostic.md（人类可读报告）

Usage:
    python experiments/g3/analyze_bypass_diagnostic.py
    python experiments/g3/analyze_bypass_diagnostic.py --tuning-dir results/selective-tuning
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TUNING_DIR = SCRIPT_DIR / "results" / "selective-tuning"

FLOWCACHE = "flowcache_lossless"
SELECTIVE_ONLY = "flowcache_selective_migrate_only"
ALWAYS = "flowcache_always_migrate"


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_summary(tuning_dir: Path) -> List[Dict]:
    """加载 summary.csv。"""
    summary_path = tuning_dir / "summary.csv"
    if not summary_path.exists():
        return []
    rows = []
    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_candidate_csv(tuning_dir: Path, candidate_id: str) -> List[Dict]:
    """加载单个候选的 csv 文件。"""
    csv_path = tuning_dir / f"{candidate_id}.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def extract_flowcache_audit(rows: List[Dict]) -> Optional[Dict]:
    """从候选 csv 行中提取 FlowCache 的审计指标。

    FlowCache 行的 global 字段在同一 cell×baseline 下相同，取第一行即可。
    """
    for r in rows:
        if r.get("baseline") == FLOWCACHE:
            return {
                "candidate_count": int(_to_float(r.get("gpu_admission_candidate_count"))),
                "bypassed_count": int(_to_float(r.get("gpu_admission_bypassed_count"))),
                "selected_count": int(_to_float(r.get("gpu_admission_selected_count"))),
                "incoming_value_total": _to_float(r.get("gpu_admission_candidate_value_index_ms_total")),
                "incumbent_value_total": _to_float(r.get("gpu_admission_incumbent_value_index_ms_total")),
                "displacement_value_total": _to_float(r.get("gpu_admission_displacement_value_index_ms_total")),
                "bypassed_prefill_ms_total": _to_float(r.get("gpu_bypassed_prefill_ms_total")),
                "transfer_ms_total": _to_float(r.get("transfer_ms_total")),
                "migrate_count": int(_to_float(r.get("migrate_count"))),
                "global_p95_cache_delay_ms": _to_float(r.get("global_p95_cache_delay_ms")),
            }
    return None


def extract_selective_only_transfer(rows: List[Dict]) -> float:
    """从候选 csv 行中提取 Selective-Migrate-Only 的 transfer_ms_total。"""
    for r in rows:
        if r.get("baseline") == SELECTIVE_ONLY:
            return _to_float(r.get("transfer_ms_total"))
    return 0.0


def extract_always_migrate_count(rows: List[Dict]) -> int:
    """从候选 csv 行中提取 Always-Migrate 的 migrate_count。"""
    for r in rows:
        if r.get("baseline") == ALWAYS:
            return int(_to_float(r.get("migrate_count")))
    return 0


def analyze_candidate(summary_row: Dict, tuning_dir: Path) -> Optional[Dict]:
    """分析单个候选的 bypass 机制。"""
    candidate_id = summary_row.get("candidate_id", "")
    if not candidate_id:
        return None

    csv_rows = load_candidate_csv(tuning_dir, candidate_id)
    if not csv_rows:
        return None

    fc_audit = extract_flowcache_audit(csv_rows)
    if fc_audit is None:
        return None

    selective_transfer = extract_selective_only_transfer(csv_rows)
    always_migrations = extract_always_migrate_count(csv_rows)

    candidate_count = fc_audit["candidate_count"]
    bypassed_count = fc_audit["bypassed_count"]
    selected_count = fc_audit["selected_count"]
    incoming_total = fc_audit["incoming_value_total"]
    incumbent_total = fc_audit["incumbent_value_total"]
    displacement_total = fc_audit["displacement_value_total"]

    # 核心诊断指标
    bypass_rate = bypassed_count / candidate_count if candidate_count > 0 else 0.0
    # incoming/incumbent 比值：>>1 说明 incoming 确实价值高
    value_ratio = incoming_total / incumbent_total if incumbent_total > 0 else float("inf")
    # displacement 占 incoming 的比例：高说明 CPU 能有效回收 victim 价值
    displacement_ratio = displacement_total / incoming_total if incoming_total > 0 else 0.0
    # 被 bypass block 的平均 prefill 成本
    avg_bypassed_prefill = (
        fc_audit["bypassed_prefill_ms_total"] / bypassed_count
        if bypassed_count > 0 else 0.0
    )
    # transfer reduction（vs selective-only）
    fc_transfer = fc_audit["transfer_ms_total"]
    transfer_reduction = (
        (selective_transfer - fc_transfer) / selective_transfer
        if selective_transfer > 0 else 0.0
    )
    # 每个 bypass 节省的 transfer（ms）
    transfer_per_bypass = (
        (selective_transfer - fc_transfer) / bypassed_count
        if bypassed_count > 0 else 0.0
    )
    # movement reduction（vs always-migrate）
    fc_migrations = fc_audit["migrate_count"]
    movement_reduction = (
        (always_migrations - fc_migrations) / always_migrations
        if always_migrations > 0 else 0.0
    )

    return {
        "candidate_id": candidate_id,
        "params": {
            "minimum_net_benefit_ms": summary_row.get("minimum_net_benefit_ms", ""),
            "cpu_admission_margin_ms": summary_row.get("cpu_admission_margin_ms", ""),
            "gpu_admission_cold_start_cost_ratio": summary_row.get("gpu_admission_cold_start_cost_ratio", ""),
            "expected_cpu_residence_steps": summary_row.get("expected_cpu_residence_steps", ""),
        },
        "bypass_rate": bypass_rate,
        "value_ratio": value_ratio,
        "displacement_ratio": displacement_ratio,
        "avg_bypassed_prefill_ms": avg_bypassed_prefill,
        "transfer_reduction": transfer_reduction,
        "transfer_per_bypass_ms": transfer_per_bypass,
        "movement_reduction": movement_reduction,
        "bypassed_count": bypassed_count,
        "selected_count": selected_count,
        "candidate_count": candidate_count,
        "incoming_value_total": incoming_total,
        "incumbent_value_total": incumbent_total,
        "displacement_value_total": displacement_total,
        "fc_transfer_ms": fc_transfer,
        "selective_only_transfer_ms": selective_transfer,
        "fc_p95_ms": fc_audit["global_p95_cache_delay_ms"],
        "fc_migrations": fc_migrations,
        "always_migrations": always_migrations,
    }


def diagnose(analyses: List[Dict]) -> Dict:
    """综合诊断。"""
    if not analyses:
        return {"verdict": "NO_DATA", "reason": "无候选数据"}

    n = len(analyses)
    bypass_rates = [a["bypass_rate"] for a in analyses]
    value_ratios = [a["value_ratio"] for a in analyses if a["value_ratio"] != float("inf")]
    displacement_ratios = [a["displacement_ratio"] for a in analyses]
    transfer_reductions = [a["transfer_reduction"] for a in analyses]
    transfer_per_bypass = [a["transfer_per_bypass_ms"] for a in analyses if a["bypassed_count"] > 0]

    # 诊断 1: incoming/incumbent 价值比值
    median_ratio = statistics.median(value_ratios) if value_ratios else 0.0
    if median_ratio > 1.5:
        value_verdict = "INCOMING_DOMINATES"
        value_reason = (
            f"incoming 价值远高于 incumbent（中位比值 {median_ratio:.2f}），"
            "bypass 少是合理的——τ-bench 的 prefix 共享结构使得大多数 incoming block "
            "确实值得准入 GPU。这是 workload 特性，不是 proxy 过保守。"
        )
    elif median_ratio > 1.1:
        value_verdict = "INCOMING_SLIGHTLY_HIGHER"
        value_reason = (
            f"incoming 价值略高于 incumbent（中位比值 {median_ratio:.2f}），"
            "doorkeeper 的 cold-start prior 可能在等价值时偏向 admit。"
            "可尝试降低 cold_start_prior 使 proxy 更激进。"
        )
    else:
        value_verdict = "APPROXIMATELY_EQUAL"
        value_reason = (
            f"incoming 与 incumbent 价值近似相等（中位比值 {median_ratio:.2f}），"
            "doorkeeper 主导决策，incoming 几乎总是被 admit。"
            "这是 proxy 过保守的信号——可改进 cold-start prior 或 hold cost 模型。"
        )

    # 诊断 2: bypass rate 的参数敏感性
    max_bypass = max(bypass_rates)
    min_bypass = min(bypass_rates)
    bypass_range = max_bypass - min_bypass
    if bypass_range < 0.02:
        param_verdict = "PARAM_INSENSITIVE"
        param_reason = (
            f"bypass rate 对参数不敏感（范围 {min_bypass:.4f} - {max_bypass:.4f}，"
            f"差值 {bypass_range:.4f}），说明 workload 特性决定 bypass 上限，"
            "调参无法显著改变。"
        )
    else:
        param_verdict = "PARAM_SENSITIVE"
        param_reason = (
            f"bypass rate 对参数敏感（范围 {min_bypass:.4f} - {max_bypass:.4f}，"
            f"差值 {bypass_range:.4f}），某些参数组合能显著提高 bypass rate。"
            "可尝试扩大参数网格。"
        )

    # 诊断 3: transfer reduction vs bypass rate
    median_transfer_reduction = statistics.median(transfer_reductions)
    median_transfer_per_bypass = statistics.median(transfer_per_bypass) if transfer_per_bypass else 0.0
    if median_transfer_per_bypass < 1.0:
        transfer_verdict = "LOW_TRANSFER_PER_BYPASS"
        transfer_reason = (
            f"每个 bypass 仅节省 {median_transfer_per_bypass:.3f} ms transfer，"
            "被旁路 block 的 transfer 成本本身很小。"
            "即使提高 bypass rate，transfer reduction 也难以达到 5% 门槛。"
            "这是 workload 特性——大部分 transfer 成本来自高价值复用 block，"
            "而这些 block 不应该被 bypass。"
        )
    else:
        transfer_verdict = "REASONABLE_TRANSFER_PER_BYPASS"
        transfer_reason = (
            f"每个 bypass 节省 {median_transfer_per_bypass:.3f} ms transfer，"
            "被旁路 block 的 transfer 成本合理。"
            "问题是 bypass rate 太低，可尝试改进 proxy 提高 bypass rate。"
        )

    # 诊断 4: displacement value
    median_displacement = statistics.median(displacement_ratios)
    if median_displacement < 0.05:
        displacement_verdict = "CPU_RECOVERY_WEAK"
        displacement_reason = (
            f"displacement value 占 incoming 价值的 {median_displacement:.4f}，"
            "CPU 几乎无法回收被驱逐 victim 的价值。"
            "这说明 selective migration 已经把低价值 block 迁移到 CPU，"
            "剩余的 GPU victim 价值高，CPU 难以回收——workload 特性。"
        )
    else:
        displacement_verdict = "CPU_RECOVERY_OK"
        displacement_reason = (
            f"displacement value 占 incoming 价值的 {median_displacement:.4f}，"
            "CPU 能有效回收被驱逐 victim 的价值。"
            "bypass 少的原因不是 displacement 不足。"
        )

    # 综合判定
    is_workload_determined = (
        value_verdict == "INCOMING_DOMINATES"
        and param_verdict == "PARAM_INSENSITIVE"
        and transfer_verdict == "LOW_TRANSFER_PER_BYPASS"
    )
    is_proxy_conservative = (
        value_verdict in ("APPROXIMATELY_EQUAL", "INCOMING_SLIGHTLY_HIGHER")
        and param_verdict == "PARAM_SENSITIVE"
    )

    if is_workload_determined:
        overall = "WORKLOAD_DETERMINED"
        overall_reason = (
            "GPU bypass 增量不足是 workload 特性决定，不是机制问题。"
            "τ-bench 的 prefix 共享结构使得大多数 incoming block 价值高于 incumbent，"
            "bypass 少是合理的。应调整研究主张：将核心贡献聚焦在选择性迁移"
            "（已验证 72-77% reduction），GPU admission 降级为可选增强。"
        )
    elif is_proxy_conservative:
        overall = "PROXY_CONSERVATIVE"
        overall_reason = (
            "GPU bypass 增量不足是 proxy 过保守导致。"
            "doorkeeper 的 cold-start prior 在等价值时偏向 admit。"
            "可改进：降低 cold_start_prior、调整 hold cost 模型、"
            "或放宽 doorkeeper 的触发条件。"
        )
    else:
        overall = "MIXED"
        overall_reason = (
            "GPU bypass 增量不足是 workload 特性和 proxy 保守的混合结果。"
            "建议：(1) 尝试改进 proxy（降低 cold_start_prior），"
            "(2) 同时准备主张调整文档作为 fallback。"
        )

    return {
        "verdict": overall,
        "reason": overall_reason,
        "n_candidates": n,
        "diagnostics": {
            "value_ratio": {
                "verdict": value_verdict,
                "reason": value_reason,
                "median": median_ratio,
            },
            "param_sensitivity": {
                "verdict": param_verdict,
                "reason": param_reason,
                "bypass_rate_range": [min_bypass, max_bypass],
            },
            "transfer_efficiency": {
                "verdict": transfer_verdict,
                "reason": transfer_reason,
                "median_transfer_per_bypass_ms": median_transfer_per_bypass,
                "median_transfer_reduction": median_transfer_reduction,
            },
            "displacement_recovery": {
                "verdict": displacement_verdict,
                "reason": displacement_reason,
                "median_displacement_ratio": median_displacement,
            },
        },
        "stats": {
            "median_bypass_rate": statistics.median(bypass_rates),
            "median_value_ratio": median_ratio,
            "median_transfer_reduction": median_transfer_reduction,
            "median_movement_reduction": statistics.median(
                [a["movement_reduction"] for a in analyses]
            ),
        },
    }


def write_report(diagnosis: Dict, analyses: List[Dict], output_path: Path) -> None:
    """输出诊断报告。"""
    lines = []
    lines.append("# G3-P1 Bypass 机制诊断报告")
    lines.append("")
    lines.append(f"**候选数**: {diagnosis['n_candidates']}")
    lines.append(f"**综合判定**: `{diagnosis['verdict']}`")
    lines.append("")
    lines.append("## 综合判定")
    lines.append("")
    lines.append(diagnosis["reason"])
    lines.append("")

    diag = diagnosis["diagnostics"]
    lines.append("## 诊断 1: incoming/incumbent 价值比值")
    lines.append("")
    d = diag["value_ratio"]
    lines.append(f"- **判定**: `{d['verdict']}`")
    lines.append(f"- **中位比值**: {d['median']:.4f}")
    lines.append(f"- **解读**: {d['reason']}")
    lines.append("")

    lines.append("## 诊断 2: bypass rate 参数敏感性")
    lines.append("")
    d = diag["param_sensitivity"]
    lines.append(f"- **判定**: `{d['verdict']}`")
    lines.append(f"- **bypass rate 范围**: {d['bypass_rate_range'][0]:.4f} - {d['bypass_rate_range'][1]:.4f}")
    lines.append(f"- **解读**: {d['reason']}")
    lines.append("")

    lines.append("## 诊断 3: transfer 效率")
    lines.append("")
    d = diag["transfer_efficiency"]
    lines.append(f"- **判定**: `{d['verdict']}`")
    lines.append(f"- **每个 bypass 节省 transfer**: {d['median_transfer_per_bypass_ms']:.4f} ms")
    lines.append(f"- **中位 transfer reduction**: {d['median_transfer_reduction']:.4f}")
    lines.append(f"- **解读**: {d['reason']}")
    lines.append("")

    lines.append("## 诊断 4: CPU displacement 回收")
    lines.append("")
    d = diag["displacement_recovery"]
    lines.append(f"- **判定**: `{d['verdict']}`")
    lines.append(f"- **中位 displacement ratio**: {d['median_displacement_ratio']:.4f}")
    lines.append(f"- **解读**: {d['reason']}")
    lines.append("")

    lines.append("## 关键统计")
    lines.append("")
    stats = diagnosis["stats"]
    lines.append(f"- 中位 bypass rate: {stats['median_bypass_rate']:.4f}")
    lines.append(f"- 中位 incoming/incumbent 比值: {stats['median_value_ratio']:.4f}")
    lines.append(f"- 中位 transfer reduction: {stats['median_transfer_reduction']:.4f}")
    lines.append(f"- 中位 movement reduction: {stats['median_movement_reduction']:.4f}")
    lines.append("")

    lines.append("## 各候选详情")
    lines.append("")
    lines.append("| candidate | bypass_rate | value_ratio | transfer_reduction | movement_reduction | transfer/bypass (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for a in sorted(analyses, key=lambda x: x["transfer_reduction"], reverse=True):
        ratio_str = f"{a['value_ratio']:.2f}" if a['value_ratio'] != float('inf') else "inf"
        tpb = f"{a['transfer_per_bypass_ms']:.3f}" if a['bypassed_count'] > 0 else "N/A"
        lines.append(
            f"| {a['candidate_id']} | {a['bypass_rate']:.4f} | {ratio_str} | "
            f"{a['transfer_reduction']:.4f} | {a['movement_reduction']:.4f} | {tpb} |"
        )
    lines.append("")

    lines.append("## 决策建议")
    lines.append("")
    if diagnosis["verdict"] == "WORKLOAD_DETERMINED":
        lines.append("1. **承认 workload 不支持 GPU bypass 作为核心机制**")
        lines.append("2. **将研究主张聚焦在选择性迁移**（已验证 72-77% movement reduction）")
        lines.append("3. **GPU admission 降级为可选增强**，不作为 G3 的核心门槛")
        lines.append("4. **调整 G3 门槛定义**：移除 gpu_bypass_transfer_reduction >= 5% 门槛")
        lines.append("5. **重跑 tune**（仅选择性迁移参数）→ 冻结 → held-out test")
    elif diagnosis["verdict"] == "PROXY_CONSERVATIVE":
        lines.append("1. **改进 likelihood proxy**：降低 cold_start_prior（0.05 → 0.01）")
        lines.append("2. **调整 hold cost 模型**：可能高估了 GPU 持有成本")
        lines.append("3. **放宽 doorkeeper 触发条件**：使等价值时更倾向 bypass")
        lines.append("4. **重跑 tune** with 改进后的 proxy")
    else:
        lines.append("1. **先尝试改进 proxy**（降低 cold_start_prior）")
        lines.append("2. **同时准备主张调整文档**作为 fallback")
        lines.append("3. **重跑 tune** → 如果仍 NO_VALID_CONFIG，执行主张调整")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="G3-P1 bypass mechanism diagnostic")
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=DEFAULT_TUNING_DIR,
        help=f"Tuning output directory (default: {DEFAULT_TUNING_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report path (default: <tuning-dir>/bypass-diagnostic.md)",
    )
    args = parser.parse_args()

    tuning_dir = args.tuning_dir
    if not tuning_dir.is_absolute():
        tuning_dir = SCRIPT_DIR / tuning_dir

    output_path = args.output or (tuning_dir / "bypass-diagnostic.md")

    print(f"Loading summary from: {tuning_dir / 'summary.csv'}")
    summary_rows = load_summary(tuning_dir)
    if not summary_rows:
        print("ERROR: summary.csv not found or empty.")
        return 1

    print(f"Analyzing {len(summary_rows)} candidates ...")
    analyses = []
    for sr in summary_rows:
        a = analyze_candidate(sr, tuning_dir)
        if a is not None:
            analyses.append(a)

    if not analyses:
        print("ERROR: no candidate data could be extracted.")
        return 1

    print(f"Diagnosing {len(analyses)} candidates ...")
    diagnosis = diagnose(analyses)

    write_report(diagnosis, analyses, output_path)

    print()
    print(f"=== VERDICT: {diagnosis['verdict']} ===")
    print(diagnosis["reason"])
    print()
    print(f"Stats:")
    for k, v in diagnosis["stats"].items():
        print(f"  {k}: {v:.4f}")

    # 也输出 JSON
    json_path = tuning_dir / "bypass-diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON written to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
