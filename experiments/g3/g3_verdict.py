"""
G3 Verdict Report Generator
============================
读取 raw_results.csv，计算 Go/No-Go 判定，输出报告。

判定条件（G3.8）：
  0. 协议完整性：真实 TTFT/throughput 有效、成本非负、强基线齐全
  1. 开销可行性：恢复 + 迁移开销 < 所节省 prefill
  2. 主收益：p95 TTFT 改善 ≥ 15%（主 cell：1 GiB c=4）
  3. 吞吐非劣：吞吐下降 ≤ 5%
  4. 优于强启发式：bootstrap CI 不含 0（vs GDSF 和 vs SizeCost-LRU）

输出：
  - g3-verdict.md（人类可读报告）
  - g3-verdict.json（机器可读判定）
"""

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent

SIMPLE_BASELINES = ["gdsf", "sizecost"]
FLOWCACHE = "flowcache_lossless"
ALWAYS_MIGRATE = "flowcache_always_migrate"
SELECTIVE_MIGRATE_ONLY = "flowcache_selective_migrate_only"
ORACLE = "oracle_cost"
NO_CACHE = "no_cache"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> List[Dict]:
    """读取 raw_results.csv。"""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v) -> bool:
    """Parse bools written by csv.DictWriter without truthy-string bugs."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_cell_baseline(rows: List[Dict]) -> Dict:
    """
    按 (capacity, concurrency, baseline) 聚合全局指标。

    Global fields are duplicated on every task row; task fields must be
    summed.  The previous implementation accidentally used only the first
    task's hits/misses/saved/miss cost.

    Returns:
        {(cap, conc): {baseline: {metric: value}}}
    """
    grouped = defaultdict(list)
    for r in rows:
        cap = _to_float(r.get("capacity_gib"))
        try:
            conc = int(r.get("concurrency", 0))
        except (TypeError, ValueError):
            continue
        bl = r.get("baseline", "")
        if cap is None or not bl:
            continue
        grouped[(cap, conc, bl)].append(r)

    result = defaultdict(lambda: defaultdict(dict))
    for (cap, conc, bl), group in grouped.items():
        first = group[0]

        def first_float(*names, default=0.0):
            for name in names:
                value = _to_float(first.get(name))
                if value is not None:
                    return value
            return default

        def sum_float(name):
            return sum((_to_float(r.get(name)) or 0.0) for r in group)

        def sum_int(name):
            total = 0
            for r in group:
                try:
                    total += int(float(r.get(name) or 0))
                except (TypeError, ValueError):
                    pass
            return total

        result[(cap, conc)][bl] = {
            "p95_ttft_ms": first_float(
                "global_p95_cache_delay_ms", "global_p95_ttft_ms"
            ),
            "p50_ttft_ms": first_float(
                "global_p50_cache_delay_ms",
                "global_p50_ttft_ms",
                "global_p95_ttft_ms",
            ),
            "miss_cost_ms": sum_float("task_miss_cost_ms"),
            "saved_prefill_ms": sum_float("task_saved_prefill_ms"),
            "transfer_ms_task_sum": sum_float("task_transfer_ms"),
            "policy_ms_task_sum": sum_float("task_policy_model_ms"),
            "block_hit_rate": first_float("global_block_hit_rate"),
            "throughput_req_per_s": first_float("global_throughput"),
            "offered_load_req_per_s": first_float(
                "global_offered_load", "global_throughput"
            ),
            "ttft_metric_valid": _to_bool(first.get("ttft_metric_valid")),
            "throughput_metric_valid": _to_bool(
                first.get("throughput_metric_valid")
            ),
            "latency_metric_scope": first.get(
                "latency_metric_scope", "legacy_unknown"
            ),
            "controller_variant": first.get(
                "controller_variant", "legacy_unknown"
            ),
            "gpu_admission_policy": first.get(
                "gpu_admission_policy", "legacy_unknown"
            ),
            "online_feature_scope": first.get(
                "online_feature_scope", "legacy_unknown"
            ),
            "future_access_index_used": _to_bool(
                first.get("future_access_index_used")
            ),
            "share_count_feature_scope": first.get(
                "share_count_feature_scope", "legacy_unknown"
            ),
            "task_split": first.get("task_split", "legacy_unknown"),
            "migrate_ms_total": first_float("migrate_ms_total"),
            "restore_ms_total": first_float("restore_ms_total"),
            "migrate_bytes_total": first_float("migrate_bytes_total"),
            "restore_bytes_total": first_float("restore_bytes_total"),
            "negative_cost_count": first_float("negative_cost_count"),
            "hits": sum_int("task_hits"),
            "misses": sum_int("task_misses"),
        }
    return dict(result)


# ---------------------------------------------------------------------------
# Headroom / improvement computation
# ---------------------------------------------------------------------------

def compute_improvement(fc_metrics: Dict, baseline_metrics: Dict,
                        metric: str = "p95_ttft_ms") -> Tuple[float, float]:
    """
    计算 FlowCache 相对 baseline 的改善。

    Returns:
        (abs_improvement, rel_improvement)
        rel_improvement = (baseline - fc) / baseline
        正值 = FlowCache 更好
    """
    fc_val = fc_metrics.get(metric, 0.0)
    bl_val = baseline_metrics.get(metric, 0.0)
    abs_imp = bl_val - fc_val
    if bl_val > 0:
        rel_imp = abs_imp / bl_val
    else:
        rel_imp = 0.0
    return abs_imp, rel_imp


def compute_throughput_change(fc_metrics: Dict,
                              baseline_metrics: Dict) -> float:
    """
    计算 FlowCache 相对 baseline 的吞吐变化。

    Returns:
        rel_change = (fc - baseline) / baseline
        正值 = FlowCache 吞吐更高（更好）
    """
    fc_val = fc_metrics.get("throughput_req_per_s", 0.0)
    bl_val = baseline_metrics.get("throughput_req_per_s", 0.0)
    if bl_val > 0:
        return (fc_val - bl_val) / bl_val
    return 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float],
                 n_bootstrap: int = 1000,
                 seed: int = 42,
                 ci_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap CI on the mean of values.

    Returns:
        (mean, ci_low, ci_high)
    """
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1.0 - ci_level
    lo_idx = max(0, int((alpha / 2.0) * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1.0 - alpha / 2.0) * n_bootstrap))
    return mean, boot_means[lo_idx], boot_means[hi_idx]


def collect_per_task_improvement(rows: List[Dict],
                                 cell: Tuple,
                                 baseline: str) -> List[float]:
    """收集 per-task 的 FlowCache vs baseline 改善值。

    G3'' 修复：新 CSV 按 task_id 分组，用 task_p95_ttft_ms 做 paired 比较，
    165 个 task 支持 bootstrap CI。
    """
    cap, conc = cell
    # task_id -> {baseline_name -> task_p95_ttft_ms}
    task_data = defaultdict(dict)
    for r in rows:
        r_cap = _to_float(r.get("capacity_gib"))
        r_conc = int(r.get("concurrency", 0))
        if r_cap != cap or r_conc != conc:
            continue
        bl = r.get("baseline", "")
        task_id = r.get("task_id", "")
        val = _to_float(r.get("task_p95_ttft_ms"))
        if val is not None:
            task_data[task_id][bl] = val

    improvements = []
    for task_id, bl_vals in task_data.items():
        fc_val = bl_vals.get(FLOWCACHE)
        bl_val = bl_vals.get(baseline)
        if fc_val is not None and bl_val is not None and bl_val > 0:
            improvements.append((bl_val - fc_val) / bl_val)
    return improvements


# ---------------------------------------------------------------------------
# Go/No-Go evaluation
# ---------------------------------------------------------------------------

def evaluate_go_no_go(rows: List[Dict], config: Dict) -> Dict:
    """Evaluate G3 without turning an invalid protocol into a route decision."""
    aggregated = aggregate_by_cell_baseline(rows)
    verdict_cfg = config.get("verdict", {})
    main_cfg = config.get("capacity", {}).get("main_cell", {})
    main_cell = (
        main_cfg.get("capacity_gib", 1),
        main_cfg.get("concurrency", 4),
    )
    main_cell_key = next(
        (
            cell for cell in aggregated
            if cell[0] == main_cell[0] and cell[1] == main_cell[1]
        ),
        None,
    )

    threshold_p95 = verdict_cfg.get("p95_ttft_threshold", 0.15)
    threshold_throughput = verdict_cfg.get("throughput_drop_threshold", 0.05)
    n_bootstrap = verdict_cfg.get("bootstrap_samples", 1000)
    result = {
        "main_cell": {
            "capacity_gib": main_cell[0],
            "concurrency": main_cell[1],
        },
        "thresholds": {
            "p95_ttft_improvement": threshold_p95,
            "throughput_drop": threshold_throughput,
        },
        "per_cell": {},
        "go_no_go": "PROTOCOL-INCOMPLETE",
        "conditions": {},
    }

    main_metrics = aggregated.get(main_cell_key, {})
    fc_metrics = main_metrics.get(FLOWCACHE, {})
    protocol_reasons = []
    if main_cell_key is None:
        protocol_reasons.append("main-cell data is missing")
    if not fc_metrics:
        protocol_reasons.append("FlowCache data is missing for the main cell")
    missing_baselines = [b for b in SIMPLE_BASELINES if b not in main_metrics]
    if missing_baselines:
        protocol_reasons.append(
            "required strong baselines are missing: "
            + ", ".join(missing_baselines)
        )
    if ALWAYS_MIGRATE not in main_metrics:
        protocol_reasons.append(
            "independent always-migrate ablation is missing"
        )
    if SELECTIVE_MIGRATE_ONLY not in main_metrics:
        protocol_reasons.append(
            "independent selective-migrate-only ablation is missing"
        )

    if fc_metrics:
        if not fc_metrics.get("ttft_metric_valid", False):
            protocol_reasons.append(
                "open-loop output is modeled cache delay, not measured TTFT"
            )
        if not fc_metrics.get("throughput_metric_valid", False):
            protocol_reasons.append(
                "arrival-window rate is offered load, not achieved throughput"
            )
        if fc_metrics.get("negative_cost_count", 0) > 0:
            protocol_reasons.append("negative modeled transfer/policy cost found")
        if fc_metrics.get("controller_variant") != "selective_value":
            protocol_reasons.append(
                "FlowCache result is not produced by selective_value"
            )
        if (
            fc_metrics.get("gpu_admission_policy")
            != "oracle_cost_proxy"
        ):
            protocol_reasons.append(
                "FlowCache result lacks causal cost-aware GPU admission"
            )
        if (
            fc_metrics.get("online_feature_scope")
            != "current_and_past_only"
            or fc_metrics.get("future_access_index_used", False)
        ):
            protocol_reasons.append(
                "online FlowCache is not fail-closed against future access"
            )
        if (
            fc_metrics.get("share_count_feature_scope")
            != "causal_past_window_including_current"
        ):
            protocol_reasons.append(
                "share_count is not marked as a causal decision-time feature"
            )
        if fc_metrics.get("task_split") != "test":
            protocol_reasons.append(
                "formal verdict requires the frozen held-out task split"
            )
        task_transfer = fc_metrics.get("transfer_ms_task_sum", 0.0)
        global_transfer = (
            fc_metrics.get("migrate_ms_total", 0.0)
            + fc_metrics.get("restore_ms_total", 0.0)
        )
        tolerance = max(1e-6, abs(global_transfer) * 1e-6)
        if abs(task_transfer - global_transfer) > tolerance:
            protocol_reasons.append(
                "per-task transfer totals do not match global transfer totals"
            )

    protocol_valid = not protocol_reasons
    result["conditions"]["protocol_valid"] = {
        "pass": protocol_valid,
        "reasons": protocol_reasons,
        "status_if_failed": "PROTOCOL-INCOMPLETE",
    }

    overhead_ms = (
        fc_metrics.get("migrate_ms_total", 0.0)
        + fc_metrics.get("restore_ms_total", 0.0)
    )
    saved_ms = fc_metrics.get("saved_prefill_ms", 0.0)
    overhead_feasible = (
        overhead_ms >= 0 and saved_ms > 0 and overhead_ms < saved_ms
    )
    result["conditions"]["overhead_feasible"] = {
        "overhead_ms": round(overhead_ms, 2),
        "saved_ms": round(saved_ms, 2),
        "pass": overhead_feasible,
        "diagnostic_only": not protocol_valid,
    }

    p95_details = {}
    p95_pass = not missing_baselines and bool(fc_metrics)
    for bl in SIMPLE_BASELINES:
        if bl not in main_metrics or not fc_metrics:
            continue
        bl_metrics = main_metrics[bl]
        _, rel_imp = compute_improvement(
            fc_metrics, bl_metrics, "p95_ttft_ms"
        )
        per_task = collect_per_task_improvement(rows, main_cell_key, bl)
        mean_imp, ci_low, ci_high = bootstrap_ci(per_task, n_bootstrap)
        passed = (
            bool(per_task)
            and rel_imp >= threshold_p95
            and ci_low > 0
        )
        p95_pass = p95_pass and passed
        p95_details[bl] = {
            "rel_improvement": round(rel_imp, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "n_tasks": len(per_task),
            "pass": passed,
        }
    result["conditions"]["p95_ttft_improvement"] = {
        "threshold": threshold_p95,
        "metric_scope": fc_metrics.get(
            "latency_metric_scope", "unknown"
        ),
        "details": p95_details,
        "pass": p95_pass,
        "diagnostic_only": not protocol_valid,
    }

    throughput_details = {}
    throughput_valid = (
        bool(fc_metrics)
        and fc_metrics.get("throughput_metric_valid", False)
        and not missing_baselines
        and all(
            main_metrics[bl].get("throughput_metric_valid", False)
            for bl in SIMPLE_BASELINES if bl in main_metrics
        )
    )
    throughput_pass = throughput_valid
    if throughput_valid:
        for bl in SIMPLE_BASELINES:
            throughput_change = compute_throughput_change(
                fc_metrics, main_metrics[bl]
            )
            passed = throughput_change >= -threshold_throughput
            throughput_pass = throughput_pass and passed
            throughput_details[bl] = {
                "rel_change": round(throughput_change, 6),
                "pass": passed,
            }
    result["conditions"]["throughput_noninferior"] = {
        "threshold_drop": threshold_throughput,
        "valid": throughput_valid,
        "details": throughput_details,
        "pass": throughput_pass,
        "reason": (
            "" if throughput_valid
            else "requires closed-loop completion timestamps"
        ),
    }

    ci_details = {}
    ci_pass = not missing_baselines and bool(fc_metrics)
    for bl in SIMPLE_BASELINES:
        if bl not in main_metrics:
            continue
        per_task = collect_per_task_improvement(rows, main_cell_key, bl)
        mean_imp, ci_low, ci_high = bootstrap_ci(per_task, n_bootstrap)
        passed = bool(per_task) and ci_low > 0
        ci_pass = ci_pass and passed
        ci_details[bl] = {
            "mean": round(mean_imp, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "n_tasks": len(per_task),
            "pass": passed,
        }
    result["conditions"]["better_than_heuristic"] = {
        "details": ci_details,
        "pass": ci_pass,
        "diagnostic_only": not protocol_valid,
    }

    for cell, bl_dict in aggregated.items():
        cap, conc = cell
        fc = bl_dict.get(FLOWCACHE, {})
        candidates = [
            (bl, bl_dict[bl].get("p95_ttft_ms", float("inf")))
            for bl in SIMPLE_BASELINES if bl in bl_dict
        ]
        best_simple, best_simple_p95 = (
            min(candidates, key=lambda item: item[1])
            if candidates else (None, 0.0)
        )
        if best_simple and fc:
            _, rel = compute_improvement(
                fc, bl_dict[best_simple], "p95_ttft_ms"
            )
            throughput_change = (
                compute_throughput_change(fc, bl_dict[best_simple])
                if (
                    fc.get("throughput_metric_valid", False)
                    and bl_dict[best_simple].get(
                        "throughput_metric_valid", False
                    )
                )
                else None
            )
        else:
            rel = 0.0
            throughput_change = None
        result["per_cell"][f"{cap}_{conc}"] = {
            "capacity_gib": cap,
            "concurrency": conc,
            "fc_p95_ttft": round(fc.get("p95_ttft_ms", 0), 2),
            "latency_metric_scope": fc.get(
                "latency_metric_scope", "unknown"
            ),
            "best_simple": best_simple,
            "best_simple_p95_ttft": round(best_simple_p95, 2),
            "p95_improvement": round(rel, 6),
            "throughput_change": (
                round(throughput_change, 6)
                if throughput_change is not None else None
            ),
        }

    all_pass = (
        overhead_feasible and p95_pass and throughput_pass and ci_pass
    )
    if protocol_valid:
        result["go_no_go"] = "GO" if all_pass else "NO-GO"
    return result


# ---------------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def build_markdown_report(verdict: Dict) -> str:
    """生成 Markdown 判定报告。"""
    lines = []
    lines.append("# G3 Verdict Report: Lossless Residency\n")
    verdict_label = {
        "GO": "✅ GO",
        "NO-GO": "❌ NO-GO",
        "PROTOCOL-INCOMPLETE": "⚠️ PROTOCOL-INCOMPLETE",
    }.get(verdict["go_no_go"], verdict["go_no_go"])
    lines.append(f"**Verdict**: {verdict_label}\n")

    mc = verdict["main_cell"]
    lines.append(f"**Main cell**: {mc['capacity_gib']} GiB, concurrency={mc['concurrency']}\n")

    lines.append("## Conditions\n")

    # 条件 0
    cond = verdict["conditions"].get("protocol_valid", {})
    status = "✅ PASS" if cond.get("pass") else "⚠️ INCOMPLETE"
    lines.append(f"### 0. Protocol Validity: {status}")
    for reason in cond.get("reasons", []):
        lines.append(f"- {reason}")
    lines.append("")

    # 条件 1
    cond = verdict["conditions"].get("overhead_feasible", {})
    status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
    lines.append(f"### 1. Overhead Feasibility: {status}")
    lines.append(f"- Overhead (migrate + restore): {cond.get('overhead_ms', 0):.2f} ms")
    lines.append(f"- Saved prefill: {cond.get('saved_ms', 0):.2f} ms\n")

    # 条件 2
    cond = verdict["conditions"].get("p95_ttft_improvement", {})
    status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
    lines.append(f"### 2. p95 TTFT Improvement ≥ {cond.get('threshold', 0.15)*100:.0f}%: {status}")
    for bl, d in cond.get("details", {}).items():
        bl_status = "✅" if d["pass"] else "❌"
        lines.append(f"- {bl_status} vs {bl}: {d['rel_improvement']*100:.2f}% "
                     f"(CI=[{d['ci_low']*100:.2f}%, {d['ci_high']*100:.2f}%])")
    lines.append("")

    # 条件 3
    cond = verdict["conditions"].get("throughput_noninferior", {})
    if not cond.get("valid", False):
        status = "⚠️ NOT MEASURED"
    else:
        status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
    lines.append(f"### 3. Throughput Non-inferior (drop ≤ {cond.get('threshold_drop', 0.05)*100:.0f}%): {status}")
    if cond.get("reason"):
        lines.append(f"- {cond['reason']}")
    for bl, d in cond.get("details", {}).items():
        bl_status = "✅" if d["pass"] else "❌"
        change = d["rel_change"] * 100
        lines.append(f"- {bl_status} vs {bl}: {change:+.2f}%")
    lines.append("")

    # 条件 4
    cond = verdict["conditions"].get("better_than_heuristic", {})
    status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
    lines.append(f"### 4. Better Than Heuristic (CI > 0): {status}")
    for bl, d in cond.get("details", {}).items():
        bl_status = "✅" if d["pass"] else "❌"
        lines.append(f"- {bl_status} vs {bl}: mean={d['mean']*100:.2f}% "
                     f"(CI=[{d['ci_low']*100:.2f}%, {d['ci_high']*100:.2f}%])")
    lines.append("")

    # 全 9 cell 摘要
    lines.append("## All Cells Summary\n")
    lines.append("| Capacity (GiB) | Concurrency | FlowCache modeled p95 cache delay | Best Simple | Best Simple p95 | Improvement | Throughput Δ |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|")
    for key, cell_data in sorted(verdict["per_cell"].items(),
                                  key=lambda x: (x[1]["capacity_gib"], x[1]["concurrency"])):
        throughput_text = (
            f"{cell_data['throughput_change']*100:+.2f}%"
            if cell_data.get("throughput_change") is not None else "N/A"
        )
        lines.append(
            f"| {cell_data['capacity_gib']} | {cell_data['concurrency']} | "
            f"{cell_data['fc_p95_ttft']:.1f} | {cell_data['best_simple'] or 'N/A'} | "
            f"{cell_data['best_simple_p95_ttft']:.1f} | "
            f"{cell_data['p95_improvement']*100:.2f}% | "
            f"{throughput_text} |"
        )
    lines.append("")

    # 失败动作
    if verdict["go_no_go"] == "PROTOCOL-INCOMPLETE":
        lines.append("## Required Next Step\n")
        lines.append(
            "先完成 G3-P1 validation 参数冻结、held-out 单 cell 验证、"
            "公平 two-tier baselines 与 closed-loop serving 测量；"
            "当前结果不得触发路线 A → B。"
        )
    elif verdict["go_no_go"] == "NO-GO":
        lines.append("## Failure Action\n")
        lines.append("按 IDEA §7 G3：路线 A No-Go，转路线 B。")
        lines.append("实现保留为工程基线，但不以无损 residency 单独投稿该主张。")
    else:
        lines.append("## Next Step\n")
        lines.append("G3 PASSED → 进入 G4（量化）→ G2（联合 R-D 控制器）。")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G3 verdict report generator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results", default=None,
                        help="Path to raw_results.csv")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_path = args.results or config.get("output", {}).get("raw_results_csv")
    if results_path is None:
        results_path = "results/raw_results.csv"
    results_path = Path(results_path)
    if not results_path.is_absolute():
        results_path = Path(__file__).parent / results_path

    rows = load_results(results_path)
    if not rows:
        print(f"No results found in {results_path}")
        return

    print(f"Loaded {len(rows)} rows from {results_path}")
    verdict = evaluate_go_no_go(rows, config)

    # 保存 JSON
    json_path = args.output_json or config.get("output", {}).get("verdict_json")
    if json_path is None:
        json_path = "g3-verdict.json"
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = Path(__file__).parent / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"Verdict JSON saved to: {json_path}")

    # 保存 Markdown
    md_path = args.output_md or config.get("output", {}).get("verdict_md")
    if md_path is None:
        md_path = "g3-verdict.md"
    md_path = Path(md_path)
    if not md_path.is_absolute():
        md_path = Path(__file__).parent / md_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_report = build_markdown_report(verdict)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Verdict report saved to: {md_path}")
    print(f"\n{'='*60}")
    print(f"G3 Verdict: {verdict['go_no_go']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
