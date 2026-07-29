"""
G3 Verdict Report Generator
============================
读取 closed-loop verdict JSON（由 run_closed_loop.py 产出）作为权威判定来源，
可选附加 open-loop raw_results.csv 作为机制诊断附录。

判定条件（G3.8，closed-loop）：
  0. 协议完整性：closed-loop verdict JSON 存在且策略齐全
  1. 主收益：p95 TTFT 改善 ≥ 15%（flowcache_lossless vs twotier_lru，主 cell）
  2. 吞吐非劣：吞吐下降 ≤ 5%（flowcache_lossless vs twotier_lru）
  3. 统计显著：bootstrap CI 排除 0（per-task 聚类）

设计变更（G3′′′）：
  - 不再从 open-loop raw_results.csv 推导 GO/NO-GO（open-loop 标记
    ttft_metric_valid=False、throughput_metric_valid=False，仅机制诊断）
  - 移除对已裁剪 baselines 的硬依赖（gdsf、sizecost、
    flowcache_always_migrate、flowcache_selective_migrate_only）
  - 主比较改为 flowcache_lossless vs twotier_lru（与 closed-loop verdict 对齐）
  - open-loop 数据若存在则作为诊断附录展示，但不参与 GO/NO-GO 判定

输出：
  - g3-verdict.md（人类可读报告）
  - g3-verdict.json（机器可读判定）
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent

# Closed-loop 主比较（与 run_closed_loop.py / config.yaml 对齐）
TREATMENT = "flowcache_lossless"
CONTROL = "twotier_lru"
NARRATIVE_BASELINE = "apc_lru"  # 论文叙事参照，不参与 verdict 门槛

# 默认 closed-loop verdict JSON 路径（相对于 experiments/g3/）
DEFAULT_CLOSED_LOOP_VERDICT = "results/closed-loop/closed-loop-verdict.json"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_closed_loop_verdict(path: Path) -> Optional[Dict]:
    """读取 closed-loop verdict JSON。返回 None 表示文件不存在。"""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_open_loop_results(csv_path: Path) -> List[Dict]:
    """读取 open-loop raw_results.csv 作为诊断附录。文件不存在时返回空列表。"""
    rows: List[Dict] = []
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


# ---------------------------------------------------------------------------
# Open-loop 诊断聚合（仅用于附录，不参与 GO/NO-GO）
# ---------------------------------------------------------------------------

def aggregate_open_loop_diagnostic(rows: List[Dict]) -> Dict:
    """按 (capacity, concurrency, baseline) 聚合 open-loop 全局指标。

    仅用于诊断附录，展示机制层（modeled cache delay）的差异。
    """
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
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

    result: Dict[Tuple, Dict[str, Any]] = {}
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

        result[(cap, conc, bl)] = {
            "p95_cache_delay_ms": first_float(
                "global_p95_cache_delay_ms", "global_p95_ttft_ms"
            ),
            "block_hit_rate": first_float("global_block_hit_rate"),
            "migrate_ms_total": first_float("migrate_ms_total"),
            "restore_ms_total": first_float("restore_ms_total"),
            "saved_prefill_ms": sum_float("task_saved_prefill_ms"),
            "ttft_metric_valid": bool(first.get("ttft_metric_valid", False)),
            "throughput_metric_valid": bool(
                first.get("throughput_metric_valid", False)
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Go/No-Go evaluation（基于 closed-loop verdict JSON）
# ---------------------------------------------------------------------------

def evaluate_go_no_go(
    closed_loop_verdict: Optional[Dict],
    open_loop_rows: List[Dict],
    config: Dict,
) -> Dict:
    """基于 closed-loop verdict JSON 计算 GO/NO-GO。

    Args:
        closed_loop_verdict: run_closed_loop.py 产出的 verdict JSON（可为 None）
        open_loop_rows: open-loop raw_results.csv 行（仅诊断附录）
        config: G3 config.yaml 解析结果

    Returns:
        判定字典，go_no_go ∈ {GO, NO-GO, PROTOCOL-INCOMPLETE}
    """
    verdict_cfg = config.get("verdict", {})
    closed_loop_cfg = config.get("closed_loop", {})
    main_cell = closed_loop_cfg.get("cell", {})

    threshold_p95 = verdict_cfg.get(
        "p95_ttft_threshold",
        closed_loop_cfg.get("verdict", {}).get(
            "p95_ttft_improvement_pct", 15.0
        ) / 100.0,
    )
    # closed-loop config 用百分比（15.0），open-loop verdict cfg 用比例（0.15）
    if threshold_p95 > 1.0:
        threshold_p95 = threshold_p95 / 100.0
    threshold_throughput = verdict_cfg.get(
        "throughput_drop_threshold",
        closed_loop_cfg.get("verdict", {}).get(
            "throughput_drop_threshold_pct", 5.0
        ) / 100.0,
    )
    if threshold_throughput > 1.0:
        threshold_throughput = threshold_throughput / 100.0

    result: Dict[str, Any] = {
        "main_cell": {
            "capacity_gib": main_cell.get("capacity_gib", 2.0),
            "concurrency": main_cell.get("concurrency", 4),
        },
        "thresholds": {
            "p95_ttft_improvement": threshold_p95,
            "throughput_drop": threshold_throughput,
        },
        "verdict_source": "closed_loop",
        "conditions": {},
        "open_loop_diagnostic": {},
        "go_no_go": "PROTOCOL-INCOMPLETE",
    }

    # --- 协议完整性：closed-loop verdict JSON 必须存在且完整 ---
    protocol_reasons: List[str] = []
    if closed_loop_verdict is None:
        protocol_reasons.append(
            f"closed-loop verdict JSON not found at "
            f"{DEFAULT_CLOSED_LOOP_VERDICT}；请先运行 "
            f"closed_loop/run_closed_loop.py"
        )
    else:
        summaries = closed_loop_verdict.get("strategy_summaries", {})
        if TREATMENT not in summaries:
            protocol_reasons.append(
                f"closed-loop 缺少 {TREATMENT} 结果"
            )
        if CONTROL not in summaries:
            protocol_reasons.append(
                f"closed-loop 缺少 {CONTROL} 结果（控制组）"
            )
        if closed_loop_verdict.get("ttft_metric_valid") is not True:
            protocol_reasons.append(
                "closed-loop ttft_metric_valid 不为 True"
            )
        if closed_loop_verdict.get("throughput_metric_valid") is not True:
            protocol_reasons.append(
                "closed-loop throughput_metric_valid 不为 True"
            )

    protocol_valid = not protocol_reasons
    result["conditions"]["protocol_valid"] = {
        "pass": protocol_valid,
        "reasons": protocol_reasons,
        "status_if_failed": "PROTOCOL-INCOMPLETE",
    }

    if not protocol_valid:
        # 无法判定，仍输出 open-loop 诊断附录
        if open_loop_rows:
            result["open_loop_diagnostic"] = _build_open_loop_appendix(
                open_loop_rows
            )
        return result

    # --- 从 closed-loop verdict 提取主判定 ---
    checks = closed_loop_verdict.get("checks", {})

    # 条件 1: p95 TTFT 改善 ≥ 15%
    c1 = checks.get("p95_ttft_improvement", {})
    p95_pass = bool(c1.get("pass", False))
    result["conditions"]["p95_ttft_improvement"] = {
        "threshold": threshold_p95,
        "control_p95_ms": c1.get("control_p95_ms", 0.0),
        "treatment_p95_ms": c1.get("treatment_p95_ms", 0.0),
        "improvement_pct": c1.get("improvement_pct", 0.0),
        "pass": p95_pass,
    }

    # 条件 2: 吞吐非劣（drop ≤ 5%）
    c2 = checks.get("throughput_drop", {})
    throughput_pass = bool(c2.get("pass", False))
    result["conditions"]["throughput_noninferior"] = {
        "threshold_drop": threshold_throughput,
        "control_throughput": c2.get("control_throughput", 0.0),
        "treatment_throughput": c2.get("treatment_throughput", 0.0),
        "drop_pct": c2.get("drop_pct", 0.0),
        "pass": throughput_pass,
    }

    # 条件 3: Bootstrap CI 排除 0（per-task）
    c3 = checks.get("bootstrap_ci_ttft", {})
    ci_pass = bool(c3.get("pass", False))
    result["conditions"]["bootstrap_ci_significant"] = {
        "mean_diff_ms": c3.get("mean_diff_ms", 0.0),
        "ci_low_ms": c3.get("ci_low_ms", 0.0),
        "ci_high_ms": c3.get("ci_high_ms", 0.0),
        "n_tasks": c3.get("n_tasks", 0),
        "ci_excludes_zero": bool(c3.get("ci_excludes_zero", False)),
        "pass": ci_pass,
    }

    # --- 总判定 ---
    all_pass = p95_pass and throughput_pass and ci_pass
    result["go_no_go"] = "GO" if all_pass else "NO-GO"

    # --- Open-loop 诊断附录（不影响判定）---
    if open_loop_rows:
        result["open_loop_diagnostic"] = _build_open_loop_appendix(open_loop_rows)

    # --- 附加策略摘要 ---
    result["strategy_summaries"] = closed_loop_verdict.get(
        "strategy_summaries", {}
    )

    return result


def _build_open_loop_appendix(rows: List[Dict]) -> Dict:
    """构建 open-loop 诊断附录（按 cell × baseline 聚合）。"""
    aggregated = aggregate_open_loop_diagnostic(rows)
    per_cell: Dict[str, Any] = {}
    for (cap, conc, bl), metrics in aggregated.items():
        key = f"{cap}_{conc}"
        per_cell.setdefault(key, {
            "capacity_gib": cap,
            "concurrency": conc,
            "baselines": {},
        })
        per_cell[key]["baselines"][bl] = {
            "p95_cache_delay_ms": round(
                metrics["p95_cache_delay_ms"], 2
            ),
            "block_hit_rate": round(metrics["block_hit_rate"], 4),
            "saved_prefill_ms": round(metrics["saved_prefill_ms"], 2),
            "migrate_ms_total": round(metrics["migrate_ms_total"], 2),
            "restore_ms_total": round(metrics["restore_ms_total"], 2),
        }
    return {
        "scope": "modeled_cache_delay_only",
        "ttft_metric_valid": False,
        "throughput_metric_valid": False,
        "verdict_role": "diagnostic_appendix_only",
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------

def build_markdown_report(verdict: Dict) -> str:
    """生成 Markdown 判定报告。"""
    lines: List[str] = []
    lines.append("# G3 Verdict Report: Lossless Residency\n")
    verdict_label = {
        "GO": "✅ GO",
        "NO-GO": "❌ NO-GO",
        "PROTOCOL-INCOMPLETE": "⚠️ PROTOCOL-INCOMPLETE",
    }.get(verdict["go_no_go"], verdict["go_no_go"])
    lines.append(f"**Verdict**: {verdict_label}\n")
    lines.append(
        f"**Verdict Source**: {verdict.get('verdict_source', 'unknown')} "
        f"(closed-loop serving)\n"
    )

    mc = verdict["main_cell"]
    lines.append(
        f"**Main cell**: {mc['capacity_gib']} GiB, "
        f"concurrency={mc['concurrency']}\n"
    )

    lines.append("## Conditions\n")

    # 条件 0: 协议完整性
    cond = verdict["conditions"].get("protocol_valid", {})
    status = "✅ PASS" if cond.get("pass") else "⚠️ INCOMPLETE"
    lines.append(f"### 0. Protocol Validity: {status}")
    for reason in cond.get("reasons", []):
        lines.append(f"- {reason}")
    lines.append("")

    # 条件 1: p95 TTFT 改善
    cond = verdict["conditions"].get("p95_ttft_improvement", {})
    if cond:
        status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
        threshold_pct = cond.get("threshold", 0.15) * 100
        lines.append(
            f"### 1. p95 TTFT Improvement ≥ {threshold_pct:.0f}%: {status}"
        )
        lines.append(
            f"- Control ({CONTROL}) p95: "
            f"{cond.get('control_p95_ms', 0):.1f} ms"
        )
        lines.append(
            f"- Treatment ({TREATMENT}) p95: "
            f"{cond.get('treatment_p95_ms', 0):.1f} ms"
        )
        lines.append(
            f"- Improvement: {cond.get('improvement_pct', 0):.2f}%"
        )
        lines.append("")

    # 条件 2: 吞吐非劣
    cond = verdict["conditions"].get("throughput_noninferior", {})
    if cond:
        status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
        threshold_drop_pct = cond.get("threshold_drop", 0.05) * 100
        lines.append(
            f"### 2. Throughput Non-inferior (drop ≤ {threshold_drop_pct:.0f}%): {status}"
        )
        lines.append(
            f"- Control throughput: "
            f"{cond.get('control_throughput', 0):.4f} req/s"
        )
        lines.append(
            f"- Treatment throughput: "
            f"{cond.get('treatment_throughput', 0):.4f} req/s"
        )
        lines.append(
            f"- Drop: {cond.get('drop_pct', 0):.2f}%"
        )
        lines.append("")

    # 条件 3: Bootstrap CI
    cond = verdict["conditions"].get("bootstrap_ci_significant", {})
    if cond:
        status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
        lines.append(
            f"### 3. Bootstrap CI Excludes 0 (per-task, {TREATMENT} vs {CONTROL}): {status}"
        )
        lines.append(
            f"- Mean diff: {cond.get('mean_diff_ms', 0):.2f} ms "
            f"(positive = treatment faster)"
        )
        lines.append(
            f"- 95% CI: [{cond.get('ci_low_ms', 0):.2f}, "
            f"{cond.get('ci_high_ms', 0):.2f}] ms"
        )
        lines.append(f"- n_tasks: {cond.get('n_tasks', 0)}")
        lines.append("")

    # 策略摘要
    summaries = verdict.get("strategy_summaries", {})
    if summaries:
        lines.append("## Strategy Summaries (Closed-loop)\n")
        lines.append(
            "| Strategy | Requests | TTFT p50 (ms) | TTFT p95 (ms) | "
            "Throughput (req/s) | Goodput |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for sname, m in summaries.items():
            lines.append(
                f"| {sname} | {m.get('successful_requests', 0)}/"
                f"{m.get('total_requests', 0)} | "
                f"{m.get('ttft_p50', 0):.1f} | "
                f"{m.get('ttft_p95', 0):.1f} | "
                f"{m.get('throughput_req_per_s', 0):.4f} | "
                f"{m.get('goodput_rate', 0)*100:.1f}% |"
            )
        lines.append("")

    # Open-loop 诊断附录
    appendix = verdict.get("open_loop_diagnostic", {})
    if appendix and appendix.get("per_cell"):
        lines.append("## Appendix: Open-loop Diagnostic (modeled cache delay)\n")
        lines.append(
            f"**Scope**: {appendix.get('scope', 'unknown')} — "
            f"ttft_metric_valid={appendix.get('ttft_metric_valid')}, "
            f"throughput_metric_valid={appendix.get('throughput_metric_valid')}\n"
        )
        lines.append(
            "Open-loop 数据仅作为机制层诊断附录，**不参与 GO/NO-GO 判定**。\n"
        )
        lines.append(
            "| Capacity (GiB) | Concurrency | Baseline | p95 cache delay (ms) | "
            "Hit rate | Saved prefill (ms) |"
        )
        lines.append("|---:|---:|---|---:|---:|---:|")
        for key, cell_data in sorted(
            appendix["per_cell"].items(),
            key=lambda x: (x[1]["capacity_gib"], x[1]["concurrency"]),
        ):
            for bl, m in sorted(cell_data["baselines"].items()):
                lines.append(
                    f"| {cell_data['capacity_gib']} | "
                    f"{cell_data['concurrency']} | {bl} | "
                    f"{m['p95_cache_delay_ms']:.1f} | "
                    f"{m['block_hit_rate']:.3f} | "
                    f"{m['saved_prefill_ms']:.1f} |"
                )
        lines.append("")

    # 失败动作
    if verdict["go_no_go"] == "PROTOCOL-INCOMPLETE":
        lines.append("## Required Next Step\n")
        lines.append(
            "请先运行 `closed_loop/run_closed_loop.py` 完成 closed-loop serving "
            "实验，生成 `results/closed-loop/closed-loop-verdict.json`。\n"
            "Open-loop 机制诊断已附录，但不能用于 GO/NO-GO 判定。"
        )
    elif verdict["go_no_go"] == "NO-GO":
        lines.append("## Failure Action\n")
        lines.append("按 IDEA §7 G3：路线 A No-Go，转路线 B。")
        lines.append("实现保留为工程基线，但不以无损 residency 单独投稿该主张。")
    else:
        lines.append("## Next Step\n")
        lines.append("G3 PASSED → 进入 G4（量化）→ G2（联合 R-D 控制器）。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="G3 verdict report generator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--closed-loop-verdict",
        default=None,
        help="Path to closed-loop-verdict.json (default: "
             "results/closed-loop/closed-loop-verdict.json)",
    )
    parser.add_argument(
        "--open-loop-results",
        default=None,
        help="Path to open-loop raw_results.csv (diagnostic appendix only)",
    )
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Closed-loop verdict JSON（权威来源）
    cl_path = args.closed_loop_verdict
    if cl_path is None:
        cl_path = DEFAULT_CLOSED_LOOP_VERDICT
    cl_path = Path(cl_path)
    if not cl_path.is_absolute():
        cl_path = Path(__file__).parent / cl_path
    closed_loop_verdict = load_closed_loop_verdict(cl_path)
    if closed_loop_verdict is None:
        print(f"Warning: closed-loop verdict not found at {cl_path}")
        print("Verdict will be PROTOCOL-INCOMPLETE.")
    else:
        print(f"Loaded closed-loop verdict from {cl_path}")

    # Open-loop raw_results.csv（诊断附录，可选）
    ol_path = args.open_loop_results
    if ol_path is None:
        ol_path = config.get("output", {}).get(
            "raw_results_csv", "results/raw_results.csv"
        )
    ol_path = Path(ol_path)
    if not ol_path.is_absolute():
        ol_path = Path(__file__).parent / ol_path
    open_loop_rows = load_open_loop_results(ol_path)
    if open_loop_rows:
        print(f"Loaded {len(open_loop_rows)} open-loop rows from {ol_path} "
              f"(diagnostic appendix)")
    else:
        print(f"No open-loop results at {ol_path} (appendix will be empty)")

    # 计算判定
    verdict = evaluate_go_no_go(closed_loop_verdict, open_loop_rows, config)

    # 保存 JSON
    json_path = args.output_json or config.get("output", {}).get("verdict_json")
    if json_path is None:
        json_path = "g3-verdict.json"
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = Path(__file__).parent / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)
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
