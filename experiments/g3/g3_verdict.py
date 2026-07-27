"""
G3 Verdict Report Generator
============================
读取 raw_results.csv，计算 Go/No-Go 判定，输出报告。

判定条件（G3.8）：
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_cell_baseline(rows: List[Dict]) -> Dict:
    """
    按 (capacity, concurrency, baseline) 聚合，跨 seed 取均值。

    Returns:
        {(cap, conc): {baseline: {metric: mean_value}}}
    """
    # 先按 (cap, conc, baseline, seed) 聚合
    cell_baseline_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        cap = _to_float(r.get("capacity_gib"))
        conc = int(r.get("concurrency", 0))
        bl = r.get("baseline", "")
        seed = int(r.get("seed", 0))
        if cap is None:
            continue
        cell_baseline_seed[(cap, conc)][bl][seed] = {
            "p95_ttft_ms": _to_float(r.get("p95_ttft_ms")) or 0.0,
            "p50_ttft_ms": _to_float(r.get("p50_ttft_ms")) or 0.0,
            "miss_cost_ms": _to_float(r.get("miss_cost_ms")) or 0.0,
            "saved_prefill_ms": _to_float(r.get("saved_prefill_ms")) or 0.0,
            "block_hit_rate": _to_float(r.get("block_hit_rate")) or 0.0,
            "throughput_req_per_s": _to_float(r.get("throughput_req_per_s")) or 0.0,
            "migrate_ms_total": _to_float(r.get("migrate_ms_total")) or 0.0,
            "restore_ms_total": _to_float(r.get("restore_ms_total")) or 0.0,
            "hits": int(r.get("hits", 0)),
            "misses": int(r.get("misses", 0)),
        }

    # 再跨 seed 取均值
    result = defaultdict(lambda: defaultdict(dict))
    for cell, bl_dict in cell_baseline_seed.items():
        for bl, seed_dict in bl_dict.items():
            metrics = {}
            for key in seed_dict[list(seed_dict.keys())[0]].keys():
                values = [s[key] for s in seed_dict.values() if key in s]
                metrics[key] = sum(values) / len(values) if values else 0.0
            result[cell][bl] = metrics
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


def collect_per_seed_improvement(rows: List[Dict],
                                 cell: Tuple,
                                 baseline: str,
                                 metric: str = "p95_ttft_ms") -> List[float]:
    """收集某个 cell 下 FlowCache vs baseline 的 per-seed 改善值。"""
    cap, conc = cell
    per_seed = []
    # 按 seed 分组
    seed_data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        r_cap = _to_float(r.get("capacity_gib"))
        r_conc = int(r.get("concurrency", 0))
        if r_cap != cap or r_conc != conc:
            continue
        bl = r.get("baseline", "")
        seed = int(r.get("seed", 0))
        val = _to_float(r.get(metric))
        if val is not None:
            seed_data[seed][bl] = val

    for seed, bl_vals in seed_data.items():
        fc_val = bl_vals.get(FLOWCACHE)
        bl_val = bl_vals.get(baseline)
        if fc_val is not None and bl_val is not None and bl_val > 0:
            per_seed.append((bl_val - fc_val) / bl_val)
    return per_seed


# ---------------------------------------------------------------------------
# Go/No-Go evaluation
# ---------------------------------------------------------------------------

def evaluate_go_no_go(rows: List[Dict], config: Dict) -> Dict:
    """评估 G3 Go/No-Go 判定。"""
    aggregated = aggregate_by_cell_baseline(rows)
    verdict_cfg = config.get("verdict", {})
    main_cell = tuple(config.get("capacity", {}).get("main_cell", {}).get("capacity_gib", 1),
                      config.get("capacity", {}).get("main_cell", {}).get("concurrency", 4))
    # 转为 tuple key
    main_cell_key = None
    for cell in aggregated:
        if cell[0] == main_cell[0] and cell[1] == main_cell[1]:
            main_cell_key = cell
            break

    threshold_p95 = verdict_cfg.get("p95_ttft_threshold", 0.15)
    threshold_throughput = verdict_cfg.get("throughput_drop_threshold", 0.05)
    n_bootstrap = verdict_cfg.get("bootstrap_samples", 1000)

    result = {
        "main_cell": {"capacity_gib": main_cell[0], "concurrency": main_cell[1]},
        "thresholds": {
            "p95_ttft_improvement": threshold_p95,
            "throughput_drop": threshold_throughput,
        },
        "per_cell": {},
        "go_no_go": "PENDING",
        "conditions": {},
    }

    # 条件 1: 开销可行性
    fc_main = aggregated.get(main_cell_key, {}).get(FLOWCACHE, {})
    overhead_ms = fc_main.get("migrate_ms_total", 0) + fc_main.get("restore_ms_total", 0)
    saved_ms = fc_main.get("saved_prefill_ms", 0)
    overhead_feasible = overhead_ms < saved_ms if saved_ms > 0 else False
    result["conditions"]["overhead_feasible"] = {
        "overhead_ms": round(overhead_ms, 2),
        "saved_ms": round(saved_ms, 2),
        "pass": overhead_feasible,
    }

    # 条件 2-4: 主 cell 判定
    if main_cell_key is None or FLOWCACHE not in aggregated.get(main_cell_key, {}):
        result["go_no_go"] = "FAIL"
        result["conditions"]["reason"] = "FlowCache data missing for main cell"
        return result

    fc_metrics = aggregated[main_cell_key][FLOWCACHE]

    # 条件 2: p95 TTFT 改善 ≥ 15%
    p95_pass = True
    p95_details = {}
    for bl in SIMPLE_BASELINES:
        if bl not in aggregated.get(main_cell_key, {}):
            continue
        bl_metrics = aggregated[main_cell_key][bl]
        _, rel_imp = compute_improvement(fc_metrics, bl_metrics, "p95_ttft_ms")
        per_seed = collect_per_seed_improvement(rows, main_cell_key, bl, "p95_ttft_ms")
        mean_imp, ci_low, ci_high = bootstrap_ci(per_seed, n_bootstrap)
        passed = (rel_imp >= threshold_p95) and (ci_low > 0)
        if not passed:
            p95_pass = False
        p95_details[bl] = {
            "rel_improvement": round(rel_imp, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "pass": passed,
        }
    result["conditions"]["p95_ttft_improvement"] = {
        "threshold": threshold_p95,
        "details": p95_details,
        "pass": p95_pass,
    }

    # 条件 3: 吞吐非劣
    throughput_pass = True
    throughput_details = {}
    for bl in SIMPLE_BASELINES:
        if bl not in aggregated.get(main_cell_key, {}):
            continue
        bl_metrics = aggregated[main_cell_key][bl]
        throughput_change = compute_throughput_change(fc_metrics, bl_metrics)
        # 吞吐下降 ≤ 5% → change >= -0.05
        passed = throughput_change >= -threshold_throughput
        if not passed:
            throughput_pass = False
        throughput_details[bl] = {
            "rel_change": round(throughput_change, 6),
            "pass": passed,
        }
    result["conditions"]["throughput_noninferior"] = {
        "threshold_drop": threshold_throughput,
        "details": throughput_details,
        "pass": throughput_pass,
    }

    # 条件 4: 优于强启发式（bootstrap CI 不含 0）
    ci_pass = True
    ci_details = {}
    for bl in SIMPLE_BASELINES:
        per_seed = collect_per_seed_improvement(rows, main_cell_key, bl, "p95_ttft_ms")
        mean_imp, ci_low, ci_high = bootstrap_ci(per_seed, n_bootstrap)
        passed = ci_low > 0
        if not passed:
            ci_pass = False
        ci_details[bl] = {
            "mean": round(mean_imp, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "pass": passed,
        }
    result["conditions"]["better_than_heuristic"] = {
        "details": ci_details,
        "pass": ci_pass,
    }

    # 全 9 cell 摘要
    for cell, bl_dict in aggregated.items():
        cap, conc = cell
        fc = bl_dict.get(FLOWCACHE, {})
        best_simple = None
        best_simple_p95 = float("inf")
        for bl in SIMPLE_BASELINES:
            if bl in bl_dict:
                p95 = bl_dict[bl].get("p95_ttft_ms", float("inf"))
                if p95 < best_simple_p95:
                    best_simple_p95 = p95
                    best_simple = bl
        if best_simple and FLOWCACHE in bl_dict:
            _, rel = compute_improvement(fc, bl_dict[best_simple], "p95_ttft_ms")
            thr_change = compute_throughput_change(fc, bl_dict[best_simple])
        else:
            rel = 0.0
            thr_change = 0.0
        result["per_cell"][f"{cap}_{conc}"] = {
            "capacity_gib": cap,
            "concurrency": conc,
            "fc_p95_ttft": round(fc.get("p95_ttft_ms", 0), 2),
            "best_simple": best_simple,
            "best_simple_p95_ttft": round(best_simple_p95, 2),
            "p95_improvement": round(rel, 6),
            "throughput_change": round(thr_change, 6),
        }

    # 最终判定
    all_pass = (overhead_feasible and p95_pass and throughput_pass and ci_pass)
    result["go_no_go"] = "GO" if all_pass else "NO-GO"

    return result


# ---------------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def build_markdown_report(verdict: Dict) -> str:
    """生成 Markdown 判定报告。"""
    lines = []
    lines.append("# G3 Verdict Report: Lossless Residency\n")
    lines.append(f"**Verdict**: {'✅ GO' if verdict['go_no_go'] == 'GO' else '❌ NO-GO'}\n")

    mc = verdict["main_cell"]
    lines.append(f"**Main cell**: {mc['capacity_gib']} GiB, concurrency={mc['concurrency']}\n")

    lines.append("## Conditions\n")

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
    status = "✅ PASS" if cond.get("pass") else "❌ FAIL"
    lines.append(f"### 3. Throughput Non-inferior (drop ≤ {cond.get('threshold_drop', 0.05)*100:.0f}%): {status}")
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
    lines.append("| Capacity (GiB) | Concurrency | FlowCache p95 TTFT | Best Simple | Best Simple p95 | Improvement | Throughput Δ |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|")
    for key, cell_data in sorted(verdict["per_cell"].items(),
                                  key=lambda x: (x[1]["capacity_gib"], x[1]["concurrency"])):
        lines.append(
            f"| {cell_data['capacity_gib']} | {cell_data['concurrency']} | "
            f"{cell_data['fc_p95_ttft']:.1f} | {cell_data['best_simple'] or 'N/A'} | "
            f"{cell_data['best_simple_p95_ttft']:.1f} | "
            f"{cell_data['p95_improvement']*100:.2f}% | "
            f"{cell_data['throughput_change']*100:+.2f}% |"
        )
    lines.append("")

    # 失败动作
    if verdict["go_no_go"] == "NO-GO":
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
