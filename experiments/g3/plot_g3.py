"""
G3 Results Plotter
===================
绘制 G3 实验结果图：
  - 双面板图：p95 TTFT + 吞吐 vs 容量
  - 按并发度分面（c=1/4/8）
  - 按 baseline 分线

输出：figures/g3-headroom.png
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent

BASELINE_ORDER = [
    "no_cache",
    "apc_lru",
    "gdsf",
    "sizecost",
    "flowcache_always_migrate",
    "flowcache_selective_migrate_only",
    "flowcache_lossless",
    "oracle_cost",
]
BASELINE_COLORS = {
    "no_cache": "#999999",
    "apc_lru": "#66b3ff",
    "gdsf": "#3399ff",
    "sizecost": "#0066cc",
    "flowcache_always_migrate": "#ffb3b3",
    "flowcache_selective_migrate_only": "#ff7f7f",
    "flowcache_lossless": "#ff3333",
    "oracle_cost": "#33cc33",
}
BASELINE_LABELS = {
    "no_cache": "No-Cache",
    "apc_lru": "APC-LRU",
    "gdsf": "GDSF",
    "sizecost": "SizeCost-LRU",
    "flowcache_always_migrate": "Always-Migrate",
    "flowcache_selective_migrate_only": "Selective-Migrate-Only",
    "flowcache_lossless": "FlowCache-Lossless",
    "oracle_cost": "Oracle-Cost (offline lookahead)",
}
BASELINE_MARKERS = {
    "no_cache": "x",
    "apc_lru": "s",
    "gdsf": "^",
    "sizecost": "D",
    "flowcache_always_migrate": "v",
    "flowcache_selective_migrate_only": "P",
    "flowcache_lossless": "o",
    "oracle_cost": "*",
}


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


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


def aggregate(rows: List[Dict]) -> Dict:
    """
    按 (capacity, concurrency, baseline) 聚合全局指标。

    G3'' 修复：新 CSV 每行是 per-task，global 指标在同行 baseline 中相同。
    取每个 (cell, baseline) 第一行的值即可。
    """
    result = defaultdict(lambda: defaultdict(dict))
    seen = set()
    for r in rows:
        cap = _to_float(r.get("capacity_gib"))
        conc = int(r.get("concurrency", 0))
        bl = r.get("baseline", "")
        if cap is None:
            continue
        key = (cap, conc, bl)
        if key not in seen:
            seen.add(key)
            result[(cap, conc)][bl] = {
                "p95_ttft_ms": _to_float(
                    r.get("global_p95_cache_delay_ms")
                    or r.get("global_p95_ttft_ms")
                ) or 0.0,
                "p50_ttft_ms": _to_float(
                    r.get("global_p50_cache_delay_ms")
                    or r.get("global_p50_ttft_ms")
                    or r.get("global_p95_ttft_ms")
                ) or 0.0,
                "throughput_req_per_s": _to_float(
                    r.get("global_offered_load")
                    or r.get("global_throughput")
                ) or 0.0,
                "ttft_metric_valid": _to_bool(r.get("ttft_metric_valid")),
                "throughput_metric_valid": _to_bool(
                    r.get("throughput_metric_valid")
                ),
                "block_hit_rate": _to_float(r.get("global_block_hit_rate")) or 0.0,
                "miss_cost_ms": _to_float(r.get("task_miss_cost_ms")) or 0.0,
            }
    return dict(result)


def plot_g3(results_path: str, output_path: str, config: Optional[Dict] = None):
    """
    绘制 G3 结果图。

    双面板（2行 × 3列）：
      - 上行：p95 TTFT vs 容量
      - 下行：吞吐 vs 容量
      - 列：并发度 c=1/4/8
    """
    rows = load_results(Path(results_path))
    if not rows:
        print(f"No results in {results_path}")
        return

    aggregated = aggregate(rows)
    if not aggregated:
        print("No aggregated data")
        return

    # 收集所有并发度和容量
    concurrencies = sorted(set(c for _, c in aggregated.keys()))
    capacities = sorted(set(cap for cap, _ in aggregated.keys()))
    all_metrics = [
        metrics
        for baseline_metrics in aggregated.values()
        for metrics in baseline_metrics.values()
    ]
    ttft_valid = bool(all_metrics) and all(
        metrics.get("ttft_metric_valid", False) for metrics in all_metrics
    )
    throughput_valid = bool(all_metrics) and all(
        metrics.get("throughput_metric_valid", False)
        for metrics in all_metrics
    )

    n_conc = len(concurrencies)
    fig, axes = plt.subplots(2, n_conc, figsize=(5 * n_conc, 8), sharex=True)
    if n_conc == 1:
        axes = axes.reshape(2, 1)

    for col, conc in enumerate(concurrencies):
        # 上行：p95 TTFT
        ax_ttft = axes[0, col]
        for bl in BASELINE_ORDER:
            # 收集该 baseline 在不同容量下的 p95 TTFT
            caps = []
            values = []
            for cap in capacities:
                cell = (cap, conc)
                if cell in aggregated and bl in aggregated[cell]:
                    caps.append(cap)
                    values.append(aggregated[cell][bl].get("p95_ttft_ms", 0))
            if caps:
                ax_ttft.plot(caps, values,
                             marker=BASELINE_MARKERS.get(bl, "o"),
                             color=BASELINE_COLORS.get(bl, "#333333"),
                             label=BASELINE_LABELS.get(bl, bl),
                             linewidth=2, markersize=7)
        ax_ttft.set_title(f"Concurrency = {conc}", fontsize=13)
        if col == 0:
            ax_ttft.set_ylabel(
                "p95 TTFT (ms)" if ttft_valid
                else "Modeled p95 cache delay (ms)",
                fontsize=12,
            )
        ax_ttft.legend(fontsize=9)
        ax_ttft.grid(True, alpha=0.3)
        ax_ttft.set_xlabel("Capacity (GiB)", fontsize=11)

        # 下行：吞吐
        ax_thr = axes[1, col]
        for bl in BASELINE_ORDER:
            caps = []
            values = []
            for cap in capacities:
                cell = (cap, conc)
                if cell in aggregated and bl in aggregated[cell]:
                    caps.append(cap)
                    values.append(aggregated[cell][bl].get("throughput_req_per_s", 0))
            if caps:
                ax_thr.plot(caps, values,
                            marker=BASELINE_MARKERS.get(bl, "o"),
                            color=BASELINE_COLORS.get(bl, "#333333"),
                            label=BASELINE_LABELS.get(bl, bl),
                            linewidth=2, markersize=7)
        if col == 0:
            ax_thr.set_ylabel(
                "Achieved throughput (req/s)" if throughput_valid
                else "Offered load (req/s; trace-fixed)",
                fontsize=12,
            )
        ax_thr.legend(fontsize=9)
        ax_thr.grid(True, alpha=0.3)
        ax_thr.set_xlabel("Capacity (GiB)", fontsize=11)

    title_metrics = (
        "p95 TTFT & Throughput"
        if ttft_valid and throughput_valid
        else "Modeled Cache Delay & Offered Load (Diagnostic)"
    )
    plt.suptitle(f"G3: Lossless Residency — {title_metrics} vs Capacity",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_hit_rate(results_path: str, output_path: str):
    """绘制 block hit rate 图（附加）。"""
    rows = load_results(Path(results_path))
    if not rows:
        return
    aggregated = aggregate(rows)
    if not aggregated:
        return

    concurrencies = sorted(set(c for _, c in aggregated.keys()))
    capacities = sorted(set(cap for cap, _ in aggregated.keys()))
    n_conc = len(concurrencies)

    fig, axes = plt.subplots(1, n_conc, figsize=(5 * n_conc, 4), sharex=True)
    if n_conc == 1:
        axes = [axes]

    for col, conc in enumerate(concurrencies):
        ax = axes[col]
        for bl in BASELINE_ORDER:
            caps = []
            values = []
            for cap in capacities:
                cell = (cap, conc)
                if cell in aggregated and bl in aggregated[cell]:
                    caps.append(cap)
                    values.append(aggregated[cell][bl].get("block_hit_rate", 0) * 100)
            if caps:
                ax.plot(caps, values,
                        marker=BASELINE_MARKERS.get(bl, "o"),
                        color=BASELINE_COLORS.get(bl, "#333333"),
                        label=BASELINE_LABELS.get(bl, bl),
                        linewidth=2, markersize=7)
        ax.set_title(f"Concurrency = {conc}", fontsize=13)
        ax.set_ylabel("Block Hit Rate (%)", fontsize=12)
        ax.set_xlabel("Capacity (GiB)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("G3: Block Hit Rate vs Capacity", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hit rate plot saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G3 results plotter")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-hitrate", default=None)
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

    output_path = args.output
    if output_path is None:
        output_path = config.get("output", {}).get("figures_dir", "figures/")
        output_path = str(Path(output_path) / "g3-headroom.png")
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    plot_g3(str(results_path), str(output_path), config)

    # 附加：hit rate 图
    hitrate_path = output_path.parent / "g3-hit-rate.png"
    plot_hit_rate(str(results_path), str(hitrate_path))


if __name__ == "__main__":
    main()
