"""
E1 Visualization Script
Generates publication-quality characterization plots.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configure Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300


SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_characterization() -> Optional[Dict]:
    """Load e1-characterization.json. Returns None if missing."""
    path = SCRIPT_DIR / "outputs" / "e1-characterization.json"
    if not path.is_file():
        print(f"[ERROR] Characterization data not found: {path}")
        print("        Run characterize_workload.py first to generate this file.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_oracle_comparison() -> Optional[Dict]:
    """Load e1-oracle-comparison.json. Returns None if missing."""
    path = SCRIPT_DIR / "outputs" / "e1-oracle-comparison.json"
    if not path.is_file():
        print(f"[ERROR] Oracle comparison data not found: {path}")
        print("        Run compare_oracle.py first to generate this file.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warn_missing(plot_name: str, field: str) -> None:
    """Log a warning when a required field is missing from the data."""
    print(f"[WARN] {plot_name}: field '{field}' not found in data. Skipping plot.")


def _approx_cdf_from_percentiles(
    stats: Dict, count: int
) -> Optional[tuple]:
    """
    Reconstruct approximate CDF data points from percentile stats.

    Returns (x_values, y_values) or None if not enough data.
    """
    required = ["median", "p95", "p99"]
    if not all(k in stats for k in required):
        return None

    # Build known (value, cumulative_probability) points
    points = []
    if "min" in stats:
        points.append((stats["min"], 0.0))
    points.append((stats["median"], 0.50))
    points.append((stats["p95"], 0.95))
    points.append((stats["p99"], 0.99))
    if "max" in stats:
        points.append((stats["max"], 1.0))

    # Sort by value
    points.sort(key=lambda p: p[0])

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    return xs, ys


# ---------------------------------------------------------------------------
# Plot 1: Exact-Prefix Overlap Distribution
# ---------------------------------------------------------------------------

def plot_overlap_histogram(data: Dict, output_path: str):
    """Plot 1: Exact-prefix overlap distribution histogram."""
    overlap = data.get("exact_prefix_overlap")
    if overlap is None:
        _warn_missing("Overlap Histogram", "exact_prefix_overlap")
        return

    overlap_ratio = overlap.get("overlap_ratio", 0.0)
    lcp = overlap.get("lcp_tokens", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    # Check for raw LCP values first
    raw_lcp = overlap.get("lcp_values")
    if raw_lcp and len(raw_lcp) > 0:
        # Use raw data for true histogram
        ax.hist(raw_lcp, bins=30, color="steelblue", edgecolor="white",
                alpha=0.85, density=False)
        mean_val = np.mean(raw_lcp)
    elif lcp and lcp.get("count", 0) > 0:
        # Approximate histogram from distribution stats
        # Use known percentiles to build binned approximation
        pcts = []
        for key in ["min", "median", "p95", "p99", "max"]:
            if key in lcp:
                pcts.append(lcp[key])
        pcts = sorted(set(pcts))
        if len(pcts) >= 3:
            # Create synthetic histogram bins
            bin_edges = []
            if len(pcts) >= 2:
                bin_edges = list(pcts)
                # Add intermediate bins if gaps are large
                refined = [bin_edges[0]]
                for i in range(1, len(bin_edges)):
                    gap = bin_edges[i] - bin_edges[i - 1]
                    if gap > bin_edges[0] * 2:
                        mid = (bin_edges[i] + bin_edges[i - 1]) / 2
                        refined.append(mid)
                    refined.append(bin_edges[i])
                bin_edges = refined

            # Assign counts proportionally
            n_bins = len(bin_edges) - 1
            if n_bins > 0:
                count_per_bin = lcp["count"] / n_bins
                counts = [count_per_bin] * n_bins
                # Weight middle bins more (mean tends toward center)
                if n_bins >= 3:
                    mid_idx = n_bins // 2
                    counts[mid_idx] *= 1.5
                    for i in range(n_bins):
                        if i != mid_idx:
                            counts[i] *= (1.0 - 0.5 / (n_bins - 1))

                ax.bar(
                    bin_edges[:-1],
                    counts,
                    width=np.diff(bin_edges),
                    align="edge",
                    color="steelblue",
                    edgecolor="white",
                    alpha=0.85,
                )
                ax.set_xlim(bin_edges[0] * 0.9, bin_edges[-1] * 1.1)
            mean_val = lcp.get("mean", 0)
        else:
            # Fallback: just annotate
            ax.text(0.5, 0.5,
                    f"Overlap Ratio = {overlap_ratio:.2%}\n"
                    f"LCP Mean = {lcp.get('mean', 'N/A')} tokens\n"
                    f"LCP Median = {lcp.get('median', 'N/A')} tokens",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {output_path} (summary annotation)")
            return
    else:
        ax.text(0.5, 0.5,
                f"Overlap Ratio = {overlap_ratio:.2%}\n"
                f"(No pair-wise distribution data available)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {output_path} (summary annotation)")
        return

    # Add mean line
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_val:.1f} tokens")

    ax.set_xlabel("重叠比例 (Overlap Ratio)", fontsize=11)
    ax.set_ylabel("频次 (Frequency)", fontsize=11)
    ax.set_title("Exact-Prefix Overlap Distribution", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Next-Use Distance CDF
# ---------------------------------------------------------------------------

def plot_next_use_cdf(data: Dict, output_path: str):
    """Plot 2: Next-use distance CDF."""
    nud = data.get("next_use_distance")
    if nud is None:
        _warn_missing("Next-Use CDF", "next_use_distance")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Check for raw distance values
    raw_distances = nud.get("raw_distances")
    if raw_distances and len(raw_distances) > 0:
        sorted_d = np.sort(raw_distances)
        y = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, y, color="steelblue", linewidth=1.5, drawstyle="steps-post")
        median_val = np.median(sorted_d)
        p95_val = np.percentile(sorted_d, 95)
        p99_val = np.percentile(sorted_d, 99)
        has_raw = True
    else:
        # Reconstruct from percentile stats
        result = _approx_cdf_from_percentiles(nud, nud.get("count", 0))
        if result is None:
            ax.text(0.5, 0.5,
                    f"Next-use distance stats only:\n"
                    f"Median = {nud.get('median', 'N/A')} steps\n"
                    f"P95 = {nud.get('p95', 'N/A')} steps\n"
                    f"P99 = {nud.get('p99', 'N/A')} steps",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {output_path} (summary annotation)")
            return
        xs, ys = result
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=1.5, markersize=6)
        ax.plot(xs, ys, color="steelblue", linewidth=1.5)  # connect with line
        median_val = nud.get("median", xs[len(xs)//2])
        p95_val = nud.get("p95", xs[-2] if len(xs) >= 2 else xs[-1])
        p99_val = nud.get("p99", xs[-1])
        has_raw = False

    # Annotate median, p95, p99
    for pct, val, color in [
        (50, median_val, "green"),
        (95, p95_val, "orange"),
        (99, p99_val, "red"),
    ]:
        ax.axhline(pct / 100, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax.axvline(val, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax.plot(val, pct / 100, "o", color=color, markersize=5)
        ax.annotate(
            f"P{pct}={val:.0f}",
            xy=(val, pct / 100),
            xytext=(10, -10 if pct < 80 else -20),
            textcoords="offset points",
            fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    # Use log scale for x-axis if heavy-tailed
    max_val = max(raw_distances) if (raw_distances and len(raw_distances) > 0) else nud.get("max", 1)
    min_val = min(raw_distances) if (raw_distances and len(raw_distances) > 0) else nud.get("min", 1)
    if max_val > 0 and (max_val / max(min_val, 1)) > 100:
        ax.set_xscale("log")
        ax.set_xlabel("下次访问距离 (Next-Use Distance, steps, log scale)", fontsize=11)
    else:
        ax.set_xlabel("下次访问距离 (Next-Use Distance, steps)", fontsize=11)

    ax.set_ylabel("累积概率 (Cumulative Probability)", fontsize=11)
    ax.set_title("Next-Use Distance CDF", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(["CDF", "P50", "P95", "P99"], loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: KV Working-Set Size Timeline
# ---------------------------------------------------------------------------

def plot_working_set(data: Dict, output_path: str):
    """Plot 3: KV working-set size over time."""
    ws = data.get("working_set")
    if ws is None:
        _warn_missing("Working Set Timeline", "working_set")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Check for per-step active block timeline data
    step_counts = data.get("working_set_timeline")
    if step_counts and len(step_counts) > 0:
        # Use actual per-step data
        steps = range(len(step_counts))
        ax.fill_between(steps, step_counts, alpha=0.3, color="steelblue")
        ax.plot(steps, step_counts, color="steelblue", linewidth=1.0)
        peak = max(step_counts)
    else:
        # Fallback: show per-workflow peaks as bar chart
        per_wf = ws.get("per_workflow_peak", [])
        if per_wf:
            wf_ids = [w["workflow_id"][:20] for w in per_wf]  # truncate long IDs
            peaks = [w["peak_active_blocks"] for w in per_wf]
            x_pos = range(len(wf_ids))
            ax.bar(x_pos, peaks, color="steelblue", edgecolor="white", alpha=0.85)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(wf_ids, rotation=45, ha="right", fontsize=7)
            peak = max(peaks) if peaks else 0
        else:
            peak = ws.get("working_set_size", 0)
            ax.text(0.5, 0.5,
                    f"Peak Working Set: {peak} blocks\n"
                    f"KV Memory: {ws.get('kv_memory_gb', 'N/A')} GB\n"
                    f"KV/VRAM: {ws.get('kv_vram_ratio', 'N/A'):.2%}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {output_path} (summary annotation)")
            return

    # Add KV budget threshold lines (25%, 50%, 100% of peak)
    kv_budgets = [0.25, 0.50, 1.00]
    colors = ["orange", "red", "darkred"]
    labels = ["25% Budget", "50% Budget", "100% Budget"]
    for budget, color, label in zip(kv_budgets, colors, labels):
        threshold = int(peak * budget)
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"{label} ({threshold} blocks)")

    ax.set_xlabel("全局步数 (Global Step)", fontsize=11)
    ax.set_ylabel("活跃块数 (Active Blocks)", fontsize=11)
    ax.set_title("KV Working-Set Size Timeline", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Oracle vs Heuristic Comparison
# ---------------------------------------------------------------------------

def plot_oracle_comparison(data: Dict, output_path: str):
    """Plot 4: Oracle vs LRU vs GDSF grouped bar chart."""
    results = data.get("results")
    if results is None:
        _warn_missing("Oracle Comparison", "results")
        return

    # Extract budget levels and hit rates
    budget_keys = sorted(results.keys(), key=lambda k: float(k.replace("budget_", "")))
    if not budget_keys:
        _warn_missing("Oracle Comparison", "results (empty)")
        return

    budgets_pct = []
    lru_rates = []
    gdsf_rates = []
    oracle_rates = []

    for bk in budget_keys:
        entry = results[bk]
        budget_float = float(bk.replace("budget_", ""))
        budgets_pct.append(f"{int(budget_float * 100)}%")
        lru_rates.append(entry.get("lru", {}).get("hit_rate", 0.0))
        gdsf_rates.append(entry.get("gdsf", {}).get("hit_rate", 0.0))
        oracle_rates.append(entry.get("oracle", {}).get("hit_rate", 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(budgets_pct))
    width = 0.22

    bars_lru = ax.bar(x - width, lru_rates, width, label="LRU",
                       color="#4C72B0", edgecolor="white", alpha=0.9)
    bars_gdsf = ax.bar(x, gdsf_rates, width, label="GDSF",
                        color="#DD8452", edgecolor="white", alpha=0.9)
    bars_oracle = ax.bar(x + width, oracle_rates, width, label="Oracle",
                          color="#55A868", edgecolor="white", alpha=0.9)

    # Add value labels on bars
    for bars, rates in [(bars_lru, lru_rates), (bars_gdsf, gdsf_rates),
                          (bars_oracle, oracle_rates)]:
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=7,
                    rotation=0)

    # Add headroom annotations (gap between oracle and best heuristic)
    for i in range(len(budgets_pct)):
        best_heuristic = max(lru_rates[i], gdsf_rates[i])
        headroom = oracle_rates[i] - best_heuristic
        if headroom > 0.005:
            mid_x = x[i]
            top_y = max(lru_rates[i], gdsf_rates[i], oracle_rates[i])
            ax.annotate(
                f"+{headroom:.1%}",
                xy=(mid_x, top_y + 0.02),
                ha="center", fontsize=7,
                color="darkgreen", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.6),
            )

    ax.set_xlabel("KV Budget Level", fontsize=11)
    ax.set_ylabel("命中率 (Hit Rate)", fontsize=11)
    ax.set_title("Oracle vs Heuristic Eviction Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets_pct)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    char_data = load_characterization()
    oracle_data = load_oracle_comparison()

    if not char_data and not oracle_data:
        print("No data files found. Run characterize_workload.py and compare_oracle.py first.")
        return

    output_dir = SCRIPT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if char_data:
        plot_overlap_histogram(char_data, str(output_dir / "e1-overlap-hist.png"))
        plot_next_use_cdf(char_data, str(output_dir / "e1-next-use-cdf.png"))
        plot_working_set(char_data, str(output_dir / "e1-working-set.png"))
    else:
        print("[SKIP] Characterization plots skipped (no characterization data).")

    if oracle_data:
        plot_oracle_comparison(oracle_data, str(output_dir / "e1-oracle-comparison.png"))
    else:
        print("[SKIP] Oracle comparison plot skipped (no oracle data).")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
