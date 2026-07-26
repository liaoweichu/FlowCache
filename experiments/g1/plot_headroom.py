"""
G1 Headroom Plotter
===================

Reads `results/raw_results.csv` and produces `figures/g1-headroom.png`,
a 2-panel figure:
  * Left panel:  miss_cost (ms)  vs  budget, one line per baseline
  * Right panel: p95 TTFT (ms)   vs  budget, one line per baseline

Values are averaged across the 3 replay seeds; error bars show min/max
across seeds.

If matplotlib is unavailable, the script degrades gracefully and prints a
warning instead of crashing (the verdict module does not depend on plots).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


# Display order: simple heuristics first, then inspired, then oracles.
BASELINE_ORDER = [
    ("lru",                   "LRU",                  "#1f77b4", "o"),
    ("gdsf",                  "GDSF",                 "#ff7f0e", "s"),
    ("sizecost",              "SizeCost",             "#2ca02c", "^"),
    ("apc_lru",               "APC-LRU",              "#d62728", "D"),
    ("pbkv_inspired",         "PBKV-Insp†",          "#9467bd", "P"),
    ("thunderagent_inspired", "ThunderAgent-Insp†",  "#8c564b", "X"),
    ("kvflow_faithful",       "KVFlow (pending)",     "#7f7f7f", "*"),
    ("belady",                "Oracle-Belady",        "#17becf", "v"),
    ("oracle_cost",           "Oracle-Cost",          "#000000", "h"),
]


# ---------------------------------------------------------------------------
# Config + CSV loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = SCRIPT_DIR / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_raw_results(csv_path: Optional[str] = None) -> List[Dict]:
    if csv_path is None:
        csv_path = SCRIPT_DIR / "results" / "raw_results.csv"
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        cleaned = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(cleaned))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: List[Dict]) -> Dict[Tuple[str, float], Dict[str, List[float]]]:
    """Group by (baseline, budget) → {miss_cost: [...], p95_ttft: [...]}."""
    out: Dict[Tuple[str, float], Dict[str, List[float]]] = defaultdict(
        lambda: {"miss_cost": [], "p95_ttft": []}
    )
    for r in rows:
        bl = r.get("baseline", "")
        try:
            budget = float(r["budget"])
        except (KeyError, TypeError, ValueError):
            continue
        mc = _to_float(r.get("miss_cost_ms"))
        tt = _to_float(r.get("p95_ttft_ms"))
        if mc is not None:
            out[(bl, budget)]["miss_cost"].append(mc)
        if tt is not None:
            out[(bl, budget)]["p95_ttft"].append(tt)
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(aggregated: Dict, budgets: List[float], out_path: Path) -> None:
    """Render the 2-panel figure to `out_path`."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping plot generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "miss_cost", "Miss cost (ms)"),
        (axes[1], "p95_ttft",  "p95 TTFT (ms)"),
    ]:
        for bl, label, color, marker in BASELINE_ORDER:
            xs: List[float] = []
            means: List[float] = []
            lo: List[float] = []
            hi: List[float] = []
            for b in budgets:
                vals = aggregated.get((bl, b), {}).get(metric, [])
                if not vals:
                    continue
                xs.append(b * 100)  # show as percent
                m = _mean(vals)
                means.append(m)
                lo.append(min(vals))
                hi.append(max(vals))
            if not xs:
                continue
            xs_sorted, means_sorted, lo_sorted, hi_sorted = zip(
                *sorted(zip(xs, means, lo, hi))
            )
            ax.plot(xs_sorted, means_sorted,
                    color=color, marker=marker, label=label, linewidth=1.5)
            ax.fill_between(xs_sorted, lo_sorted, hi_sorted,
                            color=color, alpha=0.15)
        ax.set_xlabel("KV budget (% of peak working set)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs budget")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("G1 headroom: miss cost & p95 TTFT vs budget (τ-bench, 3 seeds)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    cfg = load_config()
    csv_path = cfg["output"]["raw_results_csv"]
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    out_path = cfg["output"]["headroom_png"]
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    rows = read_raw_results(csv_path)
    if not rows:
        print(f"ERROR: no rows in {csv_path} (run run_grid.py first)")
        return
    aggregated = aggregate(rows)
    budgets = list(cfg["budgets"])
    plot(aggregated, budgets, out_path)


if __name__ == "__main__":
    main()
