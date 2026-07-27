"""
G1′ Headroom Plotter
====================
Reads ``results/raw_results.csv`` and produces a dual-panel figure
``figures/g1prime-headroom.png``:

  - **Top row**: mean ``miss_prefill_ms`` vs capacity (GiB), one line
    per baseline.
  - **Bottom row**: mean ``p95_ttft_ms`` vs capacity (GiB), one line
    per baseline.

Columns are faceted by concurrency level (c = 1 / 4 / 8). The y-axis
for the top panel is log-scaled because Oracle-Cost miss cost can be
orders of magnitude smaller than the simple heuristics at large
capacity.

Read-only w.r.t. ``experiments/e1/``.

Usage
-----
    python experiments/g1prime/plot_headroom.py
    python experiments/g1prime/plot_headroom.py --input results/raw_results.csv \\
        --output figures/g1prime-headroom.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# matplotlib import with non-interactive backend safety.
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Distinct color + marker per baseline (consistent across panels).
BASELINE_STYLE: Dict[str, Dict[str, str]] = {
    "lru":         {"color": "tab:blue",   "marker": "o"},
    "gdsf":        {"color": "tab:orange", "marker": "s"},
    "sizecost":    {"color": "tab:green",  "marker": "^"},
    "apc_lru":     {"color": "tab:red",    "marker": "D"},
    "belady":      {"color": "tab:purple", "marker": "v"},
    "oracle_cost": {"color": "tab:brown",  "marker": "*"},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "G1′ headroom plotter: dual-panel (miss cost + p95 TTFT) "
            "vs capacity, faceted by concurrency."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="Path to config.yaml (default: experiments/g1prime/config.yaml).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "results" / "raw_results.csv",
        help="Input raw_results.csv path.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "figures" / "g1prime-headroom.png",
        help="Output PNG path (default: experiments/g1prime/figures/g1prime-headroom.png).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_baseline_style(name: str) -> Dict[str, str]:
    """Return ``{color, marker}`` for a baseline, with a fallback."""
    if name in BASELINE_STYLE:
        return BASELINE_STYLE[name]
    return {"color": "tab:gray", "marker": "x"}


def aggregate_metric(
    df: pd.DataFrame,
    metric: str,
    baseline: str,
    concurrency: int,
) -> pd.Series:
    """Return ``{capacity_gib -> mean(metric)}`` for one (baseline, conc)."""
    sub = df[
        (df["baseline"] == baseline) & (df["concurrency"] == concurrency)
    ]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("capacity_gib")[metric].mean().sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("G1′ Headroom Plotter")
    print("=" * 78)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print()

    if not args.input.exists():
        sys.stderr.write(
            f"ERROR: input CSV not found: {args.input}\n"
            f"       Run experiments/g1prime/run_grid.py first.\n"
        )
        return 1

    df = pd.read_csv(args.input)
    if df.empty:
        sys.stderr.write(f"ERROR: input CSV is empty: {args.input}\n")
        return 1

    required = {"baseline", "capacity_gib", "concurrency",
                "miss_prefill_ms", "p95_ttft_ms"}
    missing = required - set(df.columns)
    if missing:
        sys.stderr.write(
            f"ERROR: input CSV missing columns: {sorted(missing)}\n"
        )
        return 1

    concurrencies: List[int] = sorted(df["concurrency"].unique().tolist())
    baselines: List[str] = sorted(df["baseline"].unique().tolist())
    capacities: List[float] = sorted(df["capacity_gib"].unique().tolist())

    print(f"  baselines   : {baselines}")
    print(f"  capacities  : {capacities}")
    print(f"  concurrencies: {concurrencies}")
    print()

    n_cols = max(1, len(concurrencies))
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 7.5),
        sharex=True,
        squeeze=False,
    )

    for col_idx, c in enumerate(concurrencies):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        for bl in baselines:
            style = get_baseline_style(bl)

            # Top: mean miss_prefill_ms
            ms = aggregate_metric(df, "miss_prefill_ms", bl, c)
            if not ms.empty:
                ax_top.plot(
                    ms.index, ms.values,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.6,
                    markersize=6,
                    label=bl,
                )

            # Bottom: mean p95_ttft_ms
            ps = aggregate_metric(df, "p95_ttft_ms", bl, c)
            if not ps.empty:
                ax_bot.plot(
                    ps.index, ps.values,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.6,
                    markersize=6,
                    label=bl,
                )

        ax_top.set_title(f"c = {c}", fontsize=11)
        ax_top.set_xscale("log", base=2)
        ax_top.set_yscale("log")
        ax_top.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        if col_idx == 0:
            ax_top.set_ylabel("Mean miss prefill (ms)", fontsize=10)

        ax_bot.set_xlabel("Capacity (GiB)", fontsize=10)
        ax_bot.set_xscale("log", base=2)
        ax_bot.set_yscale("log")
        ax_bot.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        if col_idx == 0:
            ax_bot.set_ylabel("Mean p95 TTFT (ms)", fontsize=10)

        # Legend on the bottom-right panel only (shared across panels).
        if col_idx == n_cols - 1:
            ax_bot.legend(
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
                title="baseline",
                title_fontsize=9,
            )

    # Annotate capacity tick labels with the actual GiB values.
    for ax in axes.flatten():
        ax.set_xticks(capacities)
        ax.set_xticklabels([str(int(c) if float(c).is_integer() else c)
                            for c in capacities])

    fig.suptitle(
        "G1′ Headroom: miss cost & p95 TTFT vs capacity (faceted by concurrency)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
