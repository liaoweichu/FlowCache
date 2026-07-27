"""
G1′ Verdict Generator
=====================
Reads ``results/raw_results.csv`` produced by :mod:`run_grid` and emits
the Go/No-Go verdict for the G1′ physical-prefix replay experiment.

Outputs
-------
  - ``g1prime-verdict.md``  — human-readable verdict report.
  - ``g1prime-verdict.json`` — machine-readable verdict.

Methodology
-----------
For each ``(capacity_gib, concurrency)`` cell:

  1. **Headroom (mean across episodes)**::

         oracle_miss   = mean(miss_prefill_ms | baseline == oracle_cost)
         best_simple   = min over b in {lru, gdsf, sizecost, apc_lru} of
                         mean(miss_prefill_ms | baseline == b)
         headroom_abs  = best_simple − oracle_miss   (≥ 0 expected)
         headroom_rel  = headroom_abs / oracle_miss  (if oracle_miss > 0)

     Lower ``miss_prefill_ms`` is better, so a positive ``headroom_abs``
     means the oracle beats the best simple heuristic.

  2. **Per-task cluster bootstrap** (``verdict.bootstrap_unit == task_group``):
     For each ``task_id``, compute that task's per-baseline mean
     miss cost (averaged across seeds), then the task-level
     ``headroom_rel``. Resample the 165 task groups with replacement
     ``bootstrap_samples`` times (default 1000) and recompute the mean
     each time. Take the 95 % CI lower bound.

  3. **Go/No-Go**:
     - **go** if ∃ a cell with ``headroom_rel ≥ headroom_threshold_rel``
       (default 10 %) AND ``ci_lower > 0``.
     - **no_go** otherwise.

Read-only w.r.t. ``experiments/e1/``.

Usage
-----
    python experiments/g1prime/verdict.py
    python experiments/g1prime/verdict.py --input results/raw_results.csv \\
        --output-md g1prime-verdict.md --output-json g1prime-verdict.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Default simple-heuristic baselines (max-of is the "best simple" reference).
DEFAULT_SIMPLE_BASELINES: Tuple[str, ...] = ("lru", "gdsf", "sizecost", "apc_lru")
# Oracle reference baseline (cost-aware Belady).
DEFAULT_ORACLE_BASELINE: str = "oracle_cost"
# Default metric column (lower is better → oracle should be smallest).
DEFAULT_METRIC: str = "miss_prefill_ms"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "G1′ verdict generator: compute headroom, cluster bootstrap "
            "CI, and Go/No-Go decision from raw_results.csv."
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
        help="Input raw_results.csv (default: experiments/g1prime/results/raw_results.csv).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=SCRIPT_DIR / "g1prime-verdict.md",
        help="Output markdown report path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "g1prime-verdict.json",
        help="Output JSON verdict path.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> Dict[str, Any]:
    """Load the YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Headroom (mean across episodes)
# ---------------------------------------------------------------------------
def compute_headroom_table(
    df: pd.DataFrame,
    simple_baselines: Tuple[str, ...],
    oracle_baseline: str,
    metric: str = DEFAULT_METRIC,
) -> List[Dict[str, Any]]:
    """Compute mean-episode headroom per (capacity, concurrency) cell.

    For each cell, returns a dict with:
      - ``capacity_gib``, ``concurrency``
      - ``oracle_miss_cost`` — mean metric for the oracle baseline.
      - ``best_simple_miss_cost`` — min mean metric over simple baselines.
      - ``best_simple_baseline`` — name of the arg-min simple baseline.
      - ``per_simple`` — ``{baseline -> mean metric}`` for all simples.
      - ``headroom_abs`` — ``best_simple − oracle`` (positive = oracle better).
      - ``headroom_rel`` — ``headroom_abs / oracle`` (0 if oracle ≤ 0).
    """
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    cells = (
        df[["capacity_gib", "concurrency"]]
        .drop_duplicates()
        .sort_values(["capacity_gib", "concurrency"])
        .itertuples(index=False)
    )
    for cap, conc in cells:
        sub = df[
            (df["capacity_gib"] == cap) & (df["concurrency"] == conc)
        ]
        if sub.empty:
            continue

        per_baseline = sub.groupby("baseline")[metric].mean()

        if oracle_baseline not in per_baseline.index:
            continue
        oracle_cost = float(per_baseline[oracle_baseline])

        per_simple: Dict[str, float] = {}
        for sb in simple_baselines:
            if sb in per_baseline.index and not pd.isna(per_baseline[sb]):
                per_simple[sb] = float(per_baseline[sb])

        if not per_simple:
            continue

        best_simple_name = min(per_simple, key=per_simple.get)
        best_simple = per_simple[best_simple_name]

        headroom_abs = best_simple - oracle_cost
        headroom_rel = (
            headroom_abs / oracle_cost if oracle_cost > 0 else 0.0
        )

        rows.append(
            {
                "capacity_gib": float(cap),
                "concurrency": int(conc),
                "oracle_miss_cost": oracle_cost,
                "best_simple_miss_cost": best_simple,
                "best_simple_baseline": best_simple_name,
                "per_simple": per_simple,
                "headroom_abs": headroom_abs,
                "headroom_rel": headroom_rel,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-task cluster bootstrap
# ---------------------------------------------------------------------------
def compute_per_task_headroom(
    df: pd.DataFrame,
    simple_baselines: Tuple[str, ...],
    oracle_baseline: str,
    capacity_gib: float,
    concurrency: int,
    metric: str = DEFAULT_METRIC,
) -> Dict[str, float]:
    """Compute ``{task_id -> headroom_rel}`` for one (capacity, concurrency) cell.

    For each task_id, takes the per-baseline mean across seeds (so a task
    group with 8 seeds collapses to one headroom value). Returns 0.0 for
    task groups where the oracle's mean is ≤ 0 or any simple baseline is
    missing.
    """
    sub = df[
        (df["capacity_gib"] == capacity_gib)
        & (df["concurrency"] == concurrency)
    ]
    if sub.empty:
        return {}

    out: Dict[str, float] = {}
    for tid, task_sub in sub.groupby("task_id"):
        per_baseline = task_sub.groupby("baseline")[metric].mean()
        if oracle_baseline not in per_baseline.index:
            continue
        ob = per_baseline[oracle_baseline]
        if pd.isna(ob) or ob <= 0:
            out[tid] = 0.0
            continue
        sb_costs = []
        for sb in simple_baselines:
            if sb in per_baseline.index and not pd.isna(per_baseline[sb]):
                sb_costs.append(per_baseline[sb])
        if not sb_costs:
            out[tid] = 0.0
            continue
        best_sb = min(sb_costs)
        out[tid] = float((best_sb - ob) / ob)
    return out


def bootstrap_ci(
    per_task_values: Dict[str, float],
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Cluster bootstrap on per-task headroom values.

    Resample ``len(per_task_values)`` task groups with replacement,
    ``n_samples`` times. Each resample's statistic is the mean of the
    sampled headroom values. Return the ``(mean, ci_lower, ci_upper)``
    of the bootstrap distribution.

    Args:
        per_task_values: ``{task_id -> headroom_rel}``.
        n_samples: Number of bootstrap resamples.
        ci_level: Confidence level (0..1); e.g. 0.95 for a 95 % CI.
        seed: RNG seed for reproducibility.

    Returns:
        ``(mean, ci_lower, ci_upper)``. ``(0, 0, 0)`` if input is empty.
    """
    if not per_task_values:
        return 0.0, 0.0, 0.0

    values: List[float] = list(per_task_values.values())
    n_tasks = len(values)
    mean_observed = statistics.mean(values)

    rng = random.Random(seed)
    boot: List[float] = []
    for _ in range(n_samples):
        sampled_sum = 0.0
        for _j in range(n_tasks):
            sampled_sum += values[rng.randrange(n_tasks)]
        boot.append(sampled_sum / n_tasks)

    boot.sort()
    alpha = 1.0 - ci_level
    lo_idx = max(0, int(alpha / 2.0 * n_samples))
    hi_idx = min(n_samples - 1, int((1.0 - alpha / 2.0) * n_samples))
    if hi_idx < lo_idx:
        hi_idx = lo_idx
    return mean_observed, boot[lo_idx], boot[hi_idx]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(
    overview: Dict[str, Any],
    headroom_rows: List[Dict[str, Any]],
    bootstrap_rows: List[Dict[str, Any]],
    verdict: str,
    threshold_rel: float,
    passing_cells: List[Dict[str, Any]],
    recommendation: str,
) -> str:
    """Render the verdict as a Markdown report."""
    lines: List[str] = []
    lines.append("# G1′ Verdict Report")
    lines.append("")
    lines.append("## 1. Experiment Overview")
    lines.append("")
    lines.append(f"- **CSV rows**: {overview.get('n_rows', 0):,}")
    lines.append(f"- **Episodes per cell**: {overview.get('n_episodes_per_cell', 0):,}")
    lines.append(f"- **Unique task groups**: {overview.get('n_task_groups', 0)}")
    lines.append(f"- **Unique seeds**: {overview.get('n_seeds', 0)}")
    lines.append(f"- **Capacity tiers (GiB)**: {overview.get('capacities_gib', [])}")
    lines.append(f"- **Concurrency levels**: {overview.get('concurrency_levels', [])}")
    lines.append(f"- **Baselines**: {overview.get('baselines', [])}")
    lines.append(
        f"- **Block-level accesses per cell**: "
        f"{overview.get('accesses_per_cell', 0):,}"
    )
    lines.append("")

    lines.append("## 2. Methodology")
    lines.append("")
    lines.append(
        "For each ``(capacity_gib, concurrency)`` cell, headroom is "
        "computed as the relative reduction in mean ``miss_prefill_ms`` "
        "of the Oracle-Cost baseline over the best-performing simple "
        "heuristic (LRU / GDSF / SizeCost / APC-LRU):"
    )
    lines.append("")
    lines.append("```")
    lines.append("oracle_miss   = mean(miss_prefill_ms | baseline == oracle_cost)")
    lines.append("best_simple   = min over b in {lru, gdsf, sizecost, apc_lru}")
    lines.append("                 of mean(miss_prefill_ms | baseline == b)")
    lines.append("headroom_abs  = best_simple − oracle_miss        (≥ 0 expected)")
    lines.append("headroom_rel  = headroom_abs / oracle_miss      (lower is better)")
    lines.append("```")
    lines.append("")
    lines.append(
        "Statistical significance is assessed via a cluster bootstrap "
        "resampling task groups (not individual episodes) with "
        f"{overview.get('bootstrap_samples', 1000)} resamples and a "
        f"{int(overview.get('ci_level', 0.95) * 100)}% CI."
    )
    lines.append("")

    lines.append("## 3. Headroom Table (mean across episodes)")
    lines.append("")
    lines.append(
        "| Capacity (GiB) | Concurrency | Oracle miss (ms) | "
        "Best simple (ms) | Best simple baseline | headroom_abs (ms) | "
        "headroom_rel |"
    )
    lines.append(
        "|---------------:|------------:|------------------:|"
        "-----------------:|:---------------------|------------------:|"
        "-------------:|"
    )
    for r in headroom_rows:
        lines.append(
            f"| {r['capacity_gib']} | {r['concurrency']} | "
            f"{r['oracle_miss_cost']:.2f} | "
            f"{r['best_simple_miss_cost']:.2f} | "
            f"{r['best_simple_baseline']} | "
            f"{r['headroom_abs']:.2f} | "
            f"{r['headroom_rel']*100:.2f}% |"
        )
    lines.append("")

    lines.append("## 4. Cluster Bootstrap CI (per-task resampling)")
    lines.append("")
    lines.append(
        "| Capacity (GiB) | Concurrency | n_tasks | Mean headroom_rel | "
        "CI lower | CI upper | CI lower > 0? |"
    )
    lines.append(
        "|---------------:|------------:|--------:|------------------:|"
        "---------:|---------:|:--------------|"
    )
    for r in bootstrap_rows:
        ci_pos = "YES" if r["ci_lower"] > 0 else "no"
        lines.append(
            f"| {r['capacity_gib']} | {r['concurrency']} | "
            f"{r['n_tasks']} | {r['mean_headroom_rel']*100:.2f}% | "
            f"{r['ci_lower']*100:.2f}% | {r['ci_upper']*100:.2f}% | "
            f"{ci_pos} |"
        )
    lines.append("")

    lines.append("## 5. Go/No-Go Verdict")
    lines.append("")
    threshold_pct = int(threshold_rel * 100)
    if verdict == "go":
        lines.append(
            f"**VERDICT: GO** ✅  — at least one (capacity, concurrency) "
            f"cell achieves headroom_rel ≥ {threshold_pct}% with "
            f"bootstrap CI lower > 0."
        )
    else:
        lines.append(
            f"**VERDICT: NO-GO** ❌  — no (capacity, concurrency) cell "
            f"simultaneously satisfies headroom_rel ≥ {threshold_pct}% "
            f"AND CI lower > 0."
        )
    lines.append("")
    if passing_cells:
        lines.append("Passing cells:")
        lines.append("")
        for c in passing_cells:
            lines.append(
                f"- capacity = {c['capacity_gib']} GiB, "
                f"concurrency = {c['concurrency']}: "
                f"headroom_rel = {c['headroom_rel']*100:.2f}%, "
                f"CI lower = {c['ci_lower']*100:.2f}%"
            )
        lines.append("")
    else:
        lines.append("No passing cells.")
        lines.append("")

    lines.append("## 6. Recommendation")
    lines.append("")
    lines.append(recommendation)
    lines.append("")
    lines.append("---")
    lines.append(
        "Generated by `experiments/g1prime/verdict.py` from "
        "`experiments/g1prime/results/raw_results.csv`."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("G1′ Verdict Generator")
    print("=" * 78)
    print(f"Config     : {args.config}")
    print(f"Input CSV  : {args.input}")
    print(f"Output MD  : {args.output_md}")
    print(f"Output JSON: {args.output_json}")
    print()

    if not args.input.exists():
        sys.stderr.write(
            f"ERROR: input CSV not found: {args.input}\n"
            f"       Run experiments/g1prime/run_grid.py first.\n"
        )
        return 1

    config = load_config(args.config) if args.config.exists() else {}
    vcfg = config.get("verdict", {}) or {}
    threshold_rel = float(vcfg.get("headroom_threshold_rel", 0.10))
    n_samples = int(vcfg.get("bootstrap_samples", 1000))
    ci_level = float(vcfg.get("ci_level", 0.95))
    seed = int(vcfg.get("random_seed", 42))

    simple_baselines: Tuple[str, ...] = tuple(
        (bcfg.get("name") for bcfg in (config.get("baselines", {}) or {}).get("simple_heuristic", []) or [])
    ) or DEFAULT_SIMPLE_BASELINES
    oracle_baseline = DEFAULT_ORACLE_BASELINE

    # Load CSV.
    df = pd.read_csv(args.input)
    if df.empty:
        sys.stderr.write(f"ERROR: input CSV is empty: {args.input}\n")
        return 1

    expected_cols = {"baseline", "capacity_gib", "concurrency", "task_id",
                     "seed", "miss_prefill_ms"}
    missing = expected_cols - set(df.columns)
    if missing:
        sys.stderr.write(
            f"ERROR: input CSV missing columns: {sorted(missing)}\n"
        )
        return 1

    print(f"Loaded {len(df):,} rows")
    print(f"  baselines   : {sorted(df['baseline'].unique().tolist())}")
    print(f"  capacities  : {sorted(df['capacity_gib'].unique().tolist())}")
    print(f"  concurrencies: {sorted(df['concurrency'].unique().tolist())}")
    print(f"  task groups : {df['task_id'].nunique()}")
    print(f"  seeds       : {df['seed'].nunique()}")
    print()

    # ---- Headroom table (mean across episodes) ----
    headroom_rows = compute_headroom_table(
        df, simple_baselines, oracle_baseline, DEFAULT_METRIC
    )

    # ---- Per-task cluster bootstrap per cell ----
    bootstrap_rows: List[Dict[str, Any]] = []
    for hr in headroom_rows:
        cap = hr["capacity_gib"]
        conc = hr["concurrency"]
        per_task = compute_per_task_headroom(
            df, simple_baselines, oracle_baseline, cap, conc, DEFAULT_METRIC
        )
        mean_h, ci_lo, ci_hi = bootstrap_ci(
            per_task,
            n_samples=n_samples,
            ci_level=ci_level,
            seed=seed,
        )
        bootstrap_rows.append(
            {
                "capacity_gib": cap,
                "concurrency": conc,
                "n_tasks": len(per_task),
                "mean_headroom_rel": mean_h,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }
        )

    # ---- Go/No-Go verdict ----
    # Pair each bootstrap row with its headroom table row for the threshold check.
    hr_by_cell = {(r["capacity_gib"], r["concurrency"]): r for r in headroom_rows}
    passing_cells: List[Dict[str, Any]] = []
    for br in bootstrap_rows:
        cell = (br["capacity_gib"], br["concurrency"])
        hr = hr_by_cell.get(cell)
        if hr is None:
            continue
        headroom_rel = hr["headroom_rel"]
        ci_lower = br["ci_lower"]
        if headroom_rel >= threshold_rel and ci_lower > 0:
            passing_cells.append(
                {
                    "capacity_gib": br["capacity_gib"],
                    "concurrency": br["concurrency"],
                    "headroom_rel": headroom_rel,
                    "ci_lower": ci_lower,
                    "ci_upper": br["ci_upper"],
                    "n_tasks": br["n_tasks"],
                }
            )

    verdict = "go" if passing_cells else "no_go"

    # ---- Overview stats ----
    # Block-level accesses per cell = sum of (hits + misses) over that cell's
    # episode rows. All cells replay the same trace, so this is constant
    # across cells; take the first cell as representative.
    df["_accesses"] = df["hits"].astype(int) + df["misses"].astype(int)
    first_cell_keys = (
        df[["baseline", "capacity_gib", "concurrency"]]
        .drop_duplicates()
        .iloc[0]
    )
    first_cell = df[
        (df["baseline"] == first_cell_keys["baseline"])
        & (df["capacity_gib"] == first_cell_keys["capacity_gib"])
        & (df["concurrency"] == first_cell_keys["concurrency"])
    ]
    accesses_per_cell = int(first_cell["_accesses"].sum()) if not first_cell.empty else 0
    # Episodes per cell (constant across cells; equal to unique (task, seed) pairs).
    episodes_per_cell = int(first_cell["task_id"].astype(str).str.cat(first_cell["seed"].astype(str), sep="_").nunique())
    overview = {
        "n_rows": int(len(df)),
        "n_episodes_per_cell": episodes_per_cell,
        "n_task_groups": int(df["task_id"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "capacities_gib": sorted(df["capacity_gib"].unique().tolist()),
        "concurrency_levels": sorted(df["concurrency"].unique().tolist()),
        "baselines": sorted(df["baseline"].unique().tolist()),
        "accesses_per_cell": accesses_per_cell,
        "bootstrap_samples": n_samples,
        "ci_level": ci_level,
    }

    # ---- Recommendation ----
    if verdict == "go":
        recommendation = (
            "The Oracle-Cost upper bound demonstrates a statistically "
            "significant headroom over the best simple heuristic on at "
            "least one (capacity, concurrency) cell. Proceed to **P1-A**: "
            "train a learned prefix-reuse predictor targeting the "
            "identified operating point(s), and measure how much of the "
            "oracle headroom it can capture."
        )
    else:
        recommendation = (
            "No statistically significant oracle headroom was detected "
            "on any (capacity, concurrency) cell under the current "
            "physical-prefix replay protocol. Recommended next step is "
            "**Route B**: re-examine the workload (e.g. extend to "
            "additional τ-bench domains or seeds), revisit the capacity "
            "tiers (smaller budgets amplify headroom), or relax the "
            "concurrency model before re-running G1′."
        )

    # ---- Write outputs ----
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    md_text = render_markdown(
        overview=overview,
        headroom_rows=headroom_rows,
        bootstrap_rows=bootstrap_rows,
        verdict=verdict,
        threshold_rel=threshold_rel,
        passing_cells=passing_cells,
        recommendation=recommendation,
    )
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_payload = {
        "verdict": verdict,
        "threshold_rel": threshold_rel,
        "ci_level": ci_level,
        "bootstrap_samples": n_samples,
        "bootstrap_unit": vcfg.get("bootstrap_unit", "task_group"),
        "random_seed": seed,
        "oracle_baseline": oracle_baseline,
        "simple_baselines": list(simple_baselines),
        "metric": DEFAULT_METRIC,
        "overview": overview,
        "headroom_table": headroom_rows,
        "bootstrap_table": bootstrap_rows,
        "passing_cells": passing_cells,
        "recommendation": recommendation,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    # ---- Stdout summary ----
    print("[Headroom table]")
    print(
        f"  {'cap(GiB)':>9} {'conc':>5} {'oracle':>10} {'best_simple':>12} "
        f"{'best_bl':>10} {'head_abs':>10} {'head_rel':>9}"
    )
    for r in headroom_rows:
        print(
            f"  {r['capacity_gib']:>9} {r['concurrency']:>5} "
            f"{r['oracle_miss_cost']:>10.2f} "
            f"{r['best_simple_miss_cost']:>12.2f} "
            f"{r['best_simple_baseline']:>10} "
            f"{r['headroom_abs']:>10.2f} "
            f"{r['headroom_rel']*100:>8.2f}%"
        )
    print()
    print("[Bootstrap CI]")
    print(
        f"  {'cap(GiB)':>9} {'conc':>5} {'n_tasks':>8} "
        f"{'mean':>9} {'lo':>9} {'hi':>9}"
    )
    for r in bootstrap_rows:
        print(
            f"  {r['capacity_gib']:>9} {r['concurrency']:>5} "
            f"{r['n_tasks']:>8} "
            f"{r['mean_headroom_rel']*100:>8.2f}% "
            f"{r['ci_lower']*100:>8.2f}% "
            f"{r['ci_upper']*100:>8.2f}%"
        )
    print()
    print(f"VERDICT: {verdict.upper()}")
    if passing_cells:
        print(f"  Passing cells ({len(passing_cells)}):")
        for c in passing_cells:
            print(
                f"    cap={c['capacity_gib']}GiB conc={c['concurrency']}: "
                f"headroom_rel={c['headroom_rel']*100:.2f}% "
                f"CI_lower={c['ci_lower']*100:.2f}%"
            )
    else:
        print("  No passing cells.")
    print()
    print(f"Markdown report : {args.output_md}")
    print(f"JSON verdict    : {args.output_json}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
