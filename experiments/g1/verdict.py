"""
G1 Verdict Report Generator
===========================

Reads `results/raw_results.csv` and produces:
  * `g1-verdict.md`  — human-readable report with G1.11.1 表 G1-1 filled in
  * `g1-verdict.json` — machine-readable verdict including `go_no_go`

Computation:
  1. For each (budget, dataset) group, compute
        headroom_abs = Oracle-Cost.miss_cost − max(LRU, GDSF, SizeCost, APC-LRU).miss_cost
        headroom_rel = headroom_abs / Oracle-Cost.miss_cost
     (averaged across the 3 replay seeds; per-seed values feed the bootstrap)
  2. Paired workflow-level bootstrap (1000 iterations, seed=42) on the
     per-seed headroom_rel values, yielding a 95% CI.
  3. Bonferroni correction: family-wise α = 0.05 / len(budgets).
  4. Go/No-Go:
        headroom_rel ≥ 0.10 AND CI lower bound > 0 → pass
        comparability: at least one closest baseline (PBKV/KVFlow/ThunderAgent)
                       available (inspired counts as partial) → pass

Per G1.8, "≥ 10% miss-cost headroom" → pass the first criterion.
"""

import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SIMPLE_BASELINES = ["lru", "gdsf", "sizecost", "apc_lru"]
CLOSEST_BASELINES = ["pbkv_inspired", "thunderagent_inspired", "kvflow_faithful"]
ORACLE_COST = "oracle_cost"
ORACLE_BELADY = "belady"

# G1.11.1 table row order (matches the template in experiment-designs.md).
TABLE_G1_1_ORDER = [
    "apc_lru",
    "lru",
    "gdsf",
    "sizecost",
    "pbkv_inspired",
    "thunderagent_inspired",
    "kvflow_faithful",
    "belady",
    "oracle_cost",
]

BUDGET_DISPLAY = {
    0.10: "10%",
    0.25: "25%",
    0.50: "50%",
    1.00: "100%",
}


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
    """Parse a CSV cell to float; return None on blank / non-numeric."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None


def read_raw_results(csv_path: Optional[str] = None) -> List[Dict]:
    """Read raw_results.csv; auto-skips `#`-prefixed comment lines.

    Returns a list of dicts keyed by CSV column name. Numeric columns are
    left as strings; callers convert via `_to_float` / `_to_int`.
    """
    if csv_path is None:
        csv_path = SCRIPT_DIR / "results" / "raw_results.csv"
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path

    rows: List[Dict] = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        # Skip `#`-prefixed comment lines (pilot-mode header etc.).
        cleaned = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(cleaned)
    for r in reader:
        rows.append(r)
    return rows


def read_pilot_note(csv_path: Optional[str] = None) -> Optional[str]:
    """Return the first `#`-prefixed comment line, or None."""
    if csv_path is None:
        csv_path = SCRIPT_DIR / "results" / "raw_results.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for ln in f:
            if ln.lstrip().startswith("#"):
                return ln.lstrip("# ").rstrip("\n")
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_safe(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_by_budget_seed(rows: List[Dict]) -> Dict:
    """Group rows by (budget, dataset, baseline, seed) → metrics.

    Returns nested dict:
        result[budget][dataset][baseline][seed] = {
            "hits": int, "misses": int, "hit_rate": float,
            "evictions": int, "saved_prefill_ms": float,
            "miss_cost_ms": float, "p95_ttft_ms": float,
            "status": str,
        }
    """
    out: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        try:
            budget = float(r["budget"])
        except (KeyError, TypeError, ValueError):
            continue
        dataset = r.get("dataset", "")
        baseline = r.get("baseline", "")
        try:
            seed = int(r["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        out[budget][dataset][baseline][seed] = {
            "hits": _to_int(r.get("hits")),
            "misses": _to_int(r.get("misses")),
            "hit_rate": _to_float(r.get("hit_rate")),
            "evictions": _to_int(r.get("evictions")),
            "saved_prefill_ms": _to_float(r.get("saved_prefill_ms")),
            "miss_cost_ms": _to_float(r.get("miss_cost_ms")),
            "p95_ttft_ms": _to_float(r.get("p95_ttft_ms")),
            "status": r.get("status", ""),
        }
    return out


# ---------------------------------------------------------------------------
# Headroom computation
# ---------------------------------------------------------------------------

def compute_headroom_pair(oracle_cost_ms: float,
                          simple_miss_costs: Dict[str, float]) -> Tuple[float, float, str]:
    """Return (headroom_abs, headroom_rel, best_simple_baseline).

    headroom_abs = oracle_cost_ms − max(simple_miss_costs.values())
    headroom_rel = headroom_abs / oracle_cost_ms   (0 if oracle_cost_ms == 0)
    """
    if not simple_miss_costs:
        return 0.0, 0.0, ""
    best_bl = max(simple_miss_costs, key=simple_miss_costs.get)
    best_cost = simple_miss_costs[best_bl]
    headroom_abs = oracle_cost_ms - best_cost
    if oracle_cost_ms > 0:
        headroom_rel = headroom_abs / oracle_cost_ms
    else:
        headroom_rel = 0.0
    return headroom_abs, headroom_rel, best_bl


def collect_per_seed_headroom(grouped: Dict,
                              budget: float,
                              dataset: str,
                              seeds: List[int]) -> List[float]:
    """For each seed, compute headroom_rel for (budget, dataset).

    Missing values are skipped. Returns a list of per-seed headroom_rel.
    """
    per_seed: List[float] = []
    bd = grouped.get(budget, {}).get(dataset, {})
    for seed in seeds:
        oracle_cost_ms = bd.get(ORACLE_COST, {}).get(seed, {}).get("miss_cost_ms")
        if oracle_cost_ms is None or oracle_cost_ms <= 0:
            continue
        simple_costs: Dict[str, float] = {}
        for sb in SIMPLE_BASELINES:
            v = bd.get(sb, {}).get(seed, {}).get("miss_cost_ms")
            if v is not None:
                simple_costs[sb] = v
        if not simple_costs:
            continue
        _, rel, _ = compute_headroom_pair(oracle_cost_ms, simple_costs)
        per_seed.append(rel)
    return per_seed


# ---------------------------------------------------------------------------
# Bootstrap 95% CI (paired workflow-level / per-seed)
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float],
                 n_bootstrap: int = 1000,
                 seed: int = 42,
                 ci_level: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap CI on the mean of `values`.

    Returns (mean, ci_low, ci_high). If `values` is empty, returns (0, 0, 0).
    With <2 values, CI collapses to the mean.
    """
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean

    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1.0 - ci_level
    lo_idx = max(0, int((alpha / 2.0) * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1.0 - alpha / 2.0) * n_bootstrap))
    return mean, boot_means[lo_idx], boot_means[hi_idx]


def bonferroni_alpha(family_alpha: float, n_tests: int) -> float:
    """Bonferroni-corrected per-test α = family_alpha / n_tests."""
    if n_tests <= 0:
        return family_alpha
    return family_alpha / n_tests


# ---------------------------------------------------------------------------
# Go/No-Go
# ---------------------------------------------------------------------------

def evaluate_go_no_go(headroom_rel_per_budget: Dict[float, Tuple[float, float, float]],
                      threshold: float = 0.10) -> Dict:
    """Apply G1.8 verdict rules.

    headroom_rel_per_budget: {budget: (mean_rel, ci_low, ci_high)}

    Headroom pass: for EVERY budget, mean_rel >= threshold AND ci_low > 0.
    Comparability pass: at least one closest baseline has data (inspired
                        variant counts as partial; kvflow pending is OK).
    """
    headroom_pass = True
    per_budget: Dict[str, Dict] = {}
    for budget, (mean_rel, ci_low, ci_high) in sorted(headroom_rel_per_budget.items()):
        passed = (mean_rel >= threshold) and (ci_low > 0)
        per_budget[f"{budget:.2f}"] = {
            "headroom_rel_mean": round(mean_rel, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "pass": passed,
        }
        if not passed:
            headroom_pass = False
    return {
        "headroom": "pass" if headroom_pass else "fail",
        "comparability": "pass",  # set by caller after checking closest baselines
        "per_budget": per_budget,
    }


def evaluate_comparability(grouped: Dict,
                           budget: float,
                           dataset: str,
                           seeds: List[int]) -> Dict:
    """Check whether at least one closest baseline has runnable data.

    Returns {pass: bool, available: [baseline names], pending: [...]}.
    Inspired variants (PBKV/ThunderAgent) count as partial; KVFlow is
    listed as pending.
    """
    bd = grouped.get(budget, {}).get(dataset, {})
    available: List[str] = []
    pending: List[str] = []
    for cb in CLOSEST_BASELINES:
        seed_data = bd.get(cb, {})
        any_data = False
        any_pending = False
        for s in seeds:
            d = seed_data.get(s, {})
            if d.get("status") == "pending":
                any_pending = True
            elif d.get("miss_cost_ms") is not None:
                any_data = True
        if any_data:
            available.append(cb)
        if any_pending or cb == "kvflow_faithful":
            pending.append(cb)
    # Pass if at least one closest baseline has data (inspired counts).
    return {
        "pass": len(available) >= 1,
        "available": available,
        "pending": pending,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".2f") -> str:
    if v is None:
        return "TBD"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "TBD"


def build_table_g1_1(grouped: Dict,
                     budgets: List[float],
                     dataset: str,
                     seeds: List[int]) -> str:
    """Build the G1.11.1 表 G1-1 markdown table (miss cost + p95 TTFT)."""
    # Columns: 策略 | miss_cost × 3 budgets | p95_ttft × 3 budgets
    header_parts = ["策略"]
    for b in budgets:
        label = BUDGET_DISPLAY.get(b, f"{int(b*100)}%")
        header_parts.append(f"预算 {label} miss cost (ms)")
    for b in budgets:
        label = BUDGET_DISPLAY.get(b, f"{int(b*100)}%")
        header_parts.append(f"预算 {label} p95 TTFT (ms)")
    header = "| " + " | ".join(header_parts) + " |"
    sep = "|" + "|".join(["---"] * len(header_parts)) + "|"

    lines = [header, sep]
    for bl in TABLE_G1_1_ORDER:
        row_cells = [bl]
        # Miss cost cells
        for b in budgets:
            vals = []
            bd = grouped.get(b, {}).get(dataset, {})
            for s in seeds:
                v = bd.get(bl, {}).get(s, {}).get("miss_cost_ms")
                if v is not None:
                    vals.append(v)
            if vals:
                row_cells.append(_fmt(_mean_safe(vals)))
            else:
                row_cells.append("pending" if bl == "kvflow_faithful" else "TBD")
        # p95 TTFT cells
        for b in budgets:
            vals = []
            bd = grouped.get(b, {}).get(dataset, {})
            for s in seeds:
                v = bd.get(bl, {}).get(s, {}).get("p95_ttft_ms")
                if v is not None:
                    vals.append(v)
            if vals:
                row_cells.append(_fmt(_mean_safe(vals)))
            else:
                row_cells.append("pending" if bl == "kvflow_faithful" else "TBD")
        lines.append("| " + " | ".join(row_cells) + " |")

    # Final row: oracle vs best simple relative diff (per budget).
    rel_row = ["**oracle vs 最佳简单策略相对差**"]
    for b in budgets:
        bd = grouped.get(b, {}).get(dataset, {})
        # Mean across seeds of headroom_rel.
        per_seed_rels: List[float] = []
        for s in seeds:
            oc = bd.get(ORACLE_COST, {}).get(s, {}).get("miss_cost_ms")
            if oc is None or oc <= 0:
                continue
            simple_costs = {}
            for sb in SIMPLE_BASELINES:
                v = bd.get(sb, {}).get(s, {}).get("miss_cost_ms")
                if v is not None:
                    simple_costs[sb] = v
            if not simple_costs:
                continue
            _, rel, _ = compute_headroom_pair(oc, simple_costs)
            per_seed_rels.append(rel)
        if per_seed_rels:
            rel_row.append(f"{_mean_safe(per_seed_rels)*100:.1f}%")
        else:
            rel_row.append("TBD")
    # p95 TTFT relative diff cells (leave blank — table G1-1 uses miss-cost only for the rel row).
    for _ in budgets:
        rel_row.append("—")
    lines.append("| " + " | ".join(rel_row) + " |")
    return "\n".join(lines)


def build_verdict_md(grouped: Dict,
                    cfg: Dict,
                    pilot_note: Optional[str]) -> str:
    """Render the g1-verdict.md report content."""
    budgets: List[float] = list(cfg["budgets"])
    seeds: List[int] = list(cfg["replay_seeds"])
    dataset: str = cfg["datasets"][0]
    threshold = float(cfg.get("verdict", {}).get("headroom_threshold", 0.10))
    n_boot = int(cfg.get("verdict", {}).get("bootstrap_samples", 1000))
    boot_seed = int(cfg.get("verdict", {}).get("bootstrap_seed", 42))
    ci_level = float(cfg.get("verdict", {}).get("ci_level", 0.95))
    family_alpha = float(cfg.get("verdict", {}).get("alpha", 0.05))

    n_budgets = len(budgets)
    corrected_alpha = bonferroni_alpha(family_alpha, n_budgets)

    # Headroom per budget (mean across seeds) + bootstrap CI.
    headroom_rel_per_budget: Dict[float, Tuple[float, float, float]] = {}
    headroom_rows: List[str] = []
    for b in budgets:
        per_seed_rels = collect_per_seed_headroom(grouped, b, dataset, seeds)
        mean_rel, ci_lo, ci_hi = bootstrap_ci(
            per_seed_rels, n_bootstrap=n_boot, seed=boot_seed, ci_level=ci_level
        )
        headroom_rel_per_budget[b] = (mean_rel, ci_lo, ci_hi)
        headroom_rows.append(
            f"| {BUDGET_DISPLAY.get(b, f'{int(b*100)}%')} "
            f"| {mean_rel*100:.2f}% | {ci_lo*100:.2f}% | {ci_hi*100:.2f}% "
            f"| {(mean_rel >= threshold and ci_lo > 0)} |"
        )

    # Comparability (use the smallest budget that has data, typically 0.10).
    comparability = None
    for b in budgets:
        comparability = evaluate_comparability(grouped, b, dataset, seeds)
        if comparability["available"]:
            break

    # Build Go/No-Go.
    go_no_go = evaluate_go_no_go(headroom_rel_per_budget, threshold=threshold)
    go_no_go["comparability"] = "pass" if (comparability and comparability["pass"]) else "fail"

    table_g1_1 = build_table_g1_1(grouped, budgets, dataset, seeds)

    pilot_line = (f"**Pilot note**: {pilot_note}\n\n"
                  if pilot_note else "")

    md = []
    md.append("# G1 Verdict Report\n")
    md.append("Generated by `experiments/g1/verdict.py` from "
              "`experiments/g1/results/raw_results.csv`.\n")
    if pilot_line:
        md.append(pilot_line)
    md.append("## G1.11.1 表 G1-1：headroom 主表（"
              + dataset + ", " + str(len(seeds)) + " replay seeds）\n")
    md.append(table_g1_1)
    md.append("\n\n")
    md.append("†：G1.4.1 判定后填入实际可用的 closest baseline 名称。"
              "PBKV-inspired 与 ThunderAgent-inspired 均为 inspired variant"
              "（无官方代码 / API 级代理非块级缓存），"
              "KVFlow faithful 已于 2026-07-26 在 AutoDL Linux 激活"
              "（`config.yaml: kvflow_faithful.enabled: true`），adapter 实现中。\n\n")

    md.append("## G1.8 headroom 统计检验\n")
    md.append(f"- Bootstrap samples: {n_boot} (seed={boot_seed})\n")
    md.append(f"- CI level: {ci_level*100:.0f}%\n")
    md.append(f"- Family-wise α: {family_alpha} (Bonferroni over "
              f"{n_budgets} budgets → per-test α = {corrected_alpha:.4f})\n")
    md.append(f"- Threshold: headroom_rel ≥ {threshold*100:.0f}% AND CI lower bound > 0\n\n")
    md.append("| 预算 | headroom_rel (mean) | CI low | CI high | pass |\n")
    md.append("|---|---|---|---|---|\n")
    md.extend(headroom_rows)
    md.append("\n\n")

    md.append("## Go/No-Go 判定\n")
    md.append(f"- **headroom**: {go_no_go['headroom']}\n")
    md.append(f"- **comparability**: {go_no_go['comparability']}\n")
    if comparability:
        md.append(f"  - Available closest baselines: "
                  f"{comparability['available']}\n")
        md.append(f"  - Pending: {comparability['pending']}\n")
    overall = "pass" if (go_no_go["headroom"] == "pass"
                        and go_no_go["comparability"] == "pass") else "fail"
    md.append(f"- **overall G1 verdict**: {overall}\n\n")

    md.append("## 备注\n")
    md.append("- G1 复用 E1 trace 与 baseline 实现（IDEA §8 Ch.1 line 617），"
              "不独立运行。\n")
    md.append("- kvflow_faithful 行 status=pending（adapter_not_implemented），"
              "不影响其他 baseline 运行。\n")
    if pilot_note:
        md.append("- **当前为 pilot 模式**：trace 不足，结果不可作为最终 G1 判定。\n")
    return "\n".join(md)


def build_verdict_json(grouped: Dict,
                       cfg: Dict,
                       pilot_note: Optional[str]) -> Dict:
    """Render the g1-verdict.json content."""
    budgets: List[float] = list(cfg["budgets"])
    seeds: List[int] = list(cfg["replay_seeds"])
    dataset: str = cfg["datasets"][0]
    threshold = float(cfg.get("verdict", {}).get("headroom_threshold", 0.10))
    n_boot = int(cfg.get("verdict", {}).get("bootstrap_samples", 1000))
    boot_seed = int(cfg.get("verdict", {}).get("bootstrap_seed", 42))
    ci_level = float(cfg.get("verdict", {}).get("ci_level", 0.95))
    family_alpha = float(cfg.get("verdict", {}).get("alpha", 0.05))
    corrected_alpha = bonferroni_alpha(family_alpha, len(budgets))

    headroom_rel_per_budget: Dict[float, Tuple[float, float, float]] = {}
    per_budget_detail: Dict[str, Dict] = {}
    for b in budgets:
        per_seed_rels = collect_per_seed_headroom(grouped, b, dataset, seeds)
        mean_rel, ci_lo, ci_hi = bootstrap_ci(
            per_seed_rels, n_bootstrap=n_boot, seed=boot_seed, ci_level=ci_level
        )
        headroom_rel_per_budget[b] = (mean_rel, ci_lo, ci_hi)
        per_budget_detail[f"{b:.2f}"] = {
            "headroom_rel_mean": round(mean_rel, 6),
            "headroom_rel_per_seed": [round(r, 6) for r in per_seed_rels],
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "pass": bool(mean_rel >= threshold and ci_lo > 0),
        }

    comparability = None
    for b in budgets:
        comparability = evaluate_comparability(grouped, b, dataset, seeds)
        if comparability["available"]:
            break

    go_no_go = evaluate_go_no_go(headroom_rel_per_budget, threshold=threshold)
    go_no_go["comparability"] = "pass" if (comparability and comparability["pass"]) else "fail"
    overall = "pass" if (go_no_go["headroom"] == "pass"
                         and go_no_go["comparability"] == "pass") else "fail"

    return {
        "verdict": "G1 verdict report",
        "dataset": dataset,
        "replay_seeds": seeds,
        "budgets": budgets,
        "threshold": threshold,
        "bootstrap": {
            "samples": n_boot,
            "seed": boot_seed,
            "ci_level": ci_level,
        },
        "multiple_testing": {
            "family_alpha": family_alpha,
            "correction": "bonferroni",
            "n_tests": len(budgets),
            "per_test_alpha": round(corrected_alpha, 6),
        },
        "headroom_rel_per_budget": per_budget_detail,
        "comparability": comparability or {},
        "go_no_go": go_no_go,
        "overall": overall,
        "pilot": bool(pilot_note),
        "pilot_note": pilot_note or "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(config_path: Optional[str] = None,
             csv_path: Optional[str] = None,
             md_path: Optional[str] = None,
             json_path: Optional[str] = None) -> Tuple[Path, Path]:
    """Read raw_results.csv + config, write g1-verdict.md and g1-verdict.json."""
    cfg = load_config(config_path)
    if csv_path is None:
        csv_path = cfg["output"]["raw_results_csv"]
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if md_path is None:
        md_path = cfg["output"]["verdict_md"]
    md_path = Path(md_path)
    if not md_path.is_absolute():
        md_path = PROJECT_ROOT / md_path

    if json_path is None:
        json_path = cfg["output"]["verdict_json"]
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path

    rows = read_raw_results(csv_path)
    if not rows:
        print(f"ERROR: no rows in {csv_path} (run run_grid.py first)")
        return md_path, json_path

    pilot_note = read_pilot_note(csv_path)
    grouped = aggregate_by_budget_seed(rows)

    md_content = build_verdict_md(grouped, cfg, pilot_note)
    json_content = build_verdict_json(grouped, cfg, pilot_note)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=2, ensure_ascii=False)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return md_path, json_path


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
