"""
Tests for experiments/g1/verdict.py.

Covers: headroom computation, headroom relative diff, bootstrap 95% CI
(reproducibility with fixed seed), Bonferroni correction, Go/No-Go logic.
"""

import csv
import random
import sys
from pathlib import Path

# Make experiments/g1/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import verdict as v  # noqa: E402


# ---------------------------------------------------------------------------
# SubTask 7.1: headroom computation
# ---------------------------------------------------------------------------

def test_headroom_pair_basic():
    """Oracle 100ms, max simple 80ms → headroom 20ms."""
    simple = {"lru": 80.0, "gdsf": 70.0, "sizecost": 75.0, "apc_lru": 78.0}
    abs_h, rel_h, best = v.compute_headroom_pair(100.0, simple)
    assert abs_h == 20.0
    assert best == "lru"  # 80 is the max
    assert rel_h == 0.2


def test_headroom_pair_negative_when_oracle_worse():
    """If oracle has higher miss_cost than the best simple, headroom < 0."""
    simple = {"lru": 50.0, "gdsf": 60.0}
    abs_h, rel_h, best = v.compute_headroom_pair(70.0, simple)
    assert abs_h == 10.0  # 70 − 60
    assert best == "gdsf"
    assert rel_h == 10.0 / 70.0


def test_headroom_pair_zero_oracle():
    """Oracle miss_cost = 0 → headroom_rel = 0 (no division by zero)."""
    simple = {"lru": 5.0}
    abs_h, rel_h, _ = v.compute_headroom_pair(0.0, simple)
    assert abs_h == -5.0
    assert rel_h == 0.0


def test_headroom_pair_empty_simple():
    abs_h, rel_h, best = v.compute_headroom_pair(100.0, {})
    assert abs_h == 0.0
    assert rel_h == 0.0
    assert best == ""


# ---------------------------------------------------------------------------
# SubTask 7.2: headroom relative diff
# ---------------------------------------------------------------------------

def test_headroom_rel_passes_threshold():
    """headroom_rel ≥ 0.10 → would pass G1.8 first criterion."""
    # Oracle 100, best simple 85 → rel = 0.15 → pass.
    simple = {"lru": 85.0, "gdsf": 90.0, "sizecost": 88.0, "apc_lru": 87.0}
    _, rel, _ = v.compute_headroom_pair(100.0, simple)
    assert rel >= 0.10


def test_headroom_rel_fails_threshold():
    """headroom_rel < 0.10 → fail G1.8 first criterion."""
    # Oracle 100, best simple 96 (gdsf) → rel = 0.04 → fail.
    simple = {"lru": 95.0, "gdsf": 96.0, "sizecost": 94.0, "apc_lru": 93.0}
    _, rel, _ = v.compute_headroom_pair(100.0, simple)
    assert rel < 0.10
    assert rel == 0.04  # (100 − 96) / 100


# ---------------------------------------------------------------------------
# SubTask 7.3: bootstrap 95% CI reproducibility
# ---------------------------------------------------------------------------

def test_bootstrap_ci_reproducible_with_fixed_seed():
    """Same seed → identical CI bounds."""
    values = [0.10, 0.12, 0.15, 0.09, 0.11, 0.14, 0.13, 0.10, 0.12, 0.11]
    m1, lo1, hi1 = v.bootstrap_ci(values, n_bootstrap=1000, seed=42)
    m2, lo2, hi2 = v.bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert m1 == m2
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_different_seeds_different_bounds():
    """Different seeds → generally different CI bounds (non-deterministic)."""
    values = [0.10, 0.20, 0.15, 0.30, 0.05, 0.25, 0.18, 0.22]
    _, lo1, hi1 = v.bootstrap_ci(values, n_bootstrap=1000, seed=1)
    _, lo2, hi2 = v.bootstrap_ci(values, n_bootstrap=1000, seed=2)
    # Means equal but CI bounds almost surely differ.
    assert (lo1 != lo2) or (hi1 != hi2)


def test_bootstrap_ci_mean_equals_sample_mean():
    """Bootstrap mean estimate equals the plain sample mean."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    m, lo, hi = v.bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert m == 3.0  # (1+2+3+4+5)/5
    assert lo <= m <= hi


def test_bootstrap_ci_empty_values():
    m, lo, hi = v.bootstrap_ci([], n_bootstrap=1000, seed=42)
    assert m == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_single_value():
    m, lo, hi = v.bootstrap_ci([0.42], n_bootstrap=1000, seed=42)
    assert m == 0.42
    assert lo == 0.42
    assert hi == 0.42


# ---------------------------------------------------------------------------
# SubTask 7.4: Bonferroni correction
# ---------------------------------------------------------------------------

def test_bonferroni_alpha_basic():
    """family α=0.05, n=4 budgets → per-test α=0.0125."""
    a = v.bonferroni_alpha(0.05, 4)
    assert a == 0.0125


def test_bonferroni_alpha_one_test():
    a = v.bonferroni_alpha(0.05, 1)
    assert a == 0.05


def test_bonferroni_alpha_zero_tests_safe():
    """n=0 should not divide by zero; fall back to family α."""
    a = v.bonferroni_alpha(0.05, 0)
    assert a == 0.05


def test_bonferroni_alpha_matches_g1_config():
    """G1 has 4 budget levels → corrected α = 0.05 / 4 = 0.0125."""
    cfg = v.load_config()
    n_budgets = len(cfg["budgets"])
    family_alpha = float(cfg.get("verdict", {}).get("alpha", 0.05))
    corrected = v.bonferroni_alpha(family_alpha, n_budgets)
    assert corrected == 0.0125


# ---------------------------------------------------------------------------
# SubTask 7.5: Go/No-Go logic
# ---------------------------------------------------------------------------

def test_go_no_go_passes_when_all_budgets_meet_threshold():
    """All budgets have headroom ≥ 0.10 and CI_low > 0 → pass."""
    headroom = {
        0.10: (0.15, 0.05, 0.25),
        0.25: (0.20, 0.10, 0.30),
        0.50: (0.18, 0.08, 0.28),
        1.00: (0.12, 0.02, 0.22),
    }
    result = v.evaluate_go_no_go(headroom, threshold=0.10)
    assert result["headroom"] == "pass"


def test_go_no_go_fails_when_one_budget_below_threshold():
    headroom = {
        0.10: (0.15, 0.05, 0.25),
        0.25: (0.20, 0.10, 0.30),
        0.50: (0.18, 0.08, 0.28),
        1.00: (0.05, 0.0, 0.10),  # below 0.10 threshold
    }
    result = v.evaluate_go_no_go(headroom, threshold=0.10)
    assert result["headroom"] == "fail"


def test_go_no_go_fails_when_ci_lower_bound_is_zero():
    """CI lower bound ≤ 0 → not statistically significant → fail."""
    headroom = {
        0.10: (0.15, 0.0, 0.30),  # CI_low = 0 → fail
        0.25: (0.20, 0.10, 0.30),
        0.50: (0.18, 0.08, 0.28),
        1.00: (0.12, 0.02, 0.22),
    }
    result = v.evaluate_go_no_go(headroom, threshold=0.10)
    assert result["headroom"] == "fail"


def test_go_no_go_per_budget_breakdown():
    headroom = {
        0.10: (0.05, 0.0, 0.10),    # fail (below threshold & ci_low=0)
        0.25: (0.20, 0.10, 0.30),   # pass
        0.50: (0.18, 0.08, 0.28),   # pass
        1.00: (0.12, 0.02, 0.22),   # pass
    }
    result = v.evaluate_go_no_go(headroom, threshold=0.10)
    assert result["per_budget"]["0.10"]["pass"] is False
    assert result["per_budget"]["0.25"]["pass"] is True
    assert result["headroom"] == "fail"


# ---------------------------------------------------------------------------
# Aggregation + integration: build a small fake CSV and run the pipeline.
# ---------------------------------------------------------------------------

def _write_fake_csv(path: Path):
    """Write a tiny raw_results.csv with 3 budgets × 3 seeds × 5 baselines."""
    rows = []
    # For each (budget, seed), Oracle-Cost miss_cost = 100, max simple = 80
    # → headroom_rel = 0.20 (passes 10% threshold).
    for budget in [0.10, 0.25, 0.50]:
        for seed in [1, 2, 3]:
            rows.append({"baseline": "lru", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": 100, "misses": 50, "hit_rate": 0.667,
                         "evictions": 30, "saved_prefill_ms": 1000.0,
                         "miss_cost_ms": 80.0, "p95_ttft_ms": 12.0,
                         "status": "ok"})
            rows.append({"baseline": "gdsf", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": 110, "misses": 40, "hit_rate": 0.733,
                         "evictions": 25, "saved_prefill_ms": 1100.0,
                         "miss_cost_ms": 70.0, "p95_ttft_ms": 11.0,
                         "status": "ok"})
            rows.append({"baseline": "sizecost", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": 90, "misses": 60, "hit_rate": 0.6,
                         "evictions": 35, "saved_prefill_ms": 900.0,
                         "miss_cost_ms": 75.0, "p95_ttft_ms": 13.0,
                         "status": "ok"})
            rows.append({"baseline": "apc_lru", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": 95, "misses": 55, "hit_rate": 0.633,
                         "evictions": 28, "saved_prefill_ms": 950.0,
                         "miss_cost_ms": 78.0, "p95_ttft_ms": 12.5,
                         "status": "ok"})
            rows.append({"baseline": "oracle_cost", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": 130, "misses": 20, "hit_rate": 0.867,
                         "evictions": 15, "saved_prefill_ms": 1300.0,
                         "miss_cost_ms": 100.0, "p95_ttft_ms": 8.0,
                         "status": "ok"})
            rows.append({"baseline": "kvflow_faithful", "budget": budget,
                         "dataset": "tau_bench", "seed": seed,
                         "hits": "", "misses": "", "hit_rate": "",
                         "evictions": "",
                         "saved_prefill_ms": "", "miss_cost_ms": "",
                         "p95_ttft_ms": "",
                         "status": "pending"})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=v.read_raw_results.__doc__
                                and ["baseline", "budget", "dataset", "seed",
                                     "hits", "misses", "hit_rate", "evictions",
                                     "saved_prefill_ms", "miss_cost_ms",
                                     "p95_ttft_ms", "status"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_end_to_end_pipeline_on_fake_csv(tmp_path):
    """Aggregate → headroom → bootstrap → verdict produces expected shape."""
    csv_path = tmp_path / "raw_results.csv"
    _write_fake_csv(csv_path)
    rows = v.read_raw_results(csv_path)
    assert len(rows) == 3 * 3 * 6  # 54 rows (5 baselines + kvflow pending)

    grouped = v.aggregate_by_budget_seed(rows)
    cfg = v.load_config()
    # Only use the 3 budgets present in the fake CSV.
    cfg["budgets"] = [0.10, 0.25, 0.50]
    dataset = cfg["datasets"][0]
    seeds = cfg["replay_seeds"]

    # Per-seed headroom_rel for budget 0.10 should all be 0.20
    # (oracle 100 − max simple 80 = 20, rel = 20/100 = 0.20).
    per_seed = v.collect_per_seed_headroom(grouped, 0.10, dataset, seeds)
    assert all(abs(r - 0.20) < 1e-9 for r in per_seed)
    assert len(per_seed) == 3

    # Bootstrap CI should contain 0.20.
    m, lo, hi = v.bootstrap_ci(per_seed, n_bootstrap=1000, seed=42)
    assert abs(m - 0.20) < 1e-9
    assert lo - 1e-9 <= 0.20 <= hi + 1e-9

    # Go/No-Go should pass headroom.
    headroom_per_budget = {b: v.bootstrap_ci(
        v.collect_per_seed_headroom(grouped, b, dataset, seeds),
        n_bootstrap=1000, seed=42)
        for b in cfg["budgets"]}
    result = v.evaluate_go_no_go(headroom_per_budget, threshold=0.10)
    assert result["headroom"] == "pass"


# ---------------------------------------------------------------------------
# Comparability evaluation
# ---------------------------------------------------------------------------

def test_comparability_pass_with_inspired_variants_only():
    """Inspired variants count as partial; kvflow pending → still pass."""
    grouped = {
        0.10: {
            "tau_bench": {
                "pbkv_inspired": {1: {"miss_cost_ms": 75.0, "status": "ok"}},
                "thunderagent_inspired": {1: {"miss_cost_ms": 78.0, "status": "ok"}},
                "kvflow_faithful": {1: {"status": "pending"}},
            }
        }
    }
    result = v.evaluate_comparability(grouped, 0.10, "tau_bench", [1])
    assert result["pass"] is True
    assert "pbkv_inspired" in result["available"]
    assert "kvflow_faithful" in result["pending"]


def test_comparability_fails_when_no_closest_available():
    grouped = {
        0.10: {
            "tau_bench": {
                "kvflow_faithful": {1: {"status": "pending"}},
            }
        }
    }
    result = v.evaluate_comparability(grouped, 0.10, "tau_bench", [1])
    assert result["pass"] is False
    assert result["available"] == []
    assert "kvflow_faithful" in result["pending"]
