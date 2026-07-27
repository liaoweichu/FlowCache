"""Unit tests for ``experiments/g1prime/verdict.py``.

Tests verify:

  1. ``compute_headroom_table`` returns ``headroom_abs = best_simple − oracle``.
  2. Oracle should be at least as good as the best simple heuristic
     (``headroom_abs ≥ 0``) on realistic cost-aware replay data.
  3. ``compute_per_task_headroom`` clusters by ``task_id`` (collapses seeds
     within a task to a single headroom value).
  4. Go/No-Go verdict is ``go`` when ``headroom_rel ≥ 10%`` AND CI lower > 0.
  5. Go/No-Go verdict is ``no_go`` when ``headroom_rel < 10%``.

Mock CSVs are written to ``tmp_path`` and consumed via ``verdict.main()``
with monkey-patched ``sys.argv`` so the actual ``parse_args`` path is
exercised end-to-end.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# conftest.py already puts g1prime/ on sys.path.
import verdict


G1PRIME_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = G1PRIME_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Mock CSV builders
# ---------------------------------------------------------------------------
CSV_COLUMNS = (
    "baseline,capacity_gib,concurrency,task_id,seed,domain,"
    "hits,misses,hit_rate,evictions,miss_prefill_tokens,"
    "miss_prefill_ms,p50_ttft_ms,p95_ttft_ms,resume_hit_rate,status\n"
)


def _row(baseline, cap, conc, task, seed, miss_ms, hits=80, misses=20):
    """Build one CSV row as a string. ``miss_prefill_ms`` is the key field."""
    return (
        f"{baseline},{cap},{conc},{task},{seed},airline,"
        f"{hits},{misses},0.8,0,320,"
        f"{miss_ms},5.0,10.0,0.8,ok\n"
    )


def _build_passing_csv(path: Path) -> None:
    """Build a mock CSV where Oracle beats simple heuristics by ≥10%.

    Per (cap=1, conc=1) cell, 2 task_ids × 2 seeds = 4 episodes per baseline.
    Per-task per-baseline means:
      - task_a (seeds 1, 2): simple ∈ {100, 110} → mean 105
      - task_b (seeds 1, 2): simple ∈ {90,  120} → mean 105
    Oracle: 70 (mean) → headroom_rel = (105-70)/70 = 50% ≥ 10%.
    """
    # Per-task per-seed miss_prefill_ms for simple heuristics (all 4 tied).
    simple_costs = {
        ("task_a", 1): 100,
        ("task_a", 2): 110,
        ("task_b", 1): 90,
        ("task_b", 2): 120,
    }
    rows = [CSV_COLUMNS]
    for task in ("task_a", "task_b"):
        for seed in (1, 2):
            for baseline in ("lru", "gdsf", "sizecost", "apc_lru"):
                rows.append(_row(baseline, 1, 1, task, seed,
                                 simple_costs[(task, seed)]))
            rows.append(_row("belady",      1, 1, task, seed, 80))
            rows.append(_row("oracle_cost", 1, 1, task, seed, 70))
    path.write_text("".join(rows), encoding="utf-8")


def _build_failing_csv(path: Path) -> None:
    """Build a mock CSV where Oracle does NOT beat simple heuristics.

    All baselines have identical miss_prefill_ms = 100 → headroom_rel = 0%.
    """
    rows = [CSV_COLUMNS]
    for task in ("task_a", "task_b"):
        for seed in (1, 2):
            for baseline in ("lru", "gdsf", "sizecost", "apc_lru",
                             "belady", "oracle_cost"):
                rows.append(_row(baseline, 1, 1, task, seed, 100))
    path.write_text("".join(rows), encoding="utf-8")


def _build_mixed_csv(path: Path) -> None:
    """Build a CSV with mixed per-task headroom for direct function tests.

    Layout (cap=1, conc=1, 2 tasks × 2 seeds):
      - task_a: lru=100/110 (mean 105), oracle=70/70 (mean 70)
                → headroom_rel = (105-70)/70 = 0.5
      - task_b: lru=80/80   (mean 80),  oracle=80/80 (mean 80)
                → headroom_rel = 0.0
    """
    rows = [CSV_COLUMNS]
    task_a_simple = {1: 100, 2: 110}
    task_b_simple = {1: 80,  2: 80}
    for task, simple_map in (("task_a", task_a_simple), ("task_b", task_b_simple)):
        for seed in (1, 2):
            for baseline in ("lru", "gdsf", "sizecost", "apc_lru"):
                rows.append(_row(baseline, 1, 1, task, seed, simple_map[seed]))
            rows.append(_row("belady",     1, 1, task, seed, 80))
            rows.append(_row("oracle_cost", 1, 1, task, seed,
                             70 if task == "task_a" else 80))
    path.write_text("".join(rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. compute_headroom_table — headroom_abs = best_simple − oracle
# ---------------------------------------------------------------------------
def test_headroom_calculation(tmp_path):
    """Verify ``headroom_abs = best_simple_miss − oracle_miss`` and
    ``headroom_rel = headroom_abs / oracle_miss`` (positive when oracle wins).
    """
    csv_path = tmp_path / "raw_results.csv"
    _build_passing_csv(csv_path)

    df = pd.read_csv(csv_path)
    rows = verdict.compute_headroom_table(
        df,
        simple_baselines=("lru", "gdsf", "sizecost", "apc_lru"),
        oracle_baseline="oracle_cost",
        metric="miss_prefill_ms",
    )
    assert len(rows) == 1, f"Expected 1 cell, got {len(rows)}"

    r = rows[0]
    assert r["capacity_gib"] == 1.0
    assert r["concurrency"] == 1

    # Per-baseline means across all 4 episodes (2 tasks × 2 seeds):
    #   lru/gdsf/sizecost/apc_lru = (100+110+90+120)/4 = 105
    #   oracle_cost = (70+70+70+70)/4 = 70
    assert abs(r["oracle_miss_cost"]      - 70.0) < 1e-9
    assert abs(r["best_simple_miss_cost"] - 105.0) < 1e-9

    # headroom_abs = 105 − 70 = 35 ; headroom_rel = 35 / 70 = 0.5
    assert abs(r["headroom_abs"] - 35.0) < 1e-9
    assert abs(r["headroom_rel"] - 0.5)  < 1e-9

    # The arg-min simple baseline is recorded (all four are tied at 105).
    assert r["best_simple_baseline"] in ("lru", "gdsf", "sizecost", "apc_lru")


# ---------------------------------------------------------------------------
# 2. Oracle ≥ best simple (headroom_abs ≥ 0)
# ---------------------------------------------------------------------------
def test_headroom_sign_oracle_better(tmp_path):
    """Oracle should not be worse than the best simple heuristic on
    cost-aware replay data: ``headroom_abs ≥ 0`` (structurally, the oracle
    has the option to mimic any simple heuristic).
    """
    csv_path = tmp_path / "raw_results.csv"
    _build_mixed_csv(csv_path)

    df = pd.read_csv(csv_path)
    rows = verdict.compute_headroom_table(
        df,
        simple_baselines=("lru", "gdsf", "sizecost", "apc_lru"),
        oracle_baseline="oracle_cost",
        metric="miss_prefill_ms",
    )
    assert len(rows) == 1
    r = rows[0]
    # Oracle mean = (70+70+80+80)/4 = 75 ; best simple mean = (100+110+80+80)/4 = 92.5
    # → headroom_abs = 17.5 ≥ 0 ; headroom_rel = 17.5/75 ≈ 0.2333
    assert r["headroom_abs"] >= 0.0
    assert r["headroom_rel"] >= 0.0


# ---------------------------------------------------------------------------
# 3. compute_per_task_headroom — cluster by task_id
# ---------------------------------------------------------------------------
def test_per_task_headroom_clusters_by_task_id(tmp_path):
    """``compute_per_task_headroom`` returns one entry per ``task_id``,
    each computed from the per-baseline mean across that task's seeds
    (NOT one entry per (task_id, seed) episode).
    """
    csv_path = tmp_path / "raw_results.csv"
    _build_mixed_csv(csv_path)

    df = pd.read_csv(csv_path)
    per_task = verdict.compute_per_task_headroom(
        df,
        simple_baselines=("lru", "gdsf", "sizecost", "apc_lru"),
        oracle_baseline="oracle_cost",
        capacity_gib=1.0,
        concurrency=1,
        metric="miss_prefill_ms",
    )

    # 2 task_ids → 2 entries (NOT 4 episodes).
    assert set(per_task.keys()) == {"task_a", "task_b"}

    # task_a: best_simple = (100+110)/2 = 105, oracle = (70+70)/2 = 70
    # → headroom_rel = (105-70)/70 = 0.5
    assert abs(per_task["task_a"] - 0.5) < 1e-9

    # task_b: best_simple = (80+80)/2 = 80, oracle = (80+80)/2 = 80
    # → headroom_rel = 0.0
    assert abs(per_task["task_b"] - 0.0) < 1e-9


def test_bootstrap_ci_clustering():
    """``bootstrap_ci`` resamples at the task-group level (one sample per
    task_id), not at the episode level. Each task contributes exactly one
    value to ``per_task_values``.

    With all task values equal (e.g. {a: 0.5, b: 0.5}), the bootstrap
    distribution is degenerate → mean = ci_lower = ci_upper = 0.5.
    """
    per_task = {"task_a": 0.5, "task_b": 0.5, "task_c": 0.5}
    mean, lo, hi = verdict.bootstrap_ci(per_task, n_samples=200, seed=42)
    assert abs(mean - 0.5) < 1e-9
    assert abs(lo - 0.5) < 1e-9
    assert abs(hi - 0.5) < 1e-9

    # With variance, CI should bracket the mean.
    per_task = {"t1": 0.1, "t2": 0.2, "t3": 0.3, "t4": 0.4, "t5": 0.5}
    mean, lo, hi = verdict.bootstrap_ci(per_task, n_samples=500, seed=42)
    assert lo <= mean <= hi
    assert mean > 0  # all values are positive → CI lower should be > 0


def test_bootstrap_ci_empty():
    """Empty ``per_task_values`` → ``(0, 0, 0)`` (defensive)."""
    mean, lo, hi = verdict.bootstrap_ci({})
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 4. Go/No-Go — verdict = "go" when headroom_rel ≥ 10% AND CI lower > 0
# ---------------------------------------------------------------------------
def test_go_no_go_pass(tmp_path, monkeypatch):
    """End-to-end ``verdict.main()`` returns ``go`` when ≥1 cell has
    ``headroom_rel ≥ 10%`` AND ``ci_lower > 0``.
    """
    csv_path = tmp_path / "raw_results.csv"
    md_path = tmp_path / "verdict.md"
    json_path = tmp_path / "verdict.json"
    _build_passing_csv(csv_path)

    monkeypatch.setattr(sys, "argv", [
        "verdict.py",
        "--config",     str(CONFIG_PATH),
        "--input",      str(csv_path),
        "--output-md",  str(md_path),
        "--output-json", str(json_path),
    ])
    rc = verdict.main()
    assert rc == 0

    # Read JSON verdict.
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["verdict"] == "go"
    assert len(payload["passing_cells"]) >= 1
    for cell in payload["passing_cells"]:
        assert cell["headroom_rel"] >= 0.10
        assert cell["ci_lower"] > 0

    # Markdown report should contain "GO" verdict.
    md_text = md_path.read_text(encoding="utf-8")
    assert "GO" in md_text
    assert "VERDICT" in md_text


# ---------------------------------------------------------------------------
# 5. Go/No-Go — verdict = "no_go" when headroom_rel < 10%
# ---------------------------------------------------------------------------
def test_go_no_go_fail(tmp_path, monkeypatch):
    """End-to-end ``verdict.main()`` returns ``no_go`` when no cell satisfies
    ``headroom_rel ≥ 10%`` (oracle matches but does not beat the simple
    heuristics).
    """
    csv_path = tmp_path / "raw_results.csv"
    md_path = tmp_path / "verdict.md"
    json_path = tmp_path / "verdict.json"
    _build_failing_csv(csv_path)

    monkeypatch.setattr(sys, "argv", [
        "verdict.py",
        "--config",     str(CONFIG_PATH),
        "--input",      str(csv_path),
        "--output-md",  str(md_path),
        "--output-json", str(json_path),
    ])
    rc = verdict.main()
    assert rc == 0

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["verdict"] == "no_go"
    assert payload["passing_cells"] == []

    md_text = md_path.read_text(encoding="utf-8")
    assert "NO-GO" in md_text
    assert "VERDICT" in md_text
