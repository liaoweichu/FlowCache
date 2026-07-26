"""
Tests for experiments/g1/run_grid.py.

Covers: config loading, grid expansion (8 implemented × 4 budgets × 1
dataset × 3 seeds = 96 runnable + 12 kvflow pending = 108 total), CSV
output format, and KVFlow faithful skip logic.
"""

import csv
import sys
from pathlib import Path

import yaml

# Make experiments/g1/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_grid as rg  # noqa: E402


G1_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = G1_DIR / "config.yaml"


def _load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SubTask 6.1: config loading
# ---------------------------------------------------------------------------

def test_config_loads_with_required_fields():
    cfg = rg.load_config()
    assert "budgets" in cfg
    assert "replay_seeds" in cfg
    assert "datasets" in cfg
    assert "baselines" in cfg
    assert "trace_source" in cfg


def test_config_budgets_match_spec():
    cfg = _load_cfg()
    assert cfg["budgets"] == [0.10, 0.25, 0.50, 1.00]


def test_config_replay_seeds_match_spec():
    cfg = _load_cfg()
    assert cfg["replay_seeds"] == [1, 2, 3]


def test_config_datasets_tau_bench_only():
    cfg = _load_cfg()
    assert cfg["datasets"] == ["tau_bench"]


def test_config_trace_source_points_to_e1():
    cfg = _load_cfg()
    assert "experiments/e1/traces/bf16/tau_bench" in cfg["trace_source"].replace("\\", "/")


def test_config_baselines_include_all_nine():
    """Spec requires 8 implemented + kvflow_faithful = 9 enabled baselines."""
    cfg = _load_cfg()
    enabled = rg.get_enabled_baselines(cfg)
    expected = {
        "lru", "gdsf", "sizecost", "apc_lru",
        "belady", "oracle_cost",
        "pbkv_inspired", "thunderagent_inspired",
        "kvflow_faithful",
    }
    assert set(enabled) == expected
    assert len(enabled) == 9


# ---------------------------------------------------------------------------
# SubTask 6.2: grid expansion
# ---------------------------------------------------------------------------

def test_grid_expansion_96_runnable_plus_12_pending():
    """9 baselines × 4 budgets × 1 dataset × 3 seeds = 108 total combos."""
    cfg = _load_cfg()
    grid = rg.expand_grid(cfg)
    assert len(grid) == 9 * 4 * 1 * 3  # 108

    # 8 implemented × 4 × 3 = 96 runnable
    runnable = [g for g in grid if g[0] != "kvflow_faithful"]
    assert len(runnable) == 8 * 4 * 3  # 96

    # 1 × 4 × 3 = 12 pending kvflow_faithful
    pending = [g for g in grid if g[0] == "kvflow_faithful"]
    assert len(pending) == 4 * 3  # 12


def test_grid_expansion_contains_all_budgets():
    cfg = _load_cfg()
    grid = rg.expand_grid(cfg)
    budgets_in_grid = {g[1] for g in grid}
    assert budgets_in_grid == {0.10, 0.25, 0.50, 1.00}


def test_grid_expansion_contains_all_seeds():
    cfg = _load_cfg()
    grid = rg.expand_grid(cfg)
    seeds_in_grid = {g[3] for g in grid}
    assert seeds_in_grid == {1, 2, 3}


# ---------------------------------------------------------------------------
# SubTask 6.3: CSV output format
# ---------------------------------------------------------------------------

def test_csv_columns_match_spec():
    expected = [
        "baseline", "budget", "dataset", "seed",
        "hits", "misses", "hit_rate", "evictions",
        "saved_prefill_ms", "miss_cost_ms", "p95_ttft_ms",
        "status",
    ]
    assert rg.CSV_COLUMNS == expected
    assert len(rg.CSV_COLUMNS) == 12


def test_write_csv_produces_valid_csv(tmp_path):
    rows = [
        {"baseline": "lru", "budget": 0.10, "dataset": "tau_bench", "seed": 1,
         "hits": 100, "misses": 50, "hit_rate": 0.667, "evictions": 30,
         "saved_prefill_ms": 1000.0, "miss_cost_ms": 500.0,
         "p95_ttft_ms": 12.5, "status": "ok"},
        {"baseline": "kvflow_faithful", "budget": 0.10, "dataset": "tau_bench",
         "seed": 1, "status": "pending",
         "reason": "adapter_not_implemented"},
    ]
    csv_path = tmp_path / "raw_results.csv"
    rg.write_csv(rows, csv_path, pilot_note="pilot: 28/1320 episodes available")

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # First line is the pilot comment.
    assert lines[0].startswith("# pilot: 28/1320")
    # Second line is the header.
    header = lines[1].strip().split(",")
    assert header == rg.CSV_COLUMNS
    # Data rows follow.
    assert len(lines) == 2 + 2  # comment + header + 2 rows


def test_write_csv_handles_missing_columns(tmp_path):
    """KVFlow pending rows have only `status` filled; others must default."""
    rows = [
        {"baseline": "kvflow_faithful", "budget": 0.25, "dataset": "tau_bench",
         "seed": 2, "status": "pending"},
    ]
    csv_path = tmp_path / "raw.csv"
    rg.write_csv(rows, csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)
    assert len(out_rows) == 1
    assert out_rows[0]["baseline"] == "kvflow_faithful"
    assert out_rows[0]["status"] == "pending"
    # Empty cells become "" (not None).
    assert out_rows[0]["hits"] == ""
    assert out_rows[0]["miss_cost_ms"] == ""


# ---------------------------------------------------------------------------
# SubTask 6.4: KVFlow faithful skip logic
# ---------------------------------------------------------------------------

def test_make_kvflow_pending_rows_covers_full_grid():
    cfg = _load_cfg()
    rows = rg.make_kvflow_pending_rows(cfg)
    # 4 budgets × 1 dataset × 3 seeds = 12 rows
    assert len(rows) == 4 * 1 * 3
    for r in rows:
        assert r["baseline"] == "kvflow_faithful"
        assert r["status"] == "pending"
        assert r["reason"] == "adapter_not_implemented"
        # Numeric cells left blank.
        assert r["hits"] == ""
        assert r["miss_cost_ms"] == ""


def test_kvflow_pending_rows_cover_all_budgets_and_seeds():
    cfg = _load_cfg()
    rows = rg.make_kvflow_pending_rows(cfg)
    budgets = {r["budget"] for r in rows}
    seeds = {r["seed"] for r in rows}
    assert budgets == {0.10, 0.25, 0.50, 1.00}
    assert seeds == {1, 2, 3}


def test_baseline_registry_excludes_kvflow():
    """The replay registry must not contain kvflow_faithful."""
    assert "kvflow_faithful" not in rg._BASELINE_REGISTRY
    # All 8 implemented baselines are registered.
    assert set(rg._BASELINE_REGISTRY.keys()) == {
        "lru", "gdsf", "sizecost", "apc_lru",
        "pbkv_inspired", "thunderagent_inspired",
        "belady", "oracle_cost",
    }


# ---------------------------------------------------------------------------
# Trace perturbation sanity
# ---------------------------------------------------------------------------

def test_perturb_trace_preserves_intra_workflow_order():
    """Replay-seed shuffle must not reorder accesses within a workflow."""
    trace = [
        {"block_hash": "A", "workflow_id": "w1"},
        {"block_hash": "B", "workflow_id": "w1"},
        {"block_hash": "C", "workflow_id": "w1"},
        {"block_hash": "D", "workflow_id": "w2"},
        {"block_hash": "E", "workflow_id": "w2"},
    ]
    out = rg.perturb_trace(trace, seed=1)
    # Total length preserved.
    assert len(out) == len(trace)
    # w1's blocks stay in order A→B→C; w2's stay D→E.
    w1_indices = [i for i, a in enumerate(out) if a["workflow_id"] == "w1"]
    w1_blocks = [out[i]["block_hash"] for i in w1_indices]
    assert w1_blocks == ["A", "B", "C"]
    w2_indices = [i for i, a in enumerate(out) if a["workflow_id"] == "w2"]
    w2_blocks = [out[i]["block_hash"] for i in w2_indices]
    assert w2_blocks == ["D", "E"]


def test_perturb_trace_seed_reproducibility():
    """Same seed → same shuffle order (deterministic)."""
    trace = [
        {"block_hash": str(i), "workflow_id": f"w{i}"}
        for i in range(10)
    ]
    out1 = rg.perturb_trace(trace, seed=42)
    out2 = rg.perturb_trace(trace, seed=42)
    assert out1 == out2
