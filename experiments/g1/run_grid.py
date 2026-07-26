"""
G1 Grid Runner
==============

Replays the 8 E1 baselines (LRU / GDSF / SizeCost / APC-LRU / PBKV-Inspired /
ThunderAgent-Inspired / Belady / Oracle-Cost) on the τ-bench trace under
4 budget levels × 3 replay seeds, producing `results/raw_results.csv`.

G1 does NOT re-record traces (IDEA §8 Ch.1 line 617). It imports
`experiments/e1/compare_oracle.py` directly for both the baseline classes
and `build_access_trace`.

The 9th baseline, `kvflow_faithful`, has no τ-bench adapter yet; the grid
runner emits a `status=pending` row for it without blocking other baselines.

CSV columns (12):
    baseline, budget, dataset, seed, hits, misses, hit_rate, evictions,
    saved_prefill_ms, miss_cost_ms, p95_ttft_ms, status

Reproducibility: replay seeds perturb the *inter-workflow* arrival order
(per spec §0.7 open-loop replay). Block-level access within a workflow is
preserved; only the cross-workflow interleaving order is shuffled by
`random.Random(seed)`.
"""

import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Path setup: make experiments/e1/ importable so we can reuse its baselines.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent                      # experiments/g1/
_E1_DIR = _SCRIPT_DIR.parent / "e1"                                # experiments/e1/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent                          # Prefix Caching/

if str(_E1_DIR) not in sys.path:
    sys.path.insert(0, str(_E1_DIR))

# Import E1 baselines + trace builder (compare_oracle.py lives in e1/)
import compare_oracle as co  # noqa: E402
from trace_utils import load_all_trajectories  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_EPISODES = 1320          # τ-bench pass^k: 165 tasks × 8 seeds
CSV_COLUMNS = [
    "baseline", "budget", "dataset", "seed",
    "hits", "misses", "hit_rate", "evictions",
    "saved_prefill_ms", "miss_cost_ms", "p95_ttft_ms",
    "status",
]

# Map config baseline name → E1 baseline class + access-call signature.
# `args` is a list of (kwarg_name, trace_field) tuples; `access_idx` flags
# baselines that need the trace index (Belady + Oracle-Cost).
_BASELINE_REGISTRY = {
    "lru":                   {"class": "LRUCache",                "needs_idx": False, "needs_size": False, "needs_parent": False, "needs_future": False},
    "gdsf":                  {"class": "GDSFCache",                "needs_idx": False, "needs_size": False, "needs_parent": False, "needs_future": False},
    "sizecost":              {"class": "SizeCostCache",            "needs_idx": False, "needs_size": True,  "needs_parent": False, "needs_future": False},
    "apc_lru":               {"class": "APCLRUCache",              "needs_idx": False, "needs_size": False, "needs_parent": True,  "needs_future": False},
    "pbkv_inspired":         {"class": "PBKVInspiredCache",       "needs_idx": False, "needs_size": False, "needs_parent": True,  "needs_future": True},
    "thunderagent_inspired": {"class": "ThunderAgentInspiredCache","needs_idx": False, "needs_size": False, "needs_parent": True,  "needs_future": True},
    "belady":                {"class": "BeladyOracle",             "needs_idx": True,  "needs_size": False, "needs_parent": False, "needs_future": True},
    "oracle_cost":           {"class": "OracleCostCache",          "needs_idx": True,  "needs_size": False, "needs_parent": False, "needs_future": True},
}

# Closest baselines live in the baselines/ subpackage of e1.
_BASELINE_CLASSES = {
    "LRUCache":                  co.LRUCache,
    "GDSFCache":                 co.GDSFCache,
    "SizeCostCache":             co.SizeCostCache,
    "APCLRUCache":               co.APCLRUCache,
    "BeladyOracle":              co.BeladyOracle,
    "OracleCostCache":           co.OracleCostCache,
    "PBKVInspiredCache":         co.PBKVInspiredCache,
    "ThunderAgentInspiredCache": co.ThunderAgentInspiredCache,
}

# Default per-block size when trace lacks token-range info.
_DEFAULT_BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load experiments/g1/config.yaml and return parsed dict."""
    if config_path is None:
        config_path = _SCRIPT_DIR / "config.yaml"
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = _SCRIPT_DIR / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_enabled_baselines(cfg: Dict) -> List[str]:
    """Return list of baseline names whose `enabled` flag is true."""
    out = []
    for b in cfg.get("baselines", []):
        if b.get("enabled", False):
            out.append(b["name"])
    return out


def expand_grid(cfg: Dict) -> List[Tuple[str, float, str, int]]:
    """Expand the (baseline, budget, dataset, seed) cross-product.

    Returns a list of (baseline_name, budget, dataset, seed) tuples.
    With 9 baselines (incl. kvflow_faithful) × 4 budgets × 1 dataset × 3 seeds
    this yields 108 combinations; with 8 implemented baselines it is 96
    runnable + 12 pending kvflow_faithful rows.
    """
    baselines = get_enabled_baselines(cfg)
    budgets = cfg["budgets"]
    datasets = cfg["datasets"]
    seeds = cfg["replay_seeds"]
    grid = []
    for bl in baselines:
        for budget in budgets:
            for ds in datasets:
                for seed in seeds:
                    grid.append((bl, budget, ds, seed))
    return grid


# ---------------------------------------------------------------------------
# Trace loading & seed perturbation
# ---------------------------------------------------------------------------

def _resolve_trace_source(cfg: Dict) -> Path:
    """Resolve the `trace_source` path from config (may be relative)."""
    src = cfg["trace_source"]
    p = Path(src)
    if not p.is_absolute():
        p = _PROJECT_ROOT / src
    return p


def load_trajectories(cfg: Dict) -> List[Dict]:
    """Load all τ-bench trajectories from the configured trace_source."""
    trace_dir = _resolve_trace_source(cfg)
    return load_all_trajectories(str(trace_dir))


def perturb_trace(access_trace: List[Dict], seed: int) -> List[Dict]:
    """Perturb the cross-workflow interleaving of `access_trace`.

    Open-loop replay spec (§0.7): block-level accesses within a single
    workflow stay in their recorded order; only the *cross-workflow*
    arrival order is shuffled by `random.Random(seed)`.

    Concretely: group consecutive accesses by workflow_id, shuffle the
    list of (workflow_id, chunk) groups, then concatenate. Within each
    chunk the original access order is preserved.
    """
    if not access_trace:
        return access_trace

    # Group consecutive accesses by workflow_id (preserving intra-workflow order).
    chunks: List[Tuple[str, List[Dict]]] = []
    cur_wf = None
    cur_chunk: List[Dict] = []
    for acc in access_trace:
        wf = acc.get("workflow_id", "")
        if wf != cur_wf:
            if cur_chunk:
                chunks.append((cur_wf or "", cur_chunk))
            cur_wf = wf
            cur_chunk = [acc]
        else:
            cur_chunk.append(acc)
    if cur_chunk:
        chunks.append((cur_wf or "", cur_chunk))

    rng = random.Random(seed)
    rng.shuffle(chunks)

    perturbed: List[Dict] = []
    for _, chunk in chunks:
        perturbed.extend(chunk)
    return perturbed


# ---------------------------------------------------------------------------
# Baseline replay
# ---------------------------------------------------------------------------

def _compute_p95_ttft(access_trace: List[Dict], cache) -> float:
    """Compute p95 TTFT (time-to-first-token) proxy from miss events.

    A natural per-step TTFT proxy is the per-step miss cost (ms). We
    aggregate miss_cost_ms per (workflow_id, step_id) and report the 95th
    percentile across all (workflow, step) pairs.

    For oracles + heuristics that don't track per-step cost explicitly,
    we attribute each access's `prefill_ms` to a miss or hit using the
    cache's hit/miss counters after the full replay; this gives a
    distribution-free p95 estimate consistent with the trace's prefill_ms
    field. For a *hit*, the saved prefill contributes ~0 to TTFT; for a
    *miss*, the full `prefill_ms` contributes to TTFT.

    Since we already replayed the cache, we reconstruct per-step miss
    cost by re-scanning the trace using the cache's *final* hit/miss
    state is not sound (the cache state changed mid-replay). Instead,
    we recompute per-step prefill cost from the trace itself: each
    access's `prefill_ms` is treated as a per-step TTFT sample (worst
    case). This is an upper-bound proxy; the actual p95 TTFT under a
    given cache is `miss_cost_ms / total_accesses * accesses_per_step`.

    For practicality and determinism, we use: p95 over per-access
    `prefill_ms` values that occurred as misses. The cache object exposes
    `.misses` (count) and `.miss_cost_ms` (sum), but not per-miss costs;
    so we re-replay the cache once to collect per-miss prefill_ms.

    To avoid double-replay cost, the caller passes the cache **before**
    replay; this helper replays a fresh copy. To keep the implementation
    simple and correct, we instead collect per-access prefill_ms during
    the *main* replay (see `_replay_baseline`).
    """
    # Not used; p95 is computed inside _replay_baseline.
    raise NotImplementedError


def _replay_baseline(baseline_name: str,
                     capacity: int,
                     access_trace: List[Dict],
                     future_accesses: Dict[str, List[int]],
                     block_size: int) -> Dict:
    """Replay one baseline on `access_trace`, return metrics dict.

    Returns: {hits, misses, hit_rate, evictions, saved_prefill_ms,
              miss_cost_ms, p95_ttft_ms, status}
    """
    if baseline_name not in _BASELINE_REGISTRY:
        return {
            "hits": 0, "misses": 0, "hit_rate": 0.0, "evictions": 0,
            "saved_prefill_ms": 0.0, "miss_cost_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "status": f"unknown_baseline:{baseline_name}",
        }

    spec = _BASELINE_REGISTRY[baseline_name]
    cls = _BASELINE_CLASSES[spec["class"]]

    # Construct the cache. Future-aware baselines need `future_accesses`.
    if spec["needs_future"]:
        if spec["class"] == "PBKVInspiredCache":
            cache = cls(capacity, future_accesses=future_accesses, horizon=100)
        elif spec["class"] == "ThunderAgentInspiredCache":
            cache = cls(capacity, future_accesses=future_accesses, decay_rate=0.05)
        else:
            cache = cls(capacity, future_accesses)
    else:
        cache = cls(capacity)

    # Replay + collect per-miss prefill_ms for p95 TTFT.
    miss_prefills: List[float] = []
    total = len(access_trace)

    for idx, acc in enumerate(access_trace):
        bh = acc["block_hash"]
        pre = acc["prefill_ms"]
        if spec["needs_idx"]:
            hit = cache.access(bh, idx, pre)
        elif spec["needs_size"]:
            hit = cache.access(bh, pre, acc.get("size", block_size))
        elif spec["needs_parent"]:
            hit = cache.access(bh, acc.get("parent_hash", ""), pre)
        else:
            hit = cache.access(bh, pre)
        if not hit:
            miss_prefills.append(pre)

    hit_rate = (cache.hits / total) if total else 0.0
    p95_ttft_ms = _percentile(miss_prefills, 95) if miss_prefills else 0.0

    return {
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": round(hit_rate, 6),
        "evictions": cache.evictions,
        "saved_prefill_ms": round(cache.saved_prefill_ms, 2),
        "miss_cost_ms": round(cache.miss_cost_ms, 2),
        "p95_ttft_ms": round(p95_ttft_ms, 2),
        "status": "ok",
    }


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict],
              csv_path: Path,
              pilot_note: Optional[str] = None) -> None:
    """Write the raw_results CSV with optional pilot-mode header comment."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if pilot_note:
            f.write(f"# {pilot_note}\n")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            # Ensure all columns present (fill missing with empty string).
            out = {k: row.get(k, "") for k in CSV_COLUMNS}
            writer.writerow(out)


def make_kvflow_pending_rows(cfg: Dict) -> List[Dict]:
    """Generate status=pending rows for kvflow_faithful across the grid."""
    rows = []
    budgets = cfg["budgets"]
    datasets = cfg["datasets"]
    seeds = cfg["replay_seeds"]
    for budget in budgets:
        for ds in datasets:
            for seed in seeds:
                rows.append({
                    "baseline": "kvflow_faithful",
                    "budget": budget,
                    "dataset": ds,
                    "seed": seed,
                    "hits": "", "misses": "", "hit_rate": "",
                    "evictions": "",
                    "saved_prefill_ms": "", "miss_cost_ms": "",
                    "p95_ttft_ms": "",
                    "status": "pending",
                    "reason": "adapter_not_implemented",
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grid(config_path: Optional[str] = None,
             csv_path: Optional[str] = None) -> Path:
    """Run the full G1 grid and write raw_results.csv. Returns CSV path."""
    cfg = load_config(config_path)

    # Resolve output path.
    if csv_path is None:
        csv_path = cfg["output"]["raw_results_csv"]
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path

    # Load trace once (independent of seed; seed perturbs order, not content).
    trajectories = load_trajectories(cfg)
    n_episodes = len(trajectories)
    pilot_note = None
    if n_episodes < EXPECTED_EPISODES:
        pilot_note = (f"pilot: {n_episodes}/{EXPECTED_EPISODES} episodes available")
        print(f"WARNING: {pilot_note} (results are pilot-mode)")

    if not trajectories:
        print("ERROR: no trajectories found at "
              f"{_resolve_trace_source(cfg)}")
        # Still emit headers + kvflow pending rows so downstream tools work.
        rows = make_kvflow_pending_rows(cfg)
        write_csv(rows, csv_path, pilot_note=pilot_note)
        return csv_path

    # Build the canonical (seed-independent) access trace.
    full_trace = co.build_access_trace(trajectories)
    if not full_trace:
        print("ERROR: build_access_trace returned empty list")
        rows = make_kvflow_pending_rows(cfg)
        write_csv(rows, csv_path, pilot_note=pilot_note)
        return csv_path

    # Peak working set = number of distinct block_hashes across trajectories.
    peak_ws = co.compute_peak_working_set(trajectories)
    future_accesses = co.compute_future_accesses(full_trace)
    block_size = int(cfg.get("verdict", {}).get("block_size",
                                                _DEFAULT_BLOCK_SIZE))

    print(f"Loaded {n_episodes} trajectories "
          f"({len(full_trace)} accesses, {peak_ws} unique blocks)")

    # Expand grid.
    grid = expand_grid(cfg)
    print(f"Grid size: {len(grid)} combinations "
          f"({len(get_enabled_baselines(cfg))} baselines × "
          f"{len(cfg['budgets'])} budgets × "
          f"{len(cfg['datasets'])} datasets × "
          f"{len(cfg['replay_seeds'])} seeds)")

    rows: List[Dict] = []
    for (bl, budget, ds, seed) in grid:
        if bl == "kvflow_faithful":
            # Skip replay; emit pending row.
            rows.append({
                "baseline": bl, "budget": budget, "dataset": ds, "seed": seed,
                "hits": "", "misses": "", "hit_rate": "",
                "evictions": "",
                "saved_prefill_ms": "", "miss_cost_ms": "",
                "p95_ttft_ms": "",
                "status": "pending",
                "reason": "adapter_not_implemented",
            })
            continue

        # Capacity = int(budget * peak_ws), clamped to ≥ 1.
        capacity = max(1, int(budget * peak_ws))

        # Perturb trace order by replay seed.
        perturbed = perturb_trace(full_trace, seed)
        # Re-compute future_accesses for the perturbed trace
        # (Belady / Oracle-Cost / PBKV / ThunderAgent rely on it).
        perturbed_future = co.compute_future_accesses(perturbed)

        metrics = _replay_baseline(
            baseline_name=bl,
            capacity=capacity,
            access_trace=perturbed,
            future_accesses=perturbed_future,
            block_size=block_size,
        )
        rows.append({
            "baseline": bl,
            "budget": budget,
            "dataset": ds,
            "seed": seed,
            "hits": metrics["hits"],
            "misses": metrics["misses"],
            "hit_rate": metrics["hit_rate"],
            "evictions": metrics["evictions"],
            "saved_prefill_ms": metrics["saved_prefill_ms"],
            "miss_cost_ms": metrics["miss_cost_ms"],
            "p95_ttft_ms": metrics["p95_ttft_ms"],
            "status": metrics["status"],
        })

    write_csv(rows, csv_path, pilot_note=pilot_note)
    print(f"Wrote {len(rows)} rows → {csv_path}")
    return csv_path


def main() -> None:
    run_grid()


if __name__ == "__main__":
    main()
