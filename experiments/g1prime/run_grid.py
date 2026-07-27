"""
G1′ Full Grid Runner
====================
Executes the full ``baseline × capacity × concurrency × episodes`` grid
and writes one row per ``(baseline, capacity_gib, concurrency, task_id,
seed)`` episode to ``results/raw_results.csv``.

Grid size: 6 baselines × 4 capacities × 3 concurrencies = 72 cells, each
replayed over all episodes present in the per-concurrency access trace.

Reuses the baseline cache classes from :mod:`experiments.e1.compare_oracle`
(read-only w.r.t. ``experiments/e1/``):

  - Simple heuristics: ``LRUCache``, ``GDSFCache``, ``SizeCostCache``,
    ``APCLRUCache``
  - Oracle upper bounds: ``BeladyOracle``, ``OracleCostCache``

All caches expose the same interface: ``access(block_hash, ...) -> bool``
plus the counters ``hits``, ``misses``, ``evictions``, ``saved_prefill_ms``,
``miss_cost_ms``.

Per-episode metrics written to the CSV
--------------------------------------
For each ``(baseline, capacity_gib, concurrency, task_id, seed)`` row:

  - ``hits``, ``misses``, ``hit_rate`` — block-level hit/miss counts and
    ratio within this episode.
  - ``evictions`` — number of blocks evicted while servicing this
    episode's accesses (delta attribution).
  - ``miss_prefill_tokens`` — sum of ``block_token_count`` over missed
    blocks in this episode.
  - ``miss_prefill_ms`` — sum of per-request miss cost computed via
    :func:`cost_model.compute_request_miss_cost` (request-level
    attribution: ``request_prefill_ms × (missed_tokens / num_prefix_tokens)``).
  - ``p50_ttft_ms``, ``p95_ttft_ms`` — percentiles of per-request
    ``miss_prefill_ms`` within the episode (request-level TTFT).
  - ``resume_hit_rate`` — within-episode block-level hit rate (fraction
    of this episode's accesses that hit in the cache, regardless of which
    prior episode populated the entry). Equal to ``hit_rate`` at the
    episode level by definition.
  - ``status`` — ``"ok"`` for completed cells.

Memory model
------------
For each concurrency level the trace is loaded once into memory (one
``List[Dict]`` plus a ``future_accesses`` index and an ``episode_index``).
All 6 baselines × 4 capacities = 24 replays for that concurrency level
reuse the in-memory trace. At full scale (~7.6M accesses across 1320
episodes) this is ~1.5 GB per concurrency level — acceptable.

Read-only w.r.t. ``experiments/e1/``.

Usage
-----
    python experiments/g1prime/run_grid.py
    python experiments/g1prime/run_grid.py --max-episodes 3
    python experiments/g1prime/run_grid.py --baselines lru oracle_cost \\
        --capacities 1 4 --concurrency 1 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parents[1]
E1_DIR: Path = PROJECT_ROOT / "experiments" / "e1"

# Make experiments/e1/ and experiments/g1prime/ importable.
for _p in (str(E1_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse E1 baseline classes (read-only import).
from compare_oracle import (  # noqa: E402
    APCLRUCache,
    BeladyOracle,
    GDSFCache,
    LRUCache,
    OracleCostCache,
    SizeCostCache,
)

# Reuse G1′ capacity + cost model.
from capacity import gib_to_blocks  # noqa: E402
from cost_model import (  # noqa: E402
    G0_MODEL_CONFIG,
    compute_bytes_per_block,
    compute_request_miss_cost,
)


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
BASELINE_CLASSES: Dict[str, type] = {
    "lru": LRUCache,
    "gdsf": GDSFCache,
    "sizecost": SizeCostCache,
    "apc_lru": APCLRUCache,
    "belady": BeladyOracle,
    "oracle_cost": OracleCostCache,
}

# Baselines that need future_accesses (oracle upper bounds).
ORACLE_BASELINES: Tuple[str, ...] = ("belady", "oracle_cost")

# Default output column order.
CSV_FIELDS: Tuple[str, ...] = (
    "baseline",
    "capacity_gib",
    "concurrency",
    "task_id",
    "seed",
    "domain",
    "hits",
    "misses",
    "hit_rate",
    "evictions",
    "miss_prefill_tokens",
    "miss_prefill_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "resume_hit_rate",
    "status",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "G1′ full grid runner: baseline × capacity × concurrency "
            "replay producing results/raw_results.csv."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="Path to config.yaml (default: experiments/g1prime/config.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "results" / "raw_results.csv",
        help="Output CSV path (default: experiments/g1prime/results/raw_results.csv).",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Process only the first N episodes (by first appearance in "
            "each per-concurrency trace). 0 or unset = all episodes."
        ),
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Subset of baselines to run (default: all enabled in config).",
    )
    p.add_argument(
        "--capacities",
        nargs="+",
        type=float,
        default=None,
        help="Subset of capacity budgets in GiB (default: all in config).",
    )
    p.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=None,
        help="Subset of concurrency levels (default: all in config).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> Dict[str, Any]:
    """Load the YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_enabled_baselines(config: Dict[str, Any]) -> List[str]:
    """Return the list of enabled baseline names from the config.

    Honors the ``enabled`` flag on each entry under
    ``baselines.simple_heuristic`` and ``baselines.oracle``.
    """
    names: List[str] = []
    bcfg = config.get("baselines", {}) or {}
    for group in ("simple_heuristic", "oracle"):
        for entry in bcfg.get(group, []) or []:
            if entry.get("enabled", True):
                name = entry.get("name")
                if name and name in BASELINE_CLASSES:
                    names.append(name)
    return names


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
EpisodeKey = Tuple[str, Any]  # (task_id, seed)


def load_trace(
    path: Path,
    max_episodes: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, List[int]],
    Dict[EpisodeKey, List[int]],
    List[EpisodeKey],
]:
    """Load a per-concurrency access trace into memory.

    Reads the JSONL file line by line and builds:

      - ``accesses``: ``List[Dict]`` of every block access record in
        file order (which is the synthesized global interleaved order
        for c>1 traces).
      - ``future_accesses``: ``{block_hash -> [access_idx, ...]}`` with
        indices into ``accesses``; naturally sorted by construction.
        Consumed by :class:`BeladyOracle` and :class:`OracleCostCache`.
      - ``episode_index``: ``{(task_id, seed) -> [access_idx, ...]}``
        with indices into ``accesses`` for each episode.
      - ``episode_order``: ``[(task_id, seed), ...]`` in first-seen
        order. Used to honor ``--max-episodes`` and to keep CSV rows
        deterministic.

    Args:
        path: Path to ``access_trace_c{c}.jsonl``.
        max_episodes: If set and > 0, only the first N episodes (by
            first appearance in the file) are retained. All accesses
            from later episodes are dropped.

    Returns:
        ``(accesses, future_accesses, episode_index, episode_order)``.
    """
    accesses: List[Dict[str, Any]] = []
    future: Dict[str, List[int]] = {}
    episode_index: Dict[EpisodeKey, List[int]] = {}
    episode_order: List[EpisodeKey] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ep_key: EpisodeKey = (
                rec.get("task_id", ""),
                rec.get("seed", 0),
            )
            is_new_episode = ep_key not in episode_index
            if is_new_episode:
                if (
                    max_episodes is not None
                    and max_episodes > 0
                    and len(episode_order) >= max_episodes
                ):
                    # Skip accesses from episodes beyond the limit.
                    continue
                episode_index[ep_key] = []
                episode_order.append(ep_key)

            idx = len(accesses)
            accesses.append(rec)
            bh = rec.get("block_hash", "")
            future.setdefault(bh, []).append(idx)
            episode_index[ep_key].append(idx)

    return accesses, future, episode_index, episode_order


# ---------------------------------------------------------------------------
# Percentile helper (linear interpolation, numpy-default compatible)
# ---------------------------------------------------------------------------
def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Return the ``p``-th percentile (0..100) of a sorted sequence.

    Uses linear interpolation between closest ranks (matches numpy's
    default ``linear`` method). Returns ``0.0`` for an empty sequence.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    lo = float(sorted_values[f])
    hi = float(sorted_values[c])
    return lo + (hi - lo) * (k - f)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay_baseline(
    baseline_name: str,
    capacity_gib: float,
    concurrency: int,
    capacity_blocks: int,
    accesses: List[Dict[str, Any]],
    future_accesses: Dict[str, List[int]],
    episode_index: Dict[EpisodeKey, List[int]],
    episode_order: List[EpisodeKey],
    block_size: int = 16,
) -> List[Dict[str, Any]]:
    """Replay ``accesses`` through one baseline cache.

    Args:
        baseline_name: Key in :data:`BASELINE_CLASSES`.
        capacity_gib: Capacity in GiB (for the CSV row only).
        concurrency: Concurrency level (for the CSV row only).
        capacity_blocks: Cache capacity in block count.
        accesses: The in-memory access trace (file order).
        future_accesses: Pre-computed block_hash -> [access_idx, ...];
            required by oracle baselines, ignored by heuristics.
        episode_index: ``(task_id, seed) -> [access_idx, ...]``.
        episode_order: Ordered list of episode keys.
        block_size: Block size in tokens (fallback for missing
            ``block_token_count``).

    Returns:
        List of per-episode row dicts ready for ``csv.DictWriter``.
    """
    cls = BASELINE_CLASSES[baseline_name]
    if baseline_name in ORACLE_BASELINES:
        cache = cls(capacity_blocks, future_accesses)
    else:
        cache = cls(capacity_blocks)

    # Per-episode accumulators.
    ep_hits: Dict[EpisodeKey, int] = {ek: 0 for ek in episode_order}
    ep_misses: Dict[EpisodeKey, int] = {ek: 0 for ek in episode_order}
    ep_evictions: Dict[EpisodeKey, int] = {ek: 0 for ek in episode_order}
    ep_miss_tokens: Dict[EpisodeKey, int] = {ek: 0 for ek in episode_order}
    ep_domains: Dict[EpisodeKey, str] = {ek: "" for ek in episode_order}
    # Per-episode list of per-request aggregation dicts.
    ep_requests: Dict[EpisodeKey, List[Dict[str, Any]]] = {
        ek: [] for ek in episode_order
    }

    last_evictions = 0
    current_request_id: Optional[str] = None
    current_ep_key: Optional[EpisodeKey] = None
    current_request: Optional[Dict[str, Any]] = None

    def close_request() -> None:
        if current_request is not None and current_ep_key is not None:
            ep_requests[current_ep_key].append(current_request)

    for idx, acc in enumerate(accesses):
        bh = acc.get("block_hash", "")
        prefill_ms = float(acc.get("prefill_ms", 0.0) or 0.0)
        parent_hash = acc.get("parent_hash", "") or ""
        size = int(acc.get("block_token_count", block_size) or block_size)
        num_prefix_tokens = int(acc.get("num_prefix_tokens", 0) or 0)

        rid = acc.get("request_id", "")
        ep_key: EpisodeKey = (
            acc.get("task_id", ""),
            acc.get("seed", 0),
        )
        if ep_domains.get(ep_key, "") == "":
            ep_domains[ep_key] = acc.get("domain", "")

        # Detect request boundary.
        if rid != current_request_id:
            close_request()
            current_request_id = rid
            current_ep_key = ep_key
            current_request = {
                "missed_blocks": [],  # list of {'block_token_count': int}
                "total_tokens": num_prefix_tokens,
                "prefill_ms": prefill_ms,
            }

        # Dispatch to the cache.
        if baseline_name in ORACLE_BASELINES:
            hit = cache.access(bh, idx, prefill_ms=prefill_ms)
        elif baseline_name == "apc_lru":
            hit = cache.access(bh, parent_hash=parent_hash, prefill_ms=prefill_ms)
        elif baseline_name == "sizecost":
            hit = cache.access(bh, prefill_ms=prefill_ms, size=size)
        else:  # lru, gdsf
            hit = cache.access(bh, prefill_ms=prefill_ms)

        if hit:
            ep_hits[ep_key] += 1
        else:
            ep_misses[ep_key] += 1
            ep_miss_tokens[ep_key] += size
            if current_request is not None:
                current_request["missed_blocks"].append(
                    {"block_token_count": size}
                )

        # Attribute evictions triggered by this access to its episode.
        new_ev = cache.evictions - last_evictions
        if new_ev > 0:
            ep_evictions[ep_key] += new_ev
            last_evictions = cache.evictions

    close_request()

    # Build per-episode rows.
    rows: List[Dict[str, Any]] = []
    for ep_key in episode_order:
        hits = ep_hits[ep_key]
        misses = ep_misses[ep_key]
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        # Per-request miss_prefill_ms (request-level TTFT attribution).
        request_miss_ms: List[float] = []
        for r in ep_requests[ep_key]:
            m_ms = compute_request_miss_cost(
                missed_blocks=r["missed_blocks"],
                request_total_tokens=r["total_tokens"],
                request_prefill_ms=r["prefill_ms"],
            )
            request_miss_ms.append(m_ms)

        if request_miss_ms:
            request_miss_ms.sort()
            miss_prefill_ms_sum = float(sum(request_miss_ms))
            p50 = percentile(request_miss_ms, 50.0)
            p95 = percentile(request_miss_ms, 95.0)
        else:
            miss_prefill_ms_sum = 0.0
            p50 = 0.0
            p95 = 0.0

        task_id, seed = ep_key
        rows.append(
            {
                "baseline": baseline_name,
                "capacity_gib": capacity_gib,
                "concurrency": concurrency,
                "task_id": task_id,
                "seed": seed,
                "domain": ep_domains[ep_key],
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 6),
                "evictions": ep_evictions[ep_key],
                "miss_prefill_tokens": ep_miss_tokens[ep_key],
                "miss_prefill_ms": round(miss_prefill_ms_sum, 4),
                "p50_ttft_ms": round(p50, 4),
                "p95_ttft_ms": round(p95, 4),
                "resume_hit_rate": round(hit_rate, 6),
                "status": "ok",
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("G1′ Full Grid Runner")
    print("=" * 78)
    print(f"Config : {args.config}")
    print(f"Output : {args.output}")
    if args.max_episodes is not None and args.max_episodes > 0:
        print(f"Max    : first {args.max_episodes} episodes per concurrency trace")
    print()

    if not args.config.exists():
        sys.stderr.write(f"ERROR: config not found: {args.config}\n")
        return 1

    config = load_config(args.config)

    # Resolve grid dimensions.
    budgets_gib: List[float] = (
        args.capacities
        if args.capacities
        else list(config.get("capacity", {}).get("budgets_gib", [1, 2, 4, 6]))
    )
    concurrencies: List[int] = (
        args.concurrency
        if args.concurrency
        else list(config.get("concurrency", {}).get("levels", [1, 4, 8]))
    )
    baselines: List[str] = (
        args.baselines
        if args.baselines
        else get_enabled_baselines(config)
    )

    # Validate baselines.
    unknown = [b for b in baselines if b not in BASELINE_CLASSES]
    if unknown:
        sys.stderr.write(
            f"ERROR: unknown baselines: {unknown}. "
            f"Known: {sorted(BASELINE_CLASSES.keys())}\n"
        )
        return 1

    # bytes_per_block from G0 frozen model config.
    bytes_per_block = compute_bytes_per_block(
        num_hidden_layers=G0_MODEL_CONFIG["num_hidden_layers"],
        num_kv_heads=G0_MODEL_CONFIG["num_kv_heads"],
        head_dim=G0_MODEL_CONFIG["head_dim"],
        dtype_bytes=G0_MODEL_CONFIG["dtype_bytes"],
        block_size=G0_MODEL_CONFIG["block_size"],
    )
    block_size = int(G0_MODEL_CONFIG.get("block_size", 16))

    print(f"Grid: {len(baselines)} baselines × {len(budgets_gib)} capacities "
          f"× {len(concurrencies)} concurrencies = "
          f"{len(baselines) * len(budgets_gib) * len(concurrencies)} cells")
    print(f"  baselines    : {baselines}")
    print(f"  capacities   : {budgets_gib} GiB  (bytes_per_block={bytes_per_block:,})")
    print(f"  concurrencies: {concurrencies}")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Track totals for the summary.
    total_rows = 0
    total_cells = 0
    t_global = time.time()

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()

        for c in concurrencies:
            trace_path = (
                SCRIPT_DIR / "physical_traces" / f"access_trace_c{c}.jsonl"
            )
            if not trace_path.exists():
                # Fallback to the base (sequential) trace for c=1.
                base_path = SCRIPT_DIR / "physical_traces" / "access_trace.jsonl"
                if c == 1 and base_path.exists():
                    trace_path = base_path
                else:
                    sys.stderr.write(
                        f"WARNING: trace not found for c={c}: {trace_path}; "
                        f"skipping this concurrency level.\n"
                    )
                    continue

            print(f"[c={c}] Loading trace: {trace_path}")
            t0 = time.time()
            accesses, future, episode_index, episode_order = load_trace(
                trace_path, args.max_episodes
            )
            print(
                f"      {len(accesses):,} accesses, "
                f"{len(episode_order)} episodes, "
                f"{len(future):,} unique blocks "
                f"({time.time() - t0:.1f}s)"
            )

            if not accesses or not episode_order:
                sys.stderr.write(
                    f"WARNING: empty trace or no episodes for c={c}; skipping.\n"
                )
                continue

            for cap_gib in budgets_gib:
                cap_blocks = gib_to_blocks(cap_gib, bytes_per_block)
                if cap_blocks <= 0:
                    sys.stderr.write(
                        f"WARNING: capacity {cap_gib} GiB → {cap_blocks} blocks; "
                        f"skipping.\n"
                    )
                    continue

                for bl in baselines:
                    t1 = time.time()
                    rows = replay_baseline(
                        baseline_name=bl,
                        capacity_gib=cap_gib,
                        concurrency=c,
                        capacity_blocks=cap_blocks,
                        accesses=accesses,
                        future_accesses=future,
                        episode_index=episode_index,
                        episode_order=episode_order,
                        block_size=block_size,
                    )
                    for r in rows:
                        writer.writerow(r)
                    f.flush()

                    total_rows += len(rows)
                    total_cells += 1
                    print(
                        f"      [{bl:12s}] cap={cap_gib}GiB "
                        f"blocks={cap_blocks:>6d} → {len(rows):4d} rows "
                        f"({time.time() - t1:.1f}s)"
                    )

    elapsed = time.time() - t_global
    print()
    print("[Done] Summary")
    print(f"    cells replayed : {total_cells}")
    print(f"    rows written   : {total_rows:,}")
    print(f"    output file    : {args.output}")
    print(f"    elapsed        : {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
