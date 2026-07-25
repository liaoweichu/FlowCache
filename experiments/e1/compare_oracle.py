"""
E1 Oracle vs Heuristic Comparison
==================================
Open-loop replay comparing LRU, GDSF, and Belady oracle for KV cache eviction.

Purpose: Prove that an offline oracle (future-aware) significantly outperforms
simple heuristics (LRU, GDSF), demonstrating headroom for a learned predictor.

All strategies operate on the same set of blocks and same KV budget constraint.
Comparison is in open-loop mode: same trace, same blocks, same arrival order.
"""

import bisect
import heapq
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from trace_utils import load_all_trajectories


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class LRUCache:
    """
    Standard LRU eviction using OrderedDict for O(1) access tracking.
    Evicts the block with the oldest last-access time when the cache is full.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache: OrderedDict = OrderedDict()  # block_hash -> None (value unused)
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def access(self, block_hash: str, prefill_ms: float = 0.0) -> bool:
        """
        Access a block. Returns True if hit, False if miss.

        On a hit, the block is moved to the end (most-recently-used).
        On a miss, the block is inserted; if the cache exceeds capacity,
        the least-recently-used block is evicted.
        """
        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            self.cache.move_to_end(block_hash)
            return True
        else:
            self.misses += 1
            self.miss_cost_ms += prefill_ms
            while len(self.cache) >= self.capacity:
                # popitem(last=False) removes the oldest (first) item
                self.cache.popitem(last=False)
                self.evictions += 1
            self.cache[block_hash] = None
            return False


# ---------------------------------------------------------------------------
# GDSF Cache (Greedy Dual Size Frequency)
# ---------------------------------------------------------------------------

class GDSFCache:
    """
    Greedy Dual Size Frequency (GDSF) cache eviction.

    Priority = clock + frequency / size.
    With uniform block size (size=1), priority = clock + frequency.

    The clock is updated to the evicted block's priority value on eviction.
    Uses a min-heap with lazy deletion for efficient priority-based eviction.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache: Dict[str, Dict] = {}  # block_hash -> {freq, priority}
        self._heap: List[tuple] = []      # (priority, block_hash) min-heap
        self.clock: float = 0.0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def _evict(self) -> None:
        """Evict the block with the minimum valid priority from the heap."""
        while self._heap:
            priority, block_hash = heapq.heappop(self._heap)
            if block_hash not in self.cache:
                continue
            current_priority = self.cache[block_hash]["priority"]
            if abs(current_priority - priority) < 1e-9:
                self.clock = priority
                del self.cache[block_hash]
                self.evictions += 1
                return
        # Should only reach here if heap is empty but cache reports non-empty
        # (should not happen, but fall back to arbitrary eviction)
        if self.cache:
            victim = next(iter(self.cache))
            self.clock = self.cache[victim]["priority"]
            del self.cache[victim]
            self.evictions += 1

    def access(self, block_hash: str, prefill_ms: float = 0.0) -> bool:
        """
        Access a block. Returns True if hit, False if miss.

        On a hit, increments frequency and recomputes priority.
        On a miss, inserts with frequency=1; evicts if over capacity.
        """
        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cache[block_hash]
            entry["freq"] += 1
            # size is uniform (= 1), so priority = clock + freq
            entry["priority"] = self.clock + entry["freq"]
            heapq.heappush(self._heap, (entry["priority"], block_hash))
            return True
        else:
            self.misses += 1
            self.miss_cost_ms += prefill_ms
            while len(self.cache) >= self.capacity:
                self._evict()
            freq = 1
            priority = self.clock + freq
            self.cache[block_hash] = {"freq": freq, "priority": priority}
            heapq.heappush(self._heap, (priority, block_hash))
            return False


# ---------------------------------------------------------------------------
# Belady Oracle (MIN algorithm)
# ---------------------------------------------------------------------------

class BeladyOracle:
    """
    Belady's MIN algorithm -- the offline optimal eviction policy.

    On eviction, removes the cached block whose next access is farthest
    in the future (or infinity if the block is never accessed again).

    Requires pre-computed future access information (block_hash -> sorted
    list of access indices in the trace).
    """

    def __init__(self, capacity: int, future_accesses: Dict[str, List[int]]):
        self.capacity = max(1, capacity)
        self.future_accesses = future_accesses
        self.cache: Set[str] = set()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def _next_access(self, block_hash: str, current_idx: int) -> int:
        """
        Return the next access index strictly greater than *current_idx*,
        or sys.maxsize if the block is never accessed again.
        """
        accesses = self.future_accesses.get(block_hash, [])
        if not accesses:
            return sys.maxsize
        pos = bisect.bisect_right(accesses, current_idx)
        if pos < len(accesses):
            return accesses[pos]
        return sys.maxsize

    def access(self, block_hash: str, access_idx: int,
               prefill_ms: float = 0.0) -> bool:
        """
        Access a block at the given trace index. Returns True if hit.
        """
        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            return True
        else:
            self.misses += 1
            self.miss_cost_ms += prefill_ms
            while len(self.cache) >= self.capacity and self.cache:
                victim = max(self.cache,
                             key=lambda h: self._next_access(h, access_idx))
                self.cache.remove(victim)
                self.evictions += 1
            if len(self.cache) < self.capacity:
                self.cache.add(block_hash)
            return False


# ---------------------------------------------------------------------------
# Trace building
# ---------------------------------------------------------------------------

def build_access_trace(trajectories: List[Dict]) -> List[Dict]:
    """
    Build a flat interleaved access trace from all trajectories.

    Trajectories are processed sequentially (each workflow is a contiguous
    sequence). Within a trajectory, steps are processed in step_id order.
    Every block in every step's block_assignments becomes one access.

    Each access dict:
        {block_hash, step_id, workflow_id, prefill_ms}

    prefill_ms is estimated as block_size * 0.5 ms per block (placeholder
    proportional to token count; calibrate from real measurements later).
    """
    trace: List[Dict] = []
    for traj in trajectories:
        meta = traj.get("meta", {})
        workflow_id = meta.get("workflow_id", "unknown")
        block_size = meta.get("block_size", 16)
        prefill_ms_per_block = block_size * 0.5

        for step in traj.get("steps", []):
            step_id = step.get("step_id", 0)
            for blk in step.get("block_assignments", []):
                block_hash = blk.get("block_hash", "")
                if not block_hash:
                    continue
                trace.append({
                    "block_hash": block_hash,
                    "step_id": step_id,
                    "workflow_id": workflow_id,
                    "prefill_ms": prefill_ms_per_block,
                })
    return trace


def compute_future_accesses(access_trace: List[Dict]) -> Dict[str, List[int]]:
    """
    For each unique block_hash in the trace, record all indices at which
    the block is accessed. The list is naturally sorted because we
    iterate the trace in order.
    """
    future: Dict[str, List[int]] = {}
    for idx, acc in enumerate(access_trace):
        bh = acc["block_hash"]
        if bh not in future:
            future[bh] = []
        future[bh].append(idx)
    return future


def compute_peak_working_set(trajectories: List[Dict]) -> int:
    """
    Count unique block_hashes across all trajectories.
    This is the total number of distinct blocks that could ever be referenced.
    """
    all_hashes: Set[str] = set()
    for traj in trajectories:
        for step in traj.get("steps", []):
            for blk in step.get("block_assignments", []):
                bh = blk.get("block_hash", "")
                if bh:
                    all_hashes.add(bh)
    return len(all_hashes)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _print_summary(peak_ws: int, block_size: int, total_accesses: int,
                   budgets: List[float], results: Dict) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("E1 Oracle Comparison")
    print(f"Peak working set: {peak_ws} blocks")
    print(f"Block size:       {block_size} tokens")
    print(f"Total accesses:   {total_accesses}")
    print()
    header = (f"{'Budget':>8}  {'LRU Hit%':>10}  {'GDSF Hit%':>10}  "
              f"{'Oracle Hit%':>12}  {'Headroom':>10}")
    print(header)
    print("-" * len(header))
    for budget in budgets:
        r = results[f"budget_{budget:.2f}"]
        lru_hr = r["lru"]["hit_rate"] * 100
        gdsf_hr = r["gdsf"]["hit_rate"] * 100
        orc_hr = r["oracle"]["hit_rate"] * 100
        headroom = orc_hr - max(lru_hr, gdsf_hr)
        print(f"{budget*100:>7.0f}%  {lru_hr:>9.1f}%  {gdsf_hr:>9.1f}%  "
              f"{orc_hr:>11.1f}%  {headroom:>9.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    trace_dir = script_dir / "traces" / "bf16"
    output_dir = script_dir / "outputs"
    config_path = script_dir / "config.yaml"

    # Load trajectories
    trajectories = load_all_trajectories(str(trace_dir))
    if not trajectories:
        print("No trajectories found. Run record_trajectories.py first.")
        return

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    budgets: List[float] = config.get("cache", {}).get(
        "kv_budgets", [0.10, 0.25, 0.50, 1.00]
    )
    block_size: int = int(config.get("cache", {}).get("block_size", 16))

    # Build trace and statistics
    access_trace = build_access_trace(trajectories)
    if not access_trace:
        print("Access trace is empty. No blocks to replay.")
        return

    peak_ws = compute_peak_working_set(trajectories)
    future_accesses = compute_future_accesses(access_trace)

    # Replay across all budget levels
    results: Dict[str, dict] = {}
    for budget in budgets:
        capacity = max(1, int(budget * peak_ws))

        # --- LRU ---
        lru = LRUCache(capacity)
        for acc in access_trace:
            lru.access(acc["block_hash"], acc["prefill_ms"])

        # --- GDSF ---
        gdsf = GDSFCache(capacity)
        for acc in access_trace:
            gdsf.access(acc["block_hash"], acc["prefill_ms"])

        # --- Belady Oracle ---
        oracle = BeladyOracle(capacity, future_accesses)
        for idx, acc in enumerate(access_trace):
            oracle.access(acc["block_hash"], idx, acc["prefill_ms"])

        total = len(access_trace)
        results[f"budget_{budget:.2f}"] = {
            "capacity_blocks": capacity,
            "total_accesses": total,
            "lru": {
                "hits": lru.hits,
                "misses": lru.misses,
                "hit_rate": lru.hits / total if total else 0.0,
                "evictions": lru.evictions,
                "saved_prefill_ms": round(lru.saved_prefill_ms, 2),
                "miss_cost_ms": round(lru.miss_cost_ms, 2),
            },
            "gdsf": {
                "hits": gdsf.hits,
                "misses": gdsf.misses,
                "hit_rate": gdsf.hits / total if total else 0.0,
                "evictions": gdsf.evictions,
                "saved_prefill_ms": round(gdsf.saved_prefill_ms, 2),
                "miss_cost_ms": round(gdsf.miss_cost_ms, 2),
            },
            "oracle": {
                "hits": oracle.hits,
                "misses": oracle.misses,
                "hit_rate": oracle.hits / total if total else 0.0,
                "evictions": oracle.evictions,
                "saved_prefill_ms": round(oracle.saved_prefill_ms, 2),
                "miss_cost_ms": round(oracle.miss_cost_ms, 2),
            },
        }

    # Write JSON output
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "e1-oracle-comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "peak_working_set_blocks": peak_ws,
                "block_size": block_size,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    _print_summary(peak_ws, block_size, len(access_trace), budgets, results)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
