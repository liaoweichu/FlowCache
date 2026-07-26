"""
E1 Oracle vs Heuristic Comparison
==================================
Open-loop replay comparing 7 baselines for KV cache eviction:
  - Simple heuristics: LRU, GDSF, SizeCost, APC-LRU
  - Closest-baseline inspired variant: PBKV-Inspired (GraphSAGE-style reuse
    prediction + chain-aware eviction; oracle mode uses future_accesses for
    multi-step prediction, see baselines/pbkv_inspired.py)
  - Oracle upper bounds: Belady (distance-optimal), Oracle-Cost (cost-aware)

Purpose: Prove that an offline oracle (future-aware) significantly outperforms
simple heuristics (LRU, GDSF, SizeCost, APC-LRU), demonstrating headroom for
a learned predictor. Headroom is computed as
``Oracle-Cost hit% - max(LRU, GDSF, SizeCost, APC-LRU) hit%``.

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

# baselines/ is a sibling package of this script (lives under experiments/e1/).
# Import the PBKV-inspired closest-baseline variant (see
# baselines/pbkv_inspired.py for the inspired-variant caveats).
from baselines.pbkv_inspired import PBKVInspiredCache
# Import the ThunderAgent-inspired closest-baseline variant (see
# baselines/thunderagent_inspired.py for the inspired-variant caveats).
from baselines.thunderagent_inspired import ThunderAgentInspiredCache


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
# SizeCost Cache (cost-aware GDSF variant)
# ---------------------------------------------------------------------------

class SizeCostCache:
    """
    Cost-aware GDSF cache eviction.

    Priority = clock + accumulated_cost / size, where accumulated_cost is the
    block's ``cost`` (first-seen prefill_ms) times its current frequency
    (i.e. total saved cost so far). Cost comes from the step-level prefill_ms
    apportioned per-block (see :func:`build_access_trace`), rather than the
    legacy uniform ``block_size * 0.5`` estimate.

    Class structure mirrors :class:`GDSFCache`:
    ``cache: Dict[str, Dict]`` maps block_hash to
    ``{freq, priority, cost, size}``; ``_heap`` is a min-heap of
    ``(priority, block_hash)`` tuples with lazy deletion.

    With uniform ``size`` and ``cost`` for every block, the policy reduces to
    plain GDSF; with non-uniform ``cost`` it prefers to keep expensive blocks.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache: Dict[str, Dict] = {}  # block_hash -> {freq, priority, cost, size}
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
        if self.cache:
            victim = next(iter(self.cache))
            self.clock = self.cache[victim]["priority"]
            del self.cache[victim]
            self.evictions += 1

    def access(self, block_hash: str, prefill_ms: float = 0.0,
               size: int = 16) -> bool:
        """
        Access a block. Returns True if hit, False if miss.

        On a hit, increments frequency and recomputes priority using the
        accumulated saved cost (``cost * freq``).
        On a miss, inserts with ``freq=1``, ``cost=prefill_ms``,
        ``size=size``, ``priority = clock + prefill_ms / size``; evicts if
        over capacity.
        """
        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cache[block_hash]
            entry["freq"] += 1
            accumulated_cost = entry["cost"] * entry["freq"]
            safe_size = max(1, entry["size"])
            entry["priority"] = self.clock + accumulated_cost / safe_size
            heapq.heappush(self._heap, (entry["priority"], block_hash))
            return True
        else:
            self.misses += 1
            self.miss_cost_ms += prefill_ms
            while len(self.cache) >= self.capacity:
                self._evict()
            safe_size = max(1, size)
            priority = self.clock + prefill_ms / safe_size
            self.cache[block_hash] = {
                "freq": 1,
                "priority": priority,
                "cost": prefill_ms,
                "size": size,
            }
            heapq.heappush(self._heap, (priority, block_hash))
            return False


# ---------------------------------------------------------------------------
# APC-LRU Cache (vLLM-style prefix-aware LRU)
# ---------------------------------------------------------------------------

class APCLRUCache:
    """
    vLLM APC-style prefix-aware LRU.

    Each cached block records its ``parent_hash`` and a monotonic
    ``last_access`` timestamp. Two index maps maintain the prefix tree:

      * ``parent_to_children: Dict[str, Set[str]]`` — children of each parent.
      * ``child_to_parent: Dict[str, str]`` — parent of each child.

    Access semantics (per :func:`access`):

      * Block in cache → HIT. Update its ``last_access`` and accumulate
        ``saved_prefill_ms``. Parent is *not* touched (per the G1 spec, to
        keep semantics simple and avoid double-counting on direct hits).
      * Block not in cache but ``parent_hash`` is → "prefix hit". Update the
        parent's ``last_access`` and add the parent's prefill cost to
        ``saved_prefill_ms`` (the parent's recompute is saved). The current
        block still counts as a miss.
      * Block and parent both absent → full chain miss.

    Eviction is chain-aware: when a victim X is selected (oldest
    ``last_access``), X and *all* of its descendants are evicted
    recursively, since descendant blocks cannot be reused without their
    ancestor. ``evictions`` accumulates the total number of blocks evicted.

    Capacity counts blocks (consistent with LRU / GDSF / SizeCost).
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache: Dict[str, Dict] = {}  # block_hash -> {parent_hash, last_access, prefill_ms}
        self.parent_to_children: Dict[str, Set[str]] = {}
        self.child_to_parent: Dict[str, str] = {}
        self._clock: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def _touch(self, block_hash: str) -> None:
        """Refresh the LRU timestamp of a cached block."""
        self.cache[block_hash]["last_access"] = self._clock
        self._clock += 1

    def _disconnect(self, block_hash: str) -> None:
        """Remove parent/child index entries for a block."""
        parent = self.child_to_parent.pop(block_hash, None)
        if parent is not None:
            siblings = self.parent_to_children.get(parent)
            if siblings is not None:
                siblings.discard(block_hash)
                if not siblings:
                    del self.parent_to_children[parent]
        self.parent_to_children.pop(block_hash, None)

    def _evict_chain(self, victim: str) -> int:
        """
        Recursively evict ``victim`` and all of its descendants.

        Returns the number of blocks evicted (>= 1).
        """
        evicted = 0
        stack = [victim]
        while stack:
            cur = stack.pop()
            if cur not in self.cache:
                continue
            del self.cache[cur]
            evicted += 1
            # Push descendants (snapshot to avoid mutation during iteration).
            children = list(self.parent_to_children.get(cur, set()))
            for child in children:
                if child in self.cache:
                    stack.append(child)
            self._disconnect(cur)
        return evicted

    def _evict_one(self) -> None:
        """Evict the oldest block (smallest ``last_access``) and its chain."""
        if not self.cache:
            return
        victim = min(self.cache, key=lambda h: self.cache[h]["last_access"])
        self.evictions += self._evict_chain(victim)

    def access(self, block_hash: str, parent_hash: str = "",
               prefill_ms: float = 0.0) -> bool:
        """
        Access a block. Returns True if the block itself is a hit.

        See the class docstring for the full prefix-aware semantics.
        """
        if block_hash in self.cache:
            # Direct hit on current block.
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            self._touch(block_hash)
            return True

        # Miss on current block.
        self.misses += 1
        self.miss_cost_ms += prefill_ms

        if parent_hash and parent_hash in self.cache:
            # Prefix hit: parent's recompute is saved, current block still miss.
            parent_entry = self.cache[parent_hash]
            self.saved_prefill_ms += parent_entry.get("prefill_ms", 0.0)
            self._touch(parent_hash)

        # Insert current block, evicting chain(s) if over capacity.
        while len(self.cache) >= self.capacity:
            self._evict_one()
        self.cache[block_hash] = {
            "parent_hash": parent_hash,
            "last_access": self._clock,
            "prefill_ms": prefill_ms,
        }
        self._clock += 1
        if parent_hash:
            self.parent_to_children.setdefault(parent_hash, set()).add(block_hash)
            self.child_to_parent[block_hash] = parent_hash
        return False


# ---------------------------------------------------------------------------
# Oracle-Cost Cache (cost-aware Belady upper bound)
# ---------------------------------------------------------------------------

class OracleCostCache:
    """
    Cost-aware Belady offline oracle.

    On eviction, picks the cached block minimizing
    ``block_cost[h] / next_use_distance(h, current_idx)`` — i.e. the block
    whose saved cost per unit of "time until next reuse" is smallest, meaning
    it is the least valuable to keep. If a block is never accessed again
    (``next_use_distance == sys.maxsize``) its ratio is treated as 0, giving
    it the highest eviction priority.

    Tie-break: when two ratios are equal (within 1e-9), fall back to Belady's
    rule and evict the one with the *maximum* ``next_use_distance`` (farthest
    next access).

    Class structure mirrors :class:`BeladyOracle`; additionally
    ``block_cost: Dict[str, float]`` records the first-seen ``prefill_ms`` of
    each block (subsequent accesses use this stored cost so the eviction
    decision is based on a stable per-block cost).
    """

    def __init__(self, capacity: int,
                 future_accesses: Dict[str, List[int]]):
        self.capacity = max(1, capacity)
        self.future_accesses = future_accesses
        self.cache: Set[str] = set()
        self.block_cost: Dict[str, float] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def _next_access(self, block_hash: str, current_idx: int) -> int:
        """
        Return the next access index strictly greater than *current_idx*,
        or ``sys.maxsize`` if the block is never accessed again.
        """
        accesses = self.future_accesses.get(block_hash, [])
        if not accesses:
            return sys.maxsize
        pos = bisect.bisect_right(accesses, current_idx)
        if pos < len(accesses):
            return accesses[pos]
        return sys.maxsize

    def _evict(self, current_idx: int) -> None:
        """
        Evict the block with the minimum cost-per-distance ratio.
        Ties (within 1e-9) are broken by maximum next-use distance.
        """
        if not self.cache:
            return
        best_hash = None
        best_ratio = None
        best_distance = None
        for h in self.cache:
            distance = self._next_access(h, current_idx)
            cost = self.block_cost.get(h, 0.0)
            if distance == sys.maxsize:
                ratio = 0.0
            else:
                ratio = cost / max(1, distance)
            if best_hash is None:
                best_hash = h
                best_ratio = ratio
                best_distance = distance
                continue
            if ratio < best_ratio - 1e-9:
                best_hash = h
                best_ratio = ratio
                best_distance = distance
            elif abs(ratio - best_ratio) <= 1e-9:
                # Tie-break: prefer evicting the one with larger distance.
                if distance > best_distance:
                    best_hash = h
                    best_ratio = ratio
                    best_distance = distance
        if best_hash is not None:
            self.cache.remove(best_hash)
            self.evictions += 1

    def access(self, block_hash: str, access_idx: int,
               prefill_ms: float = 0.0) -> bool:
        """
        Access a block at the given trace index. Returns True if hit.

        On first encounter with this block_hash, records
        ``block_cost[block_hash] = prefill_ms``.
        """
        if block_hash not in self.block_cost:
            self.block_cost[block_hash] = prefill_ms

        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            return True

        self.misses += 1
        self.miss_cost_ms += prefill_ms
        while len(self.cache) >= self.capacity and self.cache:
            self._evict(access_idx)
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
        {block_hash, parent_hash, step_id, workflow_id, prefill_ms, size}

    ``prefill_ms`` (block-level) is derived from the step-level ``prefill_ms``
    field, apportioned to each block proportional to that block's token span
    ``(token_range_end - token_range_start)``. If the step's ``prefill_ms`` is
    missing or zero (e.g. system/cached steps), it falls back to the legacy
    uniform estimate ``block_size * 0.5`` per block. If ``step_token_count``
    is zero, the same uniform fallback is used.

    ``size`` is the block's token count
    (``token_range_end - token_range_start``); it defaults to ``block_size``
    when token-range fields are absent.

    ``parent_hash`` is read from the block assignment (empty string when
    absent); it is consumed by APC-LRU to maintain prefix chains.
    """
    trace: List[Dict] = []
    for traj in trajectories:
        meta = traj.get("meta", {})
        workflow_id = meta.get("workflow_id", "unknown")
        block_size = meta.get("block_size", 16)
        fallback_prefill_per_block = block_size * 0.5

        for step in traj.get("steps", []):
            step_id = step.get("step_id", 0)
            step_prefill_ms = step.get("prefill_ms", 0.0) or 0.0
            step_token_count = step.get("token_count", 0) or 0

            blocks = step.get("block_assignments", [])
            if not blocks:
                continue

            # Decide per-block prefill cost for this step.
            if step_prefill_ms <= 0.0:
                # Step has no prefill cost (e.g. system/cached step) →
                # preserve legacy uniform estimate so old traces replay
                # with the same numbers when prefill_ms is absent.
                block_prefill = {
                    blk.get("block_hash", ""): fallback_prefill_per_block
                    for blk in blocks
                }
            elif step_token_count <= 0:
                # No token count to apportion by → fall back to uniform.
                block_prefill = {
                    blk.get("block_hash", ""): fallback_prefill_per_block
                    for blk in blocks
                }
            else:
                # Apportion step-level prefill_ms proportional to each
                # block's token span.
                block_prefill = {}
                total_tokens = 0
                span_list = []
                for blk in blocks:
                    bh = blk.get("block_hash", "")
                    t_start = blk.get("token_range_start", 0)
                    t_end = blk.get("token_range_end", 0)
                    span = max(0, t_end - t_start)
                    span_list.append((bh, span))
                    total_tokens += span
                if total_tokens <= 0:
                    # No usable span info → fall back to uniform.
                    for blk in blocks:
                        block_prefill[blk.get("block_hash", "")] = (
                            fallback_prefill_per_block
                        )
                else:
                    for bh, span in span_list:
                        # Per-block cost proportional to its token share.
                        block_prefill[bh] = (
                            step_prefill_ms * span / step_token_count
                        )

            for blk in blocks:
                block_hash = blk.get("block_hash", "")
                if not block_hash:
                    continue
                parent_hash = blk.get("parent_hash", "") or ""
                t_start = blk.get("token_range_start", 0)
                t_end = blk.get("token_range_end", 0)
                size = max(0, t_end - t_start)
                if size <= 0:
                    size = block_size
                trace.append({
                    "block_hash": block_hash,
                    "parent_hash": parent_hash,
                    "step_id": step_id,
                    "workflow_id": workflow_id,
                    "prefill_ms": block_prefill.get(block_hash,
                                                   fallback_prefill_per_block),
                    "size": size,
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
    """Print a formatted summary table to stdout.

    Columns: LRU%, GDSF%, SizeCost%, APC-LRU% (simple heuristics) +
    PBKV-Insp% (closest-baseline inspired variant) +
    Oracle% (Belady) + OracleCost% (cost-aware Belady).
    Headroom = ``OracleCost% - max(LRU%, GDSF%, SizeCost%, APC-LRU%)``.
    """
    print()
    print("E1 Oracle Comparison")
    print(f"Peak working set: {peak_ws} blocks")
    print(f"Block size:       {block_size} tokens")
    print(f"Total accesses:   {total_accesses}")
    print()
    header = (
        f"{'Budget':>8}  {'LRU%':>10}  {'GDSF%':>10}  "
        f"{'SizeCost%':>10}  {'APC-LRU%':>10}  {'PBKV-Insp%':>12}  "
        f"{'Oracle%':>10}  {'OracleCost%':>12}  {'Headroom':>10}"
    )
    print(header)
    print("-" * len(header))
    for budget in budgets:
        r = results[f"budget_{budget:.2f}"]
        lru_hr = r["lru"]["hit_rate"] * 100
        gdsf_hr = r["gdsf"]["hit_rate"] * 100
        sizecost_hr = r.get("sizecost", {}).get("hit_rate", 0.0) * 100
        apc_lru_hr = r.get("apc_lru", {}).get("hit_rate", 0.0) * 100
        pbkv_hr = r.get("pbkv_inspired", {}).get("hit_rate", 0.0) * 100
        orc_hr = r["oracle"]["hit_rate"] * 100
        orc_cost_hr = r.get("oracle_cost", {}).get("hit_rate", 0.0) * 100
        best_heuristic = max(lru_hr, gdsf_hr, sizecost_hr, apc_lru_hr)
        headroom = orc_cost_hr - best_heuristic
        print(f"{budget*100:>7.0f}%  {lru_hr:>9.1f}%  {gdsf_hr:>9.1f}%  "
              f"{sizecost_hr:>9.1f}%  {apc_lru_hr:>9.1f}%  "
              f"{pbkv_hr:>11.1f}%  "
              f"{orc_hr:>9.1f}%  {orc_cost_hr:>11.1f}%  "
              f"{headroom:>9.1f}%")
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

        # --- SizeCost (cost-aware GDSF) ---
        sizecost = SizeCostCache(capacity)
        for acc in access_trace:
            sizecost.access(acc["block_hash"], acc["prefill_ms"],
                            acc.get("size", block_size))

        # --- APC-LRU (prefix-aware LRU) ---
        apc_lru = APCLRUCache(capacity)
        for acc in access_trace:
            apc_lru.access(acc["block_hash"], acc.get("parent_hash", ""),
                           acc["prefill_ms"])

        # --- PBKV-Inspired (closest-baseline inspired variant) ---
        # Pass future_accesses so the multi-step prediction uses the actual
        # next-use distance (offline replay: the future trace is known).
        pbkv = PBKVInspiredCache(capacity,
                                 future_accesses=future_accesses,
                                 horizon=100)
        for acc in access_trace:
            pbkv.access(acc["block_hash"], acc.get("parent_hash", ""),
                        acc["prefill_ms"])

        # --- ThunderAgent-Inspired (closest-baseline inspired variant) ---
        # Workflow-aware time-decay: blocks from paused workflows decay as
        # 2^{-t * decay_rate}. Pass workflow_id for program-aware scheduling.
        thunderagent = ThunderAgentInspiredCache(capacity,
                                                 future_accesses=future_accesses,
                                                 decay_rate=0.05)
        for acc in access_trace:
            thunderagent.access(acc["block_hash"],
                                acc.get("parent_hash", ""),
                                acc["prefill_ms"],
                                acc.get("workflow_id", ""))

        # --- Oracle-Cost (cost-aware Belady) ---
        oracle_cost = OracleCostCache(capacity, future_accesses)
        for idx, acc in enumerate(access_trace):
            oracle_cost.access(acc["block_hash"], idx, acc["prefill_ms"])

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
            "sizecost": {
                "hits": sizecost.hits,
                "misses": sizecost.misses,
                "hit_rate": sizecost.hits / total if total else 0.0,
                "evictions": sizecost.evictions,
                "saved_prefill_ms": round(sizecost.saved_prefill_ms, 2),
                "miss_cost_ms": round(sizecost.miss_cost_ms, 2),
            },
            "apc_lru": {
                "hits": apc_lru.hits,
                "misses": apc_lru.misses,
                "hit_rate": apc_lru.hits / total if total else 0.0,
                "evictions": apc_lru.evictions,
                "saved_prefill_ms": round(apc_lru.saved_prefill_ms, 2),
                "miss_cost_ms": round(apc_lru.miss_cost_ms, 2),
            },
            "pbkv_inspired": {
                "hits": pbkv.hits,
                "misses": pbkv.misses,
                "hit_rate": pbkv.hits / total if total else 0.0,
                "evictions": pbkv.evictions,
                "saved_prefill_ms": round(pbkv.saved_prefill_ms, 2),
                "miss_cost_ms": round(pbkv.miss_cost_ms, 2),
            },
            "thunderagent_inspired": {
                "hits": thunderagent.hits,
                "misses": thunderagent.misses,
                "hit_rate": thunderagent.hits / total if total else 0.0,
                "evictions": thunderagent.evictions,
                "saved_prefill_ms": round(thunderagent.saved_prefill_ms, 2),
                "miss_cost_ms": round(thunderagent.miss_cost_ms, 2),
            },
            "oracle_cost": {
                "hits": oracle_cost.hits,
                "misses": oracle_cost.misses,
                "hit_rate": oracle_cost.hits / total if total else 0.0,
                "evictions": oracle_cost.evictions,
                "saved_prefill_ms": round(oracle_cost.saved_prefill_ms, 2),
                "miss_cost_ms": round(oracle_cost.miss_cost_ms, 2),
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
