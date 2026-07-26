"""
PBKV-Inspired Cache (INSPIRED VARIANT)
=======================================

This is NOT a faithful reproduction of PBKV (arXiv 2605.06472).
PBKV has no public official code as of 2026-07-26. This inspired variant
captures the paper's core ideas:

Faithful to paper:
- GraphSAGE-style aggregation of block neighborhood features for reuse prediction
- Workflow-history attention over recent block accesses
- Continuous reuse score (probability of re-access within horizon H)
- Multi-step prediction (estimate reuse at multiple horizons)

Differences from paper (inspired variant simplifications):
- No SGLang integration; runs on our open-loop trace replay infrastructure
- GraphSAGE replaced with a lightweight feature aggregator (mean pooling of
  block features + parent/child features), no learned weights
- Workflow-history attention replaced with exponential decay weighting
  (no learned attention parameters)
- Reuse score = weighted combination of: (1) recency, (2) parent-chain
  completeness, (3) neighborhood access frequency, (4) cost-per-distance
- Multi-step prediction: single horizon H (not multi-horizon as in paper)
- No GPU/host tier distinction (single cache tier, like other baselines)
- No prefetch (eviction-only, consistent with LRU/GDSF/SizeCost/APC-LRU)

These simplifications mean this variant likely UNDERESTIMATES PBKV's true
performance (learned GraphSAGE + attention would predict reuse more accurately
than our hand-crafted features). If this inspired variant already shows
non-trivial improvement over simple heuristics, the faithful PBKV would likely
show even larger improvement.
"""

import bisect
import math
import sys
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Reuse-score weights and decay constants (hand-tuned; paper learns these)
# ---------------------------------------------------------------------------

_W_RECENCY = 0.3
_W_PARENT_CHAIN = 0.2
_W_NEIGHBORHOOD_FREQ = 0.2
_W_COST_PER_DISTANCE = 0.3

_RECENCY_LAMBDA = 0.01          # exponential decay rate for recency
_FREQ_NORMALIZER = 5.0          # access count at which neighborhood_freq saturates
_PARENT_IN_CACHE_BONUS = 1.0    # parent_chain_factor when parent is cached
_PARENT_MISSING_BASE = 0.3      # parent_chain_factor when parent is absent
_ONLINE_COST_NORMALIZER = 1000.0  # cost normalizer for online (non-oracle) mode


class PBKVInspiredCache:
    """
    PBKV-inspired reuse-prediction cache (inspired variant; see module docstring).

    The cache maintains a per-block feature dict and a parent/child prefix-tree
    index, just like :class:`APCLRUCache`. The distinguishing component is the
    reuse-score predictor :meth:`_compute_reuse_score`, which combines four
    hand-crafted features (recency, parent-chain completeness, neighborhood
    access frequency, cost-per-distance) into a continuous score in ``[0, 1]``.

    Eviction policy: when the cache is over capacity, evict the block with the
    *minimum* reuse score (least likely to be reused). If the evicted block
    has descendants in the prefix tree, those descendants are evicted too
    (chain consistency — a descendant cannot be reused without its ancestor),
    matching :class:`APCLRUCache` semantics.

    Oracle mode: when ``future_accesses`` is supplied (offline replay), the
    cost-per-distance feature uses the actual next-access distance from the
    precomputed trace indices, mirroring PBKV's multi-step prediction. When
    ``future_accesses`` is ``None`` (online mode), the feature falls back to a
    static ``cost / 1000`` heuristic that prefers keeping expensive blocks.
    """

    def __init__(self,
                 capacity: int,
                 future_accesses: Optional[Dict[str, List[int]]] = None,
                 horizon: int = 100):
        self.capacity = max(1, capacity)
        self.future_accesses = future_accesses
        self.horizon = max(1, horizon)

        # block_hash -> {parent_hash, last_access, access_count, cost}
        self.cache: Dict[str, Dict] = {}
        # Prefix-tree indexes (mirrors APCLRUCache).
        self.parent_to_children: Dict[str, Set[str]] = {}
        self.child_to_parent: Dict[str, str] = {}

        # Counters (same schema as the other baselines).
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

        # Monotonic clock — also serves as the current trace index for the
        # oracle-mode next-use-distance lookup (incremented once per access).
        self._clock: int = 0
        # Last H accesses (workflow-history attention buffer; maintained for
        # parity with the paper, though the inspired variant uses exponential
        # decay rather than attending over this buffer directly).
        self._access_history: List[str] = []

    # ------------------------------------------------------------------
    # Reuse-score predictor (the core PBKV-inspired component)
    # ------------------------------------------------------------------

    def _next_use_distance(self, block_hash: str) -> int:
        """
        Return the distance (in accesses) from the current clock to the next
        access of ``block_hash`` strictly after the current clock, or
        ``sys.maxsize`` if the block is never accessed again.
        """
        if self.future_accesses is None:
            return sys.maxsize
        accesses = self.future_accesses.get(block_hash, [])
        if not accesses:
            return sys.maxsize
        pos = bisect.bisect_right(accesses, self._clock)
        if pos < len(accesses):
            return accesses[pos] - self._clock
        return sys.maxsize

    def _compute_reuse_score(self, block_hash: str) -> float:
        """
        Compute the continuous reuse score in ``[0, 1]`` for a cached block.

        score = w1 * recency_factor
              + w2 * parent_chain_factor
              + w3 * neighborhood_freq_factor
              + w4 * cost_per_distance_factor
        """
        entry = self.cache[block_hash]
        last_access = entry["last_access"]
        access_count = entry["access_count"]
        cost = entry["cost"]
        parent_hash = entry["parent_hash"]

        # (1) Recency factor — exponential decay (paper uses attention).
        recency_factor = math.exp(-_RECENCY_LAMBDA * (self._clock - last_access))

        # (2) Parent-chain completeness — full chain gets a bonus.
        if parent_hash and parent_hash in self.cache:
            parent_chain_factor = _PARENT_IN_CACHE_BONUS
        else:
            parent_chain_factor = _PARENT_MISSING_BASE

        # (3) Neighborhood frequency — GraphSAGE-style aggregation simplified
        # to the block's own access count.
        neighborhood_freq_factor = min(1.0, access_count / _FREQ_NORMALIZER)

        # (4) Cost-per-distance — keeps expensive-to-recompute blocks unless
        # they will not be reused for a very long time.
        if self.future_accesses is not None:
            distance = self._next_use_distance(block_hash)
            if distance == sys.maxsize:
                cost_per_distance_factor = 0.0
            else:
                cost_per_distance_factor = min(1.0, cost / max(1, distance))
        else:
            cost_per_distance_factor = min(1.0, cost / _ONLINE_COST_NORMALIZER)

        score = (
            _W_RECENCY * recency_factor
            + _W_PARENT_CHAIN * parent_chain_factor
            + _W_NEIGHBORHOOD_FREQ * neighborhood_freq_factor
            + _W_COST_PER_DISTANCE * cost_per_distance_factor
        )
        # Numerical guard — weights sum to 1.0 and each factor is in [0, 1],
        # so the score is already in [0, 1], but clamp to be safe.
        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _disconnect(self, block_hash: str) -> None:
        """Remove parent/child index entries for a block (mirrors APC-LRU)."""
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
            children = list(self.parent_to_children.get(cur, set()))
            for child in children:
                if child in self.cache:
                    stack.append(child)
            self._disconnect(cur)
        return evicted

    def _evict(self) -> None:
        """
        Evict the block with the minimum reuse score (least likely to be
        reused) and its entire descendant chain.
        """
        if not self.cache:
            return
        victim = min(self.cache, key=lambda h: self._compute_reuse_score(h))
        self.evictions += self._evict_chain(victim)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def _record_history(self, block_hash: str) -> None:
        """Append to the workflow-history buffer, trimmed to horizon H."""
        self._access_history.append(block_hash)
        if len(self._access_history) > self.horizon:
            # Trim in-place to avoid reallocating on every access.
            del self._access_history[:-self.horizon]

    def access(self, block_hash: str, parent_hash: str = "",
               prefill_ms: float = 0.0) -> bool:
        """
        Access a block. Returns True on hit, False on miss.

        On hit, refreshes recency, increments access count, and records the
        access in the workflow-history buffer.
        On miss, evicts the lowest-reuse-score block (and its chain) if the
        cache is over capacity, then inserts the new block.
        """
        if block_hash in self.cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cache[block_hash]
            entry["last_access"] = self._clock
            entry["access_count"] += 1
            self._record_history(block_hash)
            self._clock += 1
            return True

        # Miss.
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        while len(self.cache) >= self.capacity:
            self._evict()

        self.cache[block_hash] = {
            "parent_hash": parent_hash,
            "last_access": self._clock,
            "access_count": 1,
            "cost": prefill_ms,
        }
        if parent_hash:
            self.parent_to_children.setdefault(parent_hash, set()).add(block_hash)
            self.child_to_parent[block_hash] = parent_hash

        self._record_history(block_hash)
        self._clock += 1
        return False
