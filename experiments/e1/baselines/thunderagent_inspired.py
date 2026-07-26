"""
ThunderAgent-Inspired Cache (INSPIRED VARIANT)
===============================================

This is NOT a faithful reproduction of ThunderAgent (arXiv 2602.13692,
ICML 2026 Spotlight). ThunderAgent is a FastAPI proxy that routes requests
to vLLM/SGLang backends with program-aware capacity scheduling. This
inspired variant captures the paper's core ideas at the block-cache level:

Faithful to paper:
- Program-aware (workflow-aware) KV lifecycle management: blocks belonging
  to the same workflow (program_id) are managed as a group
- Time decay: priority decays as 2^{-t} where t is the time since the
  workflow's last activity (corresponds to ``--use-acting-token-decay``)
- Capacity scheduling across workflows: when evicting, prefer blocks from
  the most-paused workflow (corresponds to ``--router tr``)

Differences from paper (inspired variant simplifications):
- ThunderAgent is an API proxy (FastAPI), not a block-level cache policy;
  this variant operates at the block-cache level like LRU/GDSF/SizeCost
- No GPU memory pressure tracking (ThunderAgent uses ``--gpu-memory-pressure``);
  this variant uses a fixed capacity budget like other baselines
- No backend integration (ThunderAgent proxies to vLLM/SGLang); this variant
  runs on our open-loop trace replay infrastructure
- Time decay uses a hand-tuned decay rate (paper learns this online)
- No prefetch (eviction-only, consistent with LRU/GDSF/SizeCost/APC-LRU/
  PBKV-Inspired); ThunderAgent's restore-on-resume is approximated by the
  workflow-recency component
- Prefix-chain eviction (descendant eviction) is included for consistency
  with APC-LRU and PBKV-Inspired; ThunderAgent's original design is
  program-level, not prefix-chain-level

These simplifications mean this variant likely UNDERESTIMATES ThunderAgent's
true performance (the paper's online decay-rate tuning + GPU pressure
feedback would adapt better than our hand-tuned rate). If this inspired
variant already shows non-trivial improvement over simple heuristics, the
faithful ThunderAgent would likely show even larger improvement.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Time-decay constants (hand-tuned; paper learns these online)
# ---------------------------------------------------------------------------

_DECAY_RATE = 0.05          # 2^{-t * _DECAY_RATE}: higher = faster decay
_WORKFLOW_RECENCY_WEIGHT = 0.5   # w1: workflow-level time decay
_BLOCK_RECENCY_WEIGHT = 0.3     # w2: block-level recency (LRU within workflow)
_COST_WEIGHT = 0.2              # w3: cost-awareness (prefer keeping expensive blocks)


class ThunderAgentInspiredCache:
    """
    ThunderAgent-inspired workflow-aware time-decay cache.

    The cache maintains:
      * Per-block metadata: ``{parent_hash, last_access, access_count, cost,
        workflow_id}``
      * Per-workflow last-activity timestamp: ``workflow_last_activity[wf_id]``
      * Prefix-tree indexes (parent_to_children / child_to_parent), mirroring
        :class:`APCLRUCache` and :class:`PBKVInspiredCache`.

    Eviction policy: when the cache is over capacity, compute a priority
    score for each cached block and evict the block with the *minimum* score.
    The score combines three components:

      1. **Workflow time-decay** (ThunderAgent's ``--use-acting-token-decay``):
         ``2^{-(now - workflow_last_activity[wf_id]) * _DECAY_RATE}``.
         Blocks from active workflows score high; blocks from paused
         workflows decay exponentially.

      2. **Block recency** (LRU within workflow): ``1 / (1 + block_age)``.
         Within the same workflow, more-recently-accessed blocks score higher.

      3. **Cost factor**: ``cost / cost_normalizer``. Expensive blocks
         (high prefill_ms) score higher, mirroring SizeCost's cost-awareness.

    If the evicted block has descendants in the prefix tree, those descendants
    are evicted too (chain consistency — a descendant cannot be reused
    without its ancestor), matching :class:`APCLRUCache` semantics.
    """

    def __init__(self,
                 capacity: int,
                 future_accesses: Optional[Dict[str, List[int]]] = None,
                 decay_rate: float = _DECAY_RATE):
        self.capacity = max(1, capacity)
        self.future_accesses = future_accesses  # unused (online policy)
        self.decay_rate = max(0.0, decay_rate)

        # block_hash -> {parent_hash, last_access, access_count, cost, workflow_id}
        self.cache: Dict[str, Dict] = {}
        # Prefix-tree indexes (mirrors APCLRUCache / PBKVInspiredCache).
        self.parent_to_children: Dict[str, Set[str]] = defaultdict(set)
        self.child_to_parent: Dict[str, str] = {}

        # workflow_id -> last activity timestamp (max last_access across
        # all blocks in that workflow). This is the core ThunderAgent
        # "program-aware" state: a workflow is "paused" when its last
        # activity is far in the past.
        self.workflow_last_activity: Dict[str, int] = {}

        # Counters (same schema as the other baselines).
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

        # Monotonic clock — incremented once per access.
        self._clock: int = 0

        # Cost normalizer (updated lazily as we observe block costs).
        self._max_cost_seen: float = 1.0

    # ------------------------------------------------------------------
    # Priority score (the core ThunderAgent-inspired component)
    # ------------------------------------------------------------------

    def _compute_priority(self, block_hash: str) -> float:
        """
        Compute the priority score for a cached block.

        score = w1 * workflow_time_decay
              + w2 * block_recency_factor
              + w3 * cost_factor

        Higher score = higher priority to KEEP (less likely to be evicted).
        Eviction selects the block with the *minimum* score.
        """
        entry = self.cache[block_hash]
        workflow_id = entry["workflow_id"]

        # 1. Workflow time-decay: 2^{-(now - workflow_last_activity) * rate}
        # Active workflows (small age) → decay factor near 1.0
        # Paused workflows (large age) → decay factor near 0.0
        wf_last = self.workflow_last_activity.get(workflow_id, 0)
        wf_age = self._clock - wf_last
        # Guard against overflow for very large ages
        if wf_age > 1000 / max(self.decay_rate, 1e-6):
            workflow_decay = 0.0
        else:
            workflow_decay = pow(2.0, -wf_age * self.decay_rate)

        # 2. Block recency (LRU within workflow): 1 / (1 + block_age)
        block_age = self._clock - entry["last_access"]
        block_recency = 1.0 / (1.0 + block_age)

        # 3. Cost factor: expensive blocks are worth keeping
        cost = entry.get("cost", 0.0)
        cost_factor = cost / max(self._max_cost_seen, 1.0)

        score = (_WORKFLOW_RECENCY_WEIGHT * workflow_decay
                 + _BLOCK_RECENCY_WEIGHT * block_recency
                 + _COST_WEIGHT * min(1.0, cost_factor))
        return score

    # ------------------------------------------------------------------
    # Public API (same interface as PBKVInspiredCache, with workflow_id)
    # ------------------------------------------------------------------

    def access(self,
               block_hash: str,
               parent_hash: str = "",
               prefill_ms: float = 0.0,
               workflow_id: str = "") -> bool:
        """
        Access a block. Returns True if hit, False if miss.

        ``workflow_id`` identifies the workflow (program) this block belongs
        to; it is required for the ThunderAgent-inspired workflow-aware
        time-decay policy. When ``workflow_id`` is empty, the block is
        treated as belonging to a default workflow "".
        """
        self._clock += 1

        # Track max cost seen (for cost-factor normalization)
        if prefill_ms > self._max_cost_seen:
            self._max_cost_seen = prefill_ms

        # Update workflow last-activity on every access (hit or miss)
        if workflow_id:
            self.workflow_last_activity[workflow_id] = self._clock

        if block_hash in self.cache:
            # Hit: update metadata
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cache[block_hash]
            entry["last_access"] = self._clock
            entry["access_count"] += 1
            # Update workflow_id if it changed (shouldn't happen, but be safe)
            if workflow_id:
                entry["workflow_id"] = workflow_id
            return True
        else:
            # Miss: evict if over capacity, then insert
            self.misses += 1
            self.miss_cost_ms += prefill_ms

            while len(self.cache) >= self.capacity:
                self._evict()

            # Insert new block
            self.cache[block_hash] = {
                "parent_hash": parent_hash,
                "last_access": self._clock,
                "access_count": 1,
                "cost": prefill_ms,
                "workflow_id": workflow_id,
            }
            if parent_hash:
                self.parent_to_children[parent_hash].add(block_hash)
                self.child_to_parent[block_hash] = parent_hash
            return False

    # ------------------------------------------------------------------
    # Chain-aware eviction (mirrors APC-LRU / PBKV-Inspired)
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Evict the block with the minimum priority score, plus descendants."""
        if not self.cache:
            return

        # Compute scores for all cached blocks
        scores = {h: self._compute_priority(h) for h in self.cache}
        # Find block with minimum score (lowest priority to keep)
        victim = min(scores.items(), key=lambda x: x[1])[0]

        # Collect all descendants (recursive) — a descendant cannot be
        # reused without its ancestor, so evict the whole chain.
        to_evict: Set[str] = set()
        stack: List[str] = [victim]
        while stack:
            block = stack.pop()
            to_evict.add(block)
            stack.extend(self.parent_to_children.get(block, set()))

        # Evict all collected blocks
        for block in to_evict:
            parent = self.cache[block]["parent_hash"]
            if parent:
                children = self.parent_to_children.get(parent)
                if children and block in children:
                    children.remove(block)
                    if not children:
                        del self.parent_to_children[parent]
                if block in self.child_to_parent:
                    del self.child_to_parent[block]
            del self.cache[block]
            self.evictions += 1

    def __len__(self) -> int:
        return len(self.cache)
