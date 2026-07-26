"""
Test PBKVInspiredCache (inspired variant).

Verifies capacity boundaries, the four reuse-score factors (recency,
parent-chain, neighborhood frequency, cost-per-distance), chain eviction,
and end-to-end behavior on a synthetic trace.

Runnable as:
  - ``py -m pytest experiments/e1/baselines/test_pbkv_inspired.py -v``
    from the project root, OR
  - ``py -m pytest test_pbkv_inspired.py -v`` from inside the
    ``experiments/e1/baselines/`` directory.
"""

import math
import sys
from pathlib import Path

# Allow running from both the project root and the baselines directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pbkv_inspired import PBKVInspiredCache  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Capacity boundaries
# ---------------------------------------------------------------------------

def test_pbkv_capacity_boundaries():
    # capacity=0 clamped to 1: with alternating distinct blocks, every
    # access is a miss (the single cached block is always the wrong one).
    pbkv0 = PBKVInspiredCache(0)
    for h in ["A", "B", "A", "B"]:
        pbkv0.access(h, prefill_ms=1.0)
    assert pbkv0.hits == 0
    assert pbkv0.misses == 4

    # capacity=10: 5 unique blocks accessed twice → 5 hits, 5 misses.
    pbkv10 = PBKVInspiredCache(10)
    blocks = ["A", "B", "C", "D", "E"]
    for h in blocks + blocks:
        pbkv10.access(h, prefill_ms=1.0)
    assert pbkv10.hits == 5
    assert pbkv10.misses == 5


# ---------------------------------------------------------------------------
# 2. Reuse score: recency
# ---------------------------------------------------------------------------

def test_pbkv_reuse_score_higher_for_recent_blocks():
    # Insert A at clock 0, then advance the clock 99 times by hitting a
    # different block X (still well within capacity=10, so A is retained).
    # Finally insert B (miss) so that B is exactly 1 access old while A is
    # 100 accesses old. With all other factors equal, B's recency_factor
    # (exp(-0.01 * 1) ≈ 0.99) dominates A's (exp(-0.01 * 100) ≈ 0.37),
    # so B's reuse score must be strictly higher.
    pbkv = PBKVInspiredCache(capacity=10)
    pbkv.access("A", prefill_ms=10.0)            # A at clock 0
    for _ in range(99):
        pbkv.access("X", prefill_ms=10.0)        # X hit 98 times, clock → 100
    pbkv.access("B", prefill_ms=10.0)            # B at clock 100, clock → 101
    # At clock=101: A.last_access=0 (101 steps ago), B.last_access=100 (1 step ago).

    score_a = pbkv._compute_reuse_score("A")
    score_b = pbkv._compute_reuse_score("B")
    assert score_b > score_a, (
        f"recent block B (score={score_b:.4f}) should outrank stale block A "
        f"(score={score_a:.4f})"
    )


# ---------------------------------------------------------------------------
# 3. Reuse score: parent chain completeness
# ---------------------------------------------------------------------------

def test_pbkv_reuse_score_higher_with_parent_cached():
    # Insert P (parent), then A whose parent is P (in cache), and B whose
    # parent is Q (NOT in cache). With recency, frequency, and cost held
    # approximately equal, A's parent_chain_factor (1.0) must outrank B's
    # (0.3), so A's reuse score is strictly higher.
    pbkv = PBKVInspiredCache(capacity=10)
    pbkv.access("P", prefill_ms=10.0)
    pbkv.access("A", parent_hash="P", prefill_ms=10.0)  # parent P cached
    pbkv.access("B", parent_hash="Q", prefill_ms=10.0)   # parent Q not cached

    score_a = pbkv._compute_reuse_score("A")
    score_b = pbkv._compute_reuse_score("B")
    assert score_a > score_b, (
        f"block with cached parent (score={score_a:.4f}) should outrank "
        f"block without cached parent (score={score_b:.4f})"
    )


# ---------------------------------------------------------------------------
# 4. Reuse score: neighborhood frequency
# ---------------------------------------------------------------------------

def test_pbkv_reuse_score_higher_for_frequent_blocks():
    # A is accessed once; B is accessed five times (4 hits after the initial
    # miss). With cost and parent_chain held equal, B's higher access count
    # gives it a higher neighborhood_freq_factor (1.0 vs 0.2). B also has a
    # more recent last_access, reinforcing the ordering.
    pbkv = PBKVInspiredCache(capacity=10)
    pbkv.access("A", prefill_ms=10.0)             # A.access_count = 1
    pbkv.access("B", prefill_ms=10.0)             # B.access_count = 1
    for _ in range(4):
        pbkv.access("B", prefill_ms=10.0)         # B.access_count = 5

    score_a = pbkv._compute_reuse_score("A")
    score_b = pbkv._compute_reuse_score("B")
    assert score_b > score_a, (
        f"frequent block B (score={score_b:.4f}) should outrank rare block A "
        f"(score={score_a:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. Chain eviction (chain consistency, same semantics as APC-LRU)
# ---------------------------------------------------------------------------

def test_pbkv_chain_eviction():
    # capacity=2, build a chain A (parent="") → B (parent=A) → C (parent=B).
    # After inserting A and B, the cache is full. Inserting C must evict the
    # lowest-score block; A has parent_chain_factor=0.3 (no parent) while B
    # has parent_chain_factor=1.0 (parent A in cache), so A is evicted.
    # Chain-aware eviction recursively evicts A's descendant B as well,
    # leaving only C in the cache.
    pbkv = PBKVInspiredCache(capacity=2)
    pbkv.access("A", parent_hash="", prefill_ms=10.0)    # miss
    pbkv.access("B", parent_hash="A", prefill_ms=10.0)   # miss
    pbkv.access("C", parent_hash="B", prefill_ms=10.0)   # miss, evicts A+B

    assert len(pbkv.cache) == 1
    assert "C" in pbkv.cache
    assert "A" not in pbkv.cache
    assert "B" not in pbkv.cache
    assert pbkv.evictions >= 2  # both A and B were evicted


# ---------------------------------------------------------------------------
# 6. End-to-end run on a synthetic trace
# ---------------------------------------------------------------------------

def test_pbkv_runs_on_trace():
    # Synthetic trace mixing hits, misses, parent-chain accesses, and an
    # eviction under capacity=3. Verifies that hit/miss counters are
    # non-negative, sum to the trace length, and that saved/miss cost are
    # non-negative.
    trace = [
        ("A", "",       10.0),  # miss, insert A
        ("B", "A",      20.0),  # miss, insert B (parent A cached)
        ("C", "B",      30.0),  # miss, insert C (parent B cached)
        ("A", "",       10.0),  # hit
        ("B", "A",      20.0),  # hit
        ("D", "",       15.0),  # miss → evicts the lowest-score block/chain
        ("A", "",       10.0),  # likely hit if A survived, else miss
    ]
    pbkv = PBKVInspiredCache(capacity=3)
    for block_hash, parent_hash, prefill_ms in trace:
        pbkv.access(block_hash, parent_hash, prefill_ms)

    total = len(trace)
    assert pbkv.hits >= 0
    assert pbkv.misses >= 0
    assert pbkv.hits + pbkv.misses == total
    assert pbkv.saved_prefill_ms >= 0.0
    assert pbkv.miss_cost_ms >= 0.0
    assert pbkv.evictions >= 0
    # Sanity: every saved_prefill_ms must come from a hit, so the hit count
    # being non-zero is required for saved_prefill_ms to be non-zero.
    if pbkv.saved_prefill_ms > 0.0:
        assert pbkv.hits > 0
