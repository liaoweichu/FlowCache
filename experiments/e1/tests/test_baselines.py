"""
Test 6 baseline cache classes in compare_oracle.py.

Background: G1 needs 6 baselines (LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost)
to compute cost-aware headroom. These tests verify capacity boundaries, prefix
chain semantics (APC-LRU), and cost-aware eviction (SizeCost/Oracle-Cost).
"""

import sys
from pathlib import Path

# Make experiments/e1/ and experiments/g0/ importable (same pattern as
# other test files in this directory).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import compare_oracle as co


# ---------------------------------------------------------------------------
# 1. LRU capacity boundaries
# ---------------------------------------------------------------------------

def test_lru_capacity_boundaries():
    # capacity=0 is clamped to 1 by max(1, capacity); with alternating
    # distinct blocks every access is a miss.
    lru0 = co.LRUCache(0)
    for h in ["A", "B", "A", "B"]:
        lru0.access(h, prefill_ms=1.0)
    assert lru0.hits == 0
    assert lru0.misses == 4

    # capacity=1: only consecutive duplicates are hits.
    lru1 = co.LRUCache(1)
    for h in ["A", "A", "B", "B", "A", "A"]:
        lru1.access(h, prefill_ms=1.0)
    assert lru1.hits == 3    # 2nd A, 2nd B, 3rd A
    assert lru1.misses == 3  # 1st A, 1st B, 3rd access A (after B evicted A)

    # capacity=10: 5 unique blocks accessed twice → 5 hits, 5 misses.
    lru10 = co.LRUCache(10)
    blocks = ["A", "B", "C", "D", "E"]
    for h in blocks + blocks:
        lru10.access(h, prefill_ms=1.0)
    assert lru10.hits == 5
    assert lru10.misses == 5


# ---------------------------------------------------------------------------
# 2. GDSF capacity boundaries (same scenarios as LRU)
# ---------------------------------------------------------------------------

def test_gdsf_capacity_boundaries():
    # capacity=0 clamped to 1: alternating distinct blocks → all miss.
    gdsf0 = co.GDSFCache(0)
    for h in ["A", "B", "A", "B"]:
        gdsf0.access(h, prefill_ms=1.0)
    assert gdsf0.hits == 0
    assert gdsf0.misses == 4

    # capacity=1: only consecutive duplicates are hits.
    gdsf1 = co.GDSFCache(1)
    for h in ["A", "A", "B", "B", "A", "A"]:
        gdsf1.access(h, prefill_ms=1.0)
    assert gdsf1.hits == 3
    assert gdsf1.misses == 3

    # capacity=10: 5 unique blocks accessed twice → 5 hits, 5 misses.
    gdsf10 = co.GDSFCache(10)
    blocks = ["A", "B", "C", "D", "E"]
    for h in blocks + blocks:
        gdsf10.access(h, prefill_ms=1.0)
    assert gdsf10.hits == 5
    assert gdsf10.misses == 5


# ---------------------------------------------------------------------------
# 3. SizeCost cost-aware eviction
# ---------------------------------------------------------------------------

def test_sizecost_cost_aware_eviction():
    # Capacity=2, uniform size=16. Priority = clock + (cost * freq) / size.
    # A has low cost (10), B has high cost (100). After A is hit (freq=2),
    # A's priority = 0 + (10*2)/16 = 1.25; B's priority = 0 + 100/16 = 6.25.
    # On inserting C, the min-priority block (A) is evicted even though A
    # was accessed more recently — cost awareness keeps the expensive block.
    sc = co.SizeCostCache(2)
    sc.access("A", prefill_ms=10.0, size=16)    # miss
    sc.access("B", prefill_ms=100.0, size=16)   # miss
    assert sc.access("A", prefill_ms=10.0, size=16) is True   # hit, A.freq=2
    sc.access("C", prefill_ms=50.0, size=16)    # miss, evicts A (priority 1.25)

    # A evicted (lowest priority); B retained (expensive).
    assert "A" not in sc.cache
    assert "B" in sc.cache
    assert "C" in sc.cache

    # Accessing A again is a miss (it was evicted).
    assert sc.access("A", prefill_ms=10.0, size=16) is False


# ---------------------------------------------------------------------------
# 4. APC-LRU prefix chain hit
# ---------------------------------------------------------------------------

def test_apc_lru_prefix_chain_hit():
    # Capacity=10, build a chain root→A→B. "root" is just a label and is
    # never inserted, so A is a full miss. B's parent is A (in cache), so
    # inserting B triggers a prefix hit on A: A.last_access is refreshed
    # and A's prefill cost is added to saved_prefill_ms.
    apc = co.APCLRUCache(10)

    # Access A (parent="root" not in cache) → full miss, no prefix hit.
    apc.access("A", parent_hash="root", prefill_ms=10.0)
    assert apc.misses == 1
    assert apc.hits == 0
    assert apc.saved_prefill_ms == 0.0  # no hit, no prefix hit

    # Access B (parent="A" in cache) → miss, but A gets a prefix hit.
    apc.access("B", parent_hash="A", prefill_ms=20.0)
    assert apc.misses == 2  # B itself is a miss
    assert apc.hits == 0    # B is not a direct hit
    # A's prefill (10) was added to saved_prefill_ms via the prefix hit.
    assert apc.saved_prefill_ms == 10.0

    # Access A again → hit. A is still in cache because its last_access was
    # refreshed by the prefix hit when B was inserted.
    is_hit = apc.access("A", parent_hash="root", prefill_ms=10.0)
    assert is_hit is True
    assert apc.hits == 1
    # saved_prefill_ms: 10 (prefix hit) + 10 (direct hit) = 20.
    assert apc.saved_prefill_ms == 20.0


# ---------------------------------------------------------------------------
# 5. APC-LRU prefix chain eviction
# ---------------------------------------------------------------------------

def test_apc_lru_prefix_chain_eviction():
    # Capacity=2, build a chain A (parent="") → B (parent=A) → C (parent=B).
    # After inserting A and B the cache is full. Inserting C evicts the
    # oldest block (A); chain-aware eviction recursively evicts A's
    # descendant B as well, so both A and B are removed.
    apc = co.APCLRUCache(2)
    apc.access("A", parent_hash="", prefill_ms=10.0)   # miss
    apc.access("B", parent_hash="A", prefill_ms=20.0)  # miss, prefix hit on A
    apc.access("C", parent_hash="B", prefill_ms=30.0)  # miss, chain-evicts A+B

    # Only C remains; the whole A→B chain was evicted.
    assert len(apc.cache) == 1
    assert "C" in apc.cache
    assert "A" not in apc.cache
    assert "B" not in apc.cache
    assert apc.evictions >= 2  # both A and B were evicted


# ---------------------------------------------------------------------------
# 6. Belady capacity boundaries
# ---------------------------------------------------------------------------

def test_belady_capacity_boundaries():
    # capacity=0 clamped to 1: with alternating distinct blocks, every
    # access is a miss (the single cached block is always the wrong one).
    future1 = {"A": [0, 2], "B": [1, 3]}
    bel0 = co.BeladyOracle(0, future1)
    seq1 = ["A", "B", "A", "B"]
    for idx, h in enumerate(seq1):
        bel0.access(h, idx, prefill_ms=1.0)
    assert bel0.hits == 0
    assert bel0.misses == 4

    # capacity=10 (large enough): all hit except the first access of each
    # block.
    seq2 = ["A", "B", "C", "D", "E"] * 2
    future2 = {}
    for idx, h in enumerate(seq2):
        future2.setdefault(h, []).append(idx)
    bel10 = co.BeladyOracle(10, future2)
    for idx, h in enumerate(seq2):
        bel10.access(h, idx, prefill_ms=1.0)
    assert bel10.hits == 5
    assert bel10.misses == 5


# ---------------------------------------------------------------------------
# 7. Oracle-Cost cost-aware eviction
# ---------------------------------------------------------------------------

def test_oracle_cost_cost_aware_eviction():
    # Capacity=2. Block costs (first-seen prefill_ms): A=1000, B=1, C=50.
    # future_accesses: A → [0, 100], B → [1, 3], C → [2].
    # At idx=2 (inserting C), cache has {A, B}; need to evict one.
    #   A: next_use=100, ratio = 1000/100 = 10
    #   B: next_use=3,   ratio = 1/3     ≈ 0.333
    # Oracle-Cost evicts min ratio → B (cheap to recompute per unit distance).
    # Belady would evict max next-use distance → A (next at 100). They differ.
    future = {"A": [0, 100], "B": [1, 3], "C": [2]}
    oc = co.OracleCostCache(2, future)
    oc.access("A", 0, prefill_ms=1000.0)  # miss, insert A
    oc.access("B", 1, prefill_ms=1.0)     # miss, insert B
    oc.access("C", 2, prefill_ms=50.0)    # miss, evict B (low ratio)

    # B evicted (cheap block); A retained (expensive block).
    assert "B" not in oc.cache
    assert "A" in oc.cache

    # Access B at idx=3 → miss (B was evicted by Oracle-Cost).
    assert oc.access("B", 3, prefill_ms=1.0) is False
    # Access A at idx=100 → hit (A was retained).
    assert oc.access("A", 100, prefill_ms=1000.0) is True


# ---------------------------------------------------------------------------
# 8. Oracle-Cost vs Belady miss cost
# ---------------------------------------------------------------------------

def test_oracle_cost_vs_belady_miss_cost():
    # Same scenario as test 7. Oracle-Cost evicts the cheap block B at idx=2
    # and reloads it at idx=3 (cost=1). Belady evicts the far block A at
    # idx=2 and reloads it at idx=100 (cost=1000). So Oracle-Cost incurs a
    # much smaller miss_cost_ms.
    future = {"A": [0, 100], "B": [1, 3], "C": [2]}
    accesses = [
        (0, "A", 1000.0),
        (1, "B", 1.0),
        (2, "C", 50.0),
        (3, "B", 1.0),
        (100, "A", 1000.0),
    ]

    oc = co.OracleCostCache(2, future)
    for idx, h, cost in accesses:
        oc.access(h, idx, prefill_ms=cost)

    bel = co.BeladyOracle(2, future)
    for idx, h, cost in accesses:
        bel.access(h, idx, prefill_ms=cost)

    # Oracle-Cost reloads the cheap block (B, cost=1); Belady reloads the
    # expensive block (A, cost=1000).
    assert oc.miss_cost_ms < bel.miss_cost_ms


# ---------------------------------------------------------------------------
# 9. build_access_trace cost apportionment
# ---------------------------------------------------------------------------

def test_build_access_trace_cost_apportion():
    # 1 step with prefill_ms=500, token_count=160 (10 blocks of 16 tokens).
    # Each block's prefill_ms should be apportioned as 500 * 16 / 160 = 50.
    blocks = []
    for i in range(10):
        blocks.append({
            "block_hash": f"blk_{i}",
            "parent_hash": f"blk_{i-1}" if i > 0 else "",
            "token_range_start": i * 16,
            "token_range_end": (i + 1) * 16,
        })
    traj = {
        "meta": {"workflow_id": "wf_test", "block_size": 16},
        "steps": [{
            "step_id": 0,
            "prefill_ms": 500.0,
            "token_count": 160,
            "block_assignments": blocks,
        }],
    }
    trace = co.build_access_trace([traj])
    assert len(trace) == 10
    for i, acc in enumerate(trace):
        # 500 ms apportioned uniformly across 10 equal-token blocks → 50 ms.
        assert abs(acc["prefill_ms"] - 50.0) < 1e-9
        assert acc["size"] == 16
        assert acc["block_hash"] == f"blk_{i}"
        assert acc["parent_hash"] == (f"blk_{i-1}" if i > 0 else "")
        assert acc["step_id"] == 0
        assert acc["workflow_id"] == "wf_test"
