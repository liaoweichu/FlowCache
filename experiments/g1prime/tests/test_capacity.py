"""Unit tests for ``experiments/g1prime/capacity.py``.

Verifies the GiB → block-count conversions for the G1′ absolute KV capacity
tiers used in the Go/No-Go verdict (1 / 2 / 4 / 6 GiB).

Reference (Qwen2.5-7B-Instruct BF16, block_size=16)::

    bytes_per_block = 917,504 B
    floor(1 GiB / 917504) = 1170 blocks
    floor(2 GiB / 917504) = 2340 blocks
    floor(4 GiB / 917504) = 4681 blocks
    floor(6 GiB / 917504) = 7021 blocks
"""

import capacity


BYTES_PER_BLOCK = 917_504


# ---------------------------------------------------------------------------
# 1. Single GiB → blocks conversion
# ---------------------------------------------------------------------------
def test_gib_to_blocks_conversion():
    """1 GiB → 1170 blocks for the G0 frozen Qwen2.5-7B BF16 config."""
    assert capacity.gib_to_blocks(1, BYTES_PER_BLOCK) == 1170


# ---------------------------------------------------------------------------
# 2. Full capacity table — all verdict tiers
# ---------------------------------------------------------------------------
def test_capacity_table():
    """All 4 verdict tiers (1 / 2 / 4 / 6 GiB) → correct block counts.

    The verdict tiers are exactly ``DEFAULT_BUDGETS_GIB`` (1/2/4/6).
    """
    expected = {
        1: 1170,
        2: 2340,
        4: 4681,
        6: 7021,
    }
    for gib, expected_blocks in expected.items():
        got = capacity.gib_to_blocks(gib, BYTES_PER_BLOCK)
        assert got == expected_blocks, (
            f"{gib} GiB: expected {expected_blocks} blocks, got {got}"
        )


def test_default_budgets_match_verdict_tiers():
    """DEFAULT_BUDGETS_GIB must be exactly [1, 2, 4, 6] (verdict tiers)."""
    assert capacity.DEFAULT_BUDGETS_GIB == [1, 2, 4, 6]


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------
def test_gib_to_blocks_zero_or_negative_bpb():
    """bytes_per_block ≤ 0 → 0 blocks (defensive, avoids ZeroDivisionError)."""
    assert capacity.gib_to_blocks(1, 0) == 0
    assert capacity.gib_to_blocks(1, -1) == 0


def test_gib_to_blocks_uses_floor():
    """Verify floor semantics: 1 GiB has 1170 whole blocks + slack < 1 block.

    1 GiB = 1,073,741,824 B. 1170 × 917,504 = 1,073,479,680 B → slack
    = 262,144 B = 0.286 of a block. So 1 GiB → 1170 blocks (floor), not 1171.
    """
    one_gib_bytes = 1024 ** 3
    blocks = capacity.gib_to_blocks(1, BYTES_PER_BLOCK)
    used = blocks * BYTES_PER_BLOCK
    slack = one_gib_bytes - used
    # Slack is non-negative (we never exceed the budget) and < 1 block.
    assert 0 <= slack < BYTES_PER_BLOCK
    assert blocks == 1170  # confirmed by the slack calculation
