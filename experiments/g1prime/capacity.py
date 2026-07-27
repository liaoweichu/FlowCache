"""
G1′ Absolute KV Capacity: 1 / 2 / 4 / 6 GiB budgets grounded in a 24 GB GPU.
============================================================================

Replaces G1's percentage budgets (10 % / 25 % / 50 % / 100 % of peak
working set), which produced a 41.4 GiB KV budget at 10 % — far exceeding
the 24 GB GPU's available KV memory, and structurally forced 0 % headroom
at 100 % (since unlimited capacity = zero misses by definition).

G1′ uses absolute GiB budgets:

  - **1 / 2 / 4 / 6 GiB**: realistic KV budgets on a 24 GB GPU after
    weights (~14 GiB for Qwen2.5-7B BF16), activations, and temporary
    buffers. These are the tiers that participate in the Go/No-Go verdict.
  - **100 % (unlimited)**: sanity-check upper bound only, NOT used for
    Go/No-Go verdict (headroom is structurally zero when capacity is
    unbounded).

Block byte size is computed from the G0 frozen model config
(Qwen2.5-7B-Instruct, BF16, block_size=16); see
:func:`cost_model.compute_bytes_per_block`.

Usage:
    python experiments/g1prime/capacity.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Default capacity tiers (GiB). The unlimited (100 %) tier is a sanity-check
# only and is NOT included in the Go/No-Go verdict.
# ---------------------------------------------------------------------------
DEFAULT_BUDGETS_GIB: List[int] = [1, 2, 4, 6]

# Whether to print the unlimited sanity-check row in the capacity table.
SANITY_CHECK_UNLIMITED: bool = True


# ---------------------------------------------------------------------------
# GiB → block count
# ---------------------------------------------------------------------------
def gib_to_blocks(gib: float, bytes_per_block: int) -> int:
    """Convert a GiB budget to an integer block count.

    Formula::

        capacity_blocks = floor(gib × 1024³ / bytes_per_block)

    For Qwen2.5-7B-Instruct BF16 (bytes_per_block = 917,504 B)::

        1 GiB → floor(1,073,741,824 / 917,504) = 1,170 blocks
        2 GiB → 2,341 blocks
        4 GiB → 4,682 blocks
        6 GiB → 7,024 blocks

    Args:
        gib: Capacity in GiB (e.g. 1, 2, 4, 6).
        bytes_per_block: Byte size of one KV block, e.g. from
            :func:`cost_model.compute_bytes_per_block`.

    Returns:
        Integer block count (floor division). ``0`` if ``bytes_per_block``
        is non-positive.
    """
    if bytes_per_block <= 0:
        return 0
    return int(gib * (1024 ** 3) / bytes_per_block)


# ---------------------------------------------------------------------------
# Capacity summary
# ---------------------------------------------------------------------------
def print_capacity_table(
    bytes_per_block: int,
    budgets_gib: Optional[List[float]] = None,
    include_unlimited: bool = True,
) -> None:
    """Print the capacity-tier table (GiB → blocks → MiB).

    Args:
        bytes_per_block: Byte size of one KV block.
        budgets_gib: Capacity tiers in GiB. Defaults to
            :data:`DEFAULT_BUDGETS_GIB` (1/2/4/6 GiB).
        include_unlimited: If True, also print the "unlimited" sanity-check
            row (does NOT participate in Go/No-Go verdict).
    """
    if budgets_gib is None:
        budgets_gib = list(DEFAULT_BUDGETS_GIB)

    kib = bytes_per_block / 1024.0
    mib = bytes_per_block / (1024.0 * 1024.0)

    print(
        f"  bytes_per_block = {bytes_per_block:,} B "
        f"({kib:.1f} KiB ≈ {mib:.4f} MiB)"
    )
    print()
    print(
        f"  {'Budget (GiB)':>14s} | {'Blocks':>12s} | "
        f"{'Equivalent MiB':>16s} | Note"
    )
    print(
        f"  {'-' * 14} | {'-' * 12} | {'-' * 16} | {'-' * 32}"
    )

    for gib in budgets_gib:
        blocks = gib_to_blocks(gib, bytes_per_block)
        equiv_mib = gib * 1024.0
        print(
            f"  {gib:>14.1f} | {blocks:>12,d} | {equiv_mib:>16.1f} | "
            f"Go/No-Go verdict tier"
        )

    if include_unlimited:
        # The unlimited tier is a sanity-check upper bound; it is NOT a
        # verdict tier. Marked explicitly so the verdict module can skip it.
        print(
            f"  {'unlimited':>14s} | {'∞':>12s} | {'∞':>16s} | "
            f"sanity check only (NOT in verdict)"
        )

    print()
    print(
        f"  Note: 100 % (unlimited) is excluded from Go/No-Go because "
        f"headroom is"
    )
    print(
        f"        structurally zero when capacity is unbounded (no misses "
        f"→ no cost)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("G1′ Absolute KV Capacity — 1 / 2 / 4 / 6 GiB (24 GB GPU)")
    print("=" * 78)

    # Compute bytes_per_block from the G0 frozen model config.
    # Deferred import to avoid module-level circular dependency:
    # cost_model.main() also imports from capacity inside its main().
    try:
        from cost_model import G0_MODEL_CONFIG, compute_bytes_per_block
    except Exception as exc:
        print(f"ERROR: cannot import cost_model: {exc}", file=sys.stderr)
        return 1

    bytes_per_block = compute_bytes_per_block(
        num_hidden_layers=G0_MODEL_CONFIG["num_hidden_layers"],
        num_kv_heads=G0_MODEL_CONFIG["num_kv_heads"],
        head_dim=G0_MODEL_CONFIG["head_dim"],
        dtype_bytes=G0_MODEL_CONFIG["dtype_bytes"],
        block_size=G0_MODEL_CONFIG["block_size"],
    )

    print()
    print_capacity_table(
        bytes_per_block,
        budgets_gib=list(DEFAULT_BUDGETS_GIB),
        include_unlimited=SANITY_CHECK_UNLIMITED,
    )

    # Sanity check: verify the GiB → blocks conversion arithmetic.
    print("Sanity check (1 GiB):")
    one_gib_bytes = 1024 ** 3
    one_gib_blocks = gib_to_blocks(1, bytes_per_block)
    used_bytes = one_gib_blocks * bytes_per_block
    slack_bytes = one_gib_bytes - used_bytes
    print(f"  1 GiB = {one_gib_bytes:,} B")
    print(
        f"  blocks at 1 GiB = {one_gib_blocks:,} "
        f"(× {bytes_per_block:,} B/block = {used_bytes:,} B used, "
        f"{slack_bytes:,} B slack = {slack_bytes / bytes_per_block:.3f} block)"
    )

    print()
    print("=" * 78)
    print("Capacity OK.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
