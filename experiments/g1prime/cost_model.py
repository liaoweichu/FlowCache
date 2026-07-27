"""
G1′ Cost Model: Correct prefill_ms attribution + physical KV capacity.
======================================================================

Fixes three G1 protocol issues:

  1. **prefill_ms attribution** — G1 attributed prefill_ms to *assistant
     output blocks*; G1′ attributes it to the *request* (per assistant
     invocation). See :func:`compute_request_miss_cost`.

  2. **Fixed 8 ms fallback** — G1 used ``block_size * 0.5`` (= 8 ms) for
     system / user / tool blocks whose step-level ``prefill_ms`` was zero
     (see ``experiments/e1/compare_oracle.py`` line ~615). G1′ replaces
     this with ``per_token_prefill_rate × block_token_count``, where the
     rate is the median of measured assistant ``prefill_ms / token_count``.
     See :func:`compute_per_token_prefill_rate` and
     :func:`estimate_block_prefill_ms`.

  3. **Capacity definition** — G1 used percentages of peak working set
     (10 % ≈ 41.4 GiB), which exceeded a 24 GB GPU and structurally
     forced 0 % headroom at 100 %. G1′ uses absolute GiB budgets; see
     ``experiments/g1prime/capacity.py``.

This module is pure-Python (no Qwen tokenizer / transformers needed); it
operates on already-recorded trajectory JSON files under
``experiments/e1/traces/bf16/tau_bench/``. Read-only w.r.t.
``experiments/e1/``.

Usage:
    python experiments/g1prime/cost_model.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parents[1]
TRACE_DIR: Path = PROJECT_ROOT / "experiments" / "e1" / "traces" / "bf16" / "tau_bench"

# ---------------------------------------------------------------------------
# G0 frozen model config (Qwen2.5-7B-Instruct, BF16)
# Sourced from experiments/e1/config.yaml + Qwen2.5-7B-Instruct config.json.
# Only ``config_hash`` is stored in each trajectory; the architectural
# constants below are the G0-frozen lookup that ``config_hash`` binds to.
# ---------------------------------------------------------------------------
G0_MODEL_CONFIG: Dict[str, int] = {
    "num_hidden_layers": 28,
    "num_kv_heads": 4,        # GQA (num_key_value_heads)
    "head_dim": 128,          # hidden_size / num_attention_heads = 3584 / 28
    "dtype_bytes": 2,         # bfloat16
    "block_size": 16,         # tokens per KV block
}


# ---------------------------------------------------------------------------
# 1. bytes_per_block
# ---------------------------------------------------------------------------
def compute_bytes_per_block(
    num_hidden_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    block_size: int,
) -> int:
    """Compute the byte size of a single KV cache block.

    Formula::

        bytes_per_block = block_size
                        × num_hidden_layers
                        × 2                # K and V tensors
                        × num_kv_heads
                        × head_dim
                        × dtype_bytes

    For Qwen2.5-7B-Instruct (BF16, block_size=16)::

        16 × 28 × 2 × 4 × 128 × 2 = 917,504 B = 896 KiB ≈ 0.875 MiB

    Args:
        num_hidden_layers: Number of transformer layers (28 for Qwen2.5-7B).
        num_kv_heads: Number of KV attention heads under GQA (4).
        head_dim: Dimension of each attention head (128).
        dtype_bytes: Bytes per KV element (2 for BF16, 1 for Q8, 0.5 for Q4).
        block_size: Tokens per block (16).

    Returns:
        Byte size of one KV block (integer).
    """
    return (
        block_size
        * num_hidden_layers
        * 2  # K + V
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )


# ---------------------------------------------------------------------------
# 2. per-token prefill rate (robust median over measured assistant steps)
# ---------------------------------------------------------------------------
def compute_per_token_prefill_rate(
    trajectories: Sequence[Dict],
) -> float:
    """Compute the robust per-token prefill rate from measured assistant steps.

    Iterates every assistant step in every trajectory, collects
    ``(prefill_ms, token_count)`` pairs where both are strictly positive,
    computes ``prefill_ms / token_count`` for each, and returns the
    **median**.

    The median is chosen for robustness against outliers (very short or
    very long prefills, scheduling jitter, GPU thermal throttling, etc.).
    This rate replaces G1's fixed ``block_size * 0.5`` (= 8 ms) fallback
    for non-assistant blocks (system / user / tool).

    Args:
        trajectories: Sequence of trajectory dicts; each must have a
            ``steps`` list whose assistant entries carry ``prefill_ms``
            (float) and ``token_count`` (int).

    Returns:
        Median per-token prefill rate in ms/token. ``0.0`` if no valid
        ``(prefill_ms > 0, token_count > 0)`` assistant step is found.
    """
    rates: List[float] = []
    for traj in trajectories:
        for step in traj.get("steps", []):
            if step.get("role") != "assistant":
                continue
            prefill_ms = step.get("prefill_ms", 0.0) or 0.0
            token_count = step.get("token_count", 0) or 0
            if prefill_ms <= 0.0 or token_count <= 0:
                continue
            rates.append(prefill_ms / token_count)

    if not rates:
        return 0.0
    return float(statistics.median(rates))


# ---------------------------------------------------------------------------
# 3. block-level prefill_ms estimate (NO fixed 8 ms fallback)
# ---------------------------------------------------------------------------
def estimate_block_prefill_ms(
    block_token_count: int,
    per_token_rate: float,
) -> float:
    """Estimate a block's prefill_ms from the per-token rate.

    Replaces G1's fixed ``block_size * 0.5`` (= 8 ms) fallback used for
    system / user / tool blocks. The estimate is linear in the block's
    token count, which is the correct first-order model for prefill
    (compute-bound on attention).

    Args:
        block_token_count: Number of tokens in the block (1..block_size).
        per_token_rate: Per-token prefill rate in ms/token, e.g. from
            :func:`compute_per_token_prefill_rate`.

    Returns:
        Estimated prefill_ms for the block. ``0.0`` if either input is
        non-positive.
    """
    if block_token_count <= 0 or per_token_rate <= 0.0:
        return 0.0
    return float(block_token_count) * per_token_rate


# ---------------------------------------------------------------------------
# 4. request-level miss-cost attribution
# ---------------------------------------------------------------------------
def compute_request_miss_cost(
    missed_blocks: Sequence[Dict],
    request_total_tokens: int,
    request_prefill_ms: float,
) -> float:
    """Attribute a request's prefill_ms to its missed blocks proportionally.

    G1 attributed prefill_ms to *output blocks*; G1′ attributes it to the
    *request*. When ``k`` of the request's prefix blocks are cache misses,
    the request's miss-prefill cost is the fraction of the request's total
    tokens covered by those missed blocks, times the request's measured
    prefill_ms::

        miss_cost_ms = request_prefill_ms
                     × ( sum(missed_block_i.block_token_count)
                         / request_total_tokens )

    This is the correct request-level attribution: a request that misses
    half its prefix tokens pays half its prefill cost.

    Args:
        missed_blocks: List of block dicts; each must contain
            ``block_token_count`` (int). Other keys are ignored.
        request_total_tokens: Total tokens in the request's prefix (≥ 1).
        request_prefill_ms: The request's measured prefill_ms (≥ 0).

    Returns:
        Attributed miss cost in ms. ``0.0`` if the request has no tokens,
        no prefill, or no missed blocks.
    """
    if request_total_tokens <= 0 or request_prefill_ms <= 0.0:
        return 0.0
    total_missed_tokens = sum(
        int(b.get("block_token_count", 0)) for b in missed_blocks
    )
    if total_missed_tokens <= 0:
        return 0.0
    return request_prefill_ms * (total_missed_tokens / request_total_tokens)


# ---------------------------------------------------------------------------
# Trajectory loading (self-contained; no e1/trace_utils dependency)
# ---------------------------------------------------------------------------
def load_trajectories(trace_dir: Path) -> List[Dict]:
    """Load all trajectory JSON files from ``trace_dir``.

    Skips files starting with ``_`` (checkpoints, reports). Read-only.

    Args:
        trace_dir: Directory containing per-episode ``*.json`` files.

    Returns:
        List of trajectory dicts (sorted by filename).
    """
    if not trace_dir.exists():
        return []
    files = sorted(
        p for p in trace_dir.glob("*.json") if not p.name.startswith("_")
    )
    trajectories: List[Dict] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                trajectories.append(json.load(f))
        except Exception as exc:
            print(f"  WARN: failed to load {fp.name}: {exc}", file=sys.stderr)
    return trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("G1′ Cost Model — bytes_per_block + per-token prefill rate + capacity")
    print("=" * 78)

    # --- 1. bytes_per_block from G0 frozen model config ---
    bytes_per_block = compute_bytes_per_block(
        num_hidden_layers=G0_MODEL_CONFIG["num_hidden_layers"],
        num_kv_heads=G0_MODEL_CONFIG["num_kv_heads"],
        head_dim=G0_MODEL_CONFIG["head_dim"],
        dtype_bytes=G0_MODEL_CONFIG["dtype_bytes"],
        block_size=G0_MODEL_CONFIG["block_size"],
    )

    print("\n[1] G0 frozen model config (Qwen2.5-7B-Instruct, BF16):")
    for k, v in G0_MODEL_CONFIG.items():
        print(f"    {k:20s}: {v}")

    print(f"\n    bytes_per_block = block_size × num_hidden_layers × 2 (K+V)")
    print(f"                      × num_kv_heads × head_dim × dtype_bytes")
    print(
        f"                   = {G0_MODEL_CONFIG['block_size']} × "
        f"{G0_MODEL_CONFIG['num_hidden_layers']} × 2 × "
        f"{G0_MODEL_CONFIG['num_kv_heads']} × "
        f"{G0_MODEL_CONFIG['head_dim']} × "
        f"{G0_MODEL_CONFIG['dtype_bytes']}"
    )
    print(f"                   = {bytes_per_block:,} B")
    print(
        f"                   = {bytes_per_block / 1024:.1f} KiB "
        f"(≈ {bytes_per_block / (1024 * 1024):.4f} MiB)"
    )

    # --- 2. per-token prefill rate from measured assistant steps ---
    print(f"\n[2] Loading trajectories from:")
    print(f"    {TRACE_DIR}")
    trajectories = load_trajectories(TRACE_DIR)
    print(f"    loaded {len(trajectories)} trajectories")

    per_token_rate = 0.0
    if trajectories:
        per_token_rate = compute_per_token_prefill_rate(trajectories)

        asst_with_prefill = 0
        asst_total = 0
        for traj in trajectories:
            for step in traj.get("steps", []):
                if step.get("role") != "assistant":
                    continue
                asst_total += 1
                if (step.get("prefill_ms", 0.0) or 0.0) > 0.0 and (
                    step.get("token_count", 0) or 0
                ) > 0:
                    asst_with_prefill += 1

        print(f"    assistant steps (total)        : {asst_total}")
        print(f"    assistant steps with prefill>0 : {asst_with_prefill}")
        print(
            f"    per-token prefill rate (median): "
            f"{per_token_rate:.6f} ms/token"
        )

        # Compare with G1's fixed 8 ms / 16-token fallback.
        g1_fallback_per_block = G0_MODEL_CONFIG["block_size"] * 0.5
        g1_fallback_per_token = g1_fallback_per_block / G0_MODEL_CONFIG["block_size"]
        new_estimate_per_block = estimate_block_prefill_ms(
            G0_MODEL_CONFIG["block_size"], per_token_rate
        )
        print(f"\n    G1 fixed fallback vs G1′ per-token estimate:")
        print(
            f"      G1  per-block fallback : {g1_fallback_per_block:.3f} ms "
            f"(block_size × 0.5)"
        )
        print(
            f"      G1  per-token fallback : {g1_fallback_per_token:.6f} ms/token"
        )
        print(
            f"      G1′ per-block estimate : {new_estimate_per_block:.3f} ms "
            f"(for a full {G0_MODEL_CONFIG['block_size']}-token block)"
        )

        # --- 3. request-level miss-cost attribution demo ---
        print(f"\n[3] Request-level miss-cost attribution demo:")
        demo_missed = [{"block_token_count": 16}] * 4
        demo_cost = compute_request_miss_cost(
            missed_blocks=demo_missed,
            request_total_tokens=1024,
            request_prefill_ms=200.0,
        )
        print(f"    request_total_tokens = 1024, request_prefill_ms = 200.0")
        print(f"    missed_blocks        = 4 × 16-token blocks = 64 tokens")
        print(
            f"    → miss_cost_ms = 200.0 × (64 / 1024) = {demo_cost:.3f} ms"
        )
    else:
        print("    (no trajectories; skipping per-token rate demo)")

    # --- 4. Capacity table (delegated to capacity.py) ---
    print(f"\n[4] Absolute KV capacity table (24 GB GPU budgets):")
    try:
        # Deferred import to avoid module-level circular dependency.
        # capacity.py lives in the same directory (on sys.path[0]).
        from capacity import print_capacity_table

        print_capacity_table(bytes_per_block)
    except Exception as exc:
        print(f"    (capacity.py unavailable: {exc})")

    print()
    print("=" * 78)
    print("Cost model OK.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
