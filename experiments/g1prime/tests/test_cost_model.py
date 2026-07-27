"""Unit tests for ``experiments/g1prime/cost_model.py``.

All tests are pure-Python and do NOT require the real Qwen tokenizer or
any trajectory data. They verify:

  1. ``bytes_per_block`` for the G0 frozen Qwen2.5-7B BF16 config = 917,504 B.
  2. ``gib_to_blocks`` for 1 / 2 / 4 / 6 GiB = 1170 / 2340 / 4681 / 7021 blocks.
  3. ``compute_per_token_prefill_rate`` median computation on mock trajectories.
  4. ``compute_request_miss_cost`` per-token proportional attribution.
  5. ``estimate_block_prefill_ms`` linear scaling (NO fixed 8 ms fallback).
"""

import statistics

import cost_model
from capacity import gib_to_blocks


# ---------------------------------------------------------------------------
# 1. bytes_per_block
# ---------------------------------------------------------------------------
def test_bytes_per_block():
    """Verify bytes_per_block for Qwen2.5-7B-Instruct BF16 (block_size=16).

    Expected::

        16 × 28 × 2 × 4 × 128 × 2 = 917,504 B
    """
    b = cost_model.compute_bytes_per_block(
        num_hidden_layers=28,
        num_kv_heads=4,
        head_dim=128,
        dtype_bytes=2,
        block_size=16,
    )
    assert b == 917_504


# ---------------------------------------------------------------------------
# 2. GiB → blocks (delegates to capacity.gib_to_blocks but is exercised
#    here because cost_model.main() prints this table end-to-end).
# ---------------------------------------------------------------------------
def test_gib_to_blocks():
    """Verify 1 / 2 / 4 / 6 GiB → 1170 / 2340 / 4681 / 7021 blocks.

    floor(1024³ / 917504) = 1170
    """
    bpb = 917_504
    assert gib_to_blocks(1, bpb) == 1170
    assert gib_to_blocks(2, bpb) == 2340
    assert gib_to_blocks(4, bpb) == 4681
    assert gib_to_blocks(6, bpb) == 7021


# ---------------------------------------------------------------------------
# 3. per-token prefill rate (median over measured assistant steps)
# ---------------------------------------------------------------------------
def test_per_token_rate_median():
    """Verify median computation over measured assistant steps.

    The rate is the median of ``prefill_ms / token_count`` over every
    assistant step where both fields are strictly positive. Non-assistant
    steps and zero-token / zero-prefill steps are skipped.
    """
    trajectories = [
        {"steps": [
            {"role": "assistant", "prefill_ms": 10.0, "token_count": 20},   # 0.500
            {"role": "assistant", "prefill_ms": 30.0, "token_count": 40},   # 0.750
            {"role": "assistant", "prefill_ms": 100.0, "token_count": 100}, # 1.000
        ]},
        {"steps": [
            # Non-assistant step must be skipped.
            {"role": "user", "prefill_ms": 5.0, "token_count": 10},
            # Zero prefill must be skipped.
            {"role": "assistant", "prefill_ms": 0.0, "token_count": 10},
            # Zero token_count must be skipped.
            {"role": "assistant", "prefill_ms": 50.0, "token_count": 0},
            # Valid step contributing 0.5.
            {"role": "assistant", "prefill_ms": 50.0, "token_count": 100},  # 0.500
        ]},
    ]
    rates = [0.500, 0.750, 1.000, 0.500]
    expected = statistics.median(rates)
    got = cost_model.compute_per_token_prefill_rate(trajectories)
    assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"


def test_per_token_rate_empty_returns_zero():
    """No valid assistant steps → 0.0 (defensive)."""
    assert cost_model.compute_per_token_prefill_rate([]) == 0.0
    assert cost_model.compute_per_token_prefill_rate([{"steps": []}]) == 0.0


# ---------------------------------------------------------------------------
# 4. request-level miss-cost proportional attribution
# ---------------------------------------------------------------------------
def test_request_miss_cost_proportional():
    """Verify per-token proportional attribution of prefill_ms.

    A request with 1024 tokens and prefill_ms=200 that misses 4 blocks of
    16 tokens each (64 missed tokens) should pay::

        200 × (64 / 1024) = 12.5 ms
    """
    missed = [{"block_token_count": 16}] * 4
    cost = cost_model.compute_request_miss_cost(
        missed_blocks=missed,
        request_total_tokens=1024,
        request_prefill_ms=200.0,
    )
    assert abs(cost - 12.5) < 1e-9


def test_request_miss_cost_no_misses_is_zero():
    """No missed blocks → 0 cost (defensive)."""
    cost = cost_model.compute_request_miss_cost(
        missed_blocks=[],
        request_total_tokens=1024,
        request_prefill_ms=200.0,
    )
    assert cost == 0.0


def test_request_miss_cost_zero_prefill_is_zero():
    """request_prefill_ms = 0 → 0 cost (defensive)."""
    cost = cost_model.compute_request_miss_cost(
        missed_blocks=[{"block_token_count": 16}] * 4,
        request_total_tokens=1024,
        request_prefill_ms=0.0,
    )
    assert cost == 0.0


# ---------------------------------------------------------------------------
# 5. estimate_block_prefill_ms: NO fixed 8 ms fallback
# ---------------------------------------------------------------------------
def test_estimate_block_prefill_no_8ms_fallback():
    """Verify the linear per-token estimate replaces G1's fixed 8 ms.

    G1 used ``block_size × 0.5 = 8 ms`` regardless of ``block_token_count``.
    G1′ uses ``block_token_count × per_token_rate``, which is linear and
    distinguishes partial blocks from full blocks.
    """
    rate = 0.5  # ms/token

    # Full 16-token block: 16 × 0.5 = 8 ms (coincides with G1's fixed value).
    full = cost_model.estimate_block_prefill_ms(16, rate)
    assert abs(full - 8.0) < 1e-9

    # Half block (8 tokens): 8 × 0.5 = 4 ms — NOT 8 ms.
    half = cost_model.estimate_block_prefill_ms(8, rate)
    assert abs(half - 4.0) < 1e-9
    assert half != full, "Half-block must NOT equal full-block (G1 bug)"

    # Quarter block (4 tokens): 4 × 0.5 = 2 ms.
    quarter = cost_model.estimate_block_prefill_ms(4, rate)
    assert abs(quarter - 2.0) < 1e-9

    # Proportionality: doubling the token count doubles the estimate.
    assert abs(half - 2 * quarter) < 1e-9


def test_estimate_block_prefill_zero_inputs():
    """Zero / negative inputs → 0.0 (defensive)."""
    assert cost_model.estimate_block_prefill_ms(0, 0.5) == 0.0
    assert cost_model.estimate_block_prefill_ms(16, 0.0) == 0.0
    assert cost_model.estimate_block_prefill_ms(-1, 0.5) == 0.0
    assert cost_model.estimate_block_prefill_ms(16, -1.0) == 0.0
