"""
Test that trace_utils.compute_block_hash is the G0 8-tuple version.

Background: experiments/e1/trace_utils.py originally defined a simplified
4-tuple compute_block_hash (token_ids, parent_hash, block_idx, block_size).
G1 unifies block identity to the G0 8-tuple version (adds model_id,
revision, template_hash, config_hash, adapter_id). This test ensures
trace_utils re-exports the G0 version rather than keeping its own copy.
"""

import sys
from pathlib import Path

# Make experiments/e1/ and experiments/g0/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import experiments.e1.trace_utils as tu


def test_compute_block_hash_is_g0_version():
    """trace_utils.compute_block_hash 必须是 G0 8 元组版的再导出。"""
    from block_index import compute_block_hash as g0_hash
    assert tu.compute_block_hash is g0_hash


def test_compute_block_hash_includes_model_id():
    """不同 model_id 应产生不同 hash（G0 8 元组版特性）。"""
    h1 = tu.compute_block_hash(
        token_ids=[1, 2, 3], parent_hash="", block_idx=0, block_size=16,
        model_id="Qwen2.5-7B",
    )
    h2 = tu.compute_block_hash(
        token_ids=[1, 2, 3], parent_hash="", block_idx=0, block_size=16,
        model_id="Qwen2.5-14B",
    )
    assert h1 != h2


def test_compute_block_hash_backward_compat_default():
    """不传 model_id 等参数时仍能工作（默认空字符串）。"""
    h = tu.compute_block_hash(
        token_ids=[1, 2, 3], parent_hash="", block_idx=0, block_size=16,
    )
    assert isinstance(h, str) and len(h) == 16


def test_compute_template_hash_reexported():
    """trace_utils 应再导出 compute_template_hash / compute_config_hash / verify_parent_chain。"""
    from block_index import (
        compute_template_hash as g0_template_hash,
        compute_config_hash as g0_config_hash,
        verify_parent_chain as g0_verify_parent_chain,
    )
    assert tu.compute_template_hash is g0_template_hash
    assert tu.compute_config_hash is g0_config_hash
    assert tu.verify_parent_chain is g0_verify_parent_chain
