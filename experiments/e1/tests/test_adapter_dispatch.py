"""
Test _init_adapter dispatch in TrajectoryRecorder.

Background: G1 routes tau-bench and BFCL v3 through real adapter classes
(TauBenchAdapter / BFCLAdapter) instead of mock simulators. _init_adapter
must return the correct adapter instance based on the dataset argument,
and raise ValueError for unknown datasets. Tests skip gracefully when
the upstream packages (tau-bench / bfcl-eval) are not installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import pytest
import record_trajectories as rt


def test_init_adapter_unknown_dataset_raises():
    """_init_adapter must raise ValueError for unknown dataset names."""
    recorder = object.__new__(rt.TrajectoryRecorder)  # bypass __init__
    recorder._config = {"workload": {}}
    with pytest.raises(ValueError, match="Unknown dataset"):
        recorder._init_adapter("unknown_dataset", seed=42)


def test_init_adapter_returns_tau_bench_when_available():
    """When tau_bench is installed, --dataset tau-bench returns TauBenchAdapter."""
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._config = {
        "workload": {
            "tau_bench": {
                "user_model": "gpt-4o-mini",
                "user_provider": "openai",
                "user_temperature": 0.7,
            },
        }
    }
    try:
        adapter = recorder._init_adapter("tau-bench", seed=42, domain="retail")
        from taubench_adapter import TauBenchAdapter
        assert isinstance(adapter, TauBenchAdapter)
        adapter.close()
    except ImportError:
        pytest.skip("tau_bench not installed")


def test_init_adapter_returns_bfcl_when_available():
    """When bfcl_eval is installed, --dataset bfcl_v3 returns BFCLAdapter."""
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._config = {"workload": {}}
    try:
        adapter = recorder._init_adapter(
            "bfcl_v3", seed=42, subset="multi_turn_base"
        )
        from bfcl_adapter import BFCLAdapter
        assert isinstance(adapter, BFCLAdapter)
        adapter.close()
    except ImportError:
        pytest.skip("bfcl_eval not installed")


def test_init_adapter_bfcl_default_subset_is_multi_turn_base():
    """If subset=None for bfcl_v3, default to multi_turn_base."""
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._config = {"workload": {}}
    try:
        adapter = recorder._init_adapter("bfcl_v3", seed=42, subset=None)
        from bfcl_adapter import BFCLAdapter
        assert isinstance(adapter, BFCLAdapter)
        assert adapter.subset == "multi_turn_base"
        adapter.close()
    except ImportError:
        pytest.skip("bfcl_eval not installed")
