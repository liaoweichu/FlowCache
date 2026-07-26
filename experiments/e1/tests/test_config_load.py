"""
Test that experiments/e1/config.yaml supports G1 multi-dataset + 8 seeds + resume.

Background: G1 records 1320 episodes (tau-bench 165×8) and needs
config support for: workload.datasets list, workload.seeds (8 entries),
and output.resume flag.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_has_workload_datasets_list():
    cfg = _load_cfg()
    assert "workload" in cfg
    assert "datasets" in cfg["workload"]
    assert cfg["workload"]["datasets"] == ["tau-bench"]


def test_config_has_8_seeds():
    cfg = _load_cfg()
    seeds = cfg["workload"].get("seeds")
    assert isinstance(seeds, list) and len(seeds) == 8


def test_config_has_resume_flag():
    cfg = _load_cfg()
    assert cfg.get("output", {}).get("resume", False) is True


def test_config_has_tau_bench_full_count():
    """τ-bench must be 165 (115 retail + 50 airline), not the old 80 subset."""
    cfg = _load_cfg()
    tau = cfg["workload"].get("tau_bench", {})
    assert tau.get("tasks") == 165


def test_config_replay_has_num_replay_seeds():
    """Replay must specify num_replay_seeds (3 for G1)."""
    cfg = _load_cfg()
    assert "num_replay_seeds" in cfg.get("replay", {})
    assert cfg["replay"]["num_replay_seeds"] == 3


def test_config_trace_subdirs_present():
    """output.trace_subdirs lists per-dataset subdirectories."""
    cfg = _load_cfg()
    subdirs = cfg.get("output", {}).get("trace_subdirs")
    assert subdirs is not None
    assert set(subdirs) == {"tau_bench"}
