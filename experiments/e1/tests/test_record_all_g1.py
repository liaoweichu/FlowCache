"""Test _record_all_g1: multi-seed × multi-dataset outer loop with checkpoint/resume.

Background: G1 records 1320 episodes (165 tau-bench × 8 seeds). The outer loop iterates seed → task, writes one JSON per episode, and skips existing files on resume.

Tests use mock adapters / mock inner methods so they run without real tau-bench / GPU dependencies.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import record_trajectories as rt


# ----------------------------------------------------------------------
# Mock adapters for _init_adapter
# ----------------------------------------------------------------------

class _MockTauAdapter:
    """Mock tau-bench adapter with 2 retail tasks (test-only)."""

    def __init__(self, domain="retail", seed=0):
        self.domain = domain
        self.seed = seed

    def list_tasks(self):
        # Return 2 task objects; only task_idx is used by the recorder
        return [{"id": f"{self.domain}-0"}, {"id": f"{self.domain}-1"}]

    def close(self):
        pass


def _make_recorder(tmp_path, config=None):
    """Build a bypass-__init__ TrajectoryRecorder with mock-friendly config."""
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._block_size = 16
    recorder._device = "cpu"
    recorder._global_block_index = {}
    recorder._output_dir = tmp_path
    recorder._skip_count = 0
    recorder._oom_log = []
    if config is None:
        config = {
            "workload": {
                "datasets": ["tau-bench"],
                "seeds": [42, 123],
                "tau_bench": {"user_model": "deepseek-v4-flash"},
            },
            "output": {"resume": True},
        }
    recorder._config = config
    return recorder


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_record_all_g1_writes_tau_bench_with_seed_suffix(tmp_path):
    """tau-bench files must be {domain}-{task_idx}_seed{seed}.json under tau_bench/."""
    recorder = _make_recorder(tmp_path)
    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)

    call_log = []

    def mock_run(adapter, task_index, task_id, seed, domain):
        call_log.append((task_id, seed, domain))
        return {"meta": {"workflow_id": f"{task_id}_seed{seed}"}, "steps": []}

    recorder._run_episode_tau_bench = mock_run

    written = recorder._record_all_g1()

    # 2 domains × 2 seeds × 2 tasks = 8 files
    tau_dir = tmp_path / "tau_bench"
    assert (tau_dir / "retail-0_seed42.json").exists()
    assert (tau_dir / "retail-0_seed123.json").exists()
    assert (tau_dir / "retail-1_seed42.json").exists()
    assert (tau_dir / "retail-1_seed123.json").exists()
    assert (tau_dir / "airline-0_seed42.json").exists()
    assert (tau_dir / "airline-0_seed123.json").exists()
    assert written == 8
    assert recorder._skip_count == 0


def test_record_all_g1_resume_skips_existing(tmp_path):
    """Existing trace files must be skipped when resume=True."""
    recorder = _make_recorder(tmp_path)
    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)

    # Pre-create one file so it should be skipped
    tau_dir = tmp_path / "tau_bench"
    tau_dir.mkdir(parents=True)
    (tau_dir / "retail-0_seed42.json").write_text(json.dumps({"meta": {"workflow_id": "retail-0_seed42"}}))

    run_calls = []

    def mock_run(adapter, task_index, task_id, seed, domain):
        run_calls.append((task_id, seed))
        return {"meta": {"workflow_id": f"{task_id}_seed{seed}"}, "steps": []}

    recorder._run_episode_tau_bench = mock_run

    recorder._record_all_g1(resume=True)

    # retail-0_seed42 should have been skipped (not re-run)
    assert ("retail-0", 42) not in run_calls
    # Other 7 episodes still ran
    assert len(run_calls) == 7
    assert recorder._skip_count == 1


def test_record_all_g1_no_resume_overwrites_existing(tmp_path):
    """When resume=False, existing files must be re-recorded."""
    recorder = _make_recorder(tmp_path)
    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)

    tau_dir = tmp_path / "tau_bench"
    tau_dir.mkdir(parents=True)
    (tau_dir / "retail-0_seed42.json").write_text("OLD")

    run_calls = []

    def mock_run(adapter, task_index, task_id, seed, domain):
        run_calls.append((task_id, seed))
        return {"meta": {"workflow_id": f"{task_id}_seed{seed}"}, "steps": []}

    recorder._run_episode_tau_bench = mock_run

    recorder._record_all_g1(resume=False)

    # All 8 episodes ran (no skip)
    assert len(run_calls) == 8
    assert recorder._skip_count == 0
    # File was overwritten
    data = json.loads((tau_dir / "retail-0_seed42.json").read_text())
    assert data["meta"]["workflow_id"] == "retail-0_seed42"


def test_record_all_g1_seed_filter_single_seed(tmp_path):
    """seed_filter should restrict to a single seed."""
    recorder = _make_recorder(tmp_path)
    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)

    run_calls = []

    def mock_run(adapter, task_index, task_id, seed, domain):
        run_calls.append((task_id, seed))
        return {"meta": {"workflow_id": f"{task_id}_seed{seed}"}, "steps": []}

    recorder._run_episode_tau_bench = mock_run

    recorder._record_all_g1(seed_filter=42)

    # Only seed=42 should run: 2 domains × 2 tasks = 4
    assert all(s == 42 for _, s in run_calls)
    assert len(run_calls) == 4


def test_record_all_g1_max_episodes_caps_per_seed(tmp_path):
    """max_episodes should cap episodes per (dataset, seed)."""
    recorder = _make_recorder(tmp_path)
    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)

    run_calls = []

    def mock_run(adapter, task_index, task_id, seed, domain):
        run_calls.append((task_id, seed))
        return {"meta": {"workflow_id": f"{task_id}_seed{seed}"}, "steps": []}

    recorder._run_episode_tau_bench = mock_run

    recorder._record_all_g1(max_episodes=1)

    # 2 domains × 2 seeds × 1 episode = 4
    assert len(run_calls) == 4


def test_record_all_g1_init_error_logged_not_raised(tmp_path):
    """_init_adapter failure should be logged, not raised."""
    recorder = _make_recorder(tmp_path)

    def bad_init(dataset, seed, domain="retail", subset=None):
        raise RuntimeError("simulated init failure")

    recorder._init_adapter = bad_init
    recorder._run_episode_tau_bench = lambda *a, **kw: {"meta": {}, "steps": []}

    # Should not raise
    recorder._record_all_g1()

    assert len(recorder._oom_log) > 0
    assert "init" in recorder._oom_log[0]["error"]


def test_trace_output_excludes_global_block_index(tmp_path):
    """Trace JSON files must NOT contain the global_block_index field.

    Background: The global block index accumulates across all episodes
    (up to 1320). Writing it into every trace file causes O(n²) disk
    usage (~300GB for 1320 episodes) and leaks cross-episode metadata
    into per-episode traces. The index is kept in RAM
    (``self._global_block_index``) for the lifetime of the recorder and
    can be reconstructed from per-episode ``block_assignments`` during
    characterization if needed.
    """
    config = {
        "workload": {
            "datasets": ["tau-bench"],
            "seeds": [42],
        },
        "output": {"resume": True},
    }
    recorder = _make_recorder(tmp_path, config=config)

    def mock_run_episode_tau_bench(adapter, task_idx, task_id, seed, domain):
        return {
            "meta": {"workflow_id": f"{task_id}_seed{seed}"},
            "steps": [{"role": "system", "block_assignments": []}],
        }

    recorder._init_adapter = lambda dataset, seed, domain="retail", subset=None: _MockTauAdapter(domain=domain, seed=seed)
    recorder._run_episode_tau_bench = mock_run_episode_tau_bench

    recorder._record_all_g1(
        dataset_filter="tau-bench",
        seed_filter=42,
        max_episodes=1,
    )

    trace_files = list((tmp_path / "tau_bench").glob("*.json"))
    assert len(trace_files) >= 1  # retail-0 + airline-0 with max_episodes=1
    for tf in trace_files:
        with open(tf, "r", encoding="utf-8") as f:
            trace = json.load(f)
        assert "global_block_index" not in trace, (
            f"Trace file {tf.name} contains global_block_index (causes O(n²) "
            f"disk usage + cross-episode metadata leak)."
        )
        assert "meta" in trace
        assert "steps" in trace
