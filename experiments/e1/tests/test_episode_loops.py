"""用 Mock adapter 测试录制循环逻辑，不依赖真实 tau-bench/BFCL。

覆盖 Task 5（_run_episode_tau_bench + _run_episode_bfcl）：
  - tau-bench: LLM user simulator 风格的 conversation loop
  - BFCL: scripted user turns 风格的 conversation loop
两个数据集的录制循环结构不同，分别验证。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import record_trajectories as rt


# ----------------------------------------------------------------------
# Mock TauBenchAdapter
# ----------------------------------------------------------------------

class MockTauBenchAdapter:
    """模拟 TauBenchAdapter 的最小行为。"""

    def __init__(self, seed=0):
        self.seed = seed
        self.calls = 0

    def list_tasks(self):
        return [{"id": "mock-1", "domain": "retail"}]

    def get_system_policy(self):
        return "You are a helpful assistant."

    def get_tools_schema_for_qwen(self):
        return "Available tools: none"

    def reset(self, task_index):
        return {"observation": "Hello, I need help.", "task": {"id": "mock-1"}}

    def step_tool(self, tool_name, kwargs):
        return {"observation": '{"result":"ok"}', "reward": 0.0, "done": False, "info": {}}

    def step_respond(self, content):
        self.calls += 1
        if self.calls >= 2:
            return {"observation": "###STOP###", "reward": 1.0, "done": True, "info": {}}
        return {"observation": "Thanks.", "reward": 0.0, "done": False, "info": {}}

    def close(self):
        pass


# ----------------------------------------------------------------------
# Mock BFCLAdapter + BFCLEpisode
# ----------------------------------------------------------------------

class _MockBFCLEpisode:
    """Mock BFCLEpisode (避免依赖真实 bfcl_adapter.BFCLEpisode)。"""

    def __init__(self, entry_id, subset, seed, involved_classes, initial_config):
        self.entry_id = entry_id
        self.subset = subset
        self.seed = seed
        self.involved_classes = involved_classes
        self.initial_config = initial_config
        self.user_turns = []
        self.tool_calls = []
        self.tool_results = []
        self.agent_responses = []
        self.valid = None
        self.error_type = None
        self._gt = None
        self._entry = None
        self._model_name = f"mock_{entry_id}"
        self._backend_instances = {}


class MockBFCLAdapter:
    """模拟 BFCLAdapter 的最小行为。"""

    def __init__(self, subset="multi_turn_base"):
        self.subset = subset

    def load_entries(self):
        return [({
            "id": "bfcl-mock-1",
            "question": [[{"role": "user", "content": "Do task A."}],
                         [{"role": "user", "content": "Now do task B."}]],
            "initial_config": {},
            "involved_classes": ["MathAPI"],
        }, {"id": "bfcl-mock-1", "ground_truth": [[]]})]

    def init_episode(self, entry, gt, seed=0):
        ep = _MockBFCLEpisode(
            entry_id=entry["id"],
            subset=self.subset,
            seed=seed,
            involved_classes=entry["involved_classes"],
            initial_config=entry["initial_config"],
        )
        ep.user_turns = [m["content"] for turn in entry["question"]
                         for m in turn if m.get("role") == "user"]
        ep._gt = gt
        ep._entry = entry
        return ep

    def execute_tool_calls(self, calls, episode):
        return ["ok"] * len(calls)

    def validate_episode(self, episode):
        episode.valid = True
        return True

    def get_tool_schema_for_qwen(self, episode):
        return "Available tools: math(x)"

    def close_episode(self, episode):
        pass

    def close(self):
        pass


# ----------------------------------------------------------------------
# 共享 FakeTok
# ----------------------------------------------------------------------

def _make_fake_tok():
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

        def apply_chat_template(self, msgs, **kw):
            return " ".join(m["content"] for m in msgs)

        def __call__(self, *a, **kw):
            import torch

            class B:
                input_ids = torch.tensor([[1, 2, 3]])

                def to(self, d):
                    return self

            return B()

        pad_token = 0
        eos_token = 0
        eos_token_id = 0

    return FakeTok()


def _make_recorder(tmp_path):
    """构造一个 bypass __init__ 的 TrajectoryRecorder，配齐最小属性。"""
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._block_size = 16
    recorder._device = "cpu"
    recorder._global_block_index = {}
    recorder._output_dir = tmp_path
    recorder._config = {"model": {"name": "test-model"}}
    recorder._tokenizer = _make_fake_tok()
    recorder._model = None  # 不应在测试中调用
    return recorder


# ----------------------------------------------------------------------
# Task 5 tests
# ----------------------------------------------------------------------

def test_run_episode_tau_bench_produces_trace(tmp_path):
    """_run_episode_tau_bench 应产出合法 trace dict。"""
    recorder = _make_recorder(tmp_path)

    # 替换 _generate_response / _measure_prefill 为返回固定值
    recorder._generate_response = lambda msgs, seed=None: ("I'll help.", 4, 10.0, 5.0)
    recorder._measure_prefill = lambda inputs: 10.0

    adapter = MockTauBenchAdapter(seed=42)
    trace = recorder._run_episode_tau_bench(
        adapter=adapter, task_index=0, task_id="mock-1",
        seed=42, domain="retail",
    )
    adapter.close()

    assert trace["meta"]["dataset"] == "tau-bench"
    assert trace["meta"]["seed"] == 42
    assert trace["meta"]["task_id"] == "mock-1"
    assert trace["meta"]["model_id"]  # 必须非空
    assert trace["meta"]["template_hash"]  # 必须非空
    assert len(trace["steps"]) > 0
    # 每步必须有 arrival_time_ms 字段
    for s in trace["steps"]:
        assert "arrival_time_ms" in s


def test_run_episode_bfcl_produces_trace(tmp_path):
    """_run_episode_bfcl 应产出合法 trace dict（scripted user turns）。"""
    recorder = _make_recorder(tmp_path)
    recorder._generate_response = lambda msgs, seed=None: ("math(x=1)", 4, 10.0, 5.0)
    recorder._measure_prefill = lambda inputs: 10.0

    adapter = MockBFCLAdapter(subset="multi_turn_base")
    entries = adapter.load_entries()
    entry, gt = entries[0]
    episode = adapter.init_episode(entry, gt, seed=42)
    trace = recorder._run_episode_bfcl(
        adapter=adapter, episode=episode, seed=42,
    )
    adapter.close_episode(episode)
    adapter.close()

    assert trace["meta"]["dataset"] == "bfcl_v3"
    assert trace["meta"]["seed"] == 42
    assert trace["meta"]["bfcl_subset"] == "multi_turn_base"
    assert len(trace["steps"]) >= 4  # system + 2 user turns + 2 assistant
