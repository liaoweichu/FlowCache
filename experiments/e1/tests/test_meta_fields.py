"""Test block identity propagation in recording (Task 7).

Background: G0 unified block identity to an 8-tuple
(m, r, tau, c, a, h_parent, tokenIds, positions). The legacy
``tokenize_with_block_tracking`` only passed the 4-tuple
(token_ids, parent_hash, block_idx, block_size) to compute_block_hash,
leaving model_id / revision / template_hash / config_hash / adapter_id
as empty strings. This made blocks from different models / adapters
hash-collide, breaking cross-experiment prefix-overlap analysis.

Task 7 fixes this by:
  - Adding ``_compute_model_metadata()`` to extract model_id /
    revision / template_hash / config_hash from the loaded model.
  - Extending ``tokenize_with_block_tracking`` to accept these as
    optional kwargs (default "" preserves backward compat) and forward
    them to ``compute_block_hash``.
  - Updating ``_run_episode_tau_bench`` / ``_run_episode_bfcl`` to
    propagate metadata + adapter_id into every block hash.

Tests verify:
  - ``_compute_model_metadata()`` returns the expected dict.
  - ``tokenize_with_block_tracking`` accepts the new kwargs.
  - block_hash differs when model_id differs (no collision).
  - Traces from ``_run_episode_tau_bench`` / ``_run_episode_bfcl``
    produce block hashes that change when the model_id changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import record_trajectories as rt


# ----------------------------------------------------------------------
# Test helpers (reused from test_episode_loops.py but kept self-contained)
# ----------------------------------------------------------------------

class _MockTauAdapter:
    def get_system_policy(self):
        return "You are a helpful assistant."

    def get_tools_schema_for_qwen(self):
        return "Available tools: none"

    def reset(self, task_index):
        return {"observation": "Hello, I need help.", "task": {"id": "mock-1"}}

    def step_respond(self, content):
        return {"observation": "###STOP###", "reward": 1.0, "done": True, "info": {}}


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


def _make_recorder(tmp_path, model_name="test-model"):
    recorder = object.__new__(rt.TrajectoryRecorder)
    recorder._block_size = 16
    recorder._device = "cpu"
    recorder._global_block_index = {}
    recorder._output_dir = tmp_path
    recorder._config = {"model": {"name": model_name}}
    recorder._tokenizer = _make_fake_tok()
    recorder._model = None
    return recorder


# ----------------------------------------------------------------------
# _compute_model_metadata tests
# ----------------------------------------------------------------------

def test_compute_model_metadata_returns_dict_with_required_keys(tmp_path):
    """_compute_model_metadata must return dict with model_id/revision/template_hash/config_hash."""
    recorder = _make_recorder(tmp_path, model_name="Qwen2.5-7B")
    meta = recorder._compute_model_metadata()

    assert isinstance(meta, dict)
    assert "model_id" in meta
    assert "revision" in meta
    assert "template_hash" in meta
    assert "config_hash" in meta
    assert meta["model_id"] == "Qwen2.5-7B"


def test_compute_model_metadata_handles_missing_config(tmp_path):
    """_compute_model_metadata must not crash when _model is None."""
    recorder = _make_recorder(tmp_path)
    meta = recorder._compute_model_metadata()
    # model_id falls back to "unknown" when config absent
    recorder._config = {}
    meta2 = recorder._compute_model_metadata()
    assert meta2["model_id"] == "unknown"
    # revision should be "" when model is None
    assert meta["revision"] == ""
    assert meta2["revision"] == ""


# ----------------------------------------------------------------------
# tokenize_with_block_tracking metadata propagation
# ----------------------------------------------------------------------

def test_tokenize_accepts_metadata_kwargs(tmp_path):
    """tokenize_with_block_tracking must accept model_id/revision/template_hash/config_hash/adapter_id."""
    recorder = _make_recorder(tmp_path)
    tokens, blocks = recorder.tokenize_with_block_tracking(
        "hello world", parent_hash="",
        model_id="modelA", revision="rev1",
        template_hash="tH1", config_hash="cH1",
        adapter_id="tau_bench_v1",
    )
    assert len(blocks) > 0
    assert all("block_hash" in b for b in blocks)


def test_block_hash_differs_when_model_id_differs(tmp_path):
    """Same token_ids but different model_id must yield different block_hash."""
    recorder = _make_recorder(tmp_path)

    _tokens_a, blocks_a = recorder.tokenize_with_block_tracking(
        "hello world", parent_hash="",
        model_id="Qwen2.5-7B",
    )
    _tokens_b, blocks_b = recorder.tokenize_with_block_tracking(
        "hello world", parent_hash="",
        model_id="Qwen2.5-14B",
    )

    assert len(blocks_a) == len(blocks_b)
    # All block hashes should differ because model_id differs
    for ba, bb in zip(blocks_a, blocks_b):
        assert ba["block_hash"] != bb["block_hash"], (
            f"block_idx={ba['block_idx']} collided: {ba['block_hash']} == {bb['block_hash']}"
        )


def test_block_hash_differs_when_adapter_id_differs(tmp_path):
    """Same token_ids + model_id but different adapter_id must yield different block_hash."""
    recorder = _make_recorder(tmp_path)

    _t1, blocks_1 = recorder.tokenize_with_block_tracking(
        "hello", parent_hash="",
        model_id="Qwen2.5-7B", adapter_id="tau_bench_v1",
    )
    _t2, blocks_2 = recorder.tokenize_with_block_tracking(
        "hello", parent_hash="",
        model_id="Qwen2.5-7B", adapter_id="bfcl_v1",
    )

    assert blocks_1[0]["block_hash"] != blocks_2[0]["block_hash"]


def test_block_hash_same_when_metadata_same(tmp_path):
    """Same metadata + token_ids must yield same block_hash (deterministic)."""
    recorder = _make_recorder(tmp_path)

    _t1, blocks_1 = recorder.tokenize_with_block_tracking(
        "hello", parent_hash="",
        model_id="Qwen2.5-7B", adapter_id="tau_bench_v1",
    )
    _t2, blocks_2 = recorder.tokenize_with_block_tracking(
        "hello", parent_hash="",
        model_id="Qwen2.5-7B", adapter_id="tau_bench_v1",
    )

    assert blocks_1[0]["block_hash"] == blocks_2[0]["block_hash"]


def test_legacy_tokenize_still_works_without_metadata(tmp_path):
    """Backward compat: calling without metadata kwargs must not crash (defaults to '')."""
    recorder = _make_recorder(tmp_path)
    _t, blocks = recorder.tokenize_with_block_tracking("hello", parent_hash="")
    assert len(blocks) > 0
    # Should produce a valid 16-char hash
    assert len(blocks[0]["block_hash"]) == 16


# ----------------------------------------------------------------------
# End-to-end: _run_episode_tau_bench propagates metadata into block hashes
# ----------------------------------------------------------------------

def test_run_episode_tau_bench_propagates_model_id_into_blocks(tmp_path):
    """Two runs with different model_id should produce different block hashes."""
    recorder_a = _make_recorder(tmp_path, model_name="ModelA")
    recorder_a._generate_response = lambda msgs, seed=None: ("I'll help.", 4, 10.0, 5.0)
    recorder_a._measure_prefill = lambda inputs: 10.0

    recorder_b = _make_recorder(tmp_path, model_name="ModelB")
    recorder_b._generate_response = lambda msgs, seed=None: ("I'll help.", 4, 10.0, 5.0)
    recorder_b._measure_prefill = lambda inputs: 10.0

    adapter_a = _MockTauAdapter()
    adapter_b = _MockTauAdapter()

    trace_a = recorder_a._run_episode_tau_bench(
        adapter=adapter_a, task_index=0, task_id="mock-1",
        seed=42, domain="retail",
    )
    trace_b = recorder_b._run_episode_tau_bench(
        adapter=adapter_b, task_index=0, task_id="mock-1",
        seed=42, domain="retail",
    )

    # meta should reflect the different model_id
    assert trace_a["meta"]["model_id"] == "ModelA"
    assert trace_b["meta"]["model_id"] == "ModelB"

    # The first block (system prompt) should have different hashes because
    # model_id differs (system prompt text is the same across both runs).
    blocks_a = trace_a["steps"][0]["block_assignments"]
    blocks_b = trace_b["steps"][0]["block_assignments"]
    assert len(blocks_a) > 0
    assert len(blocks_a) == len(blocks_b)
    assert blocks_a[0]["block_hash"] != blocks_b[0]["block_hash"]


def test_run_episode_tau_bench_meta_includes_metadata_fields(tmp_path):
    """Trace meta must include model_id/revision/template_hash/config_hash/adapter_id."""
    recorder = _make_recorder(tmp_path, model_name="TestModel")
    recorder._generate_response = lambda msgs, seed=None: ("Hi", 4, 10.0, 5.0)
    recorder._measure_prefill = lambda inputs: 10.0

    adapter = _MockTauAdapter()
    trace = recorder._run_episode_tau_bench(
        adapter=adapter, task_index=0, task_id="mock-1",
        seed=42, domain="retail",
    )

    meta = trace["meta"]
    assert meta["model_id"] == "TestModel"
    assert meta["adapter_id"] == "tau_bench_v1"
    assert "revision" in meta
    assert "template_hash" in meta
    assert "config_hash" in meta
    # template_hash should be non-empty (FakeTok.apply_chat_template works)
    assert meta["template_hash"]


# ----------------------------------------------------------------------
# End-to-end: _run_episode_bfcl propagates metadata into block hashes
# -------------------------------------------------------------

# Reuse the MockBFCLAdapter / _MockBFCLEpisode pattern from test_episode_loops.py
# but keep this file self-contained.

class _MockBFCLEpisode:
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


class _MockBFCLAdapter:
    def __init__(self, subset="multi_turn_base"):
        self.subset = subset

    def load_entries(self):
        return [({
            "id": "bfcl-meta-1",
            "question": [[{"role": "user", "content": "Do task A."}],
                         [{"role": "user", "content": "Now do task B."}]],
            "initial_config": {},
            "involved_classes": ["MathAPI"],
        }, {"id": "bfcl-meta-1", "ground_truth": [[]]})]

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


def test_run_episode_bfcl_propagates_model_id_into_blocks(tmp_path):
    """Every tokenize_with_block_tracking call during _run_episode_bfcl must
    pass non-empty model_id / revision / template_hash / config_hash / adapter_id.

    This catches the bug where _run_episode_bfcl calls
    tokenize_with_block_tracking directly (bypassing the _tokenize closure
    that injects G0 8-tuple metadata) for user/assistant/tool blocks.

    A naive "two runs with different model_id → compare block hashes" test
    does NOT catch this bug, because the parent_hash chain diverges at the
    system prompt (which correctly propagates metadata), making all
    descendant block hashes differ regardless of metadata propagation.

    Instead we intercept tokenize_with_block_tracking and assert that
    every call receives non-empty metadata kwargs.
    """
    recorder = _make_recorder(tmp_path, model_name="TestBFCLModel")
    recorder._generate_response = lambda msgs, seed=None: ("math(x=1)", 4, 10.0, 5.0)
    recorder._measure_prefill = lambda inputs: 10.0

    # Intercept tokenize_with_block_tracking to record metadata kwargs.
    call_log = []
    original_tokenize = recorder.tokenize_with_block_tracking

    def _spy_tokenize(text, parent_hash="", **kwargs):
        call_log.append({
            "text_preview": text[:40],
            "model_id": kwargs.get("model_id", ""),
            "revision": kwargs.get("revision", ""),
            "template_hash": kwargs.get("template_hash", ""),
            "config_hash": kwargs.get("config_hash", ""),
            "adapter_id": kwargs.get("adapter_id", ""),
        })
        return original_tokenize(text, parent_hash=parent_hash, **kwargs)

    recorder.tokenize_with_block_tracking = _spy_tokenize

    adapter = _MockBFCLAdapter(subset="multi_turn_base")
    entries = adapter.load_entries()
    entry, gt = entries[0]
    episode = adapter.init_episode(entry, gt, seed=42)
    trace = recorder._run_episode_bfcl(
        adapter=adapter, episode=episode, seed=42,
    )
    adapter.close_episode(episode)
    adapter.close()

    # Sanity: at least 3 calls (system + user + assistant); tool calls add more.
    assert len(call_log) >= 3, (
        f"Expected ≥3 tokenize calls (system/user/assistant), got {len(call_log)}"
    )

    # Every call must include non-empty metadata.
    for i, call in enumerate(call_log):
        assert call["model_id"] == "TestBFCLModel", (
            f"call {i} (text={call['text_preview']!r}): model_id not propagated, "
            f"got {call['model_id']!r}"
        )
        assert call["adapter_id"] == "bfcl_v1", (
            f"call {i} (text={call['text_preview']!r}): adapter_id not propagated, "
            f"got {call['adapter_id']!r}"
        )
        # template_hash should be non-empty (FakeTok.apply_chat_template works)
        assert call["template_hash"], (
            f"call {i} (text={call['text_preview']!r}): template_hash empty"
        )


def test_run_episode_bfcl_meta_includes_metadata_fields(tmp_path):
    """BFCL trace meta must include model_id/revision/template_hash/config_hash/adapter_id."""
    recorder = _make_recorder(tmp_path, model_name="TestBFCLModel")
    recorder._generate_response = lambda msgs, seed=None: ("math(x=1)", 4, 10.0, 5.0)
    recorder._measure_prefill = lambda inputs: 10.0

    adapter = _MockBFCLAdapter(subset="multi_turn_base")
    entries = adapter.load_entries()
    entry, gt = entries[0]
    episode = adapter.init_episode(entry, gt, seed=42)
    trace = recorder._run_episode_bfcl(
        adapter=adapter, episode=episode, seed=42,
    )
    adapter.close_episode(episode)
    adapter.close()

    meta = trace["meta"]
    assert meta["model_id"] == "TestBFCLModel"
    assert meta["adapter_id"] == "bfcl_v1"
    assert "revision" in meta
    assert "template_hash" in meta
    assert "config_hash" in meta
    assert meta["template_hash"]
