"""Regression tests for the repaired G3 closed-loop protocol."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


G3_DIR = Path(__file__).resolve().parents[1]
CLOSED_LOOP_DIR = G3_DIR / "closed_loop"
sys.path.insert(0, str(G3_DIR))
sys.path.insert(0, str(CLOSED_LOOP_DIR))

# The bundled verification runtime intentionally omits PyYAML.  The imported
# runner only needs it in its CLI entry point.
if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda stream: {}
    sys.modules["yaml"] = yaml_stub

from flowcache_connector import (  # noqa: E402
    BlockAccessTracker,
    FlowCacheMigrationPolicy,
    release_filtered_store_refs,
)
from g3_verdict import evaluate_go_no_go  # noqa: E402
from run_closed_loop import (  # noqa: E402
    bootstrap_ci,
    compute_verdict,
    select_quick_pilot_requests,
)
from run_g3_grid import configured_full_grid_baselines  # noqa: E402
from serving_harness import (  # noqa: E402
    MetricsCollector,
    ServingHarness,
    ServingRequest,
    Strategy,
)
from two_tier_baselines import (  # noqa: E402
    TwoTierGDSFCache,
    TwoTierLRUCache,
    TwoTierSizeCostCache,
)


def make_request(index, task_id="task-0", domain="airline"):
    return ServingRequest(
        request_id=f"request-{index}",
        messages=[{"role": "user", "content": "test"}],
        task_id=task_id,
        seed=index,
        workflow_id=f"{task_id}-seed-{index}",
        domain=domain,
        step_id=index,
        arrival_time_ms=float(index),
    )


def make_output(ttft_ms, arrival=10.0, queue_ms=0.0):
    scheduled = arrival + queue_ms / 1000.0
    first = arrival + ttft_ms / 1000.0
    metrics = SimpleNamespace(
        queued_ts=arrival,
        scheduled_ts=scheduled,
        first_token_ts=first,
        last_token_ts=first + 0.01,
    )
    return SimpleNamespace(
        metrics=metrics,
        prompt_token_ids=[],
        outputs=[],
        num_cached_tokens=0,
        finished=True,
    )


class ClosedLoopMetricTests(unittest.TestCase):
    def test_ttft_includes_queueing_and_reports_service_time(self):
        request = make_request(0)
        output = make_output(ttft_ms=300.0, queue_ms=100.0)
        metrics = MetricsCollector.collect(
            "flowcache_lossless",
            [output],
            [request],
            measurement_mode="arrival_replay",
        )
        row = metrics.per_request[0]
        self.assertAlmostEqual(row.ttft, 300.0)
        self.assertAlmostEqual(row.queueing, 100.0)
        self.assertAlmostEqual(row.service_ttft, 200.0)
        self.assertTrue(row.timing_valid)
        self.assertTrue(metrics.metric_protocol_valid)

    def test_missing_output_is_a_protocol_failure(self):
        requests = [make_request(0), make_request(1)]
        metrics = MetricsCollector.collect(
            "flowcache_lossless",
            [make_output(100.0), None],
            requests,
            measurement_mode="arrival_replay",
        )
        self.assertEqual(metrics.total_requests, 2)
        self.assertEqual(metrics.missing_outputs, 1)
        self.assertEqual(metrics.failed_requests, 1)
        self.assertFalse(metrics.metric_protocol_valid)

    def test_batch_smoke_cannot_emit_formal_verdict(self):
        requests = [make_request(0), make_request(1, "task-1")]
        control = MetricsCollector.collect(
            "twotier_lru",
            [make_output(100.0), make_output(100.0)],
            requests,
            measurement_mode="batch_smoke",
        )
        treatment = MetricsCollector.collect(
            "flowcache_lossless",
            [make_output(80.0), make_output(80.0)],
            requests,
            measurement_mode="batch_smoke",
        )
        control.throughput_req_per_s = 10.0
        treatment.throughput_req_per_s = 10.0
        control.total_time_s = 1.0
        treatment.total_time_s = 1.0
        verdict = compute_verdict(
            {
                "twotier_lru": control,
                "flowcache_lossless": treatment,
            },
            bootstrap_samples=20,
            min_task_coverage=2,
        )
        self.assertEqual(verdict["verdict"], "INCOMPLETE")
        self.assertFalse(verdict["ttft_metric_valid"])

    def test_paired_arrival_replay_can_pass_synthetic_gate(self):
        requests = [
            make_request(0, "task-0"),
            make_request(1, "task-0"),
            make_request(2, "task-1"),
            make_request(3, "task-1"),
        ]
        control = MetricsCollector.collect(
            "twotier_lru",
            [make_output(100.0) for _ in requests],
            requests,
            measurement_mode="arrival_replay",
        )
        treatment = MetricsCollector.collect(
            "flowcache_lossless",
            [make_output(80.0) for _ in requests],
            requests,
            measurement_mode="arrival_replay",
        )
        control.throughput_req_per_s = 10.0
        treatment.throughput_req_per_s = 10.0
        control.total_time_s = 1.0
        treatment.total_time_s = 1.0
        verdict = compute_verdict(
            {
                "twotier_lru": control,
                "flowcache_lossless": treatment,
            },
            bootstrap_samples=50,
            min_task_coverage=2,
        )
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertGreater(
            verdict["checks"]["bootstrap_ci_ttft"]["ci_low_ms"], 0
        )

        pilot = compute_verdict(
            {
                "twotier_lru": control,
                "flowcache_lossless": treatment,
            },
            bootstrap_samples=20,
            min_task_coverage=2,
            study_scope="quick_pilot",
        )
        self.assertEqual(pilot["verdict"], "PILOT_PASS")
        report = evaluate_go_no_go(
            pilot,
            [],
            {"closed_loop": {"cell": {}}, "verdict": {}},
        )
        self.assertEqual(report["go_no_go"], "PROTOCOL-INCOMPLETE")

    def test_bootstrap_estimand_is_p95_not_task_mean(self):
        result = bootstrap_ci(
            {"a": [80.0, 80.0], "b": [800.0, 800.0]},
            {"a": [100.0, 100.0], "b": [1000.0, 1000.0]},
            n_samples=50,
        )
        self.assertAlmostEqual(result["p95_diff"], 200.0)
        self.assertEqual(result["n_tasks"], 2)

    def test_fixed_concurrency_replaces_completed_requests(self):
        class FakeLLM:
            def __init__(self):
                self.llm_engine = self
                self.active = []
                self.next_id = 0
                self.max_active = 0

            def enqueue_chat(self, messages, sampling_params, use_tqdm):
                request_id = f"internal-{self.next_id}"
                self.next_id += 1
                self.active.append(request_id)
                self.max_active = max(self.max_active, len(self.active))
                return [request_id]

            def step(self):
                request_id = self.active.pop(0)
                output = make_output(100.0)
                output.request_id = request_id
                return [output]

        requests = [make_request(index) for index in range(5)]
        fake_llm = FakeLLM()
        harness = ServingHarness("model", max_num_seqs=2)
        outputs = harness._run_closed_loop_concurrency(
            fake_llm, requests, sampling_params=object()
        )
        self.assertEqual(fake_llm.max_active, 2)
        self.assertTrue(all(output is not None for output in outputs))
        self.assertEqual(
            [output.request_id for output in outputs],
            [f"internal-{index}" for index in range(5)],
        )
        metrics = MetricsCollector.collect(
            "flowcache_lossless",
            outputs,
            requests,
            measurement_mode="closed_loop_concurrency",
        )
        self.assertTrue(metrics.metric_protocol_valid)


class QuickPilotSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_workflow_complete(self):
        requests = []
        for domain in ("airline", "retail"):
            for task_number in range(4):
                task_id = f"{domain}-{task_number}"
                for workflow_number in range(3):
                    workflow_id = f"{task_id}-seed-{workflow_number}"
                    for step in range(task_number + 2):
                        request = make_request(
                            len(requests), task_id, domain
                        )
                        request.workflow_id = workflow_id
                        request.step_id = step
                        requests.append(request)

        selected_a, manifest_a = select_quick_pilot_requests(
            requests, max_tasks=4, workflows_per_task=2, seed=7
        )
        selected_b, manifest_b = select_quick_pilot_requests(
            requests, max_tasks=4, workflows_per_task=2, seed=7
        )
        self.assertEqual(
            [request.request_id for request in selected_a],
            [request.request_id for request in selected_b],
        )
        selected_domains = {
            task_id.split("-")[0]
            for task_id in manifest_a["selected_tasks"]
        }
        self.assertEqual(selected_domains, {"airline", "retail"})
        self.assertEqual(manifest_a, manifest_b)

        selected_workflow_ids = set(manifest_a["selected_workflows"])
        for workflow_id in selected_workflow_ids:
            expected = [
                request.request_id
                for request in requests
                if request.workflow_id == workflow_id
            ]
            actual = [
                request.request_id
                for request in selected_a
                if request.workflow_id == workflow_id
            ]
            self.assertEqual(sorted(actual), sorted(expected))


class TwoTierBaselineTests(unittest.TestCase):
    def test_full_grid_includes_fair_two_tier_group(self):
        config = {
            "baselines": {
                "lower_bound": [{"name": "no_cache"}],
                "two_tier_fair": [
                    {"name": "twotier_lru"},
                    {"name": "twotier_gdsf"},
                    {"name": "disabled", "enabled": False},
                ],
            }
        }
        names = configured_full_grid_baselines(config)
        self.assertEqual(names, ["no_cache", "twotier_lru", "twotier_gdsf"])

    def test_two_tier_stats_include_movement(self):
        cache = TwoTierLRUCache(
            gpu_capacity=1,
            cpu_capacity=1,
            d2h_ms=2.0,
            h2d_ms=3.0,
        )
        cache.access("a", prefill_ms=1.0)
        cache.access("b", prefill_ms=1.0)
        cache.access("a", prefill_ms=1.0)
        stats = cache.get_stats()
        self.assertEqual(stats["migrate_to_cpu_count"], 2)
        self.assertEqual(stats["restore_to_gpu_count"], 1)
        self.assertEqual(stats["migrate_ms_total"], 4.0)
        self.assertEqual(stats["restore_ms_total"], 3.0)

    def test_cpu_hit_increments_frequency(self):
        for cache in (
            TwoTierGDSFCache(1, 2),
            TwoTierSizeCostCache(1, 2),
        ):
            cache.access("a", prefill_ms=10.0)
            cache.access("b", prefill_ms=10.0)
            cache.access("a", prefill_ms=10.0)
            self.assertEqual(cache.gpu_cache["a"]["freq"], 2)


class ConnectorSafetyTests(unittest.TestCase):
    def test_ratio_zero_migrates_nothing(self):
        tracker = BlockAccessTracker(capacity_blocks=4)
        tracker.record_request(["a", "b"], "workflow", prefill_ms=10.0)
        policy = FlowCacheMigrationPolicy(
            tracker,
            migrate_ratio=0.0,
            mode="ratio",
        )
        migrate, evict = policy.select_blocks_for_migration(
            ["a", "b"], [1, 2]
        )
        self.assertEqual(migrate, [])
        self.assertEqual(evict, [1, 2])

    def test_filter_cleanup_releases_cpu_and_gpu_refs(self):
        class Block:
            def __init__(self):
                self.reset = False

            def reset_hash(self):
                self.reset = True

        class Pool:
            def __init__(self, count):
                self.blocks = [Block() for _ in range(count)]
                self.freed = []

            def free_blocks(self, blocks):
                self.freed.extend(list(blocks))

        manager = SimpleNamespace(
            cpu_block_pool=Pool(3),
            _gpu_block_pool=Pool(3),
            _in_flight_store_gpu_blocks={1},
        )
        cpu_block = manager.cpu_block_pool.blocks[2]
        gpu_block = manager._gpu_block_pool.blocks[1]
        release_filtered_store_refs(manager, [1], [cpu_block])
        self.assertTrue(cpu_block.reset)
        self.assertIn(cpu_block, manager.cpu_block_pool.freed)
        self.assertIn(gpu_block, manager._gpu_block_pool.freed)
        self.assertNotIn(1, manager._in_flight_store_gpu_blocks)

    def test_vllm_offload_roles_are_bidirectional_and_capacity_is_fixed(self):
        captured = []

        class FakeLLM:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        class FakeKVTransferConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_config = types.ModuleType("vllm.config")
        fake_config.KVTransferConfig = FakeKVTransferConfig

        harness = ServingHarness(
            "model", kv_cache_memory_bytes=2 * 1024**3
        )
        with mock.patch.dict(
            sys.modules,
            {"vllm": fake_vllm, "vllm.config": fake_config},
        ):
            harness._create_llm(Strategy.TWOTIER_LRU)
            harness._create_llm(Strategy.FLOWCACHE_LOSSLESS)

        self.assertEqual(len(captured), 2)
        for kwargs in captured:
            self.assertEqual(kwargs["kv_cache_memory_bytes"], 2 * 1024**3)
            self.assertEqual(
                kwargs["kv_transfer_config"].kv_role, "kv_both"
            )


if __name__ == "__main__":
    unittest.main()
