"""Targeted regression tests for G3-P1 selective lossless migration."""

import json
import math
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


G3_DIR = Path(__file__).resolve().parents[1]
E1_DIR = G3_DIR.parent / "e1"
sys.path.insert(0, str(G3_DIR))
sys.path.insert(0, str(E1_DIR))

# The bundled verification Python does not ship PyYAML.  The modules under
# test only need yaml inside their CLI entry points, so a tiny import stub is
# sufficient for unit tests and avoids changing the project environment.
if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda stream: {}
    sys.modules["yaml"] = yaml_stub

from cache_manager import LosslessCacheManager
from controller import FlowCacheLosslessController
from freeze_selected_config import validate_selection
from g3_verdict import aggregate_by_cell_baseline, evaluate_go_no_go
import run_g3_grid as g3_runner
from run_g3_grid import (
    annotate_causal_share_counts,
    filter_task_split,
    instantiate_baseline,
    load_access_trace,
    replay_accesses,
)
from tune_selective_migration import summarize_candidate


def load_cost_model():
    with open(G3_DIR / "cost-model.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


class TransferCostTests(unittest.TestCase):
    def test_exact_block_measurement_precedes_negative_intercept_fit(self):
        model = load_cost_model()
        manager = LosslessCacheManager(
            gpu_capacity_blocks=1,
            cpu_capacity_blocks=1,
            block_bytes=917_504,
            cost_model=model,
        )
        self.assertAlmostEqual(
            manager._estimate_d2h_ms(),
            model["d2h_migrate"]["samples"][0]["median"],
            places=12,
        )
        self.assertAlmostEqual(
            manager._estimate_h2d_ms(),
            model["h2d_restore"]["samples"][0]["median"],
            places=12,
        )
        self.assertGreaterEqual(manager._estimate_d2h_ms(), 0.0)
        self.assertGreaterEqual(manager._estimate_h2d_ms(), 0.0)


class ControllerComplexityAndSemanticsTests(unittest.TestCase):
    def make_controller(
        self,
        gpu=2,
        cpu=4,
        policy="always_migrate",
        selective_config=None,
    ):
        return FlowCacheLosslessController(
            gpu_capacity_blocks=gpu,
            cpu_capacity_blocks=cpu,
            block_bytes=917_504,
            cost_model=load_cost_model(),
            reuse_estimator_config={
                "type": "heuristic",
                "beta": 0.005,
                "alpha": 0.5,
                "horizon": 1000,
            },
            safety_margin=0.0,
            migration_policy=policy,
            selective_migration_config=selective_config,
        )

    def test_ordered_gpu_cache_preserves_exact_lru_semantics(self):
        controller = self.make_controller(gpu=2, cpu=2)
        self.assertFalse(controller.access("a", prefill_ms=1.0))
        self.assertFalse(controller.access("b", prefill_ms=1.0))
        self.assertTrue(controller.access("a", prefill_ms=1.0))
        self.assertFalse(controller.access("c", prefill_ms=1.0))

        self.assertEqual(list(controller.manager.gpu_cache), ["a", "c"])
        self.assertEqual(list(controller.manager.cpu_cache), ["b"])
        self.assertEqual(controller.manager.migrate_to_cpu_count, 1)

    def test_lazy_heap_matches_naive_lowest_residency_value(self):
        controller = self.make_controller(gpu=1, cpu=8)
        controller._clock = 500
        blocks = {
            "a": {
                "last_access": 100, "share_count": 4, "block_idx": 0,
                "prefill_ms": 20.0, "access_count": 2,
            },
            "b": {
                "last_access": 250, "share_count": 0, "block_idx": 60,
                "prefill_ms": 10.0, "access_count": 1,
            },
            "c": {
                "last_access": 300, "share_count": 2, "block_idx": 20,
                "prefill_ms": 30.0, "access_count": 4,
            },
        }
        for block_hash, metadata in blocks.items():
            controller.manager.admit_cpu(block_hash, dict(metadata))
            controller._register_cpu_block(block_hash)

        expected = min(
            blocks,
            key=lambda block_hash: controller._cpu_residency_value_index(
                blocks[block_hash]
            ),
        )
        self.assertEqual(controller._select_cpu_victim(), expected)
        self.assertNotEqual(
            controller._select_cpu_victim(protect_hash=expected), expected
        )

    def test_old_blocks_remain_ranked_without_fixed_horizon_collapse(self):
        controller = self.make_controller(gpu=1, cpu=8)
        controller._clock = 2000
        for block_hash, last_access, prefill_ms in [
            ("old", 100, 10.0),
            ("new", 1500, 20.0),
        ]:
            controller.manager.admit_cpu(
                block_hash,
                {
                    "last_access": last_access,
                    "share_count": 1,
                    "block_idx": 0,
                    "prefill_ms": prefill_ms,
                    "access_count": 1,
                },
            )
            controller._register_cpu_block(block_hash)
        expected = min(
            controller.manager.cpu_cache,
            key=lambda block_hash: controller._cpu_residency_value_index(
                controller.manager.cpu_cache[block_hash]
            ),
        )
        self.assertEqual(controller._select_cpu_victim(), expected)
        self.assertNotEqual(
            controller._reuse_likelihood_proxy(
                controller.manager.cpu_cache["old"]
            ),
            controller._reuse_likelihood_proxy(
                controller.manager.cpu_cache["new"]
            ),
        )

    def test_lazy_heap_compaction_bounds_stale_entries(self):
        controller = self.make_controller(gpu=1, cpu=8)
        controller.manager.admit_cpu(
            "a",
            {
                "last_access": 0,
                "share_count": 1,
                "block_idx": 0,
                "prefill_ms": 10.0,
                "access_count": 1,
            },
        )
        for _ in range(4_200):
            controller._register_cpu_block("a")
        self.assertGreater(controller.cpu_heap_compaction_count, 0)
        self.assertLessEqual(len(controller._cpu_value_heap), 4_096)
        self.assertEqual(controller._select_cpu_victim(), "a")

    def test_lazy_heap_matches_scan_reference_on_deterministic_trace(self):
        class ScanReferenceController(FlowCacheLosslessController):
            def _select_cpu_victim(self, protect_hash=None):
                candidates = [
                    (block_hash, metadata)
                    for block_hash, metadata in self.manager.cpu_cache.items()
                    if block_hash != protect_hash
                ]
                if not candidates:
                    return None
                return min(
                    candidates,
                    key=lambda item: self._cpu_residency_value_index(
                        item[1]
                    ),
                )[0]

        kwargs = {
            "gpu_capacity_blocks": 17,
            "cpu_capacity_blocks": 31,
            "block_bytes": 917_504,
            "cost_model": load_cost_model(),
            "reuse_estimator_config": {
                "type": "heuristic",
                "beta": 0.005,
                "alpha": 0.5,
                "horizon": 1_000_000,
            },
            "safety_margin": 0.0,
            "migration_policy": "always_migrate",
        }
        optimized = FlowCacheLosslessController(**kwargs)
        reference = ScanReferenceController(**kwargs)
        rng = random.Random(42)
        trace = [rng.randrange(70) for _ in range(3000)]
        for index, block_id in enumerate(trace):
            access_args = {
                "block_hash": f"b-{block_id}",
                "prefill_ms": 1.0 + (block_id % 7),
                "block_idx": block_id % 60,
                "share_count": 1 + (block_id % 4),
            }
            self.assertEqual(
                optimized.access(**access_args),
                reference.access(**access_args),
                msg=f"hit/miss diverged at access {index}",
            )
            self.assertEqual(
                list(optimized.manager.gpu_cache),
                list(reference.manager.gpu_cache),
                msg=f"GPU state diverged at access {index}",
            )
            self.assertEqual(
                set(optimized.manager.cpu_cache),
                set(reference.manager.cpu_cache),
                msg=f"CPU state diverged at access {index}",
            )


class SelectiveMigrationTests(unittest.TestCase):
    def make_controller(
        self,
        gpu=1,
        cpu=2,
        policy="selective_value",
        gpu_admission_policy="always_admit",
        selective_config=None,
    ):
        selection = {
            "expected_cpu_residence_steps": 100,
            "share_count_cap": 8,
            "reuse_count_scale": 2.0,
        }
        if selective_config:
            selection.update(selective_config)
        return FlowCacheLosslessController(
            gpu_capacity_blocks=gpu,
            cpu_capacity_blocks=cpu,
            block_bytes=917_504,
            cost_model=load_cost_model(),
            reuse_estimator_config={
                "type": "heuristic",
                "horizon": 1000,
            },
            safety_margin=0.0,
            migration_policy=policy,
            gpu_admission_policy=gpu_admission_policy,
            selective_migration_config=selection,
        )

    def test_low_value_victim_is_evicted_without_migration(self):
        controller = self.make_controller()
        controller.access("low", prefill_ms=0.1, share_count=1)
        controller.access("next", prefill_ms=0.1, share_count=1)
        stats = controller.get_stats()

        self.assertNotIn("low", controller.manager.cpu_cache)
        self.assertEqual(stats["migration_candidate_count"], 1)
        self.assertEqual(stats["migration_selected_count"], 0)
        self.assertEqual(stats["migration_rejected_count"], 1)
        self.assertEqual(stats["rejected_low_value_count"], 1)

    def test_high_value_victim_is_migrated(self):
        controller = self.make_controller()
        controller.access("high", prefill_ms=100.0, share_count=8)
        controller.access("next", prefill_ms=1.0, share_count=1)
        stats = controller.get_stats()

        self.assertIn("high", controller.manager.cpu_cache)
        self.assertEqual(stats["migration_selected_count"], 1)
        self.assertEqual(stats["migrate_to_cpu_count"], 1)

    def test_cpu_competition_preserves_more_valuable_incumbent(self):
        controller = self.make_controller(cpu=1)
        controller.access("high", prefill_ms=100.0, share_count=8)
        controller.access("medium", prefill_ms=20.0, share_count=8)
        self.assertIn("high", controller.manager.cpu_cache)

        controller.access("next", prefill_ms=1.0, share_count=1)
        stats = controller.get_stats()
        self.assertEqual(list(controller.manager.cpu_cache), ["high"])
        self.assertEqual(stats["rejected_cpu_competition_count"], 1)

    def test_selective_moves_less_than_independent_always_ablation(self):
        selective = self.make_controller(cpu=3)
        always = self.make_controller(cpu=3, policy="always_migrate")
        for index in range(20):
            args = {
                "block_hash": f"b-{index}",
                "prefill_ms": 0.1,
                "share_count": 1,
            }
            selective.access(**args)
            always.access(**args)

        selective_stats = selective.get_stats()
        always_stats = always.get_stats()
        self.assertEqual(
            selective_stats["migration_candidate_count"],
            always_stats["migration_candidate_count"],
        )
        self.assertLess(
            selective_stats["migrate_to_cpu_count"],
            always_stats["migrate_to_cpu_count"],
        )
        self.assertEqual(
            always_stats["migration_candidate_count"],
            always_stats["migration_selected_count"],
        )

    def test_hot_cold_trace_keeps_hits_while_avoiding_cold_moves(self):
        selective = self.make_controller(cpu=5)
        always = self.make_controller(cpu=5, policy="always_migrate")
        trace = []
        for index in range(20):
            trace.extend(
                [
                    {
                        "block_hash": "hot",
                        "prefill_ms": 100.0,
                        "share_count": 8,
                    },
                    {
                        "block_hash": f"cold-{index}",
                        "prefill_ms": 0.1,
                        "share_count": 1,
                        "block_idx": 60,
                    },
                ]
            )
        for access in trace:
            selective.access(**access)
            always.access(**access)

        selective_stats = selective.get_stats()
        always_stats = always.get_stats()
        self.assertEqual(selective_stats["hits"], always_stats["hits"])
        self.assertLess(
            selective_stats["migrate_to_cpu_count"],
            always_stats["migrate_to_cpu_count"],
        )
        self.assertGreater(
            selective_stats["migration_rejected_count"], 0
        )

    def test_oracle_cost_proxy_bypasses_cold_without_losing_hot_hits(self):
        cost_admission = self.make_controller(
            cpu=5,
            gpu_admission_policy="oracle_cost_proxy",
        )
        migration_only = self.make_controller(cpu=5)
        trace = []
        for index in range(20):
            trace.extend(
                [
                    {
                        "block_hash": "hot",
                        "prefill_ms": 100.0,
                        "share_count": 8,
                    },
                    {
                        "block_hash": f"cold-{index}",
                        "prefill_ms": 0.1,
                        "share_count": 1,
                        "block_idx": 60,
                    },
                ]
            )

        for access in trace:
            cost_admission.access(**access)
            migration_only.access(**access)

        admission_stats = cost_admission.get_stats()
        migration_stats = migration_only.get_stats()
        self.assertEqual(admission_stats["hits"], migration_stats["hits"])
        self.assertEqual(admission_stats["migrate_to_cpu_count"], 0)
        self.assertEqual(admission_stats["restore_to_gpu_count"], 0)
        self.assertEqual(admission_stats["gpu_admission_bypassed_count"], 20)
        self.assertEqual(
            admission_stats["gpu_admission_candidate_count"],
            admission_stats["gpu_admission_bypassed_count"],
        )
        self.assertAlmostEqual(
            admission_stats["gpu_bypassed_prefill_ms_total"], 2.0
        )
        self.assertIn("hot", cost_admission.manager.gpu_cache)
        self.assertNotIn("cold-19", cost_admission.manager.gpu_cache)

    def test_oracle_cost_proxy_admits_more_valuable_incoming_block(self):
        controller = self.make_controller(
            cpu=2,
            gpu_admission_policy="oracle_cost_proxy",
        )
        controller.access(
            "low", prefill_ms=0.1, share_count=1, block_idx=60
        )
        controller.access(
            "high", prefill_ms=100.0, share_count=8, block_idx=0
        )

        stats = controller.get_stats()
        self.assertIn("high", controller.manager.gpu_cache)
        self.assertEqual(stats["gpu_admission_candidate_count"], 1)
        self.assertEqual(stats["gpu_admission_selected_count"], 1)
        self.assertEqual(stats["gpu_admission_bypassed_count"], 0)

    def test_equal_cost_cold_start_is_not_discounted_into_bypass(self):
        controller = self.make_controller(
            cpu=2,
            gpu_admission_policy="oracle_cost_proxy",
            selective_config={
                "gpu_admission_cold_start_prior": 0.05,
                "gpu_admission_cold_start_cost_ratio": 0.5,
            },
        )
        controller.access(
            "first", prefill_ms=10.0, share_count=1, block_idx=20
        )
        controller.access(
            "second", prefill_ms=10.0, share_count=1, block_idx=20
        )

        stats = controller.get_stats()
        self.assertIn("second", controller.manager.gpu_cache)
        self.assertEqual(stats["gpu_admission_selected_count"], 1)
        self.assertEqual(stats["gpu_admission_bypassed_count"], 0)
        self.assertEqual(
            stats["gpu_admission_cold_start_cost_ratio"], 0.5
        )

    def test_reused_incumbent_can_reject_equal_cost_first_seen_block(self):
        controller = self.make_controller(
            cpu=2,
            gpu_admission_policy="oracle_cost_proxy",
            selective_config={
                "gpu_admission_cold_start_prior": 0.05,
                "gpu_admission_cold_start_cost_ratio": 0.5,
            },
        )
        controller.access(
            "reused", prefill_ms=10.0, share_count=1, block_idx=20
        )
        controller.access(
            "reused", prefill_ms=10.0, share_count=1, block_idx=20
        )
        controller.access(
            "first-seen", prefill_ms=10.0, share_count=1, block_idx=20
        )

        stats = controller.get_stats()
        self.assertIn("reused", controller.manager.gpu_cache)
        self.assertNotIn("first-seen", controller.manager.gpu_cache)
        self.assertEqual(stats["gpu_admission_bypassed_count"], 1)

    def test_proxy_is_bounded_and_orders_clear_high_above_low(self):
        controller = self.make_controller()
        controller._clock = 10
        high = {
            "last_access": 9,
            "share_count": 8,
            "access_count": 5,
            "block_idx": 0,
            "prefill_ms": 50.0,
        }
        low = {
            "last_access": 0,
            "share_count": 1,
            "access_count": 1,
            "block_idx": 60,
            "prefill_ms": 50.0,
        }
        high_proxy = controller._reuse_likelihood_proxy(high)
        low_proxy = controller._reuse_likelihood_proxy(low)
        self.assertGreaterEqual(low_proxy, 0.0)
        self.assertLessEqual(high_proxy, 1.0)
        self.assertGreater(high_proxy, low_proxy)

    def test_incoming_confidence_uses_observed_reuse_not_future(self):
        controller = self.make_controller(
            gpu_admission_policy="oracle_cost_proxy"
        )
        first_seen = {
            "last_access": 0,
            "share_count": 1,
            "access_count": 1,
            "block_idx": 0,
            "prefill_ms": 10.0,
        }
        seen_again = dict(first_seen, access_count=2)
        self.assertGreater(
            controller._incoming_gpu_residency_value_index(seen_again),
            controller._incoming_gpu_residency_value_index(first_seen),
        )

    def test_causal_frequency_history_survives_physical_eviction(self):
        controller = self.make_controller()
        controller.access("a", prefill_ms=0.1, share_count=1)
        controller.access("b", prefill_ms=0.1, share_count=1)
        self.assertNotIn("a", controller.manager.gpu_cache)
        self.assertNotIn("a", controller.manager.cpu_cache)

        controller.access("a", prefill_ms=0.1, share_count=1)
        self.assertEqual(
            controller.manager.gpu_cache["a"]["access_count"], 2
        )
        self.assertEqual(controller.get_stats()["historical_blocks_tracked"], 2)


class ReplayAccountingTests(unittest.TestCase):
    def test_parameter_sweep_reuses_parsed_causal_trace(self):
        records = [
            {
                "request_id": "r1",
                "workflow_id": "w1",
                "task_id": "task-1",
                "block_hash": "a",
                "prefill_ms": 1.0,
                "block_idx": 0,
            },
            {
                "request_id": "r2",
                "workflow_id": "w2",
                "task_id": "task-2",
                "block_hash": "b",
                "prefill_ms": 1.0,
                "block_idx": 0,
            },
        ]
        config = {
            "g0": {
                "block_size": 16,
                "num_hidden_layers": 28,
                "num_kv_heads": 4,
                "head_dim": 128,
                "dtype_bytes": 2,
            },
            "trace_source": {},
            "flowcache": {
                "heuristic": {"horizon": 1000},
                "selective_migration": {
                    "share_window_accesses": 1000,
                },
            },
            "protocol_test": {
                "cell": {"capacity_gib": 1, "concurrency": 4},
                "baselines": ["no_cache"],
                "episodes": 100,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "access_trace_c4.jsonl"
            with open(trace_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            config["trace_source"]["access_trace_dir"] = temp_dir
            access_cache = {}
            with mock.patch(
                "run_g3_grid.load_access_trace",
                wraps=load_access_trace,
            ) as loader:
                g3_runner.run_grid(
                    config,
                    protocol_test=True,
                    access_cache=access_cache,
                )
                g3_runner.run_grid(
                    config,
                    protocol_test=True,
                    access_cache=access_cache,
                )

            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(access_cache), 1)

    def test_episode_cap_uses_workflow_id_and_keeps_interleaved_records(self):
        records = [
            {"workflow_id": "w1", "request_id": "r1", "block_hash": "a"},
            {"workflow_id": "w2", "request_id": "r2", "block_hash": "b"},
            {"workflow_id": "w1", "request_id": "r3", "block_hash": "c"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            with open(trace_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            selected = load_access_trace(trace_path, max_episodes=1)
        self.assertEqual([row["block_hash"] for row in selected], ["a", "c"])

    def test_share_count_annotation_is_causal_and_windowed(self):
        accesses = [
            {"block_hash": "b", "workflow_id": "w1"},
            {"block_hash": "x", "workflow_id": "other"},
            {"block_hash": "b", "workflow_id": "w2"},
            {"block_hash": "x", "workflow_id": "other"},
            {"block_hash": "x", "workflow_id": "other"},
            {"block_hash": "b", "workflow_id": "w3"},
        ]
        annotate_causal_share_counts(accesses, horizon=3)
        b_counts = [
            row["_share_count"]
            for row in accesses
            if row["block_hash"] == "b"
        ]
        self.assertEqual(b_counts, [1, 2, 1])
        self.assertEqual(
            accesses[0]["_share_count_scope"],
            "causal_past_window_including_current",
        )

    def test_online_factory_rejects_future_access_index(self):
        with self.assertRaisesRegex(
            ValueError, "future_accesses is forbidden"
        ):
            instantiate_baseline(
                "flowcache_lossless",
                capacity_blocks=1,
                cost_model=load_cost_model(),
                flowcache_config={},
                future_accesses={"future-only": [10]},
            )

    def test_flowcache_prefix_decisions_are_invariant_to_changed_future(self):
        prefix = [
            {
                "block_hash": "hot",
                "workflow_id": "w1",
                "prefill_ms": 100.0,
                "block_idx": 0,
            },
            {
                "block_hash": "cold",
                "workflow_id": "w1",
                "prefill_ms": 0.1,
                "block_idx": 60,
            },
            {
                "block_hash": "hot",
                "workflow_id": "w1",
                "prefill_ms": 100.0,
                "block_idx": 0,
            },
        ]
        future_a = [
            {"block_hash": "hot", "workflow_id": f"future-{i}"}
            for i in range(20)
        ]
        future_b = [
            {"block_hash": f"never-{i}", "workflow_id": f"future-{i}"}
            for i in range(20)
        ]
        trace_a = [dict(row) for row in prefix + future_a]
        trace_b = [dict(row) for row in prefix + future_b]
        annotate_causal_share_counts(trace_a, horizon=100)
        annotate_causal_share_counts(trace_b, horizon=100)

        def make_online_controller():
            return FlowCacheLosslessController(
                gpu_capacity_blocks=1,
                cpu_capacity_blocks=2,
                block_bytes=917_504,
                cost_model=load_cost_model(),
                reuse_estimator_config={
                    "type": "heuristic",
                    "horizon": 1000,
                },
                safety_margin=0.0,
                migration_policy="selective_value",
                gpu_admission_policy="oracle_cost_proxy",
                selective_migration_config={
                    "expected_cpu_residence_steps": 100,
                    "share_count_cap": 8,
                    "reuse_count_scale": 2.0,
                },
            )

        controller_a = make_online_controller()
        controller_b = make_online_controller()
        decisions_a = []
        decisions_b = []
        for row_a, row_b in zip(
            trace_a[:len(prefix)], trace_b[:len(prefix)]
        ):
            self.assertEqual(
                row_a["_share_count"], row_b["_share_count"]
            )
            decisions_a.append(
                controller_a.access(
                    row_a["block_hash"],
                    prefill_ms=row_a["prefill_ms"],
                    block_idx=row_a["block_idx"],
                    share_count=row_a["_share_count"],
                )
            )
            decisions_b.append(
                controller_b.access(
                    row_b["block_hash"],
                    prefill_ms=row_b["prefill_ms"],
                    block_idx=row_b["block_idx"],
                    share_count=row_b["_share_count"],
                )
            )

        self.assertEqual(decisions_a, decisions_b)
        self.assertEqual(
            list(controller_a.manager.gpu_cache),
            list(controller_b.manager.gpu_cache),
        )
        self.assertEqual(
            list(controller_a.manager.cpu_cache),
            list(controller_b.manager.cpu_cache),
        )
        self.assertEqual(
            controller_a.get_stats()["gpu_admission_bypassed_count"],
            controller_b.get_stats()["gpu_admission_bypassed_count"],
        )

    def test_task_grouped_split_is_stable_and_has_no_overlap(self):
        accesses = [
            {
                "task_id": f"task-{task}",
                "workflow_id": f"task-{task}-seed-{seed}",
            }
            for task in range(100)
            for seed in range(3)
        ]
        validation = filter_task_split(
            accesses, "validation", validation_fraction=0.2, split_seed=42
        )
        test = filter_task_split(
            accesses, "test", validation_fraction=0.2, split_seed=42
        )
        validation_tasks = {row["task_id"] for row in validation}
        test_tasks = {row["task_id"] for row in test}
        self.assertTrue(validation_tasks)
        self.assertTrue(test_tasks)
        self.assertFalse(validation_tasks & test_tasks)
        self.assertEqual(
            validation_tasks | test_tasks,
            {row["task_id"] for row in accesses},
        )
        self.assertEqual(
            validation,
            filter_task_split(
                accesses,
                "validation",
                validation_fraction=0.2,
                split_seed=42,
            ),
        )

    def test_replay_charges_movement_and_marks_open_loop_limits(self):
        controller = FlowCacheLosslessController(
            gpu_capacity_blocks=1,
            cpu_capacity_blocks=2,
            block_bytes=917_504,
            cost_model=load_cost_model(),
            reuse_estimator_config={"type": "heuristic", "horizon": 1000},
            safety_margin=0.0,
            migration_policy="always_migrate",
        )
        accesses = [
            {
                "block_hash": "a",
                "request_id": "r1",
                "task_id": "t",
                "arrival_time_ms": 0,
                "prefill_ms": 10.0,
            },
            {
                "block_hash": "b",
                "request_id": "r2",
                "task_id": "t",
                "arrival_time_ms": 1000,
                "prefill_ms": 20.0,
            },
            {
                "block_hash": "a",
                "request_id": "r3",
                "task_id": "t",
                "arrival_time_ms": 2000,
                "prefill_ms": 10.0,
            },
        ]
        global_metrics, per_task = replay_accesses(controller, accesses)
        task = per_task["t"]

        self.assertEqual(global_metrics["migrate_count"], 2)
        self.assertEqual(global_metrics["restore_count"], 1)
        self.assertAlmostEqual(
            task["task_transfer_ms"],
            global_metrics["transfer_ms_total"],
            places=12,
        )
        self.assertGreater(task["task_transfer_ms"], 0.0)
        self.assertGreater(task["task_policy_model_ms"], 0.0)
        self.assertFalse(global_metrics["ttft_metric_valid"])
        self.assertFalse(global_metrics["throughput_metric_valid"])
        self.assertEqual(
            global_metrics["latency_metric_scope"], "modeled_cache_delay"
        )
        self.assertAlmostEqual(
            global_metrics["offered_load_req_per_s"], 1.5, places=12
        )
        self.assertEqual(global_metrics["negative_cost_count"], 0)


class FrozenSelectionTests(unittest.TestCase):
    @staticmethod
    def valid_report():
        return {
            "status": "SELECTED",
            "split": {
                "unit": "task_id",
                "partition": "validation",
                "fraction": 0.2,
                "seed": 42,
            },
            "selected": {
                "minimum_net_benefit_ms": 1.0,
                "cpu_admission_margin_ms": 0.5,
                "gpu_admission_margin_ms": 0.25,
                "gpu_admission_cold_start_cost_ratio": 0.75,
                "expected_cpu_residence_steps": 64,
                "future_only_metric": 999,
            },
        }

    def test_freezer_copies_only_preregistered_parameters(self):
        params = validate_selection(
            self.valid_report(),
            expected_seed=42,
            expected_fraction=0.2,
        )
        self.assertEqual(
            set(params),
            {
                "minimum_net_benefit_ms",
                "cpu_admission_margin_ms",
                "gpu_admission_margin_ms",
                "gpu_admission_cold_start_cost_ratio",
                "expected_cpu_residence_steps",
            },
        )
        self.assertNotIn("future_only_metric", params)

    def test_freezer_rejects_no_valid_configuration(self):
        report = self.valid_report()
        report["status"] = "NO_VALID_CONFIG"
        with self.assertRaisesRegex(ValueError, "not SELECTED"):
            validate_selection(
                report,
                expected_seed=42,
                expected_fraction=0.2,
            )


class VerdictSafetyTests(unittest.TestCase):
    @staticmethod
    def row(baseline, task_id, hits, misses, ttft):
        return {
            "capacity_gib": "1",
            "concurrency": "4",
            "baseline": baseline,
            "task_id": task_id,
            "global_p95_cache_delay_ms": str(ttft),
            "global_p50_cache_delay_ms": str(ttft / 2),
            "global_block_hit_rate": "0.5",
            "global_throughput": "10",
            "global_offered_load": "10",
            "latency_metric_scope": "modeled_cache_delay",
            "ttft_metric_valid": "False",
            "throughput_metric_valid": "False",
            "controller_variant": (
                "selective_value"
                if baseline == "flowcache_lossless"
                else "not_applicable"
            ),
            "migrate_ms_total": "2" if baseline == "flowcache_lossless" else "0",
            "restore_ms_total": "1" if baseline == "flowcache_lossless" else "0",
            "negative_cost_count": "0",
            "task_miss_cost_ms": "10",
            "task_saved_prefill_ms": "20",
            "task_transfer_ms": (
                "1.5" if baseline == "flowcache_lossless" else "0"
            ),
            "task_policy_model_ms": "0.1",
            "task_hits": str(hits),
            "task_misses": str(misses),
            "task_p95_ttft_ms": str(ttft),
        }

    def test_aggregation_sums_all_task_rows(self):
        rows = [
            self.row("flowcache_lossless", "t1", 2, 1, 8),
            self.row("flowcache_lossless", "t2", 3, 4, 8),
        ]
        metrics = aggregate_by_cell_baseline(rows)[(1.0, 4)][
            "flowcache_lossless"
        ]
        self.assertEqual(metrics["hits"], 5)
        self.assertEqual(metrics["misses"], 5)
        self.assertEqual(metrics["saved_prefill_ms"], 40.0)
        self.assertEqual(metrics["miss_cost_ms"], 20.0)
        self.assertEqual(metrics["transfer_ms_task_sum"], 3.0)

    def test_open_loop_cannot_emit_go_or_no_go(self):
        rows = []
        for baseline, ttft in [
            ("flowcache_lossless", 8),
            ("gdsf", 10),
            ("sizecost", 11),
        ]:
            rows.extend(
                [
                    self.row(baseline, "t1", 2, 1, ttft),
                    self.row(baseline, "t2", 3, 4, ttft),
                ]
            )
        config = {
            "capacity": {
                "main_cell": {"capacity_gib": 1, "concurrency": 4}
            },
            "verdict": {
                "p95_ttft_threshold": 0.15,
                "throughput_drop_threshold": 0.05,
                "bootstrap_samples": 20,
            },
        }
        verdict = evaluate_go_no_go(rows, config)
        self.assertEqual(verdict["go_no_go"], "PROTOCOL-INCOMPLETE")
        self.assertFalse(verdict["conditions"]["protocol_valid"]["pass"])
        self.assertFalse(
            verdict["conditions"]["throughput_noninferior"]["valid"]
        )


class SelectiveTuningSafetyTests(unittest.TestCase):
    @staticmethod
    def constraints():
        return {
            "min_selection_rate": 0.01,
            "max_selection_rate": 0.99,
            "min_gpu_bypass_rate": 0.01,
            "max_gpu_bypass_rate": 0.99,
            "min_gpu_bypass_transfer_reduction": 0.05,
            "min_movement_reduction": 0.10,
            "max_modeled_delay_increase": 0.05,
            "max_replay_ratio": 3.0,
        }

    def test_validation_summary_accepts_constrained_candidate(self):
        rows = [
            {
                "baseline": "flowcache_lossless",
                "migration_candidate_count": "100",
                "migration_selected_count": "50",
                "migrate_count": "50",
                "gpu_admission_candidate_count": "100",
                "gpu_admission_bypassed_count": "50",
                "transfer_ms_total": "50",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "100",
                "elapsed_s": "2",
                "fallback_count": "0",
                "negative_cost_count": "0",
            },
            {
                "baseline": "flowcache_always_migrate",
                "migrate_count": "100",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "110",
                "elapsed_s": "1.5",
            },
            {
                "baseline": "flowcache_selective_migrate_only",
                "transfer_ms_total": "100",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "105",
                "elapsed_s": "1.8",
            },
            {"baseline": "oracle_cost", "elapsed_s": "1"},
        ]
        summary = summarize_candidate(rows, self.constraints())
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["movement_reduction"], 0.5)

    def test_validation_summary_rejects_always_migrate_collapse(self):
        rows = [
            {
                "baseline": "flowcache_lossless",
                "migration_candidate_count": "100",
                "migration_selected_count": "100",
                "migrate_count": "100",
                "gpu_admission_candidate_count": "100",
                "gpu_admission_bypassed_count": "0",
                "transfer_ms_total": "100",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "100",
                "elapsed_s": "2",
                "fallback_count": "0",
                "negative_cost_count": "0",
            },
            {
                "baseline": "flowcache_always_migrate",
                "migrate_count": "100",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "100",
                "elapsed_s": "1.5",
            },
            {
                "baseline": "flowcache_selective_migrate_only",
                "transfer_ms_total": "100",
                "global_p95_cache_delay_ms": "100",
                "task_modeled_service_cost_ms": "100",
                "elapsed_s": "1.8",
            },
            {"baseline": "oracle_cost", "elapsed_s": "1"},
        ]
        summary = summarize_candidate(rows, self.constraints())
        self.assertFalse(summary["valid"])
        self.assertIn("selection_rate", summary["failure_reasons"])
        self.assertIn("movement_reduction", summary["failure_reasons"])
        self.assertIn("gpu_bypass_rate", summary["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
