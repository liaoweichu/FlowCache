"""Deterministic microbenchmark for the G3-P1 controller hot path.

This measures *offline replay implementation time*.  It is intentionally
separate from modeled/serving latency and must never be reported as TTFT.
"""

import argparse
import json
import time
from pathlib import Path

from controller import FlowCacheLosslessController


SCRIPT_DIR = Path(__file__).resolve().parent


def synthetic_access(index, args):
    """Return a deterministic access; never treat it as workload evidence."""
    if args.pattern == "hot-cold":
        if index % 2 == 0:
            return {
                "block_hash": "hot",
                "prefill_ms": 100.0,
                "block_idx": 0,
                "share_count": 8,
            }
        return {
            "block_hash": f"cold-{index // 2}",
            "prefill_ms": 0.1,
            "block_idx": 60,
            "share_count": 1,
        }

    block_id = index % args.working_set
    return {
        "block_hash": f"block-{block_id}",
        "prefill_ms": 10.0,
        "block_idx": block_id % 80,
        "share_count": 1 + (block_id % 5),
    }


def run_policy(
    args,
    cost_model,
    label,
    migration_policy,
    gpu_admission_policy,
):
    controller = FlowCacheLosslessController(
        gpu_capacity_blocks=args.gpu_blocks,
        cpu_capacity_blocks=args.cpu_blocks,
        block_bytes=917_504,
        cost_model=cost_model,
        reuse_estimator_config={
            "type": "heuristic",
            "beta": 0.005,
            "alpha": 0.5,
            "horizon": 1000,
        },
        safety_margin=0.0,
        migration_policy=migration_policy,
        gpu_admission_policy=gpu_admission_policy,
        selective_migration_config={
            "minimum_net_benefit_ms": 0.0,
            "cpu_admission_margin_ms": 0.0,
            "gpu_admission_margin_ms": 0.0,
            "gpu_admission_cold_start_prior": 0.05,
            "gpu_admission_cold_start_cost_ratio": 0.5,
            "gpu_admission_confidence_scale": 1.0,
            "expected_cpu_residence_steps": 100,
            "hold_cost_weight": 1.0,
            "age_scale_capacity_multiplier": 1.0,
            "share_count_cap": 8,
            "reuse_count_scale": 2.0,
            "signal_weights": {
                "share": 0.45,
                "frequency": 0.35,
                "position": 0.20,
            },
        },
    )

    started = time.perf_counter()
    for index in range(args.accesses):
        controller.access(**synthetic_access(index, args))
    elapsed_s = time.perf_counter() - started
    stats = controller.get_stats()
    candidates = stats["migration_candidate_count"]
    selected = stats["migration_selected_count"]
    return {
        "policy": label,
        "migration_policy": migration_policy,
        "gpu_admission_policy": gpu_admission_policy,
        "elapsed_s": round(elapsed_s, 6),
        "wall_us_per_access": round(
            elapsed_s * 1_000_000 / max(1, args.accesses), 6
        ),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "migrate_count": stats["migrate_to_cpu_count"],
        "restore_count": stats["restore_to_gpu_count"],
        "migration_candidates": candidates,
        "migration_selected": selected,
        "migration_rejected": stats["migration_rejected_count"],
        "selection_rate": round(
            selected / candidates if candidates else 0.0, 6
        ),
        "modeled_movement_ms": round(
            stats["migrate_ms_total"] + stats["restore_ms_total"], 6
        ),
        "modeled_miss_cost_ms": round(stats["miss_cost_ms"], 6),
        "modeled_service_cost_ms": round(
            stats["miss_cost_ms"]
            + stats["migrate_ms_total"]
            + stats["restore_ms_total"]
            + stats["policy_model_ms_total"],
            6,
        ),
        "gpu_admission_candidates": stats[
            "gpu_admission_candidate_count"
        ],
        "gpu_admission_selected": stats[
            "gpu_admission_selected_count"
        ],
        "gpu_admission_bypassed": stats[
            "gpu_admission_bypassed_count"
        ],
        "gpu_admission_bypass_rate": round(
            stats["gpu_admission_bypass_rate"], 6
        ),
        "fallback_count": stats["fallback_count"],
        "negative_transfer_cost": (
            stats["migrate_ms_total"] < 0 or stats["restore_ms_total"] < 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accesses", type=int, default=200_000)
    parser.add_argument("--gpu-blocks", type=int, default=2_223)
    parser.add_argument("--cpu-blocks", type=int, default=4_446)
    parser.add_argument("--working-set", type=int, default=8_000)
    parser.add_argument(
        "--pattern",
        choices=["cyclic", "hot-cold"],
        default="cyclic",
    )
    parser.add_argument(
        "--policy",
        choices=[
            "oracle_cost_proxy",
            "selective_migration_only",
            "always_migrate",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    with open(SCRIPT_DIR / "cost-model.json", "r", encoding="utf-8") as handle:
        cost_model = json.load(handle)

    variants = {
        "oracle_cost_proxy": (
            "oracle_cost_proxy+selective_value",
            "selective_value",
            "oracle_cost_proxy",
        ),
        "selective_migration_only": (
            "always_admit+selective_value",
            "selective_value",
            "always_admit",
        ),
        "always_migrate": (
            "always_admit+always_migrate",
            "always_migrate",
            "always_admit",
        ),
    }
    policies = (
        list(variants)
        if args.policy == "all"
        else [args.policy]
    )
    results = [
        run_policy(args, cost_model, *variants[policy])
        for policy in policies
    ]
    output = {
        "scope": "offline_replay_implementation_time",
        "scientific_result": False,
        "accesses": args.accesses,
        "gpu_blocks": args.gpu_blocks,
        "cpu_blocks": args.cpu_blocks,
        "working_set": args.working_set,
        "pattern": args.pattern,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
