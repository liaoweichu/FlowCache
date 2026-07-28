"""Fail-closed checker for a single-cell G3-P1 replay CSV."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


FLOWCACHE = "flowcache_lossless"
ALWAYS_MIGRATE = "flowcache_always_migrate"
SELECTIVE_MIGRATE_ONLY = "flowcache_selective_migrate_only"
REQUIRED = {
    "gdsf",
    "sizecost",
    "oracle_cost",
    ALWAYS_MIGRATE,
    SELECTIVE_MIGRATE_ONLY,
    FLOWCACHE,
}


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; stdout is always retained.",
    )
    parser.add_argument("--max-replay-ratio", type=float, default=3.0)
    parser.add_argument(
        "--expected-task-split",
        choices=["all", "validation", "test"],
        default="test",
    )
    parser.add_argument("--min-selection-rate", type=float, default=0.01)
    parser.add_argument("--max-selection-rate", type=float, default=0.99)
    parser.add_argument("--min-gpu-bypass-rate", type=float, default=0.01)
    parser.add_argument("--max-gpu-bypass-rate", type=float, default=0.99)
    parser.add_argument(
        "--min-gpu-bypass-transfer-reduction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min-movement-reduction", type=float, default=0.10
    )
    parser.add_argument(
        "--max-modeled-delay-increase",
        type=float,
        default=0.05,
        help=(
            "Maximum diagnostic cache-delay/service-cost increase allowed "
            "for selective migration relative to always-migrate"
        ),
    )
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("baseline", "")].append(row)

    checks = {}
    baselines = set(grouped)
    checks["required_baselines"] = {
        "pass": REQUIRED.issubset(baselines),
        "found": sorted(baselines),
        "missing": sorted(REQUIRED - baselines),
    }

    cells = {
        (row.get("capacity_gib"), row.get("concurrency")) for row in rows
    }
    checks["single_cell"] = {
        "pass": len(cells) == 1,
        "cells": sorted([list(cell) for cell in cells]),
    }

    fc_rows = grouped.get(FLOWCACHE, [])
    fc_first = fc_rows[0] if fc_rows else {}
    transfer_details = {}
    transfer_pass = True
    for baseline in [
        ALWAYS_MIGRATE, SELECTIVE_MIGRATE_ONLY, FLOWCACHE
    ]:
        baseline_rows = grouped.get(baseline, [])
        first = baseline_rows[0] if baseline_rows else {}
        task_sum = sum(
            as_float(row.get("task_transfer_ms")) for row in baseline_rows
        )
        global_sum = (
            as_float(first.get("migrate_ms_total"))
            + as_float(first.get("restore_ms_total"))
        )
        tolerance = max(1e-6, abs(global_sum) * 1e-6)
        baseline_pass = bool(baseline_rows) and abs(
            task_sum - global_sum
        ) <= tolerance
        transfer_pass = transfer_pass and baseline_pass
        transfer_details[baseline] = {
            "pass": baseline_pass,
            "task_sum_ms": task_sum,
            "global_ms": global_sum,
            "abs_error_ms": abs(task_sum - global_sum),
        }
    checks["transfer_cost_conservation"] = {
        "pass": transfer_pass,
        "details": transfer_details,
    }
    nonnegative_details = {}
    fallback_details = {}
    for baseline in [
        ALWAYS_MIGRATE, SELECTIVE_MIGRATE_ONLY, FLOWCACHE
    ]:
        baseline_rows = grouped.get(baseline, [])
        first = baseline_rows[0] if baseline_rows else {}
        nonnegative_details[baseline] = {
            "pass": (
                bool(baseline_rows)
                and as_float(first.get("negative_cost_count")) == 0
                and as_float(first.get("migrate_ms_total")) >= 0
                and as_float(first.get("restore_ms_total")) >= 0
            ),
            "negative_cost_count": as_float(
                first.get("negative_cost_count")
            ),
        }
        fallback_details[baseline] = {
            "pass": (
                bool(baseline_rows)
                and as_float(first.get("fallback_count")) == 0
            ),
            "fallback_count": as_float(first.get("fallback_count")),
        }
    checks["nonnegative_cost"] = {
        "pass": all(
            detail["pass"] for detail in nonnegative_details.values()
        ),
        "details": nonnegative_details,
    }
    checks["zero_fallback"] = {
        "pass": all(
            detail["pass"] for detail in fallback_details.values()
        ),
        "details": fallback_details,
    }
    checks["diagnostic_scope"] = {
        "pass": (
            bool(fc_rows)
            and fc_first.get("latency_metric_scope")
            == "modeled_cache_delay"
            and not as_bool(fc_first.get("ttft_metric_valid"))
            and not as_bool(fc_first.get("throughput_metric_valid"))
            and fc_first.get("controller_variant")
            == "selective_value"
            and fc_first.get("gpu_admission_policy")
            == "oracle_cost_proxy"
            and fc_first.get("online_feature_scope")
            == "current_and_past_only"
            and not as_bool(fc_first.get("future_access_index_used"))
            and fc_first.get("share_count_feature_scope")
            == "causal_past_window_including_current"
            and fc_first.get("task_split")
            == args.expected_task_split
        ),
        "latency_metric_scope": fc_first.get("latency_metric_scope"),
        "controller_variant": fc_first.get("controller_variant"),
        "gpu_admission_policy": fc_first.get("gpu_admission_policy"),
        "online_feature_scope": fc_first.get("online_feature_scope"),
        "future_access_index_used": fc_first.get(
            "future_access_index_used"
        ),
        "share_count_feature_scope": fc_first.get(
            "share_count_feature_scope"
        ),
        "task_split": fc_first.get("task_split"),
        "expected_task_split": args.expected_task_split,
    }

    candidate_count = as_float(fc_first.get("migration_candidate_count"))
    selected_count = as_float(fc_first.get("migration_selected_count"))
    rejected_count = as_float(fc_first.get("migration_rejected_count"))
    migrate_count = as_float(fc_first.get("migrate_count"))
    checks["selection_accounting"] = {
        "pass": (
            bool(fc_rows)
            and candidate_count == selected_count + rejected_count
            and selected_count == migrate_count
        ),
        "candidates": candidate_count,
        "selected": selected_count,
        "rejected": rejected_count,
        "migrate_count": migrate_count,
    }
    gpu_admission_candidates = as_float(
        fc_first.get("gpu_admission_candidate_count")
    )
    gpu_admission_selected = as_float(
        fc_first.get("gpu_admission_selected_count")
    )
    gpu_admission_bypassed = as_float(
        fc_first.get("gpu_admission_bypassed_count")
    )
    checks["gpu_admission_accounting"] = {
        "pass": (
            bool(fc_rows)
            and gpu_admission_candidates
            == gpu_admission_selected + gpu_admission_bypassed
        ),
        "candidates": gpu_admission_candidates,
        "selected": gpu_admission_selected,
        "bypassed": gpu_admission_bypassed,
        "bypass_rate": (
            gpu_admission_bypassed / gpu_admission_candidates
            if gpu_admission_candidates else 0.0
        ),
    }
    gpu_bypass_rate = (
        gpu_admission_bypassed / gpu_admission_candidates
        if gpu_admission_candidates else 0.0
    )
    checks["nondegenerate_gpu_admission"] = {
        "pass": (
            gpu_admission_candidates > 0
            and args.min_gpu_bypass_rate
            <= gpu_bypass_rate
            <= args.max_gpu_bypass_rate
        ),
        "bypass_rate": gpu_bypass_rate,
        "accepted_range": [
            args.min_gpu_bypass_rate, args.max_gpu_bypass_rate
        ],
    }
    selection_rate = (
        selected_count / candidate_count if candidate_count else 0.0
    )
    checks["nondegenerate_selection"] = {
        "pass": (
            candidate_count > 0
            and args.min_selection_rate
            <= selection_rate
            <= args.max_selection_rate
        ),
        "selection_rate": selection_rate,
        "accepted_range": [
            args.min_selection_rate, args.max_selection_rate
        ],
        "reason_if_failed": (
            "policy is operationally too close to never-migrate or "
            "always-migrate; inspect value distribution on validation data"
        ),
    }

    always_rows = grouped.get(ALWAYS_MIGRATE, [])
    always_first = always_rows[0] if always_rows else {}
    always_migrations = as_float(always_first.get("migrate_count"))
    movement_reduction = (
        (always_migrations - selected_count) / always_migrations
        if always_migrations else 0.0
    )
    checks["movement_reduction_vs_always"] = {
        "pass": (
            bool(always_rows)
            and movement_reduction >= args.min_movement_reduction
        ),
        "selective_migrations": selected_count,
        "always_migrations": always_migrations,
        "relative_reduction": movement_reduction,
        "minimum_required_reduction": args.min_movement_reduction,
        "selective_restore_per_migration": as_float(
            fc_first.get("restore_per_migration")
        ),
        "always_restore_per_migration": as_float(
            always_first.get("restore_per_migration")
        ),
    }
    selective_only_rows = grouped.get(SELECTIVE_MIGRATE_ONLY, [])
    selective_only_first = (
        selective_only_rows[0] if selective_only_rows else {}
    )
    selective_transfer_ms = as_float(fc_first.get("transfer_ms_total"))
    selective_only_transfer_ms = as_float(
        selective_only_first.get("transfer_ms_total")
    )
    bypass_transfer_reduction = (
        (
            selective_only_transfer_ms - selective_transfer_ms
        ) / selective_only_transfer_ms
        if selective_only_transfer_ms > 0
        else 0.0
    )
    checks["gpu_bypass_increment_vs_selective_migration_only"] = {
        "pass": (
            bool(selective_only_rows)
            and bypass_transfer_reduction
            >= args.min_gpu_bypass_transfer_reduction
        ),
        "diagnostic_only": True,
        "flowcache_transfer_ms": selective_transfer_ms,
        "selective_migration_only_transfer_ms": (
            selective_only_transfer_ms
        ),
        "relative_transfer_reduction": bypass_transfer_reduction,
        "minimum_required_reduction": (
            args.min_gpu_bypass_transfer_reduction
        ),
        "flowcache_hits": as_float(fc_first.get("task_hits")),
        "note": (
            "Scientific acceptance requires paired task-level cost/hit "
            "analysis; this check only rejects a movement regression."
        ),
    }
    selective_p95 = as_float(
        fc_first.get("global_p95_cache_delay_ms")
    )
    always_p95 = as_float(
        always_first.get("global_p95_cache_delay_ms")
    )
    selective_service_cost = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in fc_rows
    )
    always_service_cost = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in always_rows
    )
    delay_limit = 1.0 + max(0.0, args.max_modeled_delay_increase)
    checks["modeled_cost_guard_vs_always"] = {
        "pass": (
            bool(fc_rows)
            and bool(always_rows)
            and always_p95 > 0
            and always_service_cost > 0
            and selective_p95 <= always_p95 * delay_limit
            and selective_service_cost <= always_service_cost * delay_limit
        ),
        "diagnostic_only": True,
        "selective_p95_cache_delay_ms": selective_p95,
        "always_p95_cache_delay_ms": always_p95,
        "selective_modeled_service_cost_ms": selective_service_cost,
        "always_modeled_service_cost_ms": always_service_cost,
        "maximum_relative_increase": args.max_modeled_delay_increase,
    }
    selective_only_p95 = as_float(
        selective_only_first.get("global_p95_cache_delay_ms")
    )
    selective_only_service_cost = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in selective_only_rows
    )
    checks["modeled_cost_guard_vs_selective_migration_only"] = {
        "pass": (
            bool(fc_rows)
            and bool(selective_only_rows)
            and selective_only_p95 > 0
            and selective_only_service_cost > 0
            and selective_p95 <= selective_only_p95 * delay_limit
            and selective_service_cost
            <= selective_only_service_cost * delay_limit
        ),
        "diagnostic_only": True,
        "flowcache_p95_cache_delay_ms": selective_p95,
        "selective_migration_only_p95_cache_delay_ms": (
            selective_only_p95
        ),
        "flowcache_modeled_service_cost_ms": selective_service_cost,
        "selective_migration_only_modeled_service_cost_ms": (
            selective_only_service_cost
        ),
        "maximum_relative_increase": args.max_modeled_delay_increase,
    }

    oracle_rows = grouped.get("oracle_cost", [])
    oracle_elapsed = (
        as_float(oracle_rows[0].get("elapsed_s")) if oracle_rows else 0.0
    )
    flowcache_elapsed = as_float(fc_first.get("elapsed_s"))
    always_elapsed = as_float(always_first.get("elapsed_s"))
    selective_only_elapsed = as_float(
        selective_only_first.get("elapsed_s")
    )
    replay_ratio = (
        flowcache_elapsed / oracle_elapsed if oracle_elapsed > 0 else math.inf
    )
    always_replay_ratio = (
        always_elapsed / oracle_elapsed if oracle_elapsed > 0 else math.inf
    )
    selective_only_replay_ratio = (
        selective_only_elapsed / oracle_elapsed
        if oracle_elapsed > 0 else math.inf
    )
    checks["replay_complexity"] = {
        "pass": (
            bool(fc_rows)
            and bool(always_rows)
            and bool(selective_only_rows)
            and bool(oracle_rows)
            and replay_ratio <= args.max_replay_ratio
            and always_replay_ratio <= args.max_replay_ratio
            and selective_only_replay_ratio <= args.max_replay_ratio
        ),
        "flowcache_elapsed_s": flowcache_elapsed,
        "always_elapsed_s": always_elapsed,
        "selective_migration_only_elapsed_s": selective_only_elapsed,
        "oracle_elapsed_s": oracle_elapsed,
        "selective_ratio": replay_ratio,
        "always_ratio": always_replay_ratio,
        "selective_migration_only_ratio": selective_only_replay_ratio,
        "engineering_threshold": args.max_replay_ratio,
    }

    passed = all(check["pass"] for check in checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "G3-P1 selective-migration engineering check; not scientific GO/NO-GO",
        "checks": checks,
        "next_if_pass": (
            "Run the full single-cell trace, add fair two-tier LRU/GDSF/oracle, "
            "then measure closed-loop serving on the main cell."
        ),
        "next_if_fail": (
            "Fix correctness/complexity failures first. If only selection "
            "rate or movement reduction fails, tune admission parameters "
            "on a task-grouped validation split, freeze them, and rerun test."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
