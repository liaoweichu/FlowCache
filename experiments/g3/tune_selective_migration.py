"""Validation-only parameter sweep for G3-P1 selective migration.

The script always uses a deterministic task-grouped validation split.  It
never mutates ``config.yaml`` and never evaluates the held-out test split.
The selected candidate must be copied into the frozen config explicitly
before a separate test run.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import yaml

from run_g3_grid import run_grid, save_results


SCRIPT_DIR = Path(__file__).resolve().parent
FLOWCACHE = "flowcache_lossless"
ALWAYS = "flowcache_always_migrate"
SELECTIVE_ONLY = "flowcache_selective_migrate_only"
ORACLE = "oracle_cost"


def parse_number_grid(text, cast):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise ValueError("parameter grid cannot be empty")
    return values


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_candidate(rows, constraints):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("baseline", ""), []).append(row)

    fc_rows = grouped.get(FLOWCACHE, [])
    always_rows = grouped.get(ALWAYS, [])
    selective_only_rows = grouped.get(SELECTIVE_ONLY, [])
    oracle_rows = grouped.get(ORACLE, [])
    if (
        not fc_rows
        or not always_rows
        or not selective_only_rows
        or not oracle_rows
    ):
        return {
            "valid": False,
            "failure_reasons": (
                "missing full/selective-only/always/oracle rows"
            ),
        }

    fc = fc_rows[0]
    always = always_rows[0]
    selective_only = selective_only_rows[0]
    oracle = oracle_rows[0]
    candidates = as_float(fc.get("migration_candidate_count"))
    selected = as_float(fc.get("migration_selected_count"))
    always_migrations = as_float(always.get("migrate_count"))
    selection_rate = selected / candidates if candidates else 0.0
    fc_migrations = as_float(fc.get("migrate_count"))
    movement_reduction = (
        (always_migrations - fc_migrations) / always_migrations
        if always_migrations else 0.0
    )
    gpu_admission_candidates = as_float(
        fc.get("gpu_admission_candidate_count")
    )
    gpu_admission_bypassed = as_float(
        fc.get("gpu_admission_bypassed_count")
    )
    gpu_bypass_rate = (
        gpu_admission_bypassed / gpu_admission_candidates
        if gpu_admission_candidates else 0.0
    )
    fc_transfer_ms = as_float(fc.get("transfer_ms_total"))
    selective_only_transfer_ms = as_float(
        selective_only.get("transfer_ms_total")
    )
    gpu_bypass_transfer_reduction = (
        (selective_only_transfer_ms - fc_transfer_ms)
        / selective_only_transfer_ms
        if selective_only_transfer_ms > 0 else 0.0
    )
    fc_p95 = as_float(fc.get("global_p95_cache_delay_ms"))
    always_p95 = as_float(always.get("global_p95_cache_delay_ms"))
    fc_service = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in fc_rows
    )
    always_service = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in always_rows
    )
    selective_only_p95 = as_float(
        selective_only.get("global_p95_cache_delay_ms")
    )
    selective_only_service = sum(
        as_float(row.get("task_modeled_service_cost_ms"))
        for row in selective_only_rows
    )
    oracle_elapsed = as_float(oracle.get("elapsed_s"))
    replay_ratio = (
        as_float(fc.get("elapsed_s")) / oracle_elapsed
        if oracle_elapsed > 0 else float("inf")
    )

    failures = []
    if not (
        constraints["min_selection_rate"]
        <= selection_rate
        <= constraints["max_selection_rate"]
    ):
        failures.append("selection_rate")
    if movement_reduction < constraints["min_movement_reduction"]:
        failures.append("movement_reduction")
    if not (
        constraints["min_gpu_bypass_rate"]
        <= gpu_bypass_rate
        <= constraints["max_gpu_bypass_rate"]
    ):
        failures.append("gpu_bypass_rate")
    if (
        gpu_bypass_transfer_reduction
        < constraints["min_gpu_bypass_transfer_reduction"]
    ):
        failures.append("gpu_bypass_transfer_reduction")
    delay_limit = 1.0 + constraints["max_modeled_delay_increase"]
    if always_p95 <= 0 or fc_p95 > always_p95 * delay_limit:
        failures.append("p95_cache_delay")
    if always_service <= 0 or fc_service > always_service * delay_limit:
        failures.append("modeled_service_cost")
    if (
        selective_only_p95 <= 0
        or fc_p95 > selective_only_p95 * delay_limit
    ):
        failures.append("p95_vs_selective_migration_only")
    if (
        selective_only_service <= 0
        or fc_service > selective_only_service * delay_limit
    ):
        failures.append("service_vs_selective_migration_only")
    if replay_ratio > constraints["max_replay_ratio"]:
        failures.append("replay_complexity")
    if as_float(fc.get("fallback_count")) != 0:
        failures.append("fallback")
    if as_float(fc.get("negative_cost_count")) != 0:
        failures.append("negative_cost")

    return {
        "valid": not failures,
        "failure_reasons": ",".join(failures),
        "selection_rate": selection_rate,
        "movement_reduction": movement_reduction,
        "selective_migrations": fc_migrations,
        "always_migrations": always_migrations,
        "gpu_bypass_rate": gpu_bypass_rate,
        "gpu_bypass_transfer_reduction": (
            gpu_bypass_transfer_reduction
        ),
        "flowcache_transfer_ms": fc_transfer_ms,
        "selective_migration_only_transfer_ms": (
            selective_only_transfer_ms
        ),
        "selective_p95_cache_delay_ms": fc_p95,
        "always_p95_cache_delay_ms": always_p95,
        "selective_modeled_service_cost_ms": fc_service,
        "always_modeled_service_cost_ms": always_service,
        "selective_migration_only_p95_cache_delay_ms": (
            selective_only_p95
        ),
        "selective_migration_only_modeled_service_cost_ms": (
            selective_only_service
        ),
        "replay_ratio_vs_oracle": replay_ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune G3-P1 admission on a task-grouped validation split"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--minimum-net-benefit-ms", default="0,0.5,1.0"
    )
    parser.add_argument(
        "--cpu-admission-margin-ms", default="0,0.5"
    )
    parser.add_argument(
        "--gpu-admission-margin-ms", default="0"
    )
    parser.add_argument(
        "--gpu-admission-cold-start-cost-ratio", default="0.5"
    )
    parser.add_argument(
        "--expected-cpu-residence-steps", default="100"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=100000)
    parser.add_argument(
        "--output-dir", default="results/selective-tuning"
    )
    parser.add_argument("--min-selection-rate", type=float, default=0.01)
    parser.add_argument("--max-selection-rate", type=float, default=0.99)
    parser.add_argument("--min-gpu-bypass-rate", type=float, default=0.0)
    parser.add_argument("--max-gpu-bypass-rate", type=float, default=0.99)
    parser.add_argument(
        "--min-gpu-bypass-transfer-reduction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-movement-reduction", type=float, default=0.10
    )
    parser.add_argument(
        "--max-modeled-delay-increase", type=float, default=0.05
    )
    parser.add_argument("--max-replay-ratio", type=float, default=3.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    net_grid = parse_number_grid(args.minimum_net_benefit_ms, float)
    margin_grid = parse_number_grid(args.cpu_admission_margin_ms, float)
    gpu_margin_grid = parse_number_grid(
        args.gpu_admission_margin_ms, float
    )
    cold_start_cost_ratio_grid = parse_number_grid(
        args.gpu_admission_cold_start_cost_ratio, float
    )
    residence_grid = parse_number_grid(
        args.expected_cpu_residence_steps, int
    )
    constraints = {
        "min_selection_rate": args.min_selection_rate,
        "max_selection_rate": args.max_selection_rate,
        "min_gpu_bypass_rate": args.min_gpu_bypass_rate,
        "max_gpu_bypass_rate": args.max_gpu_bypass_rate,
        "min_gpu_bypass_transfer_reduction": (
            args.min_gpu_bypass_transfer_reduction
        ),
        "min_movement_reduction": args.min_movement_reduction,
        "max_modeled_delay_increase": args.max_modeled_delay_increase,
        "max_replay_ratio": args.max_replay_ratio,
    }

    summaries = []
    candidate_index = 0
    protocol_baselines = list(
        base_config.get("protocol_test", {}).get("baselines", [])
    )
    invariant_baselines = [
        baseline
        for baseline in protocol_baselines
        if baseline not in {FLOWCACHE, SELECTIVE_ONLY}
    ]
    invariant_config = copy.deepcopy(base_config)
    invariant_config.setdefault("protocol_test", {})["baselines"] = (
        invariant_baselines
    )
    common_run_kwargs = {
        "protocol_test": True,
        "max_episodes": args.max_episodes,
        "task_split": "validation",
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        # The trace and its causal annotations are immutable during a sweep.
        # Keep one parsed copy instead of reparsing the full JSONL 73 times.
        "access_cache": {},
    }
    # GDSF/SizeCost/Always-Migrate/Oracle do not depend on any swept
    # parameter. Replaying them for every candidate would multiply the
    # validation runtime without changing a single decision.
    invariant_rows = run_grid(invariant_config, **common_run_kwargs)
    if not invariant_rows:
        report = {
            "status": "NO_VALIDATION_DATA",
            "scientific_result": False,
            "split": {
                "unit": "task_id",
                "partition": "validation",
                "fraction": args.validation_fraction,
                "seed": args.split_seed,
            },
            "constraints": constraints,
            "selected": None,
            "n_candidates": 0,
            "n_valid": 0,
            "baseline_replays": len(invariant_baselines),
            "baseline_replays_without_reuse": 0,
            "summary_csv": None,
            "next": (
                "The episode cap produced an empty validation partition. "
                "Increase --max-episodes or use the complete trace; do not "
                "inspect the held-out test."
            ),
        }
        report_path = output_dir / "selection.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    selective_only_cache = {}

    for net_benefit in net_grid:
        for margin in margin_grid:
            for gpu_margin in gpu_margin_grid:
                for cold_start_cost_ratio in cold_start_cost_ratio_grid:
                    for residence_steps in residence_grid:
                        candidate_index += 1
                        config = copy.deepcopy(base_config)
                        selection = config.setdefault(
                            "flowcache", {}
                        ).setdefault("selective_migration", {})
                        selection["minimum_net_benefit_ms"] = net_benefit
                        selection["cpu_admission_margin_ms"] = margin
                        selection["gpu_admission_margin_ms"] = gpu_margin
                        selection[
                            "gpu_admission_cold_start_cost_ratio"
                        ] = (
                            cold_start_cost_ratio
                        )
                        selection["expected_cpu_residence_steps"] = (
                            residence_steps
                        )

                        cpu_parameter_key = (
                            net_benefit,
                            margin,
                            residence_steps,
                        )
                        if cpu_parameter_key not in selective_only_cache:
                            selective_config = copy.deepcopy(config)
                            selective_config.setdefault(
                                "protocol_test", {}
                            )["baselines"] = [SELECTIVE_ONLY]
                            selective_only_cache[cpu_parameter_key] = run_grid(
                                selective_config, **common_run_kwargs
                            )

                        full_config = copy.deepcopy(config)
                        full_config.setdefault(
                            "protocol_test", {}
                        )["baselines"] = [FLOWCACHE]
                        full_rows = run_grid(
                            full_config, **common_run_kwargs
                        )
                        rows = (
                            invariant_rows
                            + selective_only_cache[cpu_parameter_key]
                            + full_rows
                        )
                        candidate_id = f"candidate-{candidate_index:03d}"
                        raw_path = output_dir / f"{candidate_id}.csv"
                        save_results(rows, raw_path)
                        summary = summarize_candidate(rows, constraints)
                        summary.update(
                            {
                                "candidate_id": candidate_id,
                                "minimum_net_benefit_ms": net_benefit,
                                "cpu_admission_margin_ms": margin,
                                "gpu_admission_margin_ms": gpu_margin,
                                "gpu_admission_cold_start_cost_ratio": (
                                    cold_start_cost_ratio
                                ),
                                "expected_cpu_residence_steps": (
                                    residence_steps
                                ),
                                "raw_results": str(raw_path),
                            }
                        )
                        summaries.append(summary)

    valid = [summary for summary in summaries if summary["valid"]]
    selected = (
        min(
            valid,
            key=lambda item: (
                item["selective_modeled_service_cost_ms"],
                item["selective_p95_cache_delay_ms"],
                -item["movement_reduction"],
            ),
        )
        if valid else None
    )

    summary_csv = output_dir / "summary.csv"
    fieldnames = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "status": "SELECTED" if selected else "NO_VALID_CONFIG",
        "scientific_result": False,
        "split": {
            "unit": "task_id",
            "partition": "validation",
            "fraction": args.validation_fraction,
            "seed": args.split_seed,
        },
        "constraints": constraints,
        "selected": selected,
        "n_candidates": len(summaries),
        "n_valid": len(valid),
        "baseline_replays": (
            len(invariant_baselines)
            + len(selective_only_cache)
            + len(summaries)
        ),
        "baseline_replays_without_reuse": (
            len(protocol_baselines) * len(summaries)
        ),
        "summary_csv": str(summary_csv),
        "next": (
            "Copy only the selected admission parameters into the frozen "
            "config, then run --task-split test exactly once."
            if selected
            else "Do not inspect test. Revisit the proxy/action model or "
            "declare selective migration unsupported by this workload."
        ),
    }
    report_path = output_dir / "selection.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
