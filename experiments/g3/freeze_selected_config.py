"""Freeze validation-selected G3-P1 parameters into a new config.

The source config is never modified. The script accepts only the
pre-registered parameter names and verifies that the selection came from the
task-grouped validation split before producing a held-out-test config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ALLOWED_PARAMETERS = (
    "minimum_net_benefit_ms",
    "cpu_admission_margin_ms",
    "gpu_admission_margin_ms",
    "gpu_admission_cold_start_cost_ratio",
    "expected_cpu_residence_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split-seed", type=int, default=42)
    parser.add_argument(
        "--expected-validation-fraction", type=float, default=0.2
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing frozen config.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selected {name} is not numeric: {value!r}") from exc


def validate_selection(
    report: Dict[str, Any],
    expected_seed: int,
    expected_fraction: float,
) -> Dict[str, Any]:
    if report.get("status") != "SELECTED":
        raise ValueError(
            "selection status is not SELECTED; held-out test is forbidden"
        )
    split = report.get("split") or {}
    if split.get("unit") != "task_id":
        raise ValueError("selection split unit must be task_id")
    if split.get("partition") != "validation":
        raise ValueError("selection must come from validation")
    if int(split.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"selection seed is {split.get('seed')}, expected {expected_seed}"
        )
    fraction = _as_float(split.get("fraction"), "validation fraction")
    if abs(fraction - expected_fraction) > 1e-12:
        raise ValueError(
            f"validation fraction is {fraction}, "
            f"expected {expected_fraction}"
        )

    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection has no selected candidate object")
    missing = [key for key in ALLOWED_PARAMETERS if key not in selected]
    if missing:
        raise ValueError(
            "selected candidate is missing frozen parameter(s): "
            + ", ".join(missing)
        )

    frozen = {key: selected[key] for key in ALLOWED_PARAMETERS}
    frozen["minimum_net_benefit_ms"] = _as_float(
        frozen["minimum_net_benefit_ms"],
        "minimum_net_benefit_ms",
    )
    frozen["cpu_admission_margin_ms"] = _as_float(
        frozen["cpu_admission_margin_ms"],
        "cpu_admission_margin_ms",
    )
    frozen["gpu_admission_margin_ms"] = _as_float(
        frozen["gpu_admission_margin_ms"],
        "gpu_admission_margin_ms",
    )
    ratio = _as_float(
        frozen["gpu_admission_cold_start_cost_ratio"],
        "gpu_admission_cold_start_cost_ratio",
    )
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "gpu_admission_cold_start_cost_ratio must be in [0, 1]"
        )
    frozen["gpu_admission_cold_start_cost_ratio"] = ratio
    residence = int(frozen["expected_cpu_residence_steps"])
    if residence < 0:
        raise ValueError(
            "expected_cpu_residence_steps must be nonnegative"
        )
    frozen["expected_cpu_residence_steps"] = residence
    return frozen


def atomic_dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            yaml.safe_dump(
                payload,
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    for path in (args.base_config, args.selection):
        if not path.is_file():
            raise SystemExit(f"required file not found: {path}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite frozen config: {args.output}"
        )

    with open(args.base_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.selection, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("base config is not a YAML mapping")

    try:
        parameters = validate_selection(
            report,
            args.expected_split_seed,
            args.expected_validation_fraction,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selection_config = config.setdefault(
        "flowcache", {}
    ).setdefault("selective_migration", {})
    selection_config.update(parameters)
    config["frozen_selection"] = {
        "schema": "flowcache.g3-p1-frozen-selection.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(args.base_config.resolve()),
        "base_config_sha256": sha256_file(args.base_config),
        "selection_report": str(args.selection.resolve()),
        "selection_report_sha256": sha256_file(args.selection),
        "candidate_id": report["selected"].get("candidate_id"),
        "split": report["split"],
        "parameters": parameters,
        "held_out_test_policy": "run exactly once; no test-time tuning",
    }
    atomic_dump_yaml(args.output, config)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(args.output.resolve()),
                "parameters": parameters,
                "candidate_id": report["selected"].get("candidate_id"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
