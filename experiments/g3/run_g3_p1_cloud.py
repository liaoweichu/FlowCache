"""Cloud runner for the causal G3-P1 validation/held-out protocol.

Stages are deliberately separated so validation selection can be inspected
before the held-out test is executed:

  prepare -> tune -> test

``all`` is available for a fresh run directory, but ``test`` refuses to
overwrite an existing held-out result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Run: python3 -m pip install -r "
        ".uploads/experiments/g3/requirements-cloud.txt"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
G1PRIME_DIR = SCRIPT_DIR.parent / "g1prime"
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PREFIXES = (
    G1PRIME_DIR / "physical_traces" / "request_prefixes.jsonl"
)
DEFAULT_TRACE = (
    G1PRIME_DIR / "physical_traces" / "access_trace_c4.jsonl"
)
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
REQUIRED_PROTOCOL_BASELINES = {
    "gdsf",
    "sizecost",
    "flowcache_always_migrate",
    "flowcache_selective_migrate_only",
    "oracle_cost",
    "flowcache_lossless",
}


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "tune", "test", "all"],
        required=True,
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--request-prefixes", type=Path, default=DEFAULT_PREFIXES)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=100000)
    parser.add_argument(
        "--force-rebuild-trace",
        action="store_true",
        help="Explicitly replace the shared c=4 trace and its manifest.",
    )
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="Continue when the conservative in-memory replay estimate is high.",
    )
    parser.add_argument("--skip-regression-tests", action="store_true")
    return parser.parse_args()


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
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
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def run_command(
    label: str,
    command: Iterable[str],
    run_dir: Path,
) -> None:
    command = [str(item) for item in command]
    log_path = run_dir / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{label}] {' '.join(command)}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(
            f"stage {label} failed with exit code {return_code}; "
            f"see {log_path}"
        )


def available_memory_bytes() -> Optional[int]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise SystemExit(
                f"cannot find an existing parent for path: {path}"
            )
        candidate = parent
    return candidate


def estimate_expanded_trace_bytes(
    request_prefixes: Path,
    sample_events: int = 200,
) -> int:
    """Estimate direct c=4 JSONL size from an input prefix sample."""
    sys.path.insert(0, str(G1PRIME_DIR))
    from build_physical_access_trace import build_access_record

    sampled_input = 0
    sampled_output = 0
    access_idx = 0
    sampled = 0
    with open(request_prefixes, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            sampled_input += len(line)
            for block in event.get("blocks", []) or []:
                record = build_access_record(access_idx, block, event)
                sampled_output += len(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ) + 1
                access_idx += 1
            sampled += 1
            if sampled >= sample_events:
                break
    if sampled_input <= 0:
        raise SystemExit("request_prefixes sample contains no data")
    ratio = sampled_output / sampled_input
    return int(request_prefixes.stat().st_size * ratio)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"config is not a YAML mapping: {path}")
    return config


def preflight(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9 or newer is required")
    required = [
        args.base_config,
        SCRIPT_DIR / "cost-model.json",
        SCRIPT_DIR / "run_g3_grid.py",
        SCRIPT_DIR / "tune_selective_migration.py",
        SCRIPT_DIR / "freeze_selected_config.py",
        G1PRIME_DIR / "build_concurrent_access_trace.py",
    ]
    if args.stage in {"prepare", "all"}:
        required.append(args.request_prefixes)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required file(s): " + ", ".join(missing))

    config = load_config(args.base_config)
    protocol = config.get("protocol_test") or {}
    cell = protocol.get("cell") or {}
    if int(cell.get("concurrency", -1)) != 4:
        raise SystemExit("protocol_test cell must use concurrency=4")
    found = set(protocol.get("baselines") or [])
    missing_baselines = REQUIRED_PROTOCOL_BASELINES - found
    if missing_baselines:
        raise SystemExit(
            "protocol config is missing baseline(s): "
            + ", ".join(sorted(missing_baselines))
        )

    trace_estimate = None
    free_disk = shutil.disk_usage(
        nearest_existing_parent(args.trace.parent)
    ).free
    memory_available = available_memory_bytes()
    memory_estimate = None
    if args.request_prefixes.is_file():
        trace_estimate = estimate_expanded_trace_bytes(
            args.request_prefixes
        )
        # Parsed JSON dict/list objects are substantially larger than JSONL.
        # This is intentionally conservative; the user can explicitly
        # override after checking the cloud host.
        memory_estimate = trace_estimate * 6
        required_disk = int(trace_estimate * 1.25)
        if not args.trace.exists() and free_disk < required_disk:
            raise SystemExit(
                "insufficient free disk for estimated trace: "
                f"need about {human_bytes(required_disk)}, "
                f"have {human_bytes(free_disk)}"
            )
        if (
            memory_available is not None
            and memory_available < memory_estimate
            and not args.allow_low_memory
        ):
            raise SystemExit(
                "conservative replay-memory estimate exceeds MemAvailable: "
                f"estimate {human_bytes(memory_estimate)}, "
                f"available {human_bytes(memory_available)}. "
                "Choose a larger-memory server or pass --allow-low-memory "
                "after monitoring a smoke run."
            )

    report = {
        "status": "PASS",
        "python": sys.version,
        "project_root": str(PROJECT_ROOT),
        "request_prefixes": str(args.request_prefixes.resolve()),
        "request_prefixes_bytes": (
            args.request_prefixes.stat().st_size
            if args.request_prefixes.is_file()
            else None
        ),
        "trace": str(args.trace.resolve()),
        "trace_exists": args.trace.is_file(),
        "estimated_trace_bytes": trace_estimate,
        "free_disk_bytes": free_disk,
        "estimated_replay_memory_bytes": memory_estimate,
        "available_memory_bytes": memory_available,
        "protocol_cell": cell,
        "protocol_baselines": sorted(found),
    }
    atomic_json(run_dir / "preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def validate_trace_manifest(trace: Path) -> Dict[str, Any]:
    manifest_path = Path(f"{trace}.manifest.json")
    if not trace.is_file():
        raise SystemExit(f"trace does not exist: {trace}")
    if not manifest_path.is_file():
        raise SystemExit(
            f"trace manifest does not exist: {manifest_path}; "
            "rebuild with build_concurrent_access_trace.py"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "flowcache.concurrent-access-trace.v1":
        raise SystemExit("unexpected trace manifest schema")
    if int(manifest.get("concurrency", -1)) != 4:
        raise SystemExit("trace manifest concurrency is not 4")
    if int(manifest.get("output_bytes", -1)) != trace.stat().st_size:
        raise SystemExit("trace size does not match its manifest")
    return manifest


def write_runtime_config(
    base_config: Path,
    trace: Path,
    run_dir: Path,
) -> Path:
    config = load_config(base_config)
    config.setdefault("trace_source", {})["access_trace_dir"] = str(
        trace.resolve().parent
    )
    config["cloud_run"] = {
        "schema": "flowcache.g3-p1-cloud-run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_config.resolve()),
        "base_config_sha256": sha256_file(base_config),
        "trace": str(trace.resolve()),
        "trace_manifest": str(Path(f"{trace.resolve()}.manifest.json")),
    }
    output = run_dir / "runtime-config.yaml"
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return output


def prepare(args: argparse.Namespace, run_dir: Path) -> Path:
    preflight(args, run_dir)
    manifest_path = Path(f"{args.trace}.manifest.json")
    if args.force_rebuild_trace:
        command = [
            sys.executable,
            G1PRIME_DIR / "build_concurrent_access_trace.py",
            "--input",
            args.request_prefixes,
            "--output",
            args.trace,
            "--concurrency",
            "4",
            "--manifest",
            manifest_path,
            "--overwrite",
        ]
        run_command("build-trace", command, run_dir)
    elif not args.trace.exists():
        command = [
            sys.executable,
            G1PRIME_DIR / "build_concurrent_access_trace.py",
            "--input",
            args.request_prefixes,
            "--output",
            args.trace,
            "--concurrency",
            "4",
            "--manifest",
            manifest_path,
        ]
        run_command("build-trace", command, run_dir)
    manifest = validate_trace_manifest(args.trace)
    atomic_json(run_dir / "trace-manifest-copy.json", manifest)

    if not args.skip_regression_tests:
        run_command(
            "regression-tests",
            [
                sys.executable,
                SCRIPT_DIR / "tests" / "test_protocol_repair.py",
            ],
            run_dir,
        )
        run_command(
            "hot-cold-mechanism",
            [
                sys.executable,
                SCRIPT_DIR / "benchmark_protocol.py",
                "--pattern",
                "hot-cold",
                "--accesses",
                "20000",
                "--gpu-blocks",
                "1",
                "--cpu-blocks",
                "100",
                "--policy",
                "all",
            ],
            run_dir,
        )
    return write_runtime_config(
        args.base_config, args.trace, run_dir
    )


def tune(args: argparse.Namespace, run_dir: Path) -> Path:
    validate_trace_manifest(args.trace)
    runtime_config = run_dir / "runtime-config.yaml"
    if not runtime_config.is_file():
        runtime_config = write_runtime_config(
            args.base_config, args.trace, run_dir
        )
    tuning_dir = run_dir / "tuning"
    selection = tuning_dir / "selection.json"
    if selection.exists():
        raise SystemExit(
            f"refusing to overwrite an existing validation selection: "
            f"{selection}"
        )
    run_command(
        "validation-tuning",
        [
            sys.executable,
            SCRIPT_DIR / "tune_selective_migration.py",
            "--config",
            runtime_config,
            "--validation-fraction",
            str(args.validation_fraction),
            "--split-seed",
            str(args.split_seed),
            "--max-episodes",
            str(args.max_episodes),
            "--output-dir",
            tuning_dir,
        ],
        run_dir,
    )
    with open(selection, "r", encoding="utf-8") as handle:
        selection_report = json.load(handle)
    selection_status = selection_report.get("status")
    if selection_status != "SELECTED":
        raise SystemExit(
            f"validation ended with {selection_status}; frozen config and "
            f"held-out test remain locked. See {selection}"
        )
    frozen = run_dir / "frozen-config.yaml"
    run_command(
        "freeze-selection",
        [
            sys.executable,
            SCRIPT_DIR / "freeze_selected_config.py",
            "--base-config",
            runtime_config,
            "--selection",
            selection,
            "--output",
            frozen,
            "--expected-split-seed",
            str(args.split_seed),
            "--expected-validation-fraction",
            str(args.validation_fraction),
        ],
        run_dir,
    )
    return frozen


def test(args: argparse.Namespace, run_dir: Path) -> Path:
    frozen = (
        args.frozen_config
        if args.frozen_config is not None
        else run_dir / "frozen-config.yaml"
    )
    if not frozen.is_file():
        raise SystemExit(
            f"frozen config not found: {frozen}; run --stage tune first"
        )
    results = run_dir / "g3-p1-held-out-test.csv"
    checker_report = run_dir / "g3-p1-check.json"
    if results.exists() or checker_report.exists():
        raise SystemExit(
            "held-out output already exists; refusing a second test run in "
            f"the same run directory: {run_dir}"
        )
    run_command(
        "held-out-test",
        [
            sys.executable,
            SCRIPT_DIR / "run_g3_grid.py",
            "--config",
            frozen,
            "--protocol-test",
            "--task-split",
            "test",
            "--validation-fraction",
            str(args.validation_fraction),
            "--split-seed",
            str(args.split_seed),
            "--max-episodes",
            str(args.max_episodes),
            "--output",
            results,
        ],
        run_dir,
    )
    run_command(
        "protocol-check",
        [
            sys.executable,
            SCRIPT_DIR / "check_protocol_results.py",
            results,
            "--output",
            checker_report,
            "--expected-task-split",
            "test",
        ],
        run_dir,
    )
    return results


def main() -> int:
    args = parse_args()
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (SCRIPT_DIR / "results" / f"cloud-{utc_run_id()}").resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if args.stage == "prepare":
        prepare(args, run_dir)
    elif args.stage == "tune":
        preflight(args, run_dir)
        tune(args, run_dir)
    elif args.stage == "test":
        preflight(args, run_dir)
        test(args, run_dir)
    else:
        prepare(args, run_dir)
        tune(args, run_dir)
        test(args, run_dir)

    state = {
        "status": "COMPLETED",
        "stage": args.stage,
        "run_dir": str(run_dir),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Open-loop G3-P1 diagnostics only; PASS is not a scientific "
            "G3 GO decision."
        ),
    }
    atomic_json(run_dir / f"{args.stage}-completed.json", state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
