"""Build a block-level concurrent trace directly from request prefixes.

This is the cloud-oriented replacement for the two-step
``build_physical_access_trace.py`` + ``simulate_concurrency.py`` path.
It preserves the same scheduling semantics while avoiding the large
intermediate ``access_trace.jsonl`` and never holds expanded block records in
memory.

Only one compact metadata entry per request is retained. During the second
pass, request events are fetched by byte offset and expanded directly into
``access_trace_c{N}.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from build_physical_access_trace import build_access_record
from simulate_concurrency import (
    compute_avg_active_workflows,
    interleave_requests,
    schedule_workflows,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR / "physical_traces" / "request_prefixes.jsonl"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "physical_traces" / "access_trace_c4.jsonl"
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directly build an interleaved block trace from "
            "request_prefixes.jsonl with O(number-of-requests) memory."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Smoke-only cap; unset means the complete input.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Default: <output>.manifest.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output and manifest.",
    )
    return parser.parse_args(argv)


def _decode_event(line: bytes, line_number: int) -> Dict[str, Any]:
    try:
        event = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed JSON at input line {line_number}: {exc}"
        ) from exc
    if not isinstance(event, dict):
        raise ValueError(
            f"input line {line_number} is not a JSON object"
        )
    if not event.get("request_id"):
        raise ValueError(
            f"input line {line_number} has no non-empty request_id"
        )
    if not event.get("workflow_id"):
        raise ValueError(
            f"input line {line_number} has no non-empty workflow_id"
        )
    blocks = event.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(
            f"input line {line_number} has no blocks list"
        )
    return event


def index_request_events(
    input_path: Path,
    max_requests: Optional[int] = None,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[str],
    Dict[str, List[Dict[str, Any]]],
    str,
    int,
    int,
]:
    """Index request byte offsets and build scheduling metadata."""
    requests: Dict[str, Dict[str, Any]] = {}
    request_order: List[str] = []
    input_hash = hashlib.sha256()
    processed_bytes = 0
    event_count = 0
    total_blocks_indexed = 0

    with open(input_path, "rb") as source:
        line_number = 0
        while True:
            if (
                max_requests is not None
                and max_requests > 0
                and event_count >= max_requests
            ):
                break
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue

            event = _decode_event(line, line_number)
            input_hash.update(line)
            processed_bytes += len(line)
            event_count += 1

            request_id = str(event["request_id"])
            workflow_id = str(event["workflow_id"])
            num_blocks = len(event["blocks"])
            total_blocks_indexed += num_blocks
            entry = requests.get(request_id)
            if entry is None:
                entry = {
                    "request_id": request_id,
                    "workflow_id": workflow_id,
                    "arrival_time_ms": float(
                        event.get("arrival_time_ms", 0.0) or 0.0
                    ),
                    "step_id": event.get("step_id", 0),
                    "num_blocks": 0,
                    "offsets": [],
                }
                requests[request_id] = entry
                request_order.append(request_id)
            elif entry["workflow_id"] != workflow_id:
                raise ValueError(
                    f"request_id {request_id!r} appears in multiple workflows"
                )
            entry["offsets"].append(offset)
            entry["num_blocks"] += num_blocks

            if event_count % 1000 == 0:
                print(
                    f"    indexed {event_count:,} request events, "
                    f"{total_blocks_indexed:,} "
                    "block accesses"
                )

    by_workflow: Dict[str, List[Dict[str, Any]]] = {}
    for request_id in request_order:
        request = requests[request_id]
        by_workflow.setdefault(
            request["workflow_id"], []
        ).append(request)
    for workflow_requests in by_workflow.values():
        workflow_requests.sort(
            key=lambda item: (
                item["arrival_time_ms"],
                item["step_id"],
            )
        )

    if not requests:
        raise ValueError("input contains no valid request events")
    return (
        requests,
        request_order,
        by_workflow,
        input_hash.hexdigest(),
        processed_bytes,
        event_count,
    )


def write_interleaved_trace(
    input_path: Path,
    temp_output: Path,
    requests: Dict[str, Dict[str, Any]],
    interleaved: List[Tuple[float, str, Dict[str, Any]]],
) -> Tuple[int, str, int]:
    """Expand indexed events in scheduled order into the final JSONL."""
    output_hash = hashlib.sha256()
    output_bytes = 0
    access_idx = 0

    with open(input_path, "rb") as source, open(
        temp_output, "wb"
    ) as output:
        for request_position, (
            _global_time,
            _workflow_id,
            request,
        ) in enumerate(interleaved, start=1):
            for offset in requests[request["request_id"]]["offsets"]:
                source.seek(offset)
                line = source.readline()
                event = _decode_event(line, request_position)
                if str(event["request_id"]) != request["request_id"]:
                    raise ValueError(
                        "input changed while building trace: request offset "
                        f"for {request['request_id']!r} no longer matches"
                    )
                for block in event["blocks"]:
                    record = build_access_record(
                        access_idx, block, event
                    )
                    encoded = (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    output.write(encoded)
                    output_hash.update(encoded)
                    output_bytes += len(encoded)
                    access_idx += 1

            if request_position % 1000 == 0:
                print(
                    f"    wrote {request_position:,} requests, "
                    f"{access_idx:,} block accesses"
                )

        output.flush()
        os.fsync(output.fileno())
    return access_idx, output_hash.hexdigest(), output_bytes


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
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


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    output = args.output.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else Path(f"{output}.manifest.json")
    )
    if not args.overwrite:
        existing = [
            str(path) for path in (output, manifest) if path.exists()
        ]
        if existing:
            raise SystemExit(
                "refusing to overwrite existing artifact(s): "
                + ", ".join(existing)
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 78)
    print("Direct Concurrent Access Trace Builder")
    print(f"Input       : {args.input.resolve()}")
    print(f"Output      : {output}")
    print(f"Concurrency : {args.concurrency}")
    print("=" * 78)

    (
        requests,
        request_order,
        by_workflow,
        input_sha256,
        processed_input_bytes,
        request_event_count,
    ) = index_request_events(args.input, args.max_requests)
    starts, durations = schedule_workflows(
        by_workflow, args.concurrency
    )
    interleaved = interleave_requests(by_workflow, starts)
    avg_active = compute_avg_active_workflows(
        interleaved, starts, durations
    )

    temp_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_output = Path(temp_handle.name)
    temp_handle.close()
    try:
        (
            access_count,
            output_sha256,
            output_bytes,
        ) = write_interleaved_trace(
            args.input, temp_output, requests, interleaved
        )
        os.replace(temp_output, output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise

    elapsed = time.time() - started
    payload = {
        "schema": "flowcache.concurrent-access-trace.v1",
        "builder": Path(__file__).name,
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "processed_input_bytes": processed_input_bytes,
        "source_file_bytes": args.input.stat().st_size,
        "output": str(output),
        "output_sha256": output_sha256,
        "output_bytes": output_bytes,
        "concurrency": args.concurrency,
        "request_events": request_event_count,
        "unique_requests": len(requests),
        "workflows": len(by_workflow),
        "block_accesses": access_count,
        "average_active_workflows": avg_active,
        "max_requests": args.max_requests,
        "elapsed_seconds": elapsed,
    }
    _atomic_json(manifest, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Manifest    : {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
