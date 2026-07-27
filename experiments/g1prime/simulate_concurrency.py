"""
G1′ Concurrency Simulator
==========================
Synthesizes globally-interleaved access traces at different concurrency
levels (c = 1, 4, 8) from the block-level access trace produced by
``build_physical_access_trace.py``.

Input:  ``experiments/g1prime/physical_traces/access_trace.jsonl``
        (one JSON object per block access)

Output: ``experiments/g1prime/physical_traces/access_trace_c{c}.jsonl``
        - ``access_trace_c1.jsonl``  — sequential baseline (≡ G1)
        - ``access_trace_c4.jsonl``  — 4 concurrent workflows
        - ``access_trace_c8.jsonl``  — 8 concurrent workflows

Concurrency model
-----------------
1. Group all requests by ``workflow_id``.
2. Within each workflow, sort requests by ``arrival_time_ms`` (relative to
   the workflow's own timeline).
3. Workflow duration = ``max(arrival_time_ms)`` across its requests.
4. **c = 1**: workflows execute strictly sequentially (one finishes before
   the next starts). Equivalent to G1's sequential replay.
5. **c > 1**: at most ``c`` workflows are simultaneously active. A workflow
   starts when a slot becomes free (i.e., a previous workflow's last
   request has been issued). Within the active set, requests are
   interleaved by ``workflow_global_start + arrival_time_ms``.
6. Tool-wait inactive prefixes (between requests in a workflow) compete
   with other active workflows' prefixes for cache — this is captured by
   the workflow's active interval ``[start, start + duration]``.

The output is the access_trace re-ordered by the synthesized global time,
with ``access_idx`` re-numbered. All original fields are preserved.

Memory note
-----------
Re-ordering inherently requires random access, so this script loads all
block accesses into memory (grouped by ``request_id`` then
``workflow_id``). Use ``--max-requests N`` for small-sample testing.

Read-only w.r.t. ``experiments/e1/``.

Usage
-----
    python experiments/g1prime/simulate_concurrency.py --concurrency 1
    python experiments/g1prime/simulate_concurrency.py --concurrency 4
    python experiments/g1prime/simulate_concurrency.py --concurrency 8
    python experiments/g1prime/simulate_concurrency.py --concurrency 1 \\
        --max-requests 100
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH: Path = SCRIPT_DIR / "physical_traces" / "access_trace.jsonl"
DEFAULT_OUTPUT_DIR: Path = SCRIPT_DIR / "physical_traces"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "G1′ concurrency simulator: synthesize interleaved access "
            "traces at concurrency levels c ∈ {1, 4, 8}."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input access_trace.jsonl (default: {DEFAULT_INPUT_PATH}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Default: "
            "<input_dir>/access_trace_c{concurrency}.jsonl."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrency level c (1=sequential, 4/8=concurrent). Default: 1.",
    )
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Process only the first N request events (0 or unset = all).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_access_trace(
    input_path: Path,
    max_requests: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Load the access trace and group block accesses by ``request_id``.

    Reads the jsonl file line by line; for each block access record,
    appends it to ``requests[request_id]``. The first-seen order of
    ``request_id`` values is preserved in ``request_order``.

    Args:
        input_path: Path to ``access_trace.jsonl``.
        max_requests: If set and > 0, keep only the first N requests
            (by first appearance in the file). Other requests' blocks
            are discarded.

    Returns:
        ``(requests, request_order)`` where:
        - ``requests``: ``{request_id: [block_access_record, ...]}``
          with each list preserving original block_idx order.
        - ``request_order``: ``[request_id, ...]`` in first-seen order.
    """
    requests: Dict[str, List[Dict[str, Any]]] = {}
    request_order: List[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stderr.write(f"WARNING: skip malformed line ({exc})\n")
                continue

            request_id = rec.get("request_id", "")
            if request_id not in requests:
                requests[request_id] = []
                request_order.append(request_id)
            requests[request_id].append(rec)

    if max_requests is not None and max_requests > 0 and len(request_order) > max_requests:
        for rid in request_order[max_requests:]:
            del requests[rid]
        request_order = request_order[:max_requests]

    return requests, request_order


# ---------------------------------------------------------------------------
# Workflow grouping
# ---------------------------------------------------------------------------
def group_requests_by_workflow(
    requests: Dict[str, List[Dict[str, Any]]],
    request_order: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group request-level info dicts by ``workflow_id``.

    Each request info dict contains:
        ``request_id``, ``workflow_id``, ``arrival_time_ms``,
        ``step_id``, ``num_blocks``.

    Within each workflow, requests are sorted by
    ``(arrival_time_ms, step_id)``.
    """
    by_workflow: Dict[str, List[Dict[str, Any]]] = {}
    for req_id in request_order:
        blocks = requests.get(req_id, [])
        if not blocks:
            continue
        first = blocks[0]
        wf_id = first.get("workflow_id", "")
        req_info = {
            "request_id": req_id,
            "workflow_id": wf_id,
            "arrival_time_ms": float(first.get("arrival_time_ms", 0.0)),
            "step_id": first.get("step_id", 0),
            "num_blocks": len(blocks),
        }
        by_workflow.setdefault(wf_id, []).append(req_info)

    # Sort within each workflow by (arrival_time, step_id) for determinism.
    for wf_id, reqs in by_workflow.items():
        reqs.sort(key=lambda r: (r["arrival_time_ms"], r["step_id"]))

    return by_workflow


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def schedule_workflows(
    requests_by_workflow: Dict[str, List[Dict[str, Any]]],
    c: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Schedule workflows under concurrency level ``c``.

    Args:
        requests_by_workflow: ``{workflow_id: [request_info, ...]}``.
        c: Concurrency level (max simultaneously active workflows).
            ``c <= 1`` is treated as fully sequential.

    Returns:
        ``(workflow_global_start, workflow_durations)`` where:
        - ``workflow_global_start``: ``{workflow_id: global_start_time}``.
        - ``workflow_durations``: ``{workflow_id: duration}`` with
          ``duration = max(arrival_time_ms)`` across the workflow's requests.
    """
    # Deterministic scheduling order: alphabetical by workflow_id.
    workflow_ids = sorted(requests_by_workflow.keys())

    workflow_durations: Dict[str, float] = {}
    for wf_id in workflow_ids:
        reqs = requests_by_workflow[wf_id]
        if not reqs:
            workflow_durations[wf_id] = 0.0
            continue
        workflow_durations[wf_id] = max(r["arrival_time_ms"] for r in reqs)

    workflow_global_start: Dict[str, float] = {}

    if c <= 1:
        # Sequential: each workflow starts when the previous one ends.
        current_time = 0.0
        for wf_id in workflow_ids:
            workflow_global_start[wf_id] = current_time
            current_time += workflow_durations[wf_id]
    else:
        # Concurrent: min-heap of (slot_free_time, slot_idx).
        # Each workflow is assigned to the slot that becomes free earliest.
        slots: List[Tuple[float, int]] = [(0.0, i) for i in range(c)]
        heapq.heapify(slots)
        for wf_id in workflow_ids:
            slot_free_time, slot_idx = heapq.heappop(slots)
            start_time = slot_free_time
            workflow_global_start[wf_id] = start_time
            new_end = start_time + workflow_durations[wf_id]
            heapq.heappush(slots, (new_end, slot_idx))

    return workflow_global_start, workflow_durations


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------
def interleave_requests(
    requests_by_workflow: Dict[str, List[Dict[str, Any]]],
    workflow_global_start: Dict[str, float],
) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Interleave all requests by synthesized global time.

    Each request's global time = ``workflow_global_start[wf] +
    arrival_time_ms``. The result is sorted by
    ``(global_time, workflow_id, step_id)`` for deterministic ordering
    of simultaneous events.
    """
    interleaved: List[Tuple[float, str, Dict[str, Any]]] = []
    for wf_id, reqs in requests_by_workflow.items():
        wf_start = workflow_global_start.get(wf_id, 0.0)
        for req in reqs:
            global_time = wf_start + req["arrival_time_ms"]
            interleaved.append((global_time, wf_id, req))

    interleaved.sort(key=lambda x: (x[0], x[1], x[2]["step_id"]))
    return interleaved


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_avg_active_workflows(
    interleaved: List[Tuple[float, str, Dict[str, Any]]],
    workflow_global_start: Dict[str, float],
    workflow_durations: Dict[str, float],
) -> float:
    """Compute the average number of active workflows per block access.

    A workflow is "active" at time ``t`` if
    ``workflow_global_start <= t <= workflow_global_start + duration``.

    For efficiency, we sort all workflow start times and end times, then
    for each request's global time ``t`` compute::

        active_count = (# starts ≤ t) − (# ends < t)

    Each request contributes ``num_blocks`` block accesses at its global
    time, so the average is block-access-weighted.
    """
    if not interleaved:
        return 0.0

    starts = sorted(workflow_global_start.values())
    ends = sorted(
        workflow_global_start[wf] + workflow_durations[wf]
        for wf in workflow_global_start
    )

    total_active = 0
    total_blocks = 0
    for global_time, _wf_id, req_info in interleaved:
        n_blocks = req_info["num_blocks"]
        n_started = bisect.bisect_right(starts, global_time)
        n_ended = bisect.bisect_left(ends, global_time)
        active = n_started - n_ended
        total_active += active * n_blocks
        total_blocks += n_blocks

    return total_active / total_blocks if total_blocks > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if args.concurrency < 1:
        sys.stderr.write(
            f"ERROR: --concurrency must be ≥ 1 (got {args.concurrency}).\n"
        )
        return 1

    # Resolve output path
    if args.output is not None:
        output_path: Path = args.output
    else:
        output_path = (
            args.input.parent / f"access_trace_c{args.concurrency}.jsonl"
        )

    print("=" * 78)
    print("G1′ Concurrency Simulator")
    print("=" * 78)
    print(f"Input        : {args.input}")
    print(f"Output       : {output_path}")
    print(f"Concurrency  : c={args.concurrency}")
    if args.max_requests is not None and args.max_requests > 0:
        print(f"Max requests : {args.max_requests}")
    print()

    if not args.input.exists():
        sys.stderr.write(
            f"ERROR: input file not found: {args.input}\n"
            f"       Run build_physical_access_trace.py first to generate "
            f"access_trace.jsonl.\n"
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load access trace, group by request_id ----
    print("[1] Loading access trace ...")
    t_start = time.time()
    requests_by_id, request_order = load_access_trace(
        args.input, args.max_requests
    )
    print(
        f"    Loaded {len(request_order)} requests "
        f"in {time.time() - t_start:.1f}s."
    )

    # ---- Step 2: group requests by workflow_id ----
    print("[2] Grouping requests by workflow_id ...")
    requests_by_workflow = group_requests_by_workflow(
        requests_by_id, request_order
    )
    print(f"    {len(requests_by_workflow)} workflows.")

    # ---- Step 3: schedule workflows ----
    print(f"[3] Scheduling workflows (c={args.concurrency}) ...")
    workflow_global_start, workflow_durations = schedule_workflows(
        requests_by_workflow, args.concurrency
    )
    if workflow_global_start:
        max_end = max(
            workflow_global_start[wf] + workflow_durations[wf]
            for wf in workflow_global_start
        )
        print(f"    global makespan ≈ {max_end:.1f} ms")

    # ---- Step 4: interleave requests by global time ----
    print("[4] Interleaving requests by global time ...")
    interleaved = interleave_requests(requests_by_workflow, workflow_global_start)
    print(f"    {len(interleaved)} requests in interleaved order.")

    # ---- Step 5: write re-ordered block accesses ----
    print("[5] Writing re-ordered access trace ...")
    access_idx = 0
    total_blocks = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for _global_time, _wf_id, req_info in interleaved:
            req_id = req_info["request_id"]
            for block_access in requests_by_id[req_id]:
                new_rec = dict(block_access)
                new_rec["access_idx"] = access_idx
                out_f.write(
                    json.dumps(new_rec, ensure_ascii=False, separators=(",", ":"))
                )
                out_f.write("\n")
                access_idx += 1
                total_blocks += 1

    # ---- Step 6: statistics ----
    print("[6] Computing statistics ...")
    avg_active = compute_avg_active_workflows(
        interleaved, workflow_global_start, workflow_durations
    )

    elapsed = time.time() - t_start
    print()
    print("[Done] Summary")
    print(f"    concurrency             : c={args.concurrency}")
    print(f"    total workflows         : {len(requests_by_workflow)}")
    print(f"    total requests          : {len(interleaved)}")
    print(f"    total block accesses    : {total_blocks}")
    print(f"    avg active workflows    : {avg_active:.3f}")
    print(f"    output file             : {output_path}")
    print(f"    elapsed                 : {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
