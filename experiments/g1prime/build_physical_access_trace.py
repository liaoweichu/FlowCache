"""
G1′ Physical Access Trace Builder
==================================
Expands request-level prefix events into block-level access records.

Input:  ``experiments/g1prime/physical_traces/request_prefixes.jsonl``
        (one JSON object per request event, each containing a ``blocks`` list)

Output: ``experiments/g1prime/physical_traces/access_trace.jsonl``
        (one JSON object per block access, in request-then-block_idx order)

Each request event is expanded into N block access records (one per prefix
block), preserving ``block_idx`` order. A global ``access_idx`` counter
increments across the entire stream.

Memory model: line-by-line streaming read/write — the full input file is
NOT loaded into memory. Streaming statistics (unique block hashes,
episode-internal revisit ratio) are accumulated using bounded in-memory
``set`` / ``dict`` collections keyed by ``block_hash`` and
``(workflow_id, block_hash)`` respectively.

Read-only w.r.t. ``experiments/e1/``.

Usage:
    python experiments/g1prime/build_physical_access_trace.py
    python experiments/g1prime/build_physical_access_trace.py --max-requests 100
    python experiments/g1prime/build_physical_access_trace.py \\
        --input <path> --output <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH: Path = SCRIPT_DIR / "physical_traces" / "request_prefixes.jsonl"
DEFAULT_OUTPUT_PATH: Path = SCRIPT_DIR / "physical_traces" / "access_trace.jsonl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "G1′ physical access trace builder: expand request events to "
            "block-level access records."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input request_prefixes.jsonl (default: {DEFAULT_INPUT_PATH}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output access_trace.jsonl (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Process only the first N request events (0 or unset = all).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-block access record construction
# ---------------------------------------------------------------------------
def build_access_record(
    access_idx: int,
    block: Dict[str, Any],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single block-access record from a request event + block.

    Args:
        access_idx: Global incrementing access index.
        block: Block dict with ``block_idx``, ``token_range_start``,
            ``token_range_end``, ``block_hash``, ``parent_hash``.
        event: Request event with ``request_id``, ``workflow_id``,
            ``task_id``, ``seed``, ``domain``, ``step_id``,
            ``arrival_time_ms``, ``prefill_ms``, ``num_prefix_tokens``.

    Returns:
        Block access record with the G1′ canonical field set.
    """
    token_start = int(block.get("token_range_start", 0))
    token_end = int(block.get("token_range_end", 0))
    return {
        "access_idx": access_idx,
        "block_hash": block.get("block_hash", ""),
        "parent_hash": block.get("parent_hash", ""),
        "request_id": event.get("request_id", ""),
        "workflow_id": event.get("workflow_id", ""),
        "task_id": event.get("task_id", ""),
        "seed": event.get("seed"),
        "domain": event.get("domain", ""),
        "step_id": event.get("step_id", 0),
        "block_idx": block.get("block_idx", 0),
        "arrival_time_ms": event.get("arrival_time_ms", 0.0),
        "prefill_ms": event.get("prefill_ms", 0.0),
        "num_prefix_tokens": event.get("num_prefix_tokens", 0),
        "block_token_count": token_end - token_start,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("G1′ Physical Access Trace Builder")
    print("=" * 78)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    if args.max_requests is not None and args.max_requests > 0:
        print(f"Max    : first {args.max_requests} requests")
    print()

    if not args.input.exists():
        sys.stderr.write(
            f"ERROR: input file not found: {args.input}\n"
            f"       Run recompile_prefixes.py first to generate "
            f"request_prefixes.jsonl.\n"
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Streaming statistics accumulators ----
    total_requests = 0
    total_block_accesses = 0
    unique_block_hashes: Set[str] = set()
    # (workflow_id, block_hash) -> access count within that workflow.
    # Used for episode-internal prefix revisit-ratio computation.
    workflow_block_counts: Dict[Tuple[str, str], int] = {}

    t_start = time.time()
    access_idx = 0

    with open(args.input, "r", encoding="utf-8") as in_f, \
         open(args.output, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stderr.write(f"WARNING: skip malformed line ({exc})\n")
                continue

            # Honor --max-requests BEFORE expanding this request's blocks.
            if (
                args.max_requests is not None
                and args.max_requests > 0
                and total_requests >= args.max_requests
            ):
                break

            workflow_id = event.get("workflow_id", "")
            blocks = event.get("blocks", []) or []

            for block in blocks:
                rec = build_access_record(access_idx, block, event)
                out_f.write(
                    json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
                )
                out_f.write("\n")

                block_hash = rec["block_hash"]
                access_idx += 1
                total_block_accesses += 1
                unique_block_hashes.add(block_hash)
                key = (workflow_id, block_hash)
                workflow_block_counts[key] = workflow_block_counts.get(key, 0) + 1

            total_requests += 1

            if total_requests % 1000 == 0:
                elapsed = time.time() - t_start
                print(
                    f"    [{total_requests} requests] "
                    f"blocks={total_block_accesses} elapsed={elapsed:.1f}s"
                )

    # ---- Episode-internal prefix revisit ratio ----
    # For each (workflow_id, block_hash) accessed cnt times within a workflow:
    #   the first access is "new"; the remaining (cnt - 1) are "revisits".
    # revisit_ratio = sum(cnt - 1 for cnt > 1) / total_block_accesses
    total_revisits = sum(
        cnt - 1 for cnt in workflow_block_counts.values() if cnt > 1
    )
    revisit_ratio = (
        total_revisits / total_block_accesses
        if total_block_accesses > 0
        else 0.0
    )

    elapsed = time.time() - t_start
    print()
    print("[Done] Summary")
    print(f"    total requests            : {total_requests}")
    print(f"    total block accesses      : {total_block_accesses}")
    print(f"    unique block hashes       : {len(unique_block_hashes)}")
    print(
        f"    episode-internal revisits : {total_revisits}  "
        f"({revisit_ratio * 100:.2f}% of block accesses)"
    )
    print(f"    output file               : {args.output}")
    print(f"    elapsed                   : {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
