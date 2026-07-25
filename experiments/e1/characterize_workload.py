"""
E1 Workload Characterization Script
Reads recorded trajectories and computes E1 metrics.
"""

import json
import sys
import os
import math
from pathlib import Path
from typing import List, Dict, Optional, Set
import statistics


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from trace_utils import load_all_trajectories


# ---------------------------------------------------------------------------
# Constants for KV memory estimation (Qwen2.5-7B-Instruct)
# ---------------------------------------------------------------------------
# Qwen2.5-7B-Instruct architecture
NUM_LAYERS = 28
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4   # GQA: 4 KV heads
HEAD_DIM = 128
BYTES_PER_ELEMENT = 2  # BF16
TOTAL_VRAM_GB = 24.0

# Per-token KV bytes = 2 (K+V) * num_layers * num_kv_heads * head_dim * bytes_per_element
PER_TOKEN_KV_BYTES = (
    2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_ELEMENT
)


# ---------------------------------------------------------------------------
# Metric 1: Workflow structure statistics
# ---------------------------------------------------------------------------

def compute_workflow_structure(trajectories: List[Dict]) -> Dict:
    """
    Compute per-workflow structural metrics and aggregate statistics.

    For each workflow:
      - length: total number of steps
      - depth: number of tool-call/response cycles
      - width: maximum number of tool calls in a single step (branching factor)
      - branch_rate: depth / length
      - tool_wait_duration: average decode_ms across assistant steps that
        include a tool call (proxy for tool-result wait time)

    Returns per-workflow values and summary statistics (mean, median, p95, p99).
    """
    per_wf: List[Dict] = []

    for traj in trajectories:
        meta = traj.get("meta", {})
        steps = traj.get("steps", [])
        wf_id = meta.get("workflow_id", "unknown")
        num_steps = len(steps)

        # depth: count steps that have a tool_call (assistant turns with tool use)
        tool_call_steps = [s for s in steps if s.get("tool_call") is not None]
        depth = len(tool_call_steps)

        # width: max number of tool calls in a single step (currently flat, so 0 or 1)
        width = 0
        for s in steps:
            tc = s.get("tool_call")
            if tc is not None:
                width = max(width, 1)  # each step has at most one tool_call in flat loop

        # branch_rate
        branch_rate = depth / num_steps if num_steps > 0 else 0.0

        # tool_wait_duration: average decode_ms across assistant+tool_call steps
        decode_times = []
        for s in tool_call_steps:
            if s.get("decode_ms", 0) > 0:
                decode_times.append(s["decode_ms"])
        avg_wait = statistics.mean(decode_times) if decode_times else 0.0

        per_wf.append({
            "workflow_id": wf_id,
            "length": num_steps,
            "depth": depth,
            "width": width,
            "branch_rate": round(branch_rate, 4),
            "tool_wait_duration_ms": round(avg_wait, 2),
        })

    # Aggregate statistics
    def _dist(values, name):
        if not values:
            return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        sv = sorted(values)
        n = len(sv)
        return {
            "mean": round(statistics.mean(sv), 2),
            "median": round(statistics.median(sv), 2),
            "p95": round(_percentile(sv, 95), 2),
            "p99": round(_percentile(sv, 99), 2),
            "min": sv[0],
            "max": sv[-1],
        }

    return {
        "per_workflow": per_wf,
        "summary": {
            "length": _dist([w["length"] for w in per_wf], "length"),
            "depth": _dist([w["depth"] for w in per_wf], "depth"),
            "width": _dist([w["width"] for w in per_wf], "width"),
            "branch_rate": _dist([w["branch_rate"] for w in per_wf], "branch_rate"),
            "tool_wait_duration_ms": _dist(
                [w["tool_wait_duration_ms"] for w in per_wf],
                "tool_wait_duration_ms",
            ),
        },
        "num_workflows": len(per_wf),
    }


# ---------------------------------------------------------------------------
# Metric 2: Exact-prefix overlap
# ---------------------------------------------------------------------------

def compute_exact_prefix_overlap(trajectories: List[Dict]) -> Dict:
    """
    Compute exact-prefix overlap and LCP statistics.

    overlap_ratio: fraction of total tokens that belong to blocks shared by
                   >= 2 workflows (via global_block_index.workflow_ids).
    LCP: For each pair of workflows, compute the longest common prefix of
         their ordered block_hash sequences. Report distribution in tokens.
    """
    if not trajectories:
        return {"overlap_ratio": 0.0, "lcp_tokens": {}}

    block_size = trajectories[0].get("meta", {}).get("block_size", 16)

    # --- Merge global_block_index across all trajectories ---
    merged_index: Dict[str, Dict] = {}
    for traj in trajectories:
        gbi = traj.get("global_block_index", {})
        for bhash, info in gbi.items():
            if bhash not in merged_index:
                merged_index[bhash] = dict(info)
            else:
                # Merge workflow_ids lists
                existing_ids = set(merged_index[bhash].get("workflow_ids", []))
                new_ids = set(info.get("workflow_ids", []))
                merged_index[bhash]["workflow_ids"] = sorted(existing_ids | new_ids)

    # --- overlap_ratio ---
    # Each block covers up to block_size tokens; for a partial last block,
    # the actual token range length = token_end - token_start.
    total_tokens = 0
    shared_tokens = 0
    for bhash, info in merged_index.items():
        wf_count = len(info.get("workflow_ids", []))
        block_len = info.get("token_end", 0) - info.get("token_start", 0)
        if block_len <= 0:
            block_len = block_size  # fallback
        total_tokens += block_len
        if wf_count >= 2:
            shared_tokens += block_len

    overlap_ratio = shared_tokens / total_tokens if total_tokens > 0 else 0.0

    # --- Per-workflow ordered block_hash sequences ---
    wf_block_seqs: Dict[str, List[str]] = {}
    for traj in trajectories:
        wf_id = traj.get("meta", {}).get("workflow_id", "unknown")
        seq = []
        for step in traj.get("steps", []):
            for ba in step.get("block_assignments", []):
                seq.append(ba.get("block_hash", ""))
        wf_block_seqs[wf_id] = seq

    # --- LCP for each pair ---
    wf_ids = list(wf_block_seqs.keys())
    lcp_tokens_list = []
    for i in range(len(wf_ids)):
        for j in range(i + 1, len(wf_ids)):
            seq_a = wf_block_seqs[wf_ids[i]]
            seq_b = wf_block_seqs[wf_ids[j]]
            lcp_blocks = 0
            min_len = min(len(seq_a), len(seq_b))
            for k in range(min_len):
                if seq_a[k] == seq_b[k]:
                    lcp_blocks += 1
                else:
                    break
            lcp_tokens_list.append(lcp_blocks * block_size)

    return {
        "overlap_ratio": round(overlap_ratio, 4),
        "total_tokens": total_tokens,
        "shared_tokens": shared_tokens,
        "total_unique_blocks": len(merged_index),
        "shared_blocks": sum(
            1 for info in merged_index.values()
            if len(info.get("workflow_ids", [])) >= 2
        ),
        "lcp_tokens": _distribution_stats(lcp_tokens_list),
        "num_workflow_pairs": len(lcp_tokens_list),
    }


# ---------------------------------------------------------------------------
# Metric 3: Next-use distance distribution
# ---------------------------------------------------------------------------

def compute_next_use_distance(trajectories: List[Dict]) -> Dict:
    """
    Compute next-use distance distribution across unique blocks.

    For each unique block_hash (from merged global_block_index), record the
    global step ordinal of its first appearance and its next appearance.
    Distance = next_use_step - first_use_step.
    Blocks accessed only once have infinite distance (excluded from stats).
    """
    if not trajectories:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0,
                "singleton_blocks": 0, "multi_use_blocks": 0}

    block_size = trajectories[0].get("meta", {}).get("block_size", 16)

    # Build a global step sequence: for each trajectory in order, for each step
    # in order, collect the block_hashes seen at that global step.
    # global_step_id -> set of block_hashes
    global_step_blocks: Dict[int, Set[str]] = {}
    global_step = 0
    for traj in trajectories:
        for step in traj.get("steps", []):
            block_hashes = set()
            for ba in step.get("block_assignments", []):
                bh = ba.get("block_hash", "")
                if bh:
                    block_hashes.add(bh)
            if block_hashes:
                global_step_blocks[global_step] = block_hashes
            global_step += 1

    # For each unique block_hash, find first and second appearance global steps
    block_first_seen: Dict[str, int] = {}
    block_second_seen: Dict[str, Optional[int]] = {}

    for gs in sorted(global_step_blocks.keys()):
        for bh in global_step_blocks[gs]:
            if bh not in block_first_seen:
                block_first_seen[bh] = gs
            elif bh not in block_second_seen or block_second_seen.get(bh) is None:
                block_second_seen[bh] = gs

    # Compute distances
    distances = []
    singleton_count = 0
    multi_count = 0
    for bh in block_first_seen:
        if bh in block_second_seen and block_second_seen[bh] is not None:
            dist = block_second_seen[bh] - block_first_seen[bh]
            distances.append(dist)
            multi_count += 1
        else:
            singleton_count += 1

    stats = _distribution_stats(distances)
    stats["singleton_blocks"] = singleton_count
    stats["multi_use_blocks"] = multi_count
    stats["total_unique_blocks"] = singleton_count + multi_count

    return stats


# ---------------------------------------------------------------------------
# Metric 4: Block working-set size and KV/VRAM ratio
# ---------------------------------------------------------------------------

def compute_working_set(trajectories: List[Dict], block_size: int = 16) -> Dict:
    """
    Compute block working-set size and KV/VRAM ratio.

    Simulates sequential workflow execution. For each workflow, tracks the
    set of unique block_hashes encountered (the active working set).
    Peak working set = max active_blocks across all checkpoints.

    KV memory estimate:
      per_block_bytes = block_size * PER_TOKEN_KV_BYTES
      KV_VRAM_GB = working_set_size * per_block_bytes / (1024^3)
      kv_vram_ratio = KV_VRAM_GB / TOTAL_VRAM_GB
    """
    if not trajectories:
        return {
            "working_set_size": 0,
            "kv_memory_gb": 0.0,
            "kv_vram_ratio": 0.0,
            "per_workflow_peak": [],
        }

    per_block_bytes = block_size * PER_TOKEN_KV_BYTES
    peak_overall = 0
    per_wf_peaks = []

    for traj in trajectories:
        wf_id = traj.get("meta", {}).get("workflow_id", "unknown")
        active_blocks: Set[str] = set()
        peak_this_wf = 0
        for step in traj.get("steps", []):
            for ba in step.get("block_assignments", []):
                bh = ba.get("block_hash", "")
                if bh:
                    active_blocks.add(bh)
            peak_this_wf = max(peak_this_wf, len(active_blocks))
        peak_overall = max(peak_overall, peak_this_wf)
        per_wf_peaks.append({
            "workflow_id": wf_id,
            "peak_active_blocks": peak_this_wf,
        })

    kv_memory_gb = (peak_overall * per_block_bytes) / (1024 ** 3)
    kv_vram_ratio = kv_memory_gb / TOTAL_VRAM_GB

    return {
        "working_set_size": peak_overall,
        "per_block_kv_bytes": per_block_bytes,
        "per_block_kv_mb": round(per_block_bytes / (1024 ** 2), 3),
        "kv_memory_gb": round(kv_memory_gb, 4),
        "kv_vram_ratio": round(kv_vram_ratio, 4),
        "total_vram_gb": TOTAL_VRAM_GB,
        "per_workflow_peak": per_wf_peaks,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values, pct: float) -> float:
    """Compute percentile from sorted list using linear interpolation."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _distribution_stats(values: List[float]) -> Dict:
    """Compute mean, median, p95, p99, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0,
                "min": 0, "max": 0, "count": 0}
    sv = sorted(values)
    return {
        "mean": round(statistics.mean(sv), 2),
        "median": round(statistics.median(sv), 2),
        "p95": round(_percentile(sv, 95), 2),
        "p99": round(_percentile(sv, 99), 2),
        "min": sv[0],
        "max": sv[-1],
        "count": len(sv),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown_report(results: Dict, f):
    """Write a human-readable Markdown report to file handle f."""
    f.write("# E1 Workload Characterization Report\n\n")

    # 1. Workflow Structure
    ws = results.get("workflow_structure", {})
    f.write("## 1. Workflow Structure\n\n")
    f.write(f"**Number of workflows**: {ws.get('num_workflows', 0)}\n\n")
    summary = ws.get("summary", {})
    if summary:
        f.write("| Metric | Mean | Median | P95 | P99 | Min | Max |\n")
        f.write("|--------|------|--------|-----|-----|-----|-----|\n")
        for metric_name in ["length", "depth", "width", "branch_rate", "tool_wait_duration_ms"]:
            d = summary.get(metric_name, {})
            f.write(
                f"| {metric_name} | {d.get('mean', '-')} | {d.get('median', '-')} | "
                f"{d.get('p95', '-')} | {d.get('p99', '-')} | {d.get('min', '-')} | "
                f"{d.get('max', '-')} |\n"
            )

    # Per-workflow detail
    per_wf = ws.get("per_workflow", [])
    if per_wf:
        f.write("\n### Per-Workflow\n\n")
        f.write("| workflow_id | length | depth | width | branch_rate | tool_wait_duration_ms |\n")
        f.write("|-------------|--------|-------|-------|-------------|-----------------------|\n")
        for w in per_wf:
            f.write(
                f"| {w['workflow_id']} | {w['length']} | {w['depth']} | {w['width']} | "
                f"{w['branch_rate']} | {w['tool_wait_duration_ms']} |\n"
            )

    # 2. Exact-Prefix Overlap
    epo = results.get("exact_prefix_overlap", {})
    f.write("\n## 2. Exact-Prefix Overlap\n\n")
    f.write(f"- **Overlap Ratio**: {epo.get('overlap_ratio', '-')}\n")
    f.write(f"- **Total Tokens**: {epo.get('total_tokens', '-')}\n")
    f.write(f"- **Shared Tokens**: {epo.get('shared_tokens', '-')}\n")
    f.write(f"- **Total Unique Blocks**: {epo.get('total_unique_blocks', '-')}\n")
    f.write(f"- **Shared Blocks (>=2 workflows)**: {epo.get('shared_blocks', '-')}\n")
    f.write(f"- **Workflow Pairs (for LCP)**: {epo.get('num_workflow_pairs', '-')}\n\n")

    lcp = epo.get("lcp_tokens", {})
    if lcp:
        f.write("### LCP Token Distribution\n\n")
        f.write("| Statistic | Value (tokens) |\n")
        f.write("|-----------|----------------|\n")
        f.write(f"| Mean | {lcp.get('mean', '-')} |\n")
        f.write(f"| Median | {lcp.get('median', '-')} |\n")
        f.write(f"| P95 | {lcp.get('p95', '-')} |\n")
        f.write(f"| P99 | {lcp.get('p99', '-')} |\n")
        f.write(f"| Min | {lcp.get('min', '-')} |\n")
        f.write(f"| Max | {lcp.get('max', '-')} |\n")
        f.write(f"| Count | {lcp.get('count', '-')} |\n")

    # 3. Next-Use Distance
    nud = results.get("next_use_distance", {})
    f.write("\n## 3. Next-Use Distance Distribution\n\n")
    f.write("| Statistic | Value (global steps) |\n")
    f.write("|-----------|----------------------|\n")
    f.write(f"| Mean | {nud.get('mean', '-')} |\n")
    f.write(f"| Median | {nud.get('median', '-')} |\n")
    f.write(f"| P95 | {nud.get('p95', '-')} |\n")
    f.write(f"| P99 | {nud.get('p99', '-')} |\n")
    f.write(f"| Max | {nud.get('max', '-')} |\n")
    f.write(f"| Count (multi-use blocks) | {nud.get('count', '-')} |\n")
    f.write(f"| Singleton Blocks | {nud.get('singleton_blocks', '-')} |\n")
    f.write(f"| Multi-Use Blocks | {nud.get('multi_use_blocks', '-')} |\n")
    f.write(f"| Total Unique Blocks | {nud.get('total_unique_blocks', '-')} |\n")

    # 4. Working Set
    ws2 = results.get("working_set", {})
    f.write("\n## 4. Block Working-Set Size and KV/VRAM Ratio\n\n")
    f.write(f"- **Model**: Qwen3-8B (36 layers, 32 Q heads, 8 KV heads, head_dim=128, BF16)\n")
    f.write(f"- **Per-Block KV**: {ws2.get('per_block_kv_mb', '-')} MB (block_size=16 tokens)\n")
    f.write(f"- **Working Set Size (peak)**: {ws2.get('working_set_size', '-')} blocks\n")
    f.write(f"- **KV Memory Estimate**: {ws2.get('kv_memory_gb', '-')} GB\n")
    f.write(f"- **KV/VRAM Ratio**: {ws2.get('kv_vram_ratio', '-')} ({ws2.get('total_vram_gb', 24)} GB VRAM)\n\n")

    per_wf_peak = ws2.get("per_workflow_peak", [])
    if per_wf_peak:
        f.write("### Per-Workflow Peak Active Blocks\n\n")
        f.write("| workflow_id | peak_active_blocks |\n")
        f.write("|-------------|--------------------|\n")
        for pw in per_wf_peak:
            f.write(f"| {pw['workflow_id']} | {pw['peak_active_blocks']} |\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trace_dir = SCRIPT_DIR / "traces" / "bf16"
    output_dir = SCRIPT_DIR / "outputs"

    trajectories = load_all_trajectories(str(trace_dir))
    if not trajectories:
        print(f"[E1] No trajectories found in {trace_dir}.")
        print("     Run record_trajectories.py first to generate trace data.")
        return

    print(f"[E1] Loaded {len(trajectories)} trajectories from {trace_dir}")

    results = {}
    results["workflow_structure"] = compute_workflow_structure(trajectories)
    results["exact_prefix_overlap"] = compute_exact_prefix_overlap(trajectories)
    results["next_use_distance"] = compute_next_use_distance(trajectories)
    results["working_set"] = compute_working_set(trajectories)

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e1-characterization.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[E1] JSON report saved to {json_path}")

    # Save Markdown
    md_path = output_dir / "e1-report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        _write_markdown_report(results, f)
    print(f"[E1] Markdown report saved to {md_path}")

    # Quick summary to stdout
    ws_summary = results["workflow_structure"].get("summary", {})
    epo = results["exact_prefix_overlap"]
    nud = results["next_use_distance"]
    ws2 = results["working_set"]

    print(f"\n{'='*60}")
    print("E1 Workload Characterization Summary")
    print(f"{'='*60}")
    print(f"Workflows:          {results['workflow_structure'].get('num_workflows', 0)}")
    print(f"Avg steps:          {ws_summary.get('length', {}).get('mean', '-')}")
    print(f"Avg tool calls:     {ws_summary.get('depth', {}).get('mean', '-')}")
    print(f"Overlap ratio:      {epo.get('overlap_ratio', '-')}")
    print(f"LCP median (tokens):{epo.get('lcp_tokens', {}).get('median', '-')}")
    print(f"Next-use median:    {nud.get('median', '-')} steps")
    print(f"Peak working set:   {ws2.get('working_set_size', '-')} blocks")
    print(f"KV memory:          {ws2.get('kv_memory_gb', '-')} GB")
    print(f"KV/VRAM ratio:      {ws2.get('kv_vram_ratio', '-')}")
    print(f"{'='*60}")

    print(f"\nE1 characterization complete. Reports in {output_dir}/")


if __name__ == "__main__":
    main()
