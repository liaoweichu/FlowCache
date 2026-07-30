"""§4.2 Closed-loop serving CLI runner.

Runs the 3-strategy closed-loop serving experiment:
  1. apc_lru           — vLLM native prefix caching (GPU-only LRU)
  2. twotier_lru       — SimpleCPUOffloadConnector (always migrate, control)
  3. flowcache_lossless — FlowCacheConnector (selective migrate, treatment)

Main cell: 2 GiB KV, c=4, Qwen2.5-7B-Instruct BF16.

G3 pass criteria (closed-loop):
  - p95 TTFT improvement ≥ 15% (flowcache vs twotier_lru)
  - Throughput drop ≤ 5% (flowcache vs twotier_lru)
  - Bootstrap CI excludes 0 (flowcache vs twotier_lru, task-cluster p95 diff)

Quick pilot mode (--quick-pilot):
  - 8 tasks (airline/retail × 4 each), 2 workflows per task, ~294 requests
  - Fixed in-flight submission (complete one → submit one)
  - Outputs quick-pilot-selection.json
  - Verdict = PILOT_PASS / PILOT_FAIL / INCOMPLETE (not a formal GO/NO-GO)

Note: no_cache (lower bound) was trimmed — verdict only requires
flowcache_lossless vs twotier_lru, and apc_lru is kept for narrative.

Usage:
  # Full run (all 3 strategies, all episodes)
  python run_closed_loop.py \\
    --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master \\
    --trace-dir experiments/e1/traces/bf16/tau_bench \\
    --output-dir results/closed-loop

  # Quick pilot (8 tasks, 16 workflows, ~294 requests)
  python run_closed_loop.py \\
    --model ... --trace-dir ... \\
    --quick-pilot \\
    --kv-cache-memory-gib 2.0 \\
    --cpu-capacity-gib 2.0 \\
    --gpu-memory-utilization 0.76 \\
    --max-model-len 16384 \\
    -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent dirs to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
G3_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(G3_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from serving_harness import (
    ServingHarness,
    Strategy,
    StrategyMetrics,
    TraceLoader,
    MetricsCollector,
)

logger = logging.getLogger("closed_loop")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="§4.2 Closed-loop serving experiment runner.",
    )
    p.add_argument(
        "--model", required=True,
        help="Path to Qwen2.5-7B-Instruct model.",
    )
    p.add_argument(
        "--trace-dir", required=True, type=Path,
        help="Directory of e1 τ-bench trace JSON files.",
    )
    p.add_argument(
        "--output-dir", default=Path("results/closed-loop"), type=Path,
        help="Output directory for CSV and verdict files.",
    )
    p.add_argument(
        "--strategies",
        default="apc_lru,twotier_lru,flowcache_lossless",
        help="Comma-separated strategy names (default: 3 strategies, no_cache trimmed).",
    )
    p.add_argument(
        "--max-episodes", type=int, default=None,
        help="Limit number of trace files (episodes). Default: all.",
    )
    p.add_argument(
        "--max-requests", type=int, default=None,
        help="Limit number of serving requests. Default: all.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max generation tokens per request (default: 64 for TTFT focus).",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.70,
        help="vLLM GPU memory utilization (default: 0.70).",
    )
    p.add_argument(
        "--kv-cache-memory-gib", type=float, default=None,
        help="Explicit GPU KV cache capacity in GiB. Overrides automatic "
             "inference from gpu_memory_utilization. Recommended: 2.0 GiB.",
    )
    p.add_argument(
        "--max-num-seqs", type=int, default=4,
        help="Max concurrent sequences = concurrency level (default: 4).",
    )
    p.add_argument(
        "--cpu-capacity-gib", type=float, default=2.0,
        help="CPU tier capacity in GiB (default: 2.0).",
    )
    p.add_argument(
        "--block-size", type=int, default=16,
        help="KV cache block size (default: 16).",
    )
    p.add_argument(
        "--max-model-len", type=int, default=24576,
        help="Max model context length (default: 24576, covers tau-bench max prompt).",
    )
    p.add_argument(
        "--slo-threshold-ms", type=float, default=2000.0,
        help="TTFT SLO threshold in ms (default: 2000).",
    )
    p.add_argument(
        "--flowcache-min-benefit-ms", type=float, default=0.0,
        help="FlowCache minimum_net_benefit_ms (default: 0.0).",
    )
    p.add_argument(
        "--flowcache-migrate-ratio", type=float, default=0.5,
        help="FlowCache migrate_ratio for ratio mode (default: 0.5).",
    )
    p.add_argument(
        "--flowcache-d2h-ms", type=float, default=0.10,
        help="D2H cost per block in ms (default: 0.10, cloud-calibrated).",
    )
    p.add_argument(
        "--flowcache-h2d-ms", type=float, default=0.15,
        help="H2D cost per block in ms (default: 0.15, cloud-calibrated).",
    )
    p.add_argument(
        "--flowcache-share-window", type=int, default=1000,
        help="Causal share_count sliding window size (default: 1000).",
    )
    p.add_argument(
        "--flowcache-share-cap", type=int, default=8,
        help="share_count cap (default: 8).",
    )
    p.add_argument(
        "--flowcache-prefill-ms-per-block", type=float, default=5.0,
        help="Estimated prefill cost per block in ms (default: 5.0). "
             "Must exceed h2d_ms for migration to be worthwhile.",
    )
    p.add_argument(
        "--flowcache-no-selective", action="store_true",
        help="Disable selective migration in FlowCache (always migrate all). "
             "Use for diagnosis: if this matches twotier_lru, selectivity is the "
             "sole cause of any gap.",
    )
    p.add_argument(
        "--bootstrap-samples", type=int, default=1000,
        help="Bootstrap CI samples (default: 1000).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 50 requests, 2 strategies only.",
    )
    p.add_argument(
        "--quick-pilot", action="store_true",
        help="Quick pilot: 8 tasks, 16 workflows (~294 requests), "
             "fixed in-flight, PILOT_PASS/PILOT_FAIL verdict only. "
             "Does NOT substitute for a full G3 GO/NO-GO.",
    )
    p.add_argument(
        "--arrival-replay", action="store_true",
        help="Enable arrival-time replay (submit by trace arrival_time_ms). "
             "τ-bench inter-arrival ≥1s, so this is very slow and throughput "
             "becomes arrival-rate-limited. Default: batch submit.",
    )
    p.add_argument(
        "--time-scale", type=float, default=1.0,
        help="Arrival time replay speedup factor (default: 1.0 = real time). "
             "10.0 = 10x faster (arrivals compressed).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Bootstrap CI (task-cluster global p95 difference)
# ---------------------------------------------------------------------------

def bootstrap_p95_diff(
    treatment_ttfts: Dict[str, List[float]],
    control_ttfts: Dict[str, List[float]],
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for the difference of global p95 TTFT.

    Clusters by task_id: each task's per-request values are resampled as a
    unit, then the global p95 is computed from the combined sample. This
    matches the G3 verdict metric (p95) rather than testing per-task means.

    Args:
        treatment_ttfts: {task_id: [ttft_values]} for treatment
        control_ttfts: {task_id: [ttft_values]} for control
        n_samples: number of bootstrap resamples
        ci_level: confidence level

    Returns:
        {p95_diff, ci_low, ci_high, ci_excludes_zero, n_tasks}
    """
    import random
    rng = random.Random(seed)

    common_tasks = sorted(
        set(treatment_ttfts.keys()) & set(control_ttfts.keys())
    )
    if not common_tasks:
        return {"p95_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "ci_excludes_zero": False, "n_tasks": 0}

    # Observed global p95 difference (control - treatment, positive = improvement)
    all_t = sum((treatment_ttfts[t] for t in common_tasks), [])
    all_c = sum((control_ttfts[t] for t in common_tasks), [])
    if not all_t or not all_c:
        return {"p95_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "ci_excludes_zero": False, "n_tasks": len(common_tasks)}

    def _p95(vals):
        s = sorted(vals)
        return s[int(len(s) * 0.95)]

    observed_p95_diff = _p95(all_c) - _p95(all_t)

    # Bootstrap resampling (cluster by task)
    n = len(common_tasks)
    bootstrap_diffs: List[float] = []
    for _ in range(n_samples):
        sampled_tasks = [common_tasks[rng.randrange(n)] for _ in range(n)]
        bs_t = sum((treatment_ttfts[t] for t in sampled_tasks), [])
        bs_c = sum((control_ttfts[t] for t in sampled_tasks), [])
        if bs_t and bs_c:
            bootstrap_diffs.append(_p95(bs_c) - _p95(bs_t))

    if not bootstrap_diffs:
        return {"p95_diff": observed_p95_diff, "ci_low": 0.0, "ci_high": 0.0,
                "ci_excludes_zero": False, "n_tasks": len(common_tasks)}

    bootstrap_diffs.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(alpha / 2 * len(bootstrap_diffs))
    hi_idx = int((1 - alpha / 2) * len(bootstrap_diffs))
    ci_low = bootstrap_diffs[lo_idx]
    ci_high = bootstrap_diffs[min(hi_idx, len(bootstrap_diffs) - 1)]

    return {
        "p95_diff": observed_p95_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": ci_low > 0,
        "n_tasks": len(common_tasks),
    }


# ---------------------------------------------------------------------------
# Quick pilot task/workflow selection
# ---------------------------------------------------------------------------

def select_quick_pilot_workflows(
    trace_dir: Path,
    domain_tasks: int = 4,
    workflows_per_task: int = 2,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, Any]]:
    """Select tasks and workflows for quick pilot.

    Rules:
    - airline / retail: domain_tasks tasks each
    - per task: workflows_per_task workflows closest to median request count
    - tie-breaking: fixed SHA-256 seed
    - Selection is based on request count and prompt size, NOT performance.

    Returns:
        (selected_workflow_ids, selection_metadata)
    """
    import hashlib

    # Scan all traces
    task_workflows: Dict[str, List[dict]] = {}
    for fp in sorted(trace_dir.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        try:
            with open(fp) as f:
                t = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta = t.get("meta", {})
        tid = meta.get("task_id", "")
        domain = meta.get("domain", "")
        wf_id = meta.get("workflow_id", "")
        if not tid or not wf_id:
            continue
        n_requests = sum(1 for s in t.get("steps", [])
                         if s.get("role") == "assistant")
        total_prompt = sum(
            s.get("estimated_prompt_chars", 0)
            for s in t.get("steps", [])
            if s.get("role") == "assistant"
        )
        task_workflows.setdefault(tid, []).append({
            "workflow_id": wf_id,
            "domain": domain,
            "n_requests": n_requests,
            "total_prompt": total_prompt,
        })

    # Per domain: sort tasks by median workload footprint, pick top domain_tasks
    selection: Dict[str, Any] = {"tasks": {}, "total_workflows": 0,
                                   "total_requests": 0}
    selected_wf_ids: List[str] = []

    for domain in ["airline", "retail"]:
        domain_tids = [tid for tid, wfs in task_workflows.items()
                       if any(w["domain"] == domain for w in wfs)]
        # Score each task by median request count * median prompt
        task_scores: List[Tuple[float, str]] = []
        for tid in domain_tids:
            wfs = task_workflows[tid]
            reqs = sorted(w["n_requests"] for w in wfs)
            prompts = sorted(w["total_prompt"] for w in wfs)
            med_req = reqs[len(reqs) // 2]
            med_prompt = prompts[len(prompts) // 2]
            task_scores.append((med_req * med_prompt, tid))
        task_scores.sort(key=lambda x: -x[0])

        # Pick top domain_tasks
        for _, tid in task_scores[:domain_tasks]:
            wfs = task_workflows[tid]
            # Sort by distance to median request count
            med_req = sorted(w["n_requests"] for w in wfs)[len(wfs) // 2]
            wfs_sorted = sorted(wfs, key=lambda w: abs(w["n_requests"] - med_req))
            # Tie-break with SHA-256
            rng_state = hashlib.sha256(
                f"{tid}:{seed}".encode()
            ).digest()
            import random as _random
            tie_rng = _random.Random(int.from_bytes(rng_state[:8], "big"))
            wfs_sorted.sort(key=lambda w: tie_rng.random())

            picked = wfs_sorted[:workflows_per_task]
            picked_ids = [w["workflow_id"] for w in picked]
            total_req = sum(w["n_requests"] for w in picked)
            selection["tasks"][tid] = {
                "domain": domain,
                "workflows": picked_ids,
                "n_requests": total_req,
                "n_workflows": len(picked),
            }
            selected_wf_ids.extend(picked_ids)
            selection["total_workflows"] += len(picked)
            selection["total_requests"] += total_req

    return selected_wf_ids, selection


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def compute_verdict(
    all_metrics: Dict[str, StrategyMetrics],
    bootstrap_samples: int = 1000,
    slo_threshold_ms: float = 2000.0,
    is_pilot: bool = False,
) -> Dict[str, Any]:
    """Compute G3 closed-loop verdict.

    Pass criteria:
      1. p95 TTFT improvement ≥ 15% (flowcache vs twotier_lru)
      2. Throughput drop ≤ 5% (flowcache vs twotier_lru)
      3. Bootstrap CI excludes 0 (task-cluster global p95 diff)

    Args:
        is_pilot: if True, output PILOT_PASS / PILOT_FAIL instead of PASS / FAIL
    """
    verdict_prefix = "PILOT_" if is_pilot else ""

    verdict = {
        "experiment": "g3_closed_loop_pilot" if is_pilot else "g3_closed_loop",
        "ttft_metric_valid": True,
        "throughput_metric_valid": True,
        "pass_criteria": {
            "p95_ttft_improvement_pct": ">= 15%",
            "throughput_drop_pct": "<= 5%",
            "bootstrap_ci_excludes_zero": True,
        },
        "checks": {},
        "verdict": "PENDING",
    }

    fc = all_metrics.get("flowcache_lossless")
    tt = all_metrics.get("twotier_lru")

    if fc is None or tt is None:
        verdict["verdict"] = "INCOMPLETE"
        verdict["error"] = "Missing flowcache_lossless or twotier_lru results"
        return verdict

    # Check for INCOMPLETE timestamps
    fc_invalid = sum(1 for rm in fc.per_request if not rm.timestamps_valid)
    tt_invalid = sum(1 for rm in tt.per_request if not rm.timestamps_valid)
    if fc_invalid > 0 or tt_invalid > 0:
        verdict["ttft_metric_valid"] = False
        verdict["verdict"] = "INCOMPLETE"
        verdict["error"] = (
            f"Invalid timestamps: flowcache={fc_invalid}, "
            f"twotier_lru={tt_invalid}"
        )
        return verdict

    # --- Check 1: p95 TTFT improvement ≥ 15% ---
    if tt.ttft_p95 > 0:
        ttft_improvement = (tt.ttft_p95 - fc.ttft_p95) / tt.ttft_p95
    else:
        ttft_improvement = 0.0
    check1_pass = ttft_improvement >= 0.15
    verdict["checks"]["p95_ttft_improvement"] = {
        "control_p95_ms": tt.ttft_p95,
        "treatment_p95_ms": fc.ttft_p95,
        "improvement_pct": ttft_improvement * 100,
        "threshold_pct": 15.0,
        "pass": check1_pass,
    }

    # --- Check 2: Throughput drop ≤ 5% ---
    if tt.throughput_req_per_s > 0:
        throughput_drop = (
            (tt.throughput_req_per_s - fc.throughput_req_per_s)
            / tt.throughput_req_per_s
        )
    else:
        throughput_drop = 1.0
    check2_pass = throughput_drop <= 0.05
    verdict["checks"]["throughput_drop"] = {
        "control_throughput": tt.throughput_req_per_s,
        "treatment_throughput": fc.throughput_req_per_s,
        "drop_pct": throughput_drop * 100,
        "threshold_pct": 5.0,
        "pass": check2_pass,
    }

    # --- Check 3: Bootstrap CI (task-cluster, global p95 diff) ---
    fc_by_task: Dict[str, List[float]] = {}
    tt_by_task: Dict[str, List[float]] = {}
    for rm in fc.per_request:
        if rm.timestamps_valid:
            fc_by_task.setdefault(rm.task_id, []).append(rm.ttft)
    for rm in tt.per_request:
        if rm.timestamps_valid:
            tt_by_task.setdefault(rm.task_id, []).append(rm.ttft)

    ci = bootstrap_p95_diff(fc_by_task, tt_by_task, n_samples=bootstrap_samples)
    check3_pass = ci["ci_excludes_zero"]
    verdict["checks"]["bootstrap_ci_ttft"] = {
        "p95_diff_ms": ci["p95_diff"],
        "ci_low_ms": ci["ci_low"],
        "ci_high_ms": ci["ci_high"],
        "n_tasks": ci["n_tasks"],
        "ci_excludes_zero": ci["ci_excludes_zero"],
        "pass": check3_pass,
    }

    # --- Overall verdict ---
    all_pass = check1_pass and check2_pass and check3_pass
    verdict["verdict"] = (
        f"{verdict_prefix}PASS" if all_pass else f"{verdict_prefix}FAIL"
    )

    # Add summary metrics
    verdict["strategy_summaries"] = {
        s: m.to_dict() for s, m in all_metrics.items()
    }

    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Smoke test overrides
    if args.smoke:
        args.max_requests = args.max_requests or 50
        args.strategies = "apc_lru,flowcache_lossless"
        logger.info("SMOKE MODE: max_requests=%d, strategies=%s",
                     args.max_requests, args.strategies)

    # Quick pilot overrides
    if args.quick_pilot:
        args.strategies = "twotier_lru,flowcache_lossless"
        args.max_new_tokens = 16
        logger.info("QUICK PILOT MODE: 8 tasks, 16 workflows, ~294 requests")

    # Parse strategies
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    for s in strategy_names:
        try:
            Strategy(s)
        except ValueError:
            logger.error("Unknown strategy: %s", s)
            return 1

    # Load traces
    logger.info("Loading traces from %s", args.trace_dir)

    if args.quick_pilot:
        selected_wfs, selection_meta = select_quick_pilot_workflows(args.trace_dir)
        loader = TraceLoader.from_workflow_ids(args.trace_dir, selected_wfs)
        selection_path = Path(args.output_dir) / "quick-pilot-selection.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selection_meta, f, indent=2, ensure_ascii=False)
        logger.info("Quick pilot selection saved to %s", selection_path)
    else:
        loader = TraceLoader(args.trace_dir, max_episodes=args.max_episodes)

    requests = loader.load()

    if not requests:
        logger.error("No requests loaded. Check --trace-dir.")
        return 1

    if not args.quick_pilot and args.max_requests:
        requests = requests[:args.max_requests]

    logger.info(
        "Total serving requests: %d (from %d tasks)",
        len(requests),
        len(set(r.task_id for r in requests)),
    )

    # Compute kv_cache_memory_bytes from --kv-cache-memory-gib
    kv_cache_bytes = None
    if args.kv_cache_memory_gib is not None:
        kv_cache_bytes = int(args.kv_cache_memory_gib * 1024**3)

    # Create serving harness
    harness = ServingHarness(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        cpu_capacity_bytes=int(args.cpu_capacity_gib * 1024**3),
        kv_cache_memory_bytes=kv_cache_bytes,
        flowcache_config={
            "minimum_net_benefit_ms": args.flowcache_min_benefit_ms,
            "migrate_ratio": args.flowcache_migrate_ratio,
            "d2h_ms_per_block": args.flowcache_d2h_ms,
            "h2d_ms_per_block": args.flowcache_h2d_ms,
            "share_window_accesses": args.flowcache_share_window,
            "share_count_cap": args.flowcache_share_cap,
            "prefill_ms_per_block": args.flowcache_prefill_ms_per_block,
            "no_selective": args.flowcache_no_selective,
        },
        slo_threshold_ms=args.slo_threshold_ms,
    )

    # Run each strategy
    all_metrics: Dict[str, StrategyMetrics] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sname in strategy_names:
        strategy = Strategy(sname)
        logger.info("=" * 60)
        logger.info("Running strategy: %s", sname)
        logger.info("=" * 60)

        try:
            metrics = harness.run_strategy(
                strategy=strategy,
                requests=requests,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                arrival_time_replay=args.arrival_replay,
                time_scale=args.time_scale,
            )
            all_metrics[sname] = metrics
        except Exception as e:
            logger.error("Strategy %s failed: %s", sname, e, exc_info=True)
            all_metrics[sname] = StrategyMetrics(strategy=sname)

    # Compute verdict
    logger.info("=" * 60)
    logger.info("Computing G3 closed-loop verdict")
    logger.info("=" * 60)

    verdict = compute_verdict(
        all_metrics,
        bootstrap_samples=args.bootstrap_samples,
        slo_threshold_ms=args.slo_threshold_ms,
        is_pilot=args.quick_pilot,
    )

    # Write verdict
    verdict_path = output_dir / "closed-loop-verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Verdict written to %s", verdict_path)

    # Print summary
    print("\n" + "=" * 60)
    print("G3 Closed-Loop Verdict: " + verdict["verdict"])
    print("=" * 60)

    for sname, m in all_metrics.items():
        print(f"\n  [{sname}]")
        print(f"    Requests: {m.successful_requests}/{m.total_requests}")
        print(f"    TTFT p50/p95: {m.ttft_p50:.1f} / {m.ttft_p95:.1f} ms")
        print(f"    JCT  p50/p95: {m.jct_p50:.1f} / {m.jct_p95:.1f} ms")
        print(f"    Throughput:   {m.throughput_req_per_s:.2f} req/s")
        print(f"    Goodput:      {m.goodput_rate*100:.1f}%")

    if "p95_ttft_improvement" in verdict["checks"]:
        c1 = verdict["checks"]["p95_ttft_improvement"]
        print(f"\n  Check 1 — p95 TTFT improvement: {c1['improvement_pct']:.1f}% "
              f"(threshold: {c1['threshold_pct']:.0f}%) → "
              f"{'PASS' if c1['pass'] else 'FAIL'}")

    if "throughput_drop" in verdict["checks"]:
        c2 = verdict["checks"]["throughput_drop"]
        print(f"  Check 2 — Throughput drop: {c2['drop_pct']:.1f}% "
              f"(threshold: {c2['threshold_pct']:.0f}%) → "
              f"{'PASS' if c2['pass'] else 'FAIL'}")

    if "bootstrap_ci_ttft" in verdict["checks"]:
        c3 = verdict["checks"]["bootstrap_ci_ttft"]
        print(f"  Check 3 — Bootstrap CI: [{c3['ci_low_ms']:.1f}, {c3['ci_high_ms']:.1f}] ms "
              f"(n_tasks={c3['n_tasks']}) → "
              f"{'PASS' if c3['pass'] else 'FAIL'}")

    print(f"\n  Overall: {verdict['verdict']}")
    print("=" * 60)

    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
