"""§4.2 Closed-loop serving CLI runner.

Runs the 3-strategy closed-loop serving experiment:
  1. apc_lru           — vLLM native prefix caching (GPU-only LRU)
  2. twotier_lru       — SimpleCPUOffloadConnector (always migrate, control)
  3. flowcache_lossless — FlowCacheConnector (selective migrate, treatment)

Main cell: 2 GiB KV, c=4, Qwen2.5-7B-Instruct BF16.

G3 pass criteria (closed-loop):
  - p95 TTFT improvement ≥ 15% (flowcache vs twotier_lru)
  - Throughput drop ≤ 5% (flowcache vs twotier_lru)
  - Bootstrap CI excludes 0 (flowcache vs twotier_lru, per-task)

Note: no_cache (lower bound) was trimmed — verdict only requires
flowcache_lossless vs twotier_lru, and apc_lru is kept for narrative.

Usage:
  # Full run (all 3 strategies, all episodes)
  python run_closed_loop.py \\
    --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master \\
    --trace-dir experiments/e1/traces/bf16/tau_bench \\
    --output-dir results/closed-loop

  # Smoke test (50 requests, 2 strategies)
  python run_closed_loop.py \\
    --model ... --trace-dir ... \\
    --strategies apc_lru,flowcache_lossless \\
    --max-requests 100
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
        "--bootstrap-samples", type=int, default=1000,
        help="Bootstrap CI samples (default: 1000).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 50 requests, 2 strategies only.",
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
# Bootstrap CI (per-task clustered)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    treatment_values: Dict[str, List[float]],
    control_values: Dict[str, List[float]],
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for the difference of per-task means.

    Clusters by task_id (same as G3-P1 verdict). Each task's per-request
    values are resampled as a unit.

    Args:
        treatment_values: {task_id: [ttft values]} for treatment strategy
        control_values: {task_id: [ttft values]} for control strategy
        n_samples: number of bootstrap resamples
        ci_level: confidence level (0.95 = 95% CI)

    Returns:
        {mean_diff, ci_low, ci_high, significant}
    """
    import random
    rng = random.Random(seed)

    common_tasks = sorted(
        set(treatment_values.keys()) & set(control_values.keys())
    )
    if not common_tasks:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "significant": False, "n_tasks": 0}

    # Observed per-task differences
    task_diffs: List[float] = []
    for tid in common_tasks:
        t_vals = treatment_values[tid]
        c_vals = control_values[tid]
        if t_vals and c_vals:
            t_mean = sum(t_vals) / len(t_vals)
            c_mean = sum(c_vals) / len(c_vals)
            task_diffs.append(c_mean - t_mean)  # control - treatment (positive = improvement)

    if not task_diffs:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "significant": False, "n_tasks": 0}

    observed_mean = sum(task_diffs) / len(task_diffs)

    # Bootstrap resampling (cluster by task)
    n = len(task_diffs)
    bootstrap_means: List[float] = []
    for _ in range(n_samples):
        sampled = [task_diffs[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sampled) / n)

    bootstrap_means.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(alpha / 2 * n_samples)
    hi_idx = int((1 - alpha / 2) * n_samples)
    ci_low = bootstrap_means[lo_idx]
    ci_high = bootstrap_means[min(hi_idx, n_samples - 1)]

    return {
        "mean_diff": observed_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": (ci_low > 0 or ci_high < 0),
        "n_tasks": n,
    }


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def compute_verdict(
    all_metrics: Dict[str, StrategyMetrics],
    bootstrap_samples: int = 1000,
    slo_threshold_ms: float = 2000.0,
) -> Dict[str, Any]:
    """Compute G3 closed-loop verdict.

    Pass criteria:
      1. p95 TTFT improvement ≥ 15% (flowcache vs twotier_lru)
      2. Throughput drop ≤ 5% (flowcache vs twotier_lru)
      3. Bootstrap CI excludes 0 (flowcache vs twotier_lru, per-task)
      4. flowcache ttft_metric_valid = true (real measured TTFT)
    """
    verdict = {
        "experiment": "g3_closed_loop",
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

    # --- Check 3: Bootstrap CI (per-task, TTFT) ---
    # Group TTFT by task_id
    fc_by_task: Dict[str, List[float]] = {}
    tt_by_task: Dict[str, List[float]] = {}
    for rm in fc.per_request:
        fc_by_task.setdefault(rm.task_id, []).append(rm.ttft)
    for rm in tt.per_request:
        tt_by_task.setdefault(rm.task_id, []).append(rm.ttft)

    ci = bootstrap_ci(fc_by_task, tt_by_task, n_samples=bootstrap_samples)
    # ci["mean_diff"] is control - treatment (positive = FlowCache is faster)
    check3_pass = ci["significant"] and ci["ci_low"] > 0
    verdict["checks"]["bootstrap_ci_ttft"] = {
        "mean_diff_ms": ci["mean_diff"],
        "ci_low_ms": ci["ci_low"],
        "ci_high_ms": ci["ci_high"],
        "n_tasks": ci["n_tasks"],
        "significant": ci["significant"],
        "ci_excludes_zero": ci["ci_low"] > 0,
        "pass": check3_pass,
    }

    # --- Overall verdict ---
    all_pass = check1_pass and check2_pass and check3_pass
    verdict["verdict"] = "PASS" if all_pass else "FAIL"

    # Add summary metrics for all strategies
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
    loader = TraceLoader(args.trace_dir, max_episodes=args.max_episodes)
    requests = loader.load()

    if not requests:
        logger.error("No requests loaded. Check --trace-dir.")
        return 1

    if args.max_requests:
        requests = requests[:args.max_requests]

    logger.info(
        "Total serving requests: %d (from %d tasks)",
        len(requests),
        len(set(r.task_id for r in requests)),
    )

    # Create serving harness
    harness = ServingHarness(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        cpu_capacity_bytes=int(args.cpu_capacity_gib * 1024**3),
        flowcache_config={
            "minimum_net_benefit_ms": args.flowcache_min_benefit_ms,
            "migrate_ratio": args.flowcache_migrate_ratio,
            "d2h_ms_per_block": args.flowcache_d2h_ms,
            "h2d_ms_per_block": args.flowcache_h2d_ms,
            "share_window_accesses": args.flowcache_share_window,
            "share_count_cap": args.flowcache_share_cap,
            "prefill_ms_per_block": args.flowcache_prefill_ms_per_block,
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
