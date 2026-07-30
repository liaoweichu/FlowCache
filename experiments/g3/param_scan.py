"""Quick parameter scan for FlowCache connector.
Tests prefill_ms_per_block and minimum_net_benefit_ms combinations
on 50 requests to find the sweet spot where cache hit rate is high
but migration is reduced.
"""
import json
import subprocess
import sys
from pathlib import Path

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master"
TRACE_DIR = "experiments/e1/traces/bf16/tau_bench"
BASE_OUT = Path("experiments/g3/results/param-scan")
BASE_OUT.mkdir(parents=True, exist_ok=True)

# Grid: prefill_ms_per_block × minimum_net_benefit_ms
grid = [
    # (prefill_ms, min_benefit_ms, label)
    (1.0,  0.0,  "prefill1_min0"),
    (1.0,  2.0,  "prefill1_min2"),
    (1.0,  5.0,  "prefill1_min5"),
    (2.0,  0.0,  "prefill2_min0"),
    (2.0,  2.0,  "prefill2_min2"),
    (2.0,  5.0,  "prefill2_min5"),
    (5.0,  0.0,  "prefill5_min0"),    # current default
    (5.0,  2.0,  "prefill5_min2"),
    (5.0,  5.0,  "prefill5_min5"),
]

results = {}

for prefill_ms, min_benefit, label in grid:
    out_dir = BASE_OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        "experiments/g3/closed_loop/run_closed_loop.py",
        "--model", MODEL,
        "--trace-dir", TRACE_DIR,
        "--output-dir", str(out_dir),
        "--quick-pilot",
        "--kv-cache-memory-gib", "2.0",
        "--cpu-capacity-gib", "2.0",
        "--gpu-memory-utilization", "0.76",
        "--max-model-len", "16384",
        "--max-requests", "50",
        "--flowcache-prefill-ms-per-block", str(prefill_ms),
        "--flowcache-min-benefit-ms", str(min_benefit),
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    # Parse verdict
    verdict_path = out_dir / "closed-loop-verdict.json"
    if verdict_path.exists():
        with open(verdict_path) as f:
            v = json.load(f)
        results[label] = {
            "prefill_ms": prefill_ms,
            "min_benefit": min_benefit,
            "verdict": v.get("verdict", "UNKNOWN"),
            "ttft_p95_flowcache": v.get("strategy_summaries", {}).get(
                "flowcache_lossless", {}).get("ttft_p95", 0),
            "ttft_p95_twotier": v.get("strategy_summaries", {}).get(
                "twotier_lru", {}).get("ttft_p95", 0),
            "throughput_fc": v.get("strategy_summaries", {}).get(
                "flowcache_lossless", {}).get("throughput_req_per_s", 0),
            "throughput_tt": v.get("strategy_summaries", {}).get(
                "twotier_lru", {}).get("throughput_req_per_s", 0),
        }
        improvement = (results[label]["ttft_p95_twotier"] - results[label]["ttft_p95_flowcache"]) / max(1, results[label]["ttft_p95_twotier"]) * 100
        results[label]["improvement_pct"] = improvement
        print(f"  p95 TTFT: fc={results[label]['ttft_p95_flowcache']:.0f} twotier={results[label]['ttft_p95_twotier']:.0f} (+{improvement:.1f}%)")
        print(f"  Throughput: fc={results[label]['throughput_fc']:.2f} twotier={results[label]['throughput_tt']:.2f}")
        print(f"  Verdict: {results[label]['verdict']}")
    else:
        print(f"  FAILED: no verdict file")
        results[label] = {"error": "no verdict", "stderr": r.stderr[-500:]}

# Save summary
with open(BASE_OUT / "param-scan-summary.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("Parameter scan complete. Summary saved to param-scan-summary.json")
print(f"{'='*60}")

# Print summary table
print("\n| Prefill ms | Min benefit | p95 TTFT fc | p95 TTFT tt | Improv % | Throughput fc | Throughput tt |")
print("|---|---:|---:|---:|---:|---:|---:|")
for label, r in results.items():
    if "error" in r:
        continue
    print(f"| {r['prefill_ms']:.0f} | {r['min_benefit']:.0f} | {r['ttft_p95_flowcache']:.0f} | {r['ttft_p95_twotier']:.0f} | {r['improvement_pct']:+.1f} | {r['throughput_fc']:.2f} | {r['throughput_tt']:.2f} |")
