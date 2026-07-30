"""Compare memory footprint: SimpleCPUOffloadConnector vs FlowCacheConnector (non-selective).
Runs each with 10 requests, captures GPU/CPU memory allocation from vLLM engine logs.
"""
import subprocess, sys, re

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/snapshots/master"
TRACE = "experiments/e1/traces/bf16/tau_bench"

def run_and_capture(label, extra_args=None):
    cmd = [
        sys.executable, "experiments/g3/closed_loop/run_closed_loop.py",
        "--model", MODEL, "--trace-dir", TRACE,
        "--output-dir", f"/tmp/memtest_{label}",
        "--strategies", label,
        "--max-requests", "10",
        "--kv-cache-memory-gib", "2.0",
        "--cpu-capacity-gib", "2.0",
        "--gpu-memory-utilization", "0.76",
        "--max-model-len", "16384",
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    combined = r.stdout + r.stderr
    
    # Extract memory info
    import re
    patterns = {
        "gpu_free_mem": r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)",
        "kv_cache_size": r"GPU KV cache size: ([\d,]+) tokens",
        "cpu_blocks": r"Allocating (\d+) CPU blocks",
        "cpu_size": r"per_rank=([\d.]+) GB",
        "gpu_weight": r"Actual usage is ([\d.]+) GiB for weight",
        "gpu_activation": r"([\d.]+) GiB for peak activation",
        "gpu_cudagraph": r"([\d.]+) GiB for CUDAGraph memory",
        "gpu_nontorch": r"([\d.]+) GiB for non-torch memory",
        "kv_cache_memory": r"Current kv cache memory in use is ([\d.]+) GiB",
        "suggested_kv": r"`--kv-cache-memory=(\d+)`",
        "max_concurrency": r"Maximum concurrency for .* tokens per request: ([\d.]+)x",
    }
    
    result = {"label": label}
    for key, pat in patterns.items():
        m = re.search(pat, combined)
        if m:
            result[key] = m.group(1)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in result.items():
        if k != "label":
            print(f"  {k}: {v}")
    return result

print("Running twotier_lru (SimpleCPUOffloadConnector)...")
tt = run_and_capture("twotier_lru")

print("\nRunning flowcache_lossless (FlowCacheConnector, non-selective)...")  
fc = run_and_capture("flowcache_lossless", ["--flowcache-no-selective"])

# Summary comparison
print(f"\n{'='*60}")
print("  Summary Comparison")
print(f"{'='*60}")
print(f"  {'Metric':<30} {'twotier_lru':>15} {'flowcache(ns)':>15}")
for key in sorted(set(tt.keys()) | set(fc.keys())):
    if key == "label":
        continue
    tv = tt.get(key, "N/A")
    fv = fc.get(key, "N/A")
    print(f"  {key:<30} {tv:>15} {fv:>15}")
