"""
G3 Grid Runner
==============
全网格运行器：9 cell × 6 baselines × 495 episodes × 3 replay seeds。

读取 G1′ 的物理前缀访问流（access_trace_c{1,4,8}.jsonl），
对每个 (capacity, concurrency, baseline, seed) 组合重放访问流，
收集 request 级指标，输出 raw_results.csv。

指标：
  - miss_prefill_ms: request 级 miss cost
  - p95_ttft_ms: 95% 分位 request miss cost
  - block_hit_rate: block 级命中率
  - saved_prefill_ms: 节省的 prefill 时间
  - throughput_req_per_s: 吞吐（requests/s）
  - migrate_ms_total / restore_ms_total: 迁移/恢复开销（仅 FlowCache）

用法：
  python run_g3_grid.py --config config.yaml
  python run_g3_grid.py --config config.yaml --smoke-test  # W8 冒烟
  python run_g3_grid.py --config config.yaml --max-episodes 100  # 小样本
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# 添加 experiments/e1/ 到路径以复用 baseline 类
SCRIPT_DIR = Path(__file__).resolve().parent
E1_DIR = SCRIPT_DIR.parent / "e1"
sys.path.insert(0, str(E1_DIR))

from compare_oracle import (
    LRUCache, GDSFCache, SizeCostCache, APCLRUCache, BeladyOracle, OracleCostCache
)
from controller import FlowCacheLosslessController, NoCacheBaseline


# ---------------------------------------------------------------------------
# Capacity conversion
# ---------------------------------------------------------------------------

def gib_to_blocks(gib: float, block_bytes: int) -> int:
    """将 GiB 转换为 block 数（向下取整）。"""
    if gib <= 0 or block_bytes <= 0:
        return 0
    return int(gib * 1024**3 // block_bytes)


def compute_block_bytes(g0_config: Dict) -> int:
    """计算每个 block 的字节数。"""
    return (g0_config["block_size"]
            * g0_config["num_hidden_layers"]
            * 2  # K + V
            * g0_config["num_kv_heads"]
            * g0_config["head_dim"]
            * g0_config["dtype_bytes"])


# ---------------------------------------------------------------------------
# Access trace loading
# ---------------------------------------------------------------------------

def load_access_trace(trace_path: Path, max_episodes: Optional[int] = None) -> List[Dict]:
    """
    加载 G1′ 的 access_trace JSONL 文件。

    每行是一个 access record，包含：
      block_hash, parent_hash, request_id, workflow_id, task_id, seed,
      domain, step_id, block_idx, arrival_time_ms, prefill_ms,
      num_prefix_tokens, block_token_count
    """
    accesses = []
    seen_requests = set()
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            accesses.append(record)
            seen_requests.add(record.get("request_id", ""))
            if max_episodes and len(seen_requests) > max_episodes:
                break
    return accesses


def filter_by_seed(accesses: List[Dict], seed: int) -> List[Dict]:
    """按 replay seed 过滤 access records。"""
    return [a for a in accesses if a.get("seed", 0) == seed]


def compute_share_counts(accesses: List[Dict], horizon: int = 1000) -> Dict[str, int]:
    """
    预计算每个 block_hash 的 share_count（H 窗口内访问该 block 的不同 workflow 数）。

    为了效率，使用滑动窗口近似。
    """
    # block_hash -> set of workflow_ids (简化版：全 trace 的共享度)
    block_workflows = defaultdict(set)
    for acc in accesses:
        bh = acc.get("block_hash", "")
        wf = acc.get("workflow_id", "")
        block_workflows[bh].add(wf)
    return {bh: len(wfs) for bh, wfs in block_workflows.items()}


# ---------------------------------------------------------------------------
# Baseline instantiation
# ---------------------------------------------------------------------------

def instantiate_baseline(name: str,
                         capacity_blocks: int,
                         cost_model: Optional[Dict] = None,
                         flowcache_config: Optional[Dict] = None,
                         block_bytes: int = 917504,
                         future_accesses: Optional[Dict] = None):
    """
    根据 baseline 名称实例化对应的 cache 类。

    Args:
        name: baseline 名称
        capacity_blocks: GPU 容量（block 数）
        cost_model: 成本模型（FlowCache 用）
        flowcache_config: FlowCache 配置
        block_bytes: 每个 block 的字节数
        future_accesses: Belady/OracleCost 的未来访问信息
    """
    if name == "no_cache":
        return NoCacheBaseline(capacity=0)
    if name == "lru":
        return LRUCache(capacity_blocks)
    if name == "gdsf":
        return GDSFCache(capacity_blocks)
    if name == "sizecost":
        return SizeCostCache(capacity_blocks)
    if name == "apc_lru":
        return APCLRUCache(capacity_blocks)
    if name == "belady":
        return BeladyOracle(capacity_blocks, future_accesses or {})
    if name == "oracle_cost":
        return OracleCostCache(capacity_blocks, future_accesses or {})
    if name == "flowcache_lossless":
        fc_cfg = flowcache_config or {}
        heuristic_cfg = fc_cfg.get("heuristic", {})
        return FlowCacheLosslessController(
            gpu_capacity_blocks=capacity_blocks,
            cpu_capacity_blocks=fc_cfg.get("cpu_capacity_blocks", -1),
            block_bytes=block_bytes,
            cost_model=cost_model or {},
            reuse_estimator_config=heuristic_cfg,
            safety_margin=fc_cfg.get("safety_margin", 0.10),
            score_lambda=fc_cfg.get("score_lambda", 0.1),
            fallback=fc_cfg.get("fallback", "sizecost"),
        )
    raise ValueError(f"Unknown baseline: {name}")


def access_baseline(cache, record: Dict, access_idx: int):
    """统一调用不同 baseline 的 access 方法。"""
    name = type(cache).__name__
    block_hash = record.get("block_hash", "")
    parent_hash = record.get("parent_hash", "")
    prefill_ms = record.get("prefill_ms", 0.0)
    block_idx = record.get("block_idx", 0)
    share_count = record.get("_share_count", 0)

    # FlowCache-Lossless 需要额外参数
    if isinstance(cache, FlowCacheLosslessController):
        return cache.access(
            block_hash, parent_hash=parent_hash, prefill_ms=prefill_ms,
            block_idx=block_idx, share_count=share_count
        )
    # NoCache
    if isinstance(cache, NoCacheBaseline):
        return cache.access(block_hash, prefill_ms=prefill_ms)
    # APC-LRU 需要 parent_hash
    if isinstance(cache, APCLRUCache):
        return cache.access(block_hash, parent_hash=parent_hash, prefill_ms=prefill_ms)
    # Belady / OracleCost 需要 access_idx
    if isinstance(cache, (BeladyOracle, OracleCostCache)):
        return cache.access(block_hash, access_idx=access_idx, prefill_ms=prefill_ms)
    # SizeCost 需要 size
    if isinstance(cache, SizeCostCache):
        size = record.get("block_token_count", 16)
        return cache.access(block_hash, prefill_ms=prefill_ms, size=size)
    # LRU / GDSF
    return cache.access(block_hash, prefill_ms=prefill_ms)


# ---------------------------------------------------------------------------
# Replay and metrics collection
# ---------------------------------------------------------------------------

def compute_future_accesses(accesses: List[Dict]) -> Dict[str, List[int]]:
    """预计算 future_accesses（Belady/OracleCost 用）。"""
    future = defaultdict(list)
    for idx, acc in enumerate(accesses):
        bh = acc.get("block_hash", "")
        future[bh].append(idx)
    return dict(future)


def replay_accesses(cache, accesses: List[Dict]) -> Dict:
    """
    重放访问流，收集指标。

    Returns:
        包含各种指标的字典
    """
    # request 级 miss cost 收集
    request_miss_cost = defaultdict(float)
    request_total_cost = defaultdict(float)
    request_blocks = defaultdict(int)
    request_hits = defaultdict(int)

    for idx, record in enumerate(accesses):
        is_hit = access_baseline(cache, record, idx)
        req_id = record.get("request_id", "")
        prefill_ms = record.get("prefill_ms", 0.0)

        request_blocks[req_id] += 1
        if is_hit:
            request_hits[req_id] += 1
        else:
            request_miss_cost[req_id] += prefill_ms
        request_total_cost[req_id] += prefill_ms

    # 计算 request 级指标
    miss_costs = list(request_miss_cost.values())
    total_costs = list(request_total_cost.values())

    # p95 TTFT = request 级 miss cost 的 P95
    if miss_costs:
        miss_costs_sorted = sorted(miss_costs)
        p95_idx = int(len(miss_costs_sorted) * 0.95)
        p95_ttft_ms = miss_costs_sorted[min(p95_idx, len(miss_costs_sorted) - 1)]
        p50_ttft_ms = miss_costs_sorted[len(miss_costs_sorted) // 2]
    else:
        p95_ttft_ms = 0.0
        p50_ttft_ms = 0.0

    # 吞吐：requests / total_time
    n_requests = len(request_miss_cost)
    total_arrival_ms = max(
        (a.get("arrival_time_ms", 0) for a in accesses),
        default=0
    )
    if total_arrival_ms > 0:
        throughput = n_requests / (total_arrival_ms / 1000.0)
    else:
        throughput = 0.0

    # block 级指标
    stats = cache.get_stats() if hasattr(cache, "get_stats") else {
        "hits": getattr(cache, "hits", 0),
        "misses": getattr(cache, "misses", 0),
        "evictions": getattr(cache, "evictions", 0),
        "saved_prefill_ms": getattr(cache, "saved_prefill_ms", 0.0),
        "miss_cost_ms": getattr(cache, "miss_cost_ms", 0.0),
    }

    total_blocks = stats["hits"] + stats["misses"]
    block_hit_rate = stats["hits"] / total_blocks if total_blocks > 0 else 0.0

    return {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "evictions": stats["evictions"],
        "block_hit_rate": block_hit_rate,
        "saved_prefill_ms": stats["saved_prefill_ms"],
        "miss_cost_ms": stats["miss_cost_ms"],
        "p50_ttft_ms": p50_ttft_ms,
        "p95_ttft_ms": p95_ttft_ms,
        "throughput_req_per_s": throughput,
        "n_requests": n_requests,
        "migrate_ms_total": stats.get("migrate_ms_total", 0.0),
        "restore_ms_total": stats.get("restore_ms_total", 0.0),
        "migrate_count": stats.get("migrate_to_cpu_count", 0),
        "restore_count": stats.get("restore_to_gpu_count", 0),
        "fallback_count": stats.get("fallback_count", 0),
    }


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------

def run_grid(config: Dict, smoke_test: bool = False,
             max_episodes: Optional[int] = None) -> List[Dict]:
    """运行全网格，返回结果行列表。"""
    g0 = config["g0"]
    block_bytes = compute_block_bytes(g0)

    # 加载成本模型
    cost_model_path = Path(__file__).parent / "cost-model.json"
    cost_model = {}
    if cost_model_path.exists():
        with open(cost_model_path, "r", encoding="utf-8") as f:
            cost_model = json.load(f)
        print(f"Loaded cost model from {cost_model_path}")
    else:
        print("Warning: cost-model.json not found, using empty cost model")

    # 确定运行的 cells
    if smoke_test:
        cells = [config["smoke_test"]["cell"]]
        baselines_to_run = config["smoke_test"]["baselines"]
        seeds = [0]
        episodes = config["smoke_test"].get("episodes", 100)
        print(f"[SMOKE TEST] cell={cells[0]}, baselines={baselines_to_run}, "
              f"episodes={episodes}")
    else:
        cells = config["grid"]["cells"]
        baselines_to_run = []
        for group in ["lower_bound", "simple_heuristic", "oracle", "flowcache"]:
            for bl in config["baselines"].get(group, []):
                if bl.get("enabled", True):
                    baselines_to_run.append(bl["name"])
        seeds = config["grid"]["replay_seeds"]
        episodes = config["grid"].get("episodes_per_cell", 495)
        if max_episodes:
            episodes = max_episodes

    flowcache_config = config.get("flowcache", {})
    results = []

    for cell in cells:
        cap_gib = cell["capacity_gib"]
        conc = cell["concurrency"]
        capacity_blocks = gib_to_blocks(cap_gib, block_bytes)

        # 加载对应并发度的 access trace
        trace_path = (Path(__file__).parent.parent / "g1prime" / "physical_traces"
                      / f"access_trace_c{conc}.jsonl")
        if not trace_path.exists():
            print(f"Warning: trace file {trace_path} not found, skipping cell")
            continue

        print(f"\nCell: {cap_gib} GiB (capacity={capacity_blocks} blocks), "
              f"concurrency={conc}")

        # 加载 trace
        all_accesses = load_access_trace(trace_path, max_episodes=episodes)
        print(f"  Loaded {len(all_accesses)} accesses")

        # 预计算 share_count
        share_counts = compute_share_counts(all_accesses)
        for acc in all_accesses:
            acc["_share_count"] = share_counts.get(acc.get("block_hash", ""), 0)

        # 预计算 future_accesses（所有 seed 共用）
        future_accesses = compute_future_accesses(all_accesses)

        for seed in seeds:
            seed_accesses = filter_by_seed(all_accesses, seed) if seeds != [0] else all_accesses
            if not seed_accesses:
                # 如果 seed 过滤后为空，用全部
                seed_accesses = all_accesses

            # 重新计算 future_accesses for this seed's subset
            if seeds != [0]:
                future_accesses = compute_future_accesses(seed_accesses)

            print(f"  Seed {seed}: {len(seed_accesses)} accesses, "
                  f"{len(set(a.get('request_id','') for a in seed_accesses))} requests")

            for baseline_name in baselines_to_run:
                t0 = time.perf_counter()
                cache = instantiate_baseline(
                    baseline_name, capacity_blocks, cost_model,
                    flowcache_config, block_bytes, future_accesses
                )
                metrics = replay_accesses(cache, seed_accesses)
                elapsed = time.perf_counter() - t0

                row = {
                    "capacity_gib": cap_gib,
                    "concurrency": conc,
                    "baseline": baseline_name,
                    "seed": seed,
                    "capacity_blocks": capacity_blocks,
                    "n_accesses": len(seed_accesses),
                    "elapsed_s": round(elapsed, 2),
                    **metrics,
                }
                results.append(row)
                print(f"    {baseline_name}: p95_ttft={metrics['p95_ttft_ms']:.1f} ms, "
                      f"hit_rate={metrics['block_hit_rate']:.3f}, "
                      f"miss_cost={metrics['miss_cost_ms']:.1f} ms "
                      f"({elapsed:.1f}s)")

    return results


def save_results(results: List[Dict], output_path: Path) -> None:
    """保存结果到 CSV。"""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_path} ({len(results)} rows)")


def main():
    parser = argparse.ArgumentParser(description="G3 grid runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run W8 smoke test (main cell × 4 baselines × 100 episodes)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes per cell (for testing)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: config output.raw_results_csv)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run grid
    results = run_grid(config, smoke_test=args.smoke_test,
                       max_episodes=args.max_episodes)

    # Save
    output_path = args.output or config.get("output", {}).get("raw_results_csv")
    if output_path is None:
        output_path = "results/raw_results.csv"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    save_results(results, output_path)


if __name__ == "__main__":
    main()
