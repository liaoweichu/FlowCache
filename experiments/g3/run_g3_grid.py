"""
G3′ Grid Runner
===============
全网格运行器：9 cell × 6 baselines × 1320 episodes（单次运行，无 replay seed）。

G3′ 修复（相对原 G3）：
  1. 移除 replay_seeds 概念：trace 中所有 1320 episodes 一次跑完
  2. 输出 task_id 级别的 per-task 指标，用于 165 个 task 聚类 bootstrap
  3. FlowCache 使用修复后的 controller（migrate_threshold=0.01, safety_margin=0.05）

读取 G1′ 的物理前缀访问流（access_trace_c{1,4,8}.jsonl），
对每个 (capacity, concurrency, baseline) 组合重放访问流，
收集 per-task 指标，输出 raw_results.csv（每行 = cell × baseline × task_id）。

指标：
  - task_miss_cost_ms: task 级 miss cost 总和
  - task_p95_ttft_ms: task 内 request miss cost 的 P95
  - task_hit_rate: task 级 block 命中率
  - block_hit_rate: 全局 block 命中率（聚合行）
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
from typing import Dict, List, Optional, Tuple

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
            safety_margin=fc_cfg.get("safety_margin", 0.05),
            score_lambda=fc_cfg.get("score_lambda", 0.1),
            fallback=fc_cfg.get("fallback", "sizecost"),
            migrate_threshold=fc_cfg.get("migrate_threshold", 0.01),
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


def replay_accesses(cache, accesses: List[Dict]) -> Tuple[Dict, Dict[str, Dict]]:
    """
    重放访问流，收集全局指标和 per-task 指标。

    G3′：不再按 seed 分组，而是按 task_id 收集 per-task 指标，
    用于 165 个 task 聚类 bootstrap。

    G3'' 修复：
    - Bug 3: throughput 用总 request 数（含全命中），不是有 miss 的 request 数
    - Bug 4: p95 TTFT 包含全命中 request（TTFT=0），不是只算有 miss 的 request

    Returns:
        (global_metrics, per_task_metrics)
        - global_metrics: 全局聚合指标
        - per_task_metrics: {task_id: {metric: value}}
    """
    # per-request TTFT（miss cost）收集——所有 request 都计入
    request_ttft = defaultdict(float)  # request_id -> 总 miss cost（TTFT 近似）
    request_to_task = {}  # request_id -> task_id
    task_request_ttfts = defaultdict(list)  # task_id -> [ttft_per_request]
    task_hits = defaultdict(int)
    task_misses = defaultdict(int)
    task_saved = defaultdict(float)
    task_miss_cost = defaultdict(float)

    for idx, record in enumerate(accesses):
        is_hit = access_baseline(cache, record, idx)
        req_id = record.get("request_id", "")
        task_id = record.get("task_id", "")
        prefill_ms = record.get("prefill_ms", 0.0)

        request_to_task[req_id] = task_id
        # G3'': 所有 request 都初始化 TTFT=0（命中则保持 0，miss 则累加）
        if req_id not in request_ttft:
            request_ttft[req_id] = 0.0

        if is_hit:
            task_hits[task_id] += 1
            task_saved[task_id] += prefill_ms
        else:
            task_misses[task_id] += 1
            task_miss_cost[task_id] += prefill_ms
            request_ttft[req_id] += prefill_ms  # TTFT = 该 request 的总 miss cost

    # 构建 per-task 的 request TTFT 列表（含全命中 request，TTFT=0）
    for req_id, ttft in request_ttft.items():
        task_id = request_to_task.get(req_id, "")
        task_request_ttfts[task_id].append(ttft)

    # 计算 per-task 指标
    per_task: Dict[str, Dict] = {}
    all_task_ids = set(task_hits.keys()) | set(task_misses.keys())
    for task_id in all_task_ids:
        req_ttfts = task_request_ttfts.get(task_id, [])
        hits = task_hits.get(task_id, 0)
        misses = task_misses.get(task_id, 0)
        total_blocks = hits + misses
        hit_rate = hits / total_blocks if total_blocks > 0 else 0.0

        # task 级 p95 TTFT（所有 request 的 TTFT 的 P95，含全命中 TTFT=0）
        if req_ttfts:
            req_ttfts_sorted = sorted(req_ttfts)
            p95_idx = int(len(req_ttfts_sorted) * 0.95)
            task_p95 = req_ttfts_sorted[min(p95_idx, len(req_ttfts_sorted) - 1)]
            task_p50 = req_ttfts_sorted[len(req_ttfts_sorted) // 2]
        else:
            task_p95 = 0.0
            task_p50 = 0.0

        per_task[task_id] = {
            "task_miss_cost_ms": task_miss_cost.get(task_id, 0.0),
            "task_saved_prefill_ms": task_saved.get(task_id, 0.0),
            "task_p95_ttft_ms": task_p95,
            "task_p50_ttft_ms": task_p50,
            "task_hits": hits,
            "task_misses": misses,
            "task_block_hit_rate": hit_rate,
            "task_n_requests": len(req_ttfts),
        }

    # 全局指标——G3'': 所有 request 的 TTFT 的 P95
    all_ttfts = list(request_ttft.values())
    if all_ttfts:
        all_ttfts_sorted = sorted(all_ttfts)
        p95_idx = int(len(all_ttfts_sorted) * 0.95)
        p95_ttft_ms = all_ttfts_sorted[min(p95_idx, len(all_ttfts_sorted) - 1)]
        p50_ttft_ms = all_ttfts_sorted[len(all_ttfts_sorted) // 2]
    else:
        p95_ttft_ms = 0.0
        p50_ttft_ms = 0.0

    # G3'' Bug 3 修复：throughput 用总 request 数（含全命中）
    n_requests = len(request_ttft)
    total_arrival_ms = max(
        (a.get("arrival_time_ms", 0) for a in accesses),
        default=0
    )
    throughput = n_requests / (total_arrival_ms / 1000.0) if total_arrival_ms > 0 else 0.0

    stats = cache.get_stats() if hasattr(cache, "get_stats") else {
        "hits": getattr(cache, "hits", 0),
        "misses": getattr(cache, "misses", 0),
        "evictions": getattr(cache, "evictions", 0),
        "saved_prefill_ms": getattr(cache, "saved_prefill_ms", 0.0),
        "miss_cost_ms": getattr(cache, "miss_cost_ms", 0.0),
    }

    total_blocks = stats["hits"] + stats["misses"]
    block_hit_rate = stats["hits"] / total_blocks if total_blocks > 0 else 0.0

    global_metrics = {
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
        "n_tasks": len(per_task),
    }

    return global_metrics, per_task


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------

def run_grid(config: Dict, smoke_test: bool = False,
             max_episodes: Optional[int] = None) -> List[Dict]:
    """
    运行全网格，返回 per-task 结果行列表。

    G3′：不再有 replay seed 循环，每个 cell × baseline 跑一次全部 trace，
    输出每个 task_id 一行（用于聚类 bootstrap）。
    """
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
        episodes = config["grid"].get("episodes_per_cell", 1320)
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

        # 加载 trace（G3′: 一次性加载全部，不再按 seed 过滤）
        all_accesses = load_access_trace(trace_path, max_episodes=episodes)
        print(f"  Loaded {len(all_accesses)} accesses, "
              f"{len(set(a.get('request_id','') for a in all_accesses))} requests, "
              f"{len(set(a.get('task_id','') for a in all_accesses))} tasks")

        # 预计算 share_count
        share_counts = compute_share_counts(all_accesses)
        for acc in all_accesses:
            acc["_share_count"] = share_counts.get(acc.get("block_hash", ""), 0)

        # 预计算 future_accesses（Belady/OracleCost 用）
        future_accesses = compute_future_accesses(all_accesses)

        for baseline_name in baselines_to_run:
            t0 = time.perf_counter()
            cache = instantiate_baseline(
                baseline_name, capacity_blocks, cost_model,
                flowcache_config, block_bytes, future_accesses
            )
            global_metrics, per_task = replay_accesses(cache, all_accesses)
            elapsed = time.perf_counter() - t0

            # 输出每个 task_id 一行（用于 bootstrap）
            for task_id, task_metrics in per_task.items():
                row = {
                    "capacity_gib": cap_gib,
                    "concurrency": conc,
                    "baseline": baseline_name,
                    "task_id": task_id,
                    "capacity_blocks": capacity_blocks,
                    "n_accesses": len(all_accesses),
                    "elapsed_s": round(elapsed, 2),
                    # 全局指标（同一 baseline 在同一 cell 下相同，便于聚合）
                    "global_block_hit_rate": global_metrics["block_hit_rate"],
                    "global_p95_ttft_ms": global_metrics["p95_ttft_ms"],
                    "global_throughput": global_metrics["throughput_req_per_s"],
                    "migrate_ms_total": global_metrics["migrate_ms_total"],
                    "restore_ms_total": global_metrics["restore_ms_total"],
                    "migrate_count": global_metrics["migrate_count"],
                    "restore_count": global_metrics["restore_count"],
                    "fallback_count": global_metrics["fallback_count"],
                    "n_tasks": global_metrics["n_tasks"],
                    # per-task 指标（bootstrap 单元）
                    **task_metrics,
                }
                results.append(row)

            print(f"    {baseline_name}: global_p95={global_metrics['p95_ttft_ms']:.1f} ms, "
                  f"hit_rate={global_metrics['block_hit_rate']:.3f}, "
                  f"migrate={global_metrics['migrate_count']}, "
                  f"restore={global_metrics['restore_count']} "
                  f"({elapsed:.1f}s, {len(per_task)} tasks)")

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
    parser = argparse.ArgumentParser(description="G3′ grid runner")
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
