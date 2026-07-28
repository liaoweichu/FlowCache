"""
G3-P1 Selective-Migration Runner
=================================
全网格运行器：9 cell × configured baselines × 1320 episodes
（单次运行，无 replay seed）。

G3′ 修复（相对原 G3）：
  1. 移除 replay_seeds 概念：trace 中所有 1320 episodes 一次跑完
  2. 输出 task_id 级别的 per-task 指标，用于 165 个 task 聚类 bootstrap
  3. FlowCache 使用修复后的 controller（migrate_threshold=0.01, safety_margin=0.05）

读取 G1′ 的物理前缀访问流（access_trace_c{1,4,8}.jsonl），
对每个 (capacity, concurrency, baseline) 组合重放访问流，
收集 per-task 指标，输出 raw_results.csv（每行 = cell × baseline × task_id）。

指标：
  - task_miss_cost_ms: task 级 miss cost 总和
  - task_p95_cache_delay_ms: miss + D2H + H2D + controller 的建模 P95
  - task_hit_rate: task 级 block 命中率
  - block_hit_rate: 全局 block 命中率（聚合行）
  - migrate_ms_total / restore_ms_total: 迁移/恢复开销（仅 FlowCache）

注意：open-loop 只能输出 modeled cache delay 和 offered load；真实 TTFT、
queueing、throughput/goodput 必须由 closed-loop serving 实验测量。

用法：
  python run_g3_grid.py --config config.yaml
  python run_g3_grid.py --config config.yaml --smoke-test  # W8 冒烟
  python run_g3_grid.py --config config.yaml --protocol-test  # G3-P1 单 cell
  python run_g3_grid.py --config config.yaml --max-episodes 100  # 小样本
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict, deque
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

OFFLINE_FUTURE_BASELINES = frozenset({"belady", "oracle_cost"})


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
    selected_episodes = set()
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # workflow_id is the episode-level unit; request_id is only one
            # LLM call within an episode.  The old code capped unique request
            # IDs and also retained the first block of request N+1.
            episode_id = (
                record.get("workflow_id")
                or record.get("episode_id")
                or record.get("request_id", "")
            )
            if (
                max_episodes
                and episode_id not in selected_episodes
                and len(selected_episodes) >= max_episodes
            ):
                # Continue scanning because accesses from selected workflows
                # may be interleaved with later workflows under concurrency.
                continue
            selected_episodes.add(episode_id)
            accesses.append(record)
    return accesses


def annotate_causal_share_counts(
    accesses: List[Dict], horizon: int = 1000
) -> None:
    """Annotate decision-time workflow sharing without future leakage.

    For access ``i``, ``_share_count`` is the number of distinct workflows
    that have accessed the same physical block within the trailing
    ``horizon`` accesses, including the current access.  The annotation is
    causal: no access after ``i`` contributes to its value.
    """
    horizon = max(1, int(horizon))
    window = deque()
    per_block = defaultdict(Counter)

    for index, access in enumerate(accesses):
        while window and index - window[0][0] >= horizon:
            _, old_block, old_workflow = window.popleft()
            counts = per_block[old_block]
            counts[old_workflow] -= 1
            if counts[old_workflow] <= 0:
                del counts[old_workflow]
            if not counts:
                del per_block[old_block]

        block_hash = access.get("block_hash", "")
        workflow_id = (
            access.get("workflow_id")
            or access.get("episode_id")
            or access.get("request_id", "")
        )
        per_block[block_hash][workflow_id] += 1
        window.append((index, block_hash, workflow_id))
        access["_share_count"] = len(per_block[block_hash])
        access["_share_count_scope"] = (
            "causal_past_window_including_current"
        )


def filter_task_split(
    accesses: List[Dict],
    split: str = "all",
    validation_fraction: float = 0.2,
    split_seed: int = 42,
) -> List[Dict]:
    """Deterministically group-split a trace by ``task_id``.

    Every workflow/seed belonging to one task stays in the same split.
    Hashing is stable across Python processes and independent of trace order.
    """
    if split == "all":
        return accesses
    if split not in {"validation", "test"}:
        raise ValueError("task split must be all, validation, or test")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    def is_validation(task_id: str) -> bool:
        payload = f"{split_seed}:{task_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64)
        return bucket < validation_fraction

    keep_validation = split == "validation"
    return [
        access for access in accesses
        if is_validation(str(access.get("task_id", "")))
        == keep_validation
    ]


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
        future_accesses: 仅允许 Belady/OracleCost 使用；任何在线
            baseline 收到非 None 值都会 fail-closed
    """
    if (
        name not in OFFLINE_FUTURE_BASELINES
        and future_accesses is not None
    ):
        raise ValueError(
            f"future_accesses is forbidden for online baseline {name}"
        )
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
    if name in {
        "flowcache_lossless",
        "flowcache_selective_migrate_only",
        "flowcache_always_migrate",
    }:
        fc_cfg = flowcache_config or {}
        heuristic_cfg = fc_cfg.get("heuristic", {})
        migration_policy = (
            "always_migrate"
            if name == "flowcache_always_migrate"
            else fc_cfg.get("controller_variant", "selective_value")
        )
        gpu_admission_policy = (
            fc_cfg.get("gpu_admission_policy", "oracle_cost_proxy")
            if name == "flowcache_lossless"
            else "always_admit"
        )
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
            migration_policy=migration_policy,
            gpu_admission_policy=gpu_admission_policy,
            selective_migration_config=fc_cfg.get(
                "selective_migration", {}
            ),
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
    """Replay one trace and collect *modeled cache-delay* diagnostics.

    The previous runner called miss-only cost "TTFT" and computed
    ``requests / arrival-window`` as policy throughput.  The repaired replay
    now charges synchronous D2H/H2D and modeled controller cost to each
    request.  It deliberately marks real TTFT and throughput as unavailable:
    those require a closed-loop serving run with queueing and completion
    timestamps.
    """
    request_delay = defaultdict(float)
    request_to_task = {}
    task_request_delays = defaultdict(list)
    task_hits = defaultdict(int)
    task_misses = defaultdict(int)
    task_saved = defaultdict(float)
    task_miss_cost = defaultdict(float)
    task_migrate = defaultdict(float)
    task_restore = defaultdict(float)
    task_policy = defaultdict(float)
    task_migration_candidates = defaultdict(int)
    task_migration_selected = defaultdict(int)
    task_migration_rejected = defaultdict(int)
    task_candidate_value_index = defaultdict(float)
    task_gpu_admission_candidates = defaultdict(int)
    task_gpu_admission_selected = defaultdict(int)
    task_gpu_admission_bypassed = defaultdict(int)

    negative_cost_count = 0
    for idx, record in enumerate(accesses):
        is_hit = access_baseline(cache, record, idx)
        req_id = record.get("request_id", "")
        task_id = record.get("task_id", "")
        prefill_ms = max(0.0, float(record.get("prefill_ms", 0.0) or 0.0))

        migrate_ms = float(getattr(cache, "last_migrate_ms", 0.0) or 0.0)
        restore_ms = float(getattr(cache, "last_restore_ms", 0.0) or 0.0)
        policy_ms = float(getattr(cache, "last_policy_model_ms", 0.0) or 0.0)
        migration_candidates = int(
            getattr(cache, "last_migration_candidate_count", 0) or 0
        )
        migration_selected = int(
            getattr(cache, "last_migration_selected_count", 0) or 0
        )
        migration_rejected = int(
            getattr(cache, "last_migration_rejected_count", 0) or 0
        )
        candidate_value_index = float(
            getattr(cache, "last_candidate_value_index_ms", 0.0) or 0.0
        )
        gpu_admission_candidates = int(
            getattr(cache, "last_gpu_admission_candidate_count", 0) or 0
        )
        gpu_admission_selected = int(
            getattr(cache, "last_gpu_admission_selected_count", 0) or 0
        )
        gpu_admission_bypassed = int(
            getattr(cache, "last_gpu_admission_bypassed_count", 0) or 0
        )
        if migrate_ms < 0 or restore_ms < 0 or policy_ms < 0:
            negative_cost_count += 1
        migrate_ms = max(0.0, migrate_ms)
        restore_ms = max(0.0, restore_ms)
        policy_ms = max(0.0, policy_ms)

        request_to_task[req_id] = task_id
        request_delay[req_id] += migrate_ms + restore_ms + policy_ms
        task_migrate[task_id] += migrate_ms
        task_restore[task_id] += restore_ms
        task_policy[task_id] += policy_ms
        task_migration_candidates[task_id] += migration_candidates
        task_migration_selected[task_id] += migration_selected
        task_migration_rejected[task_id] += migration_rejected
        task_candidate_value_index[task_id] += candidate_value_index
        task_gpu_admission_candidates[task_id] += (
            gpu_admission_candidates
        )
        task_gpu_admission_selected[task_id] += gpu_admission_selected
        task_gpu_admission_bypassed[task_id] += gpu_admission_bypassed

        if is_hit:
            task_hits[task_id] += 1
            task_saved[task_id] += prefill_ms
        else:
            task_misses[task_id] += 1
            task_miss_cost[task_id] += prefill_ms
            request_delay[req_id] += prefill_ms

    for req_id, delay in request_delay.items():
        task_id = request_to_task.get(req_id, "")
        task_request_delays[task_id].append(delay)

    per_task: Dict[str, Dict] = {}
    all_task_ids = set(task_hits) | set(task_misses)
    for task_id in all_task_ids:
        req_delays = task_request_delays.get(task_id, [])
        hits = task_hits.get(task_id, 0)
        misses = task_misses.get(task_id, 0)
        total_blocks = hits + misses
        hit_rate = hits / total_blocks if total_blocks else 0.0

        if req_delays:
            sorted_delays = sorted(req_delays)
            p95_idx = min(
                int(len(sorted_delays) * 0.95), len(sorted_delays) - 1
            )
            task_p95 = sorted_delays[p95_idx]
            task_p50 = sorted_delays[len(sorted_delays) // 2]
        else:
            task_p95 = 0.0
            task_p50 = 0.0

        migrate_total = task_migrate.get(task_id, 0.0)
        restore_total = task_restore.get(task_id, 0.0)
        policy_total = task_policy.get(task_id, 0.0)
        per_task[task_id] = {
            "task_miss_cost_ms": task_miss_cost.get(task_id, 0.0),
            "task_saved_prefill_ms": task_saved.get(task_id, 0.0),
            "task_migrate_ms": migrate_total,
            "task_restore_ms": restore_total,
            "task_transfer_ms": migrate_total + restore_total,
            "task_policy_model_ms": policy_total,
            "task_migration_candidate_count": (
                task_migration_candidates.get(task_id, 0)
            ),
            "task_migration_selected_count": (
                task_migration_selected.get(task_id, 0)
            ),
            "task_migration_rejected_count": (
                task_migration_rejected.get(task_id, 0)
            ),
            "task_candidate_value_index_ms": (
                task_candidate_value_index.get(task_id, 0.0)
            ),
            "task_gpu_admission_candidate_count": (
                task_gpu_admission_candidates.get(task_id, 0)
            ),
            "task_gpu_admission_selected_count": (
                task_gpu_admission_selected.get(task_id, 0)
            ),
            "task_gpu_admission_bypassed_count": (
                task_gpu_admission_bypassed.get(task_id, 0)
            ),
            "task_modeled_service_cost_ms": (
                task_miss_cost.get(task_id, 0.0)
                + migrate_total + restore_total + policy_total
            ),
            # Legacy column names are retained for existing bootstrap code,
            # but their scope is explicitly marked in the global columns.
            "task_p95_ttft_ms": task_p95,
            "task_p50_ttft_ms": task_p50,
            "task_p95_cache_delay_ms": task_p95,
            "task_p50_cache_delay_ms": task_p50,
            "task_hits": hits,
            "task_misses": misses,
            "task_block_hit_rate": hit_rate,
            "task_n_requests": len(req_delays),
        }

    all_delays = list(request_delay.values())
    if all_delays:
        sorted_delays = sorted(all_delays)
        p95_idx = min(int(len(sorted_delays) * 0.95), len(sorted_delays) - 1)
        p95_delay_ms = sorted_delays[p95_idx]
        p50_delay_ms = sorted_delays[len(sorted_delays) // 2]
    else:
        p95_delay_ms = 0.0
        p50_delay_ms = 0.0

    # Arrival-window rate describes the offered trace, not achieved service
    # throughput: every policy sees exactly the same value.
    n_requests = len(request_delay)
    arrivals = [
        float(a.get("arrival_time_ms", 0.0) or 0.0) for a in accesses
    ]
    arrival_span_ms = (max(arrivals) - min(arrivals)) if arrivals else 0.0
    offered_load = (
        n_requests / (arrival_span_ms / 1000.0)
        if arrival_span_ms > 0 else 0.0
    )

    stats = cache.get_stats() if hasattr(cache, "get_stats") else {
        "hits": getattr(cache, "hits", 0),
        "misses": getattr(cache, "misses", 0),
        "evictions": getattr(cache, "evictions", 0),
        "saved_prefill_ms": getattr(cache, "saved_prefill_ms", 0.0),
        "miss_cost_ms": getattr(cache, "miss_cost_ms", 0.0),
    }
    total_blocks = stats["hits"] + stats["misses"]
    block_hit_rate = stats["hits"] / total_blocks if total_blocks else 0.0

    migrate_total = float(stats.get("migrate_ms_total", 0.0) or 0.0)
    restore_total = float(stats.get("restore_ms_total", 0.0) or 0.0)
    policy_total = float(stats.get("policy_model_ms_total", 0.0) or 0.0)
    global_metrics = {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "evictions": stats["evictions"],
        "block_hit_rate": block_hit_rate,
        "saved_prefill_ms": stats["saved_prefill_ms"],
        "miss_cost_ms": stats["miss_cost_ms"],
        "p50_ttft_ms": p50_delay_ms,
        "p95_ttft_ms": p95_delay_ms,
        "p50_cache_delay_ms": p50_delay_ms,
        "p95_cache_delay_ms": p95_delay_ms,
        "latency_metric_scope": "modeled_cache_delay",
        "ttft_metric_valid": False,
        "throughput_req_per_s": offered_load,
        "offered_load_req_per_s": offered_load,
        "throughput_metric_valid": False,
        "n_requests": n_requests,
        "migrate_ms_total": migrate_total,
        "restore_ms_total": restore_total,
        "transfer_ms_total": migrate_total + restore_total,
        "migrate_bytes_total": stats.get("migrate_bytes_total", 0),
        "restore_bytes_total": stats.get("restore_bytes_total", 0),
        "migrate_count": stats.get("migrate_to_cpu_count", 0),
        "restore_count": stats.get("restore_to_gpu_count", 0),
        "policy_model_ms_total": policy_total,
        "replay_wall_ms_total": stats.get("replay_wall_ms_total", 0.0),
        "replay_wall_us_per_access": stats.get(
            "replay_wall_us_per_access", 0.0
        ),
        "controller_variant": stats.get("controller_variant", "not_applicable"),
        "gpu_admission_policy": stats.get(
            "gpu_admission_policy", "not_applicable"
        ),
        "policy_stack": stats.get("policy_stack", "not_applicable"),
        "gpu_admission_margin_ms": stats.get(
            "gpu_admission_margin_ms", 0.0
        ),
        "gpu_admission_cold_start_prior": stats.get(
            "gpu_admission_cold_start_prior", 0.0
        ),
        "gpu_admission_cold_start_cost_ratio": stats.get(
            "gpu_admission_cold_start_cost_ratio", 0.0
        ),
        "gpu_admission_confidence_scale": stats.get(
            "gpu_admission_confidence_scale", 0.0
        ),
        "online_feature_scope": stats.get(
            "online_feature_scope", "not_applicable"
        ),
        "future_access_index_used": stats.get(
            "future_access_index_used", False
        ),
        "gpu_admission_candidate_count": stats.get(
            "gpu_admission_candidate_count", 0
        ),
        "gpu_admission_selected_count": stats.get(
            "gpu_admission_selected_count", 0
        ),
        "gpu_admission_bypassed_count": stats.get(
            "gpu_admission_bypassed_count", 0
        ),
        "gpu_admission_selection_rate": stats.get(
            "gpu_admission_selection_rate", 0.0
        ),
        "gpu_admission_bypass_rate": stats.get(
            "gpu_admission_bypass_rate", 0.0
        ),
        "gpu_bypassed_prefill_ms_total": stats.get(
            "gpu_bypassed_prefill_ms_total", 0.0
        ),
        "gpu_admission_candidate_value_index_ms_total": stats.get(
            "gpu_admission_candidate_value_index_ms_total", 0.0
        ),
        "gpu_admission_incumbent_value_index_ms_total": stats.get(
            "gpu_admission_incumbent_value_index_ms_total", 0.0
        ),
        "gpu_admission_displacement_value_index_ms_total": stats.get(
            "gpu_admission_displacement_value_index_ms_total", 0.0
        ),
        "migration_candidate_count": stats.get(
            "migration_candidate_count", 0
        ),
        "migration_selected_count": stats.get(
            "migration_selected_count", 0
        ),
        "migration_rejected_count": stats.get(
            "migration_rejected_count", 0
        ),
        "rejected_low_value_count": stats.get(
            "rejected_low_value_count", 0
        ),
        "rejected_cpu_competition_count": stats.get(
            "rejected_cpu_competition_count", 0
        ),
        "rejected_no_cpu_slot_count": stats.get(
            "rejected_no_cpu_slot_count", 0
        ),
        "cpu_admission_replacement_count": stats.get(
            "cpu_admission_replacement_count", 0
        ),
        "cpu_heap_compaction_count": stats.get(
            "cpu_heap_compaction_count", 0
        ),
        "migration_selection_rate": stats.get(
            "migration_selection_rate", 0.0
        ),
        "restore_per_migration": stats.get("restore_per_migration", 0.0),
        "candidate_value_index_ms_total": stats.get(
            "candidate_value_index_ms_total", 0.0
        ),
        "selected_value_index_ms_total": stats.get(
            "selected_value_index_ms_total", 0.0
        ),
        "rejected_value_index_ms_total": stats.get(
            "rejected_value_index_ms_total", 0.0
        ),
        "cpu_hold_cost_per_block_ms": stats.get(
            "cpu_hold_cost_per_block_ms", 0.0
        ),
        "fallback_count": stats.get("fallback_count", 0),
        "negative_cost_count": negative_cost_count,
        "n_tasks": len(per_task),
    }
    return global_metrics, per_task


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------

def run_grid(config: Dict, smoke_test: bool = False,
             protocol_test: bool = False,
             max_episodes: Optional[int] = None,
             task_split: str = "all",
             validation_fraction: float = 0.2,
             split_seed: int = 42,
             access_cache: Optional[Dict] = None) -> List[Dict]:
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
    if protocol_test:
        protocol_cfg = config["protocol_test"]
        cells = [protocol_cfg["cell"]]
        baselines_to_run = protocol_cfg["baselines"]
        episodes = protocol_cfg.get("episodes", 100)
        if max_episodes:
            episodes = max_episodes
        print(f"[PROTOCOL TEST] cell={cells[0]}, "
              f"baselines={baselines_to_run}, episodes={episodes}")
    elif smoke_test:
        cells = [config["smoke_test"]["cell"]]
        baselines_to_run = config["smoke_test"]["baselines"]
        episodes = config["smoke_test"].get("episodes", 100)
        if max_episodes:
            episodes = max_episodes
        print(f"[SMOKE TEST] cell={cells[0]}, baselines={baselines_to_run}, "
              f"episodes={episodes}")
    else:
        cells = config["grid"]["cells"]
        baselines_to_run = []
        for group in [
            "lower_bound",
            "simple_heuristic",
            "tiered_ablation",
            "oracle",
            "flowcache",
        ]:
            for bl in config["baselines"].get(group, []):
                if bl.get("enabled", True):
                    baselines_to_run.append(bl["name"])
        episodes = config["grid"].get("episodes_per_cell", 1320)
        if max_episodes:
            episodes = max_episodes

    flowcache_config = config.get("flowcache", {})
    trace_dir = Path(
        config.get("trace_source", {}).get(
            "access_trace_dir", "../g1prime/physical_traces/"
        )
    )
    if not trace_dir.is_absolute():
        trace_dir = (SCRIPT_DIR / trace_dir).resolve()
    results = []

    for cell in cells:
        cap_gib = cell["capacity_gib"]
        conc = cell["concurrency"]
        capacity_blocks = gib_to_blocks(cap_gib, block_bytes)

        # 加载对应并发度的 access trace
        trace_path = trace_dir / f"access_trace_c{conc}.jsonl"
        if not trace_path.exists():
            print(f"Warning: trace file {trace_path} not found, skipping cell")
            continue

        print(f"\nCell: {cap_gib} GiB (capacity={capacity_blocks} blocks), "
              f"concurrency={conc}")

        selection_cfg = flowcache_config.get("selective_migration", {})
        share_horizon = selection_cfg.get(
            "share_window_accesses",
            flowcache_config.get("heuristic", {}).get("horizon", 1000),
        )
        trace_stat = trace_path.stat()
        access_cache_key = (
            str(trace_path.resolve()),
            trace_stat.st_size,
            trace_stat.st_mtime_ns,
            episodes,
            task_split,
            validation_fraction,
            split_seed,
            share_horizon,
        )
        if (
            access_cache is not None
            and access_cache_key in access_cache
        ):
            all_accesses = access_cache[access_cache_key]
            print("  Reusing parsed/causally-annotated access trace")
        else:
            # Parse, group-split and annotate only once per sweep. This cache
            # is process-local and contains no future-access index.
            all_accesses = load_access_trace(
                trace_path, max_episodes=episodes
            )
            all_accesses = filter_task_split(
                all_accesses,
                split=task_split,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            annotate_causal_share_counts(all_accesses, share_horizon)
            if access_cache is not None:
                access_cache[access_cache_key] = all_accesses
        print(f"  Loaded {len(all_accesses)} accesses, "
            f"{len(set(a.get('request_id','') for a in all_accesses))} requests, "
            f"{len(set(a.get('task_id','') for a in all_accesses))} tasks, "
            f"task_split={task_split}")

        # Future index is built lazily below, only for offline oracle rows.
        # Future indices are physically isolated from every online policy.
        # Build them lazily only for an explicitly offline oracle.
        oracle_future_accesses = None

        for baseline_name in baselines_to_run:
            t0 = time.perf_counter()
            future_accesses = None
            if baseline_name in OFFLINE_FUTURE_BASELINES:
                if oracle_future_accesses is None:
                    oracle_future_accesses = compute_future_accesses(
                        all_accesses
                    )
                future_accesses = oracle_future_accesses
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
                    "task_split": task_split,
                    "validation_fraction": validation_fraction,
                    "split_seed": split_seed,
                    # Keep enough precision for small-sample complexity
                    # diagnostics; two decimals made the ≤3× ratio unstable.
                    "elapsed_s": round(elapsed, 6),
                    # 全局指标（同一 baseline 在同一 cell 下相同，便于聚合）
                    "global_block_hit_rate": global_metrics["block_hit_rate"],
                    "global_p50_ttft_ms": global_metrics["p50_ttft_ms"],
                    "global_p95_ttft_ms": global_metrics["p95_ttft_ms"],
                    "global_p50_cache_delay_ms": global_metrics["p50_cache_delay_ms"],
                    "global_p95_cache_delay_ms": global_metrics["p95_cache_delay_ms"],
                    "latency_metric_scope": global_metrics["latency_metric_scope"],
                    "ttft_metric_valid": global_metrics["ttft_metric_valid"],
                    "global_throughput": global_metrics["throughput_req_per_s"],
                    "global_offered_load": global_metrics["offered_load_req_per_s"],
                    "throughput_metric_valid": global_metrics["throughput_metric_valid"],
                    "migrate_ms_total": global_metrics["migrate_ms_total"],
                    "restore_ms_total": global_metrics["restore_ms_total"],
                    "transfer_ms_total": global_metrics["transfer_ms_total"],
                    "migrate_bytes_total": global_metrics["migrate_bytes_total"],
                    "restore_bytes_total": global_metrics["restore_bytes_total"],
                    "migrate_count": global_metrics["migrate_count"],
                    "restore_count": global_metrics["restore_count"],
                    "policy_model_ms_total": global_metrics["policy_model_ms_total"],
                    "replay_wall_ms_total": global_metrics["replay_wall_ms_total"],
                    "replay_wall_us_per_access": global_metrics["replay_wall_us_per_access"],
                    "controller_variant": global_metrics["controller_variant"],
                    "gpu_admission_policy": global_metrics["gpu_admission_policy"],
                    "policy_stack": global_metrics["policy_stack"],
                    "gpu_admission_margin_ms": global_metrics["gpu_admission_margin_ms"],
                    "gpu_admission_cold_start_prior": global_metrics["gpu_admission_cold_start_prior"],
                    "gpu_admission_cold_start_cost_ratio": global_metrics["gpu_admission_cold_start_cost_ratio"],
                    "gpu_admission_confidence_scale": global_metrics["gpu_admission_confidence_scale"],
                    "online_feature_scope": global_metrics["online_feature_scope"],
                    "future_access_index_used": global_metrics["future_access_index_used"],
                    "share_count_feature_scope": (
                        "causal_past_window_including_current"
                    ),
                    "gpu_admission_candidate_count": global_metrics["gpu_admission_candidate_count"],
                    "gpu_admission_selected_count": global_metrics["gpu_admission_selected_count"],
                    "gpu_admission_bypassed_count": global_metrics["gpu_admission_bypassed_count"],
                    "gpu_admission_selection_rate": global_metrics["gpu_admission_selection_rate"],
                    "gpu_admission_bypass_rate": global_metrics["gpu_admission_bypass_rate"],
                    "gpu_bypassed_prefill_ms_total": global_metrics["gpu_bypassed_prefill_ms_total"],
                    "gpu_admission_candidate_value_index_ms_total": global_metrics["gpu_admission_candidate_value_index_ms_total"],
                    "gpu_admission_incumbent_value_index_ms_total": global_metrics["gpu_admission_incumbent_value_index_ms_total"],
                    "gpu_admission_displacement_value_index_ms_total": global_metrics["gpu_admission_displacement_value_index_ms_total"],
                    "migration_candidate_count": global_metrics["migration_candidate_count"],
                    "migration_selected_count": global_metrics["migration_selected_count"],
                    "migration_rejected_count": global_metrics["migration_rejected_count"],
                    "rejected_low_value_count": global_metrics["rejected_low_value_count"],
                    "rejected_cpu_competition_count": global_metrics["rejected_cpu_competition_count"],
                    "rejected_no_cpu_slot_count": global_metrics["rejected_no_cpu_slot_count"],
                    "cpu_admission_replacement_count": global_metrics["cpu_admission_replacement_count"],
                    "cpu_heap_compaction_count": global_metrics["cpu_heap_compaction_count"],
                    "migration_selection_rate": global_metrics["migration_selection_rate"],
                    "restore_per_migration": global_metrics["restore_per_migration"],
                    "candidate_value_index_ms_total": global_metrics["candidate_value_index_ms_total"],
                    "selected_value_index_ms_total": global_metrics["selected_value_index_ms_total"],
                    "rejected_value_index_ms_total": global_metrics["rejected_value_index_ms_total"],
                    "cpu_hold_cost_per_block_ms": global_metrics["cpu_hold_cost_per_block_ms"],
                    "fallback_count": global_metrics["fallback_count"],
                    "negative_cost_count": global_metrics["negative_cost_count"],
                    "n_tasks": global_metrics["n_tasks"],
                    # per-task 指标（bootstrap 单元）
                    **task_metrics,
                }
                results.append(row)

            print(f"    {baseline_name}: modeled_p95_cache_delay="
                  f"{global_metrics['p95_cache_delay_ms']:.1f} ms, "
                  f"hit_rate={global_metrics['block_hit_rate']:.3f}, "
                  f"migrate={global_metrics['migrate_count']}, "
                  f"restore={global_metrics['restore_count']}, "
                  f"select={global_metrics['migration_selection_rate']:.3f} "
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
    parser = argparse.ArgumentParser(
        description="G3-P1 selective-migration runner"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run W8 smoke test (main cell × 4 baselines × 100 episodes)")
    parser.add_argument("--protocol-test", action="store_true",
                        help="Run G3-P1 selective migration on one diagnostic cell")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes per cell (for testing)")
    parser.add_argument(
        "--task-split",
        choices=["all", "validation", "test"],
        default="all",
        help="Deterministic task-grouped split for parameter tuning",
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.2
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--minimum-net-benefit-ms", type=float)
    parser.add_argument("--cpu-admission-margin-ms", type=float)
    parser.add_argument("--expected-cpu-residence-steps", type=int)
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: config output.raw_results_csv)")
    args = parser.parse_args()
    if args.smoke_test and args.protocol_test:
        parser.error("--smoke-test and --protocol-test are mutually exclusive")

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    selection_cfg = config.setdefault("flowcache", {}).setdefault(
        "selective_migration", {}
    )
    if args.minimum_net_benefit_ms is not None:
        selection_cfg["minimum_net_benefit_ms"] = (
            args.minimum_net_benefit_ms
        )
    if args.cpu_admission_margin_ms is not None:
        selection_cfg["cpu_admission_margin_ms"] = (
            args.cpu_admission_margin_ms
        )
    if args.expected_cpu_residence_steps is not None:
        selection_cfg["expected_cpu_residence_steps"] = (
            args.expected_cpu_residence_steps
        )

    # Run grid
    results = run_grid(config, smoke_test=args.smoke_test,
                       protocol_test=args.protocol_test,
                       max_episodes=args.max_episodes,
                       task_split=args.task_split,
                       validation_fraction=args.validation_fraction,
                       split_seed=args.split_seed)

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
