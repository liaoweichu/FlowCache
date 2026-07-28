"""§4.2 Closed-loop serving harness.

Replays τ-bench request traces through vLLM with different cache strategies,
measuring real TTFT, JCT, throughput, and SLO goodput.

Strategies:
  1. no_cache          — vLLM with enable_prefix_caching=False
  2. apc_lru           — vLLM with enable_prefix_caching=True (native LRU)
  3. twotier_lru       — vLLM + SimpleCPUOffloadConnector (always migrate)
  4. flowcache_lossless — vLLM + FlowCacheConnector (selective migrate)

Main cell: 2 GiB KV cache, c=4 concurrency, Qwen2.5-7B-Instruct BF16.

Usage:
  from closed_loop.serving_harness import ServingHarness, Strategy
  harness = ServingHarness(model_path, gpu_mem_util=0.70, max_num_seqs=4)
  harness.run_strategy(Strategy.APC_LRU, requests, output_csv)
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    NO_CACHE = "no_cache"
    APC_LRU = "apc_lru"
    TWOTIER_LRU = "twotier_lru"
    FLOWCACHE_LOSSLESS = "flowcache_lossless"


# ---------------------------------------------------------------------------
# Request data structure
# ---------------------------------------------------------------------------

@dataclass
class ServingRequest:
    """A single request to be served by vLLM."""
    request_id: str
    messages: List[Dict[str, str]]   # chat messages for llm.chat()
    task_id: str
    seed: int
    workflow_id: str
    domain: str
    step_id: int
    arrival_time_ms: float           # original trace arrival time
    num_prefix_tokens: int = 0       # for reference


# ---------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    """Metrics for a single served request."""
    request_id: str
    task_id: str
    seed: int
    workflow_id: str
    domain: str
    step_id: int
    # Real measured timings (seconds)
    ttft: float = 0.0                # time to first token
    jct: float = 0.0                 # job completion time
    # Token counts
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Cache info
    cached_tokens: int = 0           # prefix-cached tokens (from vLLM)
    # Status
    success: bool = True
    error: str = ""


@dataclass
class StrategyMetrics:
    """Aggregated metrics for one strategy run."""
    strategy: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    # TTFT (ms)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    ttft_mean: float = 0.0
    # JCT (ms)
    jct_p50: float = 0.0
    jct_p95: float = 0.0
    jct_p99: float = 0.0
    jct_mean: float = 0.0
    # Throughput
    throughput_req_per_s: float = 0.0
    total_time_s: float = 0.0
    # SLO goodput
    slo_threshold_ms: float = 0.0
    goodput_rate: float = 0.0
    # Cache
    avg_cached_tokens: float = 0.0
    total_cached_tokens: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    # Per-request metrics (for bootstrap CI)
    per_request: List[RequestMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if k != "per_request"}


# ---------------------------------------------------------------------------
# Trace loader: reads e1 traces and builds ServingRequest list
# ---------------------------------------------------------------------------

class TraceLoader:
    """Loads τ-bench traces and builds serving requests.

    Reads e1 trace JSON files (experiments/e1/traces/bf16/tau_bench/*.json),
    reconstructs message histories for each assistant step, and produces
    ServingRequest objects in arrival-time order.
    """

    def __init__(self, trace_dir: Path, max_episodes: Optional[int] = None):
        self.trace_dir = Path(trace_dir)
        self.max_episodes = max_episodes

    def load(self) -> List[ServingRequest]:
        """Load all traces and return serving requests sorted by arrival time."""
        requests: List[ServingRequest] = []
        # Filter out checkpoint/metadata files (prefixed with "_")
        trace_files = sorted(
            p for p in self.trace_dir.glob("*.json") if not p.name.startswith("_")
        )
        if self.max_episodes:
            trace_files = trace_files[:self.max_episodes]

        logger.info("Loading %d trace files from %s", len(trace_files), self.trace_dir)

        for tf in trace_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skip %s: %s", tf.name, e)
                continue

            requests.extend(self._extract_requests(trace))

        # Sort by arrival time (simulates real arrival order)
        requests.sort(key=lambda r: r.arrival_time_ms)
        logger.info("Loaded %d serving requests", len(requests))
        return requests

    def _extract_requests(self, trace: Dict[str, Any]) -> List[ServingRequest]:
        """Extract serving requests from a single trace.

        For each assistant step, collect the full message history and create
        a ServingRequest. This matches the recompile_prefixes.py logic.
        """
        meta = trace.get("meta", {})
        steps = trace.get("steps", [])
        workflow_id = meta.get("workflow_id", "")
        task_id = meta.get("task_id", "")
        seed = meta.get("seed", 0)
        domain = meta.get("domain", "")

        requests: List[ServingRequest] = []
        message_history: List[Dict[str, str]] = []

        for step in steps:
            role = step.get("role", "")
            content = step.get("content", "")
            step_id = step.get("step_id", 0)
            arrival_ms = step.get("arrival_time_ms", 0.0)

            if role == "assistant":
                # This is a request point — the model needs to generate
                req = ServingRequest(
                    request_id=f"{workflow_id}__s{step_id}",
                    messages=list(message_history),  # snapshot
                    task_id=task_id,
                    seed=seed,
                    workflow_id=workflow_id,
                    domain=domain,
                    step_id=step_id,
                    arrival_time_ms=arrival_ms,
                    num_prefix_tokens=step.get("num_prefix_tokens", 0),
                )
                requests.append(req)
                # Add the assistant response to history (for subsequent requests)
                message_history.append({"role": "assistant", "content": content})
            elif role in ("system", "user", "tool"):
                # Add to message history
                msg_role = "tool" if role == "tool" else role
                message_history.append({"role": msg_role, "content": content})

        return requests


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects and aggregates per-request metrics from vLLM outputs."""

    @staticmethod
    def collect(
        strategy: str,
        outputs: List[Any],
        requests: List[ServingRequest],
        slo_threshold_ms: float = 2000.0,
    ) -> StrategyMetrics:
        """Collect metrics from vLLM RequestOutput list.

        vLLM assigns its own internal request_ids that do NOT match the
        ServingRequest.request_id field. However, llm.chat() returns outputs
        in the SAME ORDER as the input messages_list, so we match by index.

        Args:
            strategy: strategy name
            outputs: vLLM RequestOutput list (same order as requests)
            requests: original ServingRequest list
            slo_threshold_ms: TTFT SLO threshold in ms
        """
        metrics = StrategyMetrics(strategy=strategy, slo_threshold_ms=slo_threshold_ms)

        ttfts: List[float] = []
        jcts: List[float] = []
        total_cached = 0
        total_prompt = 0
        total_output = 0

        for idx, output in enumerate(outputs):
            # Match by index — llm.chat() preserves input order in outputs
            req = requests[idx] if idx < len(requests) else None

            rm = RequestMetrics(
                request_id=(req.request_id if req else str(idx)),
                task_id=req.task_id if req else "",
                seed=req.seed if req else 0,
                workflow_id=req.workflow_id if req else "",
                domain=req.domain if req else "",
                step_id=req.step_id if req else 0,
            )

            # Extract metrics from vLLM RequestOutput
            try:
                m = output.metrics
                # vLLM metrics: arrival_time, first_token_time, finished_time
                # (all in seconds, monotonic clock)
                if hasattr(m, "arrival_time") and hasattr(m, "first_token_time"):
                    if m.first_token_time and m.arrival_time is not None:
                        rm.ttft = (m.first_token_time - m.arrival_time) * 1000.0  # ms
                    if hasattr(m, "finished_time") and m.finished_time:
                        rm.jct = (m.finished_time - m.arrival_time) * 1000.0  # ms
            except (AttributeError, TypeError):
                pass

            # Token counts
            try:
                rm.prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
                rm.output_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
            except (AttributeError, IndexError):
                pass

            # Cached tokens (prefix cache hits)
            try:
                if hasattr(output, "metrics") and hasattr(output.metrics, "num_cached_tokens"):
                    rm.cached_tokens = output.metrics.num_cached_tokens or 0
            except AttributeError:
                pass

            rm.success = output.finished

            metrics.per_request.append(rm)
            if rm.success:
                metrics.successful_requests += 1
                ttfts.append(rm.ttft)
                jcts.append(rm.jct)
                total_cached += rm.cached_tokens
                total_prompt += rm.prompt_tokens
                total_output += rm.output_tokens
            else:
                metrics.failed_requests += 1

        metrics.total_requests = len(outputs)

        # Compute percentiles
        if ttfts:
            ttfts_sorted = sorted(ttfts)
            n = len(ttfts_sorted)
            metrics.ttft_p50 = ttfts_sorted[int(n * 0.50)]
            metrics.ttft_p95 = ttfts_sorted[int(n * 0.95)]
            metrics.ttft_p99 = ttfts_sorted[min(int(n * 0.99), n - 1)]
            metrics.ttft_mean = sum(ttfts) / len(ttfts)

        if jcts:
            jcts_sorted = sorted(jcts)
            n = len(jcts_sorted)
            metrics.jct_p50 = jcts_sorted[int(n * 0.50)]
            metrics.jct_p95 = jcts_sorted[int(n * 0.95)]
            metrics.jct_p99 = jcts_sorted[min(int(n * 0.99), n - 1)]
            metrics.jct_mean = sum(jcts) / len(jcts)

        # Throughput — note: this is a placeholder; the real wall-clock
        # throughput is set by ServingHarness.run_strategy() which wraps
        # the llm.chat() call with perf_counter. The per-request TTFT/JCT
        # values are durations (ms), not timestamps, so they cannot be
        # used to compute wall-clock span.
        metrics.total_time_s = 0.0
        metrics.throughput_req_per_s = 0.0

        # SLO goodput
        if ttfts:
            slo_count = sum(1 for t in ttfts if t <= slo_threshold_ms)
            metrics.goodput_rate = slo_count / len(ttfts)

        # Cache stats
        metrics.total_cached_tokens = total_cached
        metrics.total_prompt_tokens = total_prompt
        metrics.total_output_tokens = total_output
        metrics.avg_cached_tokens = (
            total_cached / max(1, metrics.successful_requests)
        )

        return metrics

    @staticmethod
    def to_csv(metrics: StrategyMetrics, output_path: Path) -> None:
        """Write per-request metrics to CSV."""
        import csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "strategy", "request_id", "task_id", "seed", "workflow_id",
                "domain", "step_id", "ttft_ms", "jct_ms",
                "prompt_tokens", "output_tokens", "cached_tokens", "success",
            ])
            for rm in metrics.per_request:
                writer.writerow([
                    metrics.strategy, rm.request_id, rm.task_id, rm.seed,
                    rm.workflow_id, rm.domain, rm.step_id,
                    f"{rm.ttft:.3f}", f"{rm.jct:.3f}",
                    rm.prompt_tokens, rm.output_tokens, rm.cached_tokens,
                    rm.success,
                ])

    @staticmethod
    def summary_csv(metrics: StrategyMetrics, output_path: Path) -> None:
        """Write summary metrics to CSV."""
        import csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "strategy", "total_requests", "successful_requests",
                "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "ttft_mean_ms",
                "jct_p50_ms", "jct_p95_ms", "jct_p99_ms", "jct_mean_ms",
                "throughput_req_per_s", "goodput_rate",
                "avg_cached_tokens", "total_cached_tokens",
            ])
            d = metrics.to_dict()
            writer.writerow([
                d["strategy"], d["total_requests"], d["successful_requests"],
                f"{d['ttft_p50']:.3f}", f"{d['ttft_p95']:.3f}",
                f"{d['ttft_p99']:.3f}", f"{d['ttft_mean']:.3f}",
                f"{d['jct_p50']:.3f}", f"{d['jct_p95']:.3f}",
                f"{d['jct_p99']:.3f}", f"{d['jct_mean']:.3f}",
                f"{d['throughput_req_per_s']:.4f}", f"{d['goodput_rate']:.4f}",
                f"{d['avg_cached_tokens']:.1f}", d["total_cached_tokens"],
            ])


# ---------------------------------------------------------------------------
# vLLM serving harness
# ---------------------------------------------------------------------------

class ServingHarness:
    """Configures vLLM and runs serving experiments.

    For each strategy, creates a fresh vLLM LLM instance with the appropriate
    cache configuration, processes all requests, and collects metrics.
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.70,
        max_num_seqs: int = 4,
        block_size: int = 16,
        max_model_len: int = 8192,
        cpu_capacity_bytes: int = 2 * (1024**3),
        dtype: str = "bfloat16",
        flowcache_config: Optional[Dict[str, Any]] = None,
        slo_threshold_ms: float = 2000.0,
    ):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.dtype = dtype
        self.flowcache_config = flowcache_config or {}
        self.slo_threshold_ms = slo_threshold_ms

    def run_strategy(
        self,
        strategy: Strategy,
        requests: List[ServingRequest],
        output_dir: Path,
        max_requests: Optional[int] = None,
        max_new_tokens: int = 64,
    ) -> StrategyMetrics:
        """Run one strategy and return metrics.

        Args:
            strategy: cache strategy to use
            requests: serving requests (will be truncated if max_requests)
            output_dir: directory for output CSV files
            max_requests: limit number of requests (for smoke tests)
            max_new_tokens: max generation length per request
        """
        if max_requests:
            requests = requests[:max_requests]

        logger.info(
            "=== Strategy: %s, requests: %d, max_seqs: %d ===",
            strategy.value, len(requests), self.max_num_seqs,
        )

        # Build vLLM config for this strategy
        llm = self._create_llm(strategy)

        try:
            # Prepare chat messages
            messages_list = [req.messages for req in requests]

            # Sampling parameters — short generation for TTFT measurement
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=42,
            )

            logger.info("Sending %d requests to vLLM...", len(requests))
            t_start = time.perf_counter()

            # Use llm.chat() for chat-template-aware prefix caching
            outputs = llm.chat(
                messages_list,
                sampling_params=sampling_params,
            )

            wall_time = time.perf_counter() - t_start
            logger.info(
                "vLLM completed %d requests in %.1fs (wall)", len(outputs), wall_time
            )

            # Collect metrics
            metrics = MetricsCollector.collect(
                strategy=strategy.value,
                outputs=outputs,
                requests=requests,
                slo_threshold_ms=self.slo_threshold_ms,
            )

            # Override total_time with wall clock (more accurate)
            metrics.total_time_s = wall_time
            metrics.throughput_req_per_s = (
                metrics.successful_requests / max(0.1, wall_time)
            )

            # Write outputs
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            MetricsCollector.to_csv(
                metrics, output_dir / f"closed-loop-{strategy.value}.csv"
            )
            MetricsCollector.summary_csv(
                metrics, output_dir / f"closed-loop-summary-{strategy.value}.csv"
            )

            # Log summary
            logger.info(
                "  TTFT p50/p95/p99: %.1f / %.1f / %.1f ms",
                metrics.ttft_p50, metrics.ttft_p95, metrics.ttft_p99,
            )
            logger.info(
                "  JCT  p50/p95/p99: %.1f / %.1f / %.1f ms",
                metrics.jct_p50, metrics.jct_p95, metrics.jct_p99,
            )
            logger.info(
                "  Throughput: %.2f req/s, Goodput: %.1f%%",
                metrics.throughput_req_per_s, metrics.goodput_rate * 100,
            )
            logger.info(
                "  Cached tokens: avg=%.1f, total=%d",
                metrics.avg_cached_tokens, metrics.total_cached_tokens,
            )

            return metrics

        finally:
            # Clean up vLLM instance to free GPU memory
            del llm
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("vLLM instance cleaned up")

    def _create_llm(self, strategy: Strategy):
        """Create a vLLM LLM instance configured for the given strategy."""
        from vllm import LLM

        common_kwargs = dict(
            model=self.model_path,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            disable_log_stats=False,
        )

        if strategy == Strategy.NO_CACHE:
            common_kwargs["enable_prefix_caching"] = False
            logger.info("vLLM config: no prefix caching")
            return LLM(**common_kwargs)

        if strategy == Strategy.APC_LRU:
            common_kwargs["enable_prefix_caching"] = True
            logger.info("vLLM config: prefix caching (LRU)")
            return LLM(**common_kwargs)

        # Two-tier strategies need a KV connector
        common_kwargs["enable_prefix_caching"] = True

        if strategy == Strategy.TWOTIER_LRU:
            # Use vLLM's built-in SimpleCPUOffloadConnector (always migrate)
            from vllm.config import KVTransferConfig
            kt_cfg = KVTransferConfig(
                kv_connector="SimpleCPUOffloadConnector",
                kv_connector_module=(
                    "vllm.distributed.kv_transfer.kv_connector.v1."
                    "simple_cpu_offload_connector"
                ),
                kv_role="producer",
                kv_connector_extra_config={
                    "cpu_bytes_to_use": str(self.cpu_capacity_bytes),
                },
            )
            common_kwargs["kv_transfer_config"] = kt_cfg
            logger.info(
                "vLLM config: SimpleCPUOffloadConnector, cpu=%.1f GiB",
                self.cpu_capacity_bytes / (1024**3),
            )
            return LLM(**common_kwargs)

        if strategy == Strategy.FLOWCACHE_LOSSLESS:
            # Use FlowCacheConnector (selective migrate)
            from vllm.config import KVTransferConfig
            fc_cfg = {
                "cpu_bytes_to_use": str(self.cpu_capacity_bytes),
                "selective_migration": "true",
                "migration_mode": "threshold",
                "minimum_net_benefit_ms": str(
                    self.flowcache_config.get("minimum_net_benefit_ms", 0.0)
                ),
                "migrate_ratio": str(
                    self.flowcache_config.get("migrate_ratio", 0.5)
                ),
                "d2h_ms_per_block": str(
                    self.flowcache_config.get("d2h_ms_per_block", 0.10)
                ),
                "h2d_ms_per_block": str(
                    self.flowcache_config.get("h2d_ms_per_block", 0.15)
                ),
                "share_window_accesses": str(
                    self.flowcache_config.get("share_window_accesses", 1000)
                ),
                "share_count_cap": str(
                    self.flowcache_config.get("share_count_cap", 8)
                ),
                # Estimated prefill cost per block (ms) — without this,
                # V_b = proxy * max(0 - h2d, 0) - d2h = -d2h < 0 for all
                # blocks, and NO blocks would ever be migrated.
                "prefill_ms_per_block": str(
                    self.flowcache_config.get("prefill_ms_per_block", 5.0)
                ),
            }
            kt_cfg = KVTransferConfig(
                kv_connector="FlowCacheConnector",
                kv_connector_module="closed_loop.flowcache_connector",
                kv_role="producer",
                kv_connector_extra_config=fc_cfg,
            )
            common_kwargs["kv_transfer_config"] = kt_cfg
            logger.info(
                "vLLM config: FlowCacheConnector (selective), cpu=%.1f GiB, "
                "mode=%s, min_benefit=%.3f, prefill_ms/block=%.1f",
                self.cpu_capacity_bytes / (1024**3),
                fc_cfg["migration_mode"],
                float(fc_cfg["minimum_net_benefit_ms"]),
                float(fc_cfg["prefill_ms_per_block"]),
            )
            return LLM(**common_kwargs)

        raise ValueError(f"Unknown strategy: {strategy}")
