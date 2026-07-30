"""FlowCacheConnector: selective-migration KV connector for vLLM V1.

Extends SimpleCPUOffloadConnector to add value-aware selective migration:
  - On request_finished, evaluate each block's reuse value (R_b)
  - Only store high-R blocks to CPU (selective D2H migration)
  - Low-R blocks are evicted directly (no D2H cost)

This directly tests the G3 hypothesis: selective migration reduces D2H/H2D
traffic and improves p95 TTFT vs always-migrate (SimpleCPUOffloadConnector).

Registration (vLLM v0.7+):
  kv_transfer_config:
    kv_connector: "FlowCacheConnector"
    kv_connector_module: "closed_loop.flowcache_connector"
    kv_connector_extra_config:
      cpu_bytes_to_use: <bytes>        # CPU tier capacity
      selective_migration: true         # enable selective migration
      minimum_net_benefit_ms: 0.0       # V_b threshold
      migrate_ratio: 0.5               # fraction of blocks to migrate [0,1]

Action space A₀ (lossless): GPU BF16 ↔ CPU BF16 ↔ Evict. No quantization.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# ---------------------------------------------------------------------------
# Block access tracker (online, causal — no future information)
# ---------------------------------------------------------------------------

@dataclass
class BlockMeta:
    """Per-block metadata for R-value computation."""
    block_hash: str
    access_count: int = 0          # total times this block was accessed
    workflow_ids: set = field(default_factory=set)  # workflows that touched this block
    last_access_step: int = 0      # step of last access
    block_idx: int = 0             # position in prefix (0 = earliest)
    prefill_ms: float = 0.0        # estimated prefill cost for this block


class BlockAccessTracker:
    """Online tracker for block access history (causal, no future leakage).

    Maintains per-block metadata needed to compute the reuse value R_b at
    request_finished time. Only uses current and past information.
    """

    def __init__(
        self,
        capacity_blocks: int,
        share_window_accesses: int = 1000,
        share_count_cap: int = 8,
        beta: float = 0.005,
        alpha: float = 0.5,
        signal_weights: Optional[Dict[str, float]] = None,
    ):
        self.capacity_blocks = max(1, capacity_blocks)
        self.share_window = share_window_accesses
        self.share_count_cap = share_count_cap
        self.beta = beta
        self.alpha = alpha

        weights = signal_weights or {}
        self.w_share = weights.get("share", 0.45)
        self.w_freq = weights.get("frequency", 0.35)
        self.w_position = weights.get("position", 0.20)
        self.w_total = self.w_share + self.w_freq + self.w_position
        if self.w_total <= 0:
            raise ValueError("signal weights sum to zero")

        self._blocks: Dict[str, BlockMeta] = {}
        self._step: int = 0
        # Sliding window of (block_hash, workflow_id) for share_count
        self._access_window: deque = deque()
        # Running max of access_count for O(1) freq normalization
        # (avoids O(N) scan in compute_proxy)
        self._max_access_count: int = 1

    def record_request(
        self,
        block_hashes: List[str],
        workflow_id: str,
        block_idx_offset: int = 0,
        prefill_ms: float = 0.0,
    ) -> None:
        """Record block accesses for a new request.

        Args:
            block_hashes: hashes of blocks in this request's prefix
            workflow_id: identifier for the workflow/task this request belongs to
            block_idx_offset: starting index for position weighting
            prefill_ms: estimated prefill cost per block
        """
        for i, bhash in enumerate(block_hashes):
            if not bhash:
                continue
            meta = self._blocks.get(bhash)
            if meta is None:
                meta = BlockMeta(
                    block_hash=bhash,
                    block_idx=i + block_idx_offset,
                    prefill_ms=prefill_ms,
                )
                self._blocks[bhash] = meta
            meta.access_count += 1
            meta.workflow_ids.add(workflow_id)
            meta.last_access_step = self._step
            meta.prefill_ms = max(meta.prefill_ms, prefill_ms)

            # Update running max (O(1) instead of O(N) scan in compute_proxy)
            if meta.access_count > self._max_access_count:
                self._max_access_count = meta.access_count

            self._access_window.append((bhash, workflow_id))
            self._step += 1

        # Trim sliding window
        while len(self._access_window) > self.share_window:
            self._access_window.popleft()

    def compute_proxy(self, block_hash: str) -> float:
        """Compute bounded reuse likelihood proxy p_b in [0, 1].

        proxy_b = exp(-age / (kappa * B_G)) * normalized_signals
        """
        meta = self._blocks.get(block_hash)
        if meta is None:
            return 0.0

        age = self._step - meta.last_access_step
        kappa = 1.0
        decay = math.exp(-age / (kappa * self.capacity_blocks))

        # Share count from sliding window (causal)
        share_count = 0
        seen_workflows = set()
        for bh, wid in reversed(self._access_window):
            if bh == block_hash:
                seen_workflows.add(wid)
                if len(seen_workflows) >= self.share_count_cap:
                    break
        share_count = min(len(seen_workflows), self.share_count_cap)

        # Normalized signals — use running max for O(1) lookup
        max_freq = self._max_access_count
        freq_norm = min(1.0, meta.access_count / max(1, max_freq))
        share_norm = min(1.0, share_count / max(1, self.share_count_cap))

        # Position weight: early blocks have higher reuse
        position_norm = max(0.0, 1.0 - meta.block_idx / max(1, self.capacity_blocks))

        signal = (
            self.w_share * share_norm
            + self.w_freq * freq_norm
            + self.w_position * position_norm
        ) / self.w_total

        return decay * signal

    def compute_migration_value(
        self,
        block_hash: str,
        d2h_ms: float,
        h2d_ms: float,
        hold_cost_ms: float = 0.0,
    ) -> float:
        """Compute net migration value V_b.

        V_b = proxy_b * max(prefill_ms - h2d_ms, 0) - d2h_ms - hold_cost_ms

        V_b > 0 → migrate to CPU (worth storing for future reuse)
        V_b <= 0 → evict directly (not worth the D2H cost)
        """
        proxy = self.compute_proxy(block_hash)
        meta = self._blocks.get(block_hash)
        prefill_ms = meta.prefill_ms if meta else 0.0
        saved_prefill = max(prefill_ms - h2d_ms, 0.0)
        return proxy * saved_prefill - d2h_ms - hold_cost_ms

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tracked_blocks": len(self._blocks),
            "total_steps": self._step,
            "window_size": len(self._access_window),
        }


# ---------------------------------------------------------------------------
# Migration policy
# ---------------------------------------------------------------------------

class FlowCacheMigrationPolicy:
    """Selective migration policy: decide which blocks to store to CPU.

    Two modes:
      1. threshold: migrate if V_b > minimum_net_benefit_ms
      2. ratio: migrate top-K blocks by V_b (K = migrate_ratio * num_blocks)
    """

    def __init__(
        self,
        tracker: BlockAccessTracker,
        d2h_ms_per_block: float = 0.1,
        h2d_ms_per_block: float = 0.15,
        hold_cost_ms: float = 0.0,
        minimum_net_benefit_ms: float = 0.0,
        migrate_ratio: float = 0.5,
        mode: str = "threshold",
    ):
        self.tracker = tracker
        self.d2h_ms = d2h_ms_per_block
        self.h2d_ms = h2d_ms_per_block
        self.hold_cost_ms = hold_cost_ms
        self.minimum_net_benefit_ms = minimum_net_benefit_ms
        self.migrate_ratio = max(0.0, min(1.0, migrate_ratio))
        self.mode = mode

        # Audit counters
        self.total_blocks_seen: int = 0
        self.total_blocks_migrated: int = 0
        self.total_blocks_evicted: int = 0

    def select_blocks_for_migration(
        self,
        block_hashes: List[str],
        block_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Select which blocks to migrate to CPU.

        Returns:
            (migrate_ids, evict_ids) — GPU block IDs to migrate and to evict.
        """
        self.total_blocks_seen += len(block_hashes)

        # Compute V_b for each block
        values: List[Tuple[float, int, str]] = []
        for idx, bhash in enumerate(block_hashes):
            if not bhash or idx >= len(block_ids):
                continue
            v = self.tracker.compute_migration_value(
                bhash, self.d2h_ms, self.h2d_ms, self.hold_cost_ms
            )
            values.append((v, block_ids[idx], bhash))

        if not values:
            return [], list(block_ids)

        if self.mode == "ratio":
            # Migrate top-K by value.
            # When migrate_ratio is 0, K=0 → migrate nothing.
            # When migrate_ratio is 1, K=len(values) → migrate all.
            values.sort(key=lambda x: -x[0])
            k = int(len(values) * self.migrate_ratio)
            migrate_set = set(vid for _, vid, _ in values[:k])
        else:
            # Threshold: migrate if V_b > minimum_net_benefit_ms
            migrate_set = set(
                vid for v, vid, _ in values
                if v > self.minimum_net_benefit_ms
            )

        migrate_ids = [vid for vid in block_ids if vid in migrate_set]
        evict_ids = [vid for vid in block_ids if vid not in migrate_set]

        self.total_blocks_migrated += len(migrate_ids)
        self.total_blocks_evicted += len(evict_ids)
        return migrate_ids, evict_ids

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_blocks_seen": self.total_blocks_seen,
            "total_blocks_migrated": self.total_blocks_migrated,
            "total_blocks_evicted": self.total_blocks_evicted,
            "migration_rate": (
                self.total_blocks_migrated / max(1, self.total_blocks_seen)
            ),
            "policy_mode": self.mode,
            "migrate_ratio": self.migrate_ratio,
            "minimum_net_benefit_ms": self.minimum_net_benefit_ms,
        }


# ---------------------------------------------------------------------------
# FlowCacheConnector (vLLM KVConnectorBase_V1)
# ---------------------------------------------------------------------------

# Default D2H/H2D costs (ms per block) — calibrated on RTX 4090 / A100
DEFAULT_D2H_MS = 0.10
DEFAULT_H2D_MS = 0.15


def _make_flowcache_connector_cls():
    """Import and subclass SimpleCPUOffloadConnector lazily.

    This avoids import errors when vLLM is not installed (e.g., local Windows
    dev). The actual connector class is only needed on the cloud server.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
        SimpleCPUOffloadConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )

    class FlowCacheConnector(SimpleCPUOffloadConnector):
        """CPU KV cache offloading with FlowCache selective migration.

        Extends SimpleCPUOffloadConnector:
          - Always-migrate (parent): stores ALL finished-request blocks to CPU
          - Selective-migrate (this): stores only high-R blocks to CPU

        The selective decision is made in request_finished(), which filters
        block_ids by R-value before delegating to the parent scheduler.

        Configuration via kv_connector_extra_config:
          selective_migration: bool (default True)
          minimum_net_benefit_ms: float (default 0.0)
          migrate_ratio: float (default 0.5, used when mode="ratio")
          migration_mode: str ("threshold" | "ratio", default "threshold")
          d2h_ms_per_block: float (default 0.10)
          h2d_ms_per_block: float (default 0.15)
          share_window_accesses: int (default 1000)
          share_count_cap: int (default 8)
          beta: float (default 0.005)
          alpha: float (default 0.5)
        """

        def __init__(
            self,
            vllm_config: "VllmConfig",
            role: "KVConnectorRole",
            kv_cache_config: "KVCacheConfig",
        ):
            super().__init__(vllm_config, role, kv_cache_config)

            extra_config = (
                self._kv_transfer_config.kv_connector_extra_config or {}
            )

            # Parse selective_migration — handles both bool and string types
            # (vLLM serializes extra_config values as strings)
            _sel = extra_config.get("selective_migration", True)
            if isinstance(_sel, str):
                self._selective_enabled = _sel.lower() in ("true", "1", "yes")
            else:
                self._selective_enabled = bool(_sel)

            # Estimated prefill cost per block (ms). Used as the "saved" cost
            # when a block is a cache hit. Without this, V_b is always negative
            # and no blocks would ever be migrated.
            # Default: 5.0 ms/block (Qwen2.5-7B, block_size=16, RTX 4090).
            self._prefill_ms_per_block = float(
                extra_config.get("prefill_ms_per_block", 5.0)
            )

            if not self._selective_enabled:
                logger.info(
                    "FlowCacheConnector: selective_migration=False, "
                    "behaving as SimpleCPUOffloadConnector (always-migrate)."
                )
                self._policy: Optional[FlowCacheMigrationPolicy] = None
                self._tracker: Optional[BlockAccessTracker] = None
                return

            if role != KVConnectorRole.SCHEDULER:
                self._policy = None
                self._tracker = None
                return

            # --- Scheduler-side: initialize tracker and policy ---
            gpu_blocks = kv_cache_config.num_blocks if kv_cache_config else 1024
            cpu_bytes = int(
                extra_config.get("cpu_bytes_to_use", 8 * (1024**3))
            )
            block_bytes = (
                kv_cache_config.kv_cache_tensors[0].size
                // max(1, kv_cache_config.num_blocks)
            ) if kv_cache_config and kv_cache_config.kv_cache_tensors else 917504

            signal_weights = {
                "share": float(extra_config.get("w_share", 0.45)),
                "frequency": float(extra_config.get("w_freq", 0.35)),
                "position": float(extra_config.get("w_position", 0.20)),
            }

            self._tracker = BlockAccessTracker(
                capacity_blocks=gpu_blocks,
                share_window_accesses=int(
                    extra_config.get("share_window_accesses", 1000)
                ),
                share_count_cap=int(extra_config.get("share_count_cap", 8)),
                beta=float(extra_config.get("beta", 0.005)),
                alpha=float(extra_config.get("alpha", 0.5)),
                signal_weights=signal_weights,
            )

            self._policy = FlowCacheMigrationPolicy(
                tracker=self._tracker,
                d2h_ms_per_block=float(
                    extra_config.get("d2h_ms_per_block", DEFAULT_D2H_MS)
                ),
                h2d_ms_per_block=float(
                    extra_config.get("h2d_ms_per_block", DEFAULT_H2D_MS)
                ),
                hold_cost_ms=float(extra_config.get("hold_cost_ms", 0.0)),
                minimum_net_benefit_ms=float(
                    extra_config.get("minimum_net_benefit_ms", 0.0)
                ),
                migrate_ratio=float(extra_config.get("migrate_ratio", 0.5)),
                mode=str(extra_config.get("migration_mode", "threshold")),
            )

            # Per-request block hash cache (for request_finished lookup)
            self._request_block_hashes: Dict[str, List[Any]] = {}
            self._request_workflow_ids: Dict[str, str] = {}

            # Monkey-patch SimpleCPUOffloadScheduler.prepare_store_specs so
            # FlowCache can filter the blocks that SimpleCPUOffload would
            # migrate to CPU. Without this patch, request_finished() is the
            # only override point, but SimpleCPUOffloadScheduler.request_finished()
            # only cleans metadata and never schedules migration; the real
            # migration decision happens in _prepare_eager_store_specs().
            if self.scheduler_manager is not None:
                self._patch_prepare_store_specs()

            logger.info(
                "FlowCacheConnector: selective_migration=True, "
                "gpu_blocks=%d, block_bytes=%d, mode=%s, "
                "minimum_net_benefit_ms=%.3f, migrate_ratio=%.2f",
                gpu_blocks,
                block_bytes,
                self._policy.mode,
                self._policy.minimum_net_benefit_ms,
                self._policy.migrate_ratio,
            )

        # --- Selective-migration patch for SimpleCPUOffloadScheduler ---

        def _patch_prepare_store_specs(self) -> None:
            """Patch scheduler_manager.prepare_store_specs to filter migrations.

            SimpleCPUOffloadScheduler performs migration in
            _prepare_eager_store_specs(), not in request_finished(). To make
            selective migration effective we wrap prepare_store_specs and drop
            low-value blocks from the store list before the store event is
            created.
            """
            sm = self.scheduler_manager
            if sm is None or hasattr(sm, "_flowcache_store_specs_patched"):
                return
            sm._flowcache_store_specs_patched = True
            orig_prepare = sm.prepare_store_specs

            def _wrapped_prepare(scheduler_output: "SchedulerOutput"):
                return self._selective_store_specs(orig_prepare, scheduler_output)

            sm.prepare_store_specs = _wrapped_prepare

        def _selective_store_specs(
            self,
            orig_prepare: Any,
            scheduler_output: "SchedulerOutput",
        ) -> Tuple[List[int], List[int], List[str]]:
            """Wrap prepare_store_specs and filter low-value blocks."""
            store_gpu, store_cpu, store_req_ids = orig_prepare(scheduler_output)
            if (
                not store_gpu
                or not self._selective_enabled
                or self._policy is None
                or self._tracker is None
            ):
                return store_gpu, store_cpu, store_req_ids

            sm = self.scheduler_manager
            if sm is None:
                return store_gpu, store_cpu, store_req_ids

            from vllm.v1.core.kv_cache_utils import get_block_hash

            keep_gpu: List[int] = []
            keep_cpu: List[int] = []
            free_cpu_blocks: List[Any] = []
            dropped_gpu: List[int] = []

            for gpu_id, cpu_id in zip(store_gpu, store_cpu):
                # vLLM v0.26.0+: BlockPool.blocks is a list indexed by block_id.
                gpu_block = (
                    sm._gpu_block_pool.blocks[gpu_id]
                    if 0 <= gpu_id < len(sm._gpu_block_pool.blocks)
                    else None
                )
                if gpu_block is None or gpu_block.block_hash is None:
                    keep_gpu.append(gpu_id)
                    keep_cpu.append(cpu_id)
                    continue

                # Strip group id to match tracker keys (bytes)
                base_hash = get_block_hash(gpu_block.block_hash)
                # Tracker keys are the raw block-hash bytes from request.block_hashes
                v = self._policy.tracker.compute_migration_value(
                    base_hash,
                    self._policy.d2h_ms,
                    self._policy.h2d_ms,
                    self._policy.hold_cost_ms,
                )
                if v > self._policy.minimum_net_benefit_ms:
                    keep_gpu.append(gpu_id)
                    keep_cpu.append(cpu_id)
                else:
                    cpu_block = (
                        sm.cpu_block_pool.blocks[cpu_id]
                        if 0 <= cpu_id < len(sm.cpu_block_pool.blocks)
                        else None
                    )
                    if cpu_block is not None:
                        free_cpu_blocks.append(cpu_block)
                    dropped_gpu.append(gpu_id)

            if dropped_gpu:
                # Release CPU blocks that were allocated but will not be
                # transferred; reset their hashes so they are not mistaken for
                # valid cached data by the load path.
                for blk in free_cpu_blocks:
                    blk.reset_hash()
                sm.cpu_block_pool.free_blocks(free_cpu_blocks)
                sm._in_flight_store_gpu_blocks.difference_update(dropped_gpu)
                # Release GPU touch references (fix GPU ref leak)
                for gid in dropped_gpu:
                    if 0 <= gid < len(sm._gpu_block_pool.blocks):
                        gb = sm._gpu_block_pool.blocks[gid]
                        if hasattr(gb, '_touch_count') and gb._touch_count > 0:
                            gb._touch_count -= 1
                logger.debug(
                    "FlowCacheConnector: dropped %d low-value blocks from store",
                    len(dropped_gpu),
                )

            if not keep_gpu:
                return [], [], []
            return keep_gpu, keep_cpu, store_req_ids

        # --- Intercept request lifecycle for tracking ---

        def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
        ) -> Tuple[Optional[int], bool]:
            """Check CPU cache for hits (delegates to parent)."""
            # Record block hashes for this request (for tracking)
            if self._tracker is not None and hasattr(request, "block_hashes"):
                req_id = request.request_id
                # Extract workflow/task ID from request_id or prompt
                workflow_id = self._extract_workflow_id(request)
                self._request_block_hashes[req_id] = list(
                    request.block_hashes
                ) if request.block_hashes else []
                self._request_workflow_ids[req_id] = workflow_id

            return super().get_num_new_matched_tokens(
                request, num_computed_tokens
            )

        def update_state_after_alloc(
            self,
            request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int,
        ) -> None:
            """Record block access after allocation (updates tracker)."""
            # Record access in tracker for R-value computation
            if self._tracker is not None:
                req_id = request.request_id
                block_hashes = self._request_block_hashes.get(req_id, [])
                workflow_id = self._request_workflow_ids.get(req_id, req_id)

                if block_hashes:
                    self._tracker.record_request(
                        block_hashes=block_hashes,
                        workflow_id=workflow_id,
                        prefill_ms=self._prefill_ms_per_block,
                    )

            super().update_state_after_alloc(
                request, blocks, num_external_tokens
            )

        def request_finished(
            self,
            request: "Request",
            block_ids: List[int],
        ) -> Tuple[bool, Optional[Dict[str, Any]]]:
            """Selective migration: only store high-R blocks to CPU.

            This is the KEY override. The parent (SimpleCPUOffloadConnector)
            stores ALL blocks to CPU. FlowCache filters block_ids by R-value:
              - High-R blocks → store to CPU (migrate)
              - Low-R (new) blocks → evict directly (no D2H cost)
              - Cache-hit blocks → always migrate (proven reuse value)

            block_ids = [prefix_0 ... prefix_N, generated_0 ... generated_M].
            We evaluate only prefix blocks that have hashes in the tracker
            (= newly computed). Cache-hit prefix blocks (no hash tracked)
            are migrated unconditionally. Generated blocks are not migrated.
            """
            if not self._selective_enabled or self._policy is None:
                return super().request_finished(request, block_ids)

            req_id = request.request_id
            block_hashes = self._request_block_hashes.get(req_id, [])

            if not block_hashes:
                # No hashes at all — migrate everything (fallback to parent)
                result = super().request_finished(request, block_ids)
                self._cleanup_request(req_id)
                return result

            # Determine prefix count from request metadata
            # block_ids = [prefix_blocks..., generated_blocks...]
            # Use num_prompt_tokens or num_computed_tokens to split.
            prompt_tokens = getattr(request, 'num_prompt_tokens', 0)
            block_size = getattr(request, 'block_size', 16)
            if hasattr(self, '_vllm_config'):
                block_size = self._vllm_config.scheduler_config.block_size
            n_prefix = prompt_tokens // max(1, block_size)
            n_prefix = min(n_prefix, len(block_ids))
            prefix_ids = block_ids[:n_prefix]

            # Split prefix blocks: hashed (newly computed) vs unhashed (cache hits).
            # Cache-hit blocks have proven reuse value → always migrate.
            hash_set = set(block_hashes)
            hashed_ids: List[int] = []
            cached_ids: List[int] = []
            for bid, bhash in zip(prefix_ids, block_hashes[:len(prefix_ids)]):
                if bhash and bhash in hash_set:
                    hashed_ids.append(bid)
                else:
                    cached_ids.append(bid)
            # Remaining prefix blocks beyond block_hashes length are cache hits
            if len(prefix_ids) > len(block_hashes):
                cached_ids.extend(prefix_ids[len(block_hashes):])

            # Evaluate newly-computed blocks for selective migration
            migrate_ids, evict_ids = self._policy.select_blocks_for_migration(
                block_hashes=[h for h in block_hashes[:len(hashed_ids)] if h],
                block_ids=hashed_ids,
            )

            # Always migrate cache-hit blocks (they've proven their value)
            final_migrate = list(cached_ids) + migrate_ids

            if not final_migrate:
                self._cleanup_request(req_id)
                return False, None

            # Store high-R + cache-hit blocks to CPU via parent
            result = super().request_finished(request, final_migrate)
            self._cleanup_request(req_id)
            return result

        def _cleanup_request(self, req_id: str) -> None:
            """Clean up per-request tracking state."""
            self._request_block_hashes.pop(req_id, None)
            self._request_workflow_ids.pop(req_id, None)

        def _extract_workflow_id(self, request: "Request") -> str:
            """Extract workflow ID for share_count clustering.

            vLLM assigns its own internal request_ids (integer strings) that
            do NOT carry task metadata. Since we cannot access the original
            ServingRequest's task_id from inside the connector, we use the
            request_id directly as the workflow_id.

            This means share_count counts distinct requests that touched a
            block — a block shared by N requests (e.g. system prompt) gets
            share_count = min(N, share_count_cap), which is still a strong
            signal for reuse value. The per-task bootstrap CI is computed
            in run_closed_loop.py where the original task_id is available
            via index-based output matching.
            """
            return request.request_id

        def get_migration_stats(self) -> Dict[str, Any]:
            """Return selective migration audit statistics."""
            if self._policy is None:
                return {"selective_migration": False}
            stats = self._policy.get_stats()
            stats["selective_migration"] = True
            stats["tracker"] = self._tracker.get_stats() if self._tracker else {}
            return stats

    return FlowCacheConnector


# Cached connector class (created on first import)
_FlowCacheConnectorCls = None


def get_flowcache_connector_class():
    """Get or create the FlowCacheConnector class (lazy vLLM import)."""
    global _FlowCacheConnectorCls
    if _FlowCacheConnectorCls is None:
        _FlowCacheConnectorCls = _make_flowcache_connector_cls()
    return _FlowCacheConnectorCls


# vLLM looks for this top-level name when loading the connector
# We use a proxy that defers the actual class creation
class FlowCacheConnector:
    """Proxy class — delegates to the real connector after lazy vLLM import.

    vLLM instantiates connectors by calling the class with (vllm_config,
    role, kv_cache_config). We forward to the lazily-created real class.

    vLLM also queries class-level methods (e.g. requires_piecewise_for_cudagraph)
    before instantiation. We lazily create the real class and delegate those
    calls as well.
    """

    _real_cls = None

    @classmethod
    def _ensure_real_cls(cls):
        if cls._real_cls is None:
            cls._real_cls = _make_flowcache_connector_cls()
        return cls._real_cls

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        real_cls = cls._ensure_real_cls()
        return real_cls.get_required_kvcache_layout(vllm_config)

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        real_cls = cls._ensure_real_cls()
        return real_cls.requires_piecewise_for_cudagraph(extra_config)

    @classmethod
    def build_kv_connector_stats(cls, data: dict[str, Any] | None = None):
        real_cls = cls._ensure_real_cls()
        return real_cls.build_kv_connector_stats(data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: Any,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Any | None:
        real_cls = cls._ensure_real_cls()
        return real_cls.build_prom_metrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def __new__(
        cls,
        vllm_config: "VllmConfig",
        role: Any,
        kv_cache_config: Any,
    ):
        real_cls = cls._ensure_real_cls()
        return real_cls(vllm_config, role, kv_cache_config)
