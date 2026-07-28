"""
Lossless Two-Tier Cache Manager
================================
无损三层动作 cache manager（动作空间 A₀）：

  GPU BF16  ↔  CPU pinned BF16  ↔  Evict

提供底层存储原语，不包含决策逻辑（由 controller.py 调用）。

跟踪指标（与 baseline 类一致）：
  hits, misses, evictions, saved_prefill_ms, miss_cost_ms
额外跟踪：
  gpu_evictions, cpu_evictions
  migrate_to_cpu_count, restore_to_gpu_count
  migrate_ms_total, restore_ms_total  (开销)

容量以 block 数为单位：
  gpu_capacity_blocks: GPU 可容纳的 block 数
  cpu_capacity_blocks: CPU pinned buffer 可容纳的 block 数（-1 = 无限制）
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set


class LosslessCacheManager:
    """
    Two-tier lossless cache: GPU BF16 ↔ CPU pinned BF16 ↔ Evict.

    The manager provides storage primitives; eviction/migration decisions
    are made by the controller (controller.py). The controller calls:
      - lookup_gpu / lookup_cpu: check presence
      - admit_gpu: insert a block into GPU (must have space)
      - migrate_to_cpu: move a block from GPU to CPU
      - restore_to_gpu: move a block from CPU to GPU
      - evict_from_gpu / evict_from_cpu: remove a block

    All access accounting (hits, misses, saved/miss cost) is done by the
    controller via record_hit() / record_miss() to keep the manager focused
    on storage state.
    """

    def __init__(self,
                 gpu_capacity_blocks: int,
                 cpu_capacity_blocks: int = -1,
                 block_bytes: int = 917504,
                 cost_model: Optional[Dict] = None):
        """
        Args:
            gpu_capacity_blocks: GPU 可容纳的 block 数
            cpu_capacity_blocks: CPU pinned buffer block 数（-1 = 无限制）
            block_bytes: 每个 block 的字节数（用于成本估算）
            cost_model: 冻结的成本模型（来自 cost-model.json）
        """
        self.gpu_capacity = max(1, gpu_capacity_blocks)
        self.cpu_capacity = cpu_capacity_blocks  # -1 = unlimited
        self.block_bytes = block_bytes
        self.cost_model = cost_model or {}
        # The model and block size are frozen for one run.  Resolve transfer
        # estimates once instead of re-sorting calibration samples on every
        # migration/restoration decision.
        self._d2h_ms_per_block = self._estimate_transfer_ms("d2h_migrate")
        self._h2d_ms_per_block = self._estimate_transfer_ms("h2d_restore")

        # GPU pool: block_hash -> metadata.  OrderedDict makes the first item
        # the exact LRU victim, so eviction is O(1) rather than a full scan.
        self.gpu_cache: "OrderedDict[str, Dict]" = OrderedDict()
        # CPU insertion order is retained for diagnostics/fallbacks.  The
        # value-aware controller maintains a separate lazy heap.
        self.cpu_cache: "OrderedDict[str, Dict]" = OrderedDict()

        # 访问统计（与 baseline 一致）
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

        # 迁移/恢复统计
        self.gpu_evictions: int = 0
        self.cpu_evictions: int = 0
        self.migrate_to_cpu_count: int = 0
        self.restore_to_gpu_count: int = 0
        self.migrate_ms_total: float = 0.0
        self.restore_ms_total: float = 0.0

        # 时钟（用于 age 计算）
        self._clock: int = 0

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_gpu(self, block_hash: str) -> bool:
        """检查 block 是否在 GPU 池中。"""
        return block_hash in self.gpu_cache

    def lookup_cpu(self, block_hash: str) -> bool:
        """检查 block 是否在 CPU 池中。"""
        return block_hash in self.cpu_cache

    def lookup(self, block_hash: str) -> str:
        """
        检查 block 位置。

        Returns:
            "gpu" | "cpu" | "miss"
        """
        if block_hash in self.gpu_cache:
            return "gpu"
        if block_hash in self.cpu_cache:
            return "cpu"
        return "miss"

    # ------------------------------------------------------------------
    # GPU pool operations
    # ------------------------------------------------------------------

    def admit_gpu(self, block_hash: str, metadata: Optional[Dict] = None) -> None:
        """将 block 插入 GPU 池（调用方需确保有空间）。"""
        self.gpu_cache[block_hash] = metadata or {
            "parent_hash": "",
            "last_access": self._clock,
            "prefill_ms": 0.0,
            "block_idx": 0,
        }
        self.gpu_cache.move_to_end(block_hash)
        self._clock += 1

    def touch_gpu(self, block_hash: str) -> None:
        """更新 GPU 池中 block 的 last_access。"""
        if block_hash in self.gpu_cache:
            self.gpu_cache[block_hash]["last_access"] = self._clock
            self.gpu_cache.move_to_end(block_hash)
            self._clock += 1

    def evict_from_gpu(self, block_hash: str) -> None:
        """从 GPU 池淘汰 block。"""
        if block_hash in self.gpu_cache:
            del self.gpu_cache[block_hash]
            self.gpu_evictions += 1
            self.evictions += 1

    def gpu_full(self) -> bool:
        """GPU 池是否已满。"""
        return len(self.gpu_cache) >= self.gpu_capacity

    def gpu_size(self) -> int:
        """GPU 池当前 block 数。"""
        return len(self.gpu_cache)

    def gpu_available(self) -> int:
        """GPU 池剩余可用 block 数。"""
        return max(0, self.gpu_capacity - len(self.gpu_cache))

    # ------------------------------------------------------------------
    # CPU pool operations
    # ------------------------------------------------------------------

    def admit_cpu(self, block_hash: str, metadata: Optional[Dict] = None) -> None:
        """将 block 插入 CPU 池（调用方需确保有空间）。"""
        self.cpu_cache[block_hash] = metadata or {
            "parent_hash": "",
            "last_access": self._clock,
            "prefill_ms": 0.0,
            "block_idx": 0,
        }
        self.cpu_cache.move_to_end(block_hash)
        self._clock += 1

    def touch_cpu(self, block_hash: str) -> None:
        """更新 CPU 池中 block 的 last_access。"""
        if block_hash in self.cpu_cache:
            self.cpu_cache[block_hash]["last_access"] = self._clock
            self.cpu_cache.move_to_end(block_hash)
            self._clock += 1

    def evict_from_cpu(self, block_hash: str) -> None:
        """从 CPU 池淘汰 block。"""
        if block_hash in self.cpu_cache:
            del self.cpu_cache[block_hash]
            self.cpu_evictions += 1
            self.evictions += 1

    def cpu_full(self) -> bool:
        """CPU 池是否已满（-1 = 无限制，永远不满）。"""
        if self.cpu_capacity < 0:
            return False
        return len(self.cpu_cache) >= self.cpu_capacity

    def cpu_size(self) -> int:
        """CPU 池当前 block 数。"""
        return len(self.cpu_cache)

    # ------------------------------------------------------------------
    # Migration / Restoration (with cost tracking)
    # ------------------------------------------------------------------

    def migrate_to_cpu(self, block_hash: str) -> float:
        """
        将 block 从 GPU 迁移到 CPU（D2H）。

        Returns:
            迁移耗时 (ms)
        """
        if block_hash not in self.gpu_cache:
            return 0.0

        # 估算 D2H 成本
        migrate_ms = self._estimate_d2h_ms()

        # 移动 block
        metadata = self.gpu_cache.pop(block_hash)
        if not self.cpu_full():
            self.cpu_cache[block_hash] = metadata
            self.cpu_cache.move_to_end(block_hash)
        else:
            # CPU 也满了，直接淘汰
            self.cpu_evictions += 1
            self.evictions += 1

        self.migrate_to_cpu_count += 1
        self.migrate_ms_total += migrate_ms
        return migrate_ms

    def restore_to_gpu(self, block_hash: str) -> float:
        """
        将 block 从 CPU 恢复到 GPU（H2D）。

        调用方需确保 GPU 有空间（或先淘汰）。

        Returns:
            恢复耗时 (ms)
        """
        if block_hash not in self.cpu_cache:
            return 0.0

        # 估算 H2D 成本
        restore_ms = self._estimate_h2d_ms()

        # 移动 block
        metadata = self.cpu_cache.pop(block_hash)
        self.gpu_cache[block_hash] = metadata
        self.gpu_cache.move_to_end(block_hash)

        self.restore_to_gpu_count += 1
        self.restore_ms_total += restore_ms
        return restore_ms

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_d2h_ms(self) -> float:
        """从成本模型估算 D2H 迁移一个 block 的 ms。"""
        return self._d2h_ms_per_block

    def _estimate_h2d_ms(self) -> float:
        """从成本模型估算 H2D 恢复一个 block 的 ms。"""
        return self._h2d_ms_per_block

    def estimate_d2h_ms(self) -> float:
        """Public read-only D2H estimate for controller admission decisions."""
        return self._estimate_d2h_ms()

    def estimate_h2d_ms(self) -> float:
        """Public read-only H2D estimate for controller admission decisions."""
        return self._estimate_h2d_ms()

    def _estimate_transfer_ms(self, section_name: str) -> float:
        """Estimate one transfer without allowing an invalid negative cost.

        The calibration file contains exact-size measurements as well as a
        global linear regression.  A negative fitted intercept made the old
        one-block D2H estimate negative.  Exact measured medians now take
        precedence; intermediate sizes use piecewise interpolation; only
        out-of-range sizes fall back to the clamped regression.
        """
        section = self.cost_model.get(section_name, {})
        samples = []
        for sample in section.get("samples", []):
            try:
                nbytes = int(sample["bytes"])
                median = float(sample["median"])
            except (KeyError, TypeError, ValueError):
                continue
            if nbytes > 0 and median >= 0:
                samples.append((nbytes, median))
        samples.sort()

        for nbytes, median in samples:
            if nbytes == self.block_bytes:
                return median

        if len(samples) >= 2:
            for (x0, y0), (x1, y1) in zip(samples, samples[1:]):
                if x0 < self.block_bytes < x1:
                    ratio = (self.block_bytes - x0) / (x1 - x0)
                    return max(0.0, y0 + ratio * (y1 - y0))

        intercept = float(section.get("intercept", 0.0) or 0.0)
        slope = float(section.get("slope_per_byte", 0.0) or 0.0)
        fitted = intercept + slope * self.block_bytes
        if samples and fitted <= 0:
            # A non-positive extrapolation is physically impossible.  Use the
            # nearest measured point rather than reporting "negative time".
            nearest = min(samples, key=lambda item: abs(item[0] - self.block_bytes))
            return nearest[1]
        return max(0.0, fitted)

    def estimate_prefill_ms(self, parent_length_tokens: int = 0,
                            concurrency: int = 1) -> float:
        """
        从成本模型估算 prefill 一个 block 的 ms。

        Args:
            parent_length_tokens: 父前缀长度（tokens）
            concurrency: 并发度
        """
        prefill = self.cost_model.get("prefill", {}).get("params", {})
        # 查找最接近的 parent_length
        if not prefill:
            # fallback: 0.02 ms/token × 16 tokens
            return 0.02 * 16

        # 找到最接近的 parent_length key
        available_plens = sorted(int(k) for k in prefill.keys())
        closest_plen = min(available_plens,
                          key=lambda p: abs(p - parent_length_tokens))
        plen_dict = prefill[str(closest_plen)] if str(closest_plen) in prefill else prefill[available_plens[0]]

        # 找到最接近的 concurrency
        available_concs = sorted(int(k) for k in plen_dict.keys())
        closest_conc = min(available_concs,
                          key=lambda c: abs(c - concurrency))
        return plen_dict[str(closest_conc)]["median"]

    # ------------------------------------------------------------------
    # Accounting (called by controller)
    # ------------------------------------------------------------------

    def record_hit(self, prefill_ms: float) -> None:
        """记录一次命中。"""
        self.hits += 1
        self.saved_prefill_ms += prefill_ms

    def record_miss(self, prefill_ms: float) -> None:
        """记录一次未命中。"""
        self.misses += 1
        self.miss_cost_ms += prefill_ms

    # ------------------------------------------------------------------
    # Snapshot / reset
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """返回所有统计指标。"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "saved_prefill_ms": self.saved_prefill_ms,
            "miss_cost_ms": self.miss_cost_ms,
            "gpu_evictions": self.gpu_evictions,
            "cpu_evictions": self.cpu_evictions,
            "migrate_to_cpu_count": self.migrate_to_cpu_count,
            "restore_to_gpu_count": self.restore_to_gpu_count,
            "migrate_ms_total": self.migrate_ms_total,
            "restore_ms_total": self.restore_ms_total,
            "migrate_bytes_total": self.migrate_to_cpu_count * self.block_bytes,
            "restore_bytes_total": self.restore_to_gpu_count * self.block_bytes,
            "gpu_size": self.gpu_size(),
            "cpu_size": self.cpu_size(),
        }

    def get_gpu_blocks(self) -> List[str]:
        """返回 GPU 池中所有 block_hash（用于 controller 决策）。"""
        return list(self.gpu_cache.keys())

    def get_cpu_blocks(self) -> List[str]:
        """返回 CPU 池中所有 block_hash。"""
        return list(self.cpu_cache.keys())

    def get_block_metadata(self, block_hash: str) -> Optional[Dict]:
        """获取 block 的元数据（从 GPU 或 CPU 池）。"""
        if block_hash in self.gpu_cache:
            return self.gpu_cache[block_hash]
        if block_hash in self.cpu_cache:
            return self.cpu_cache[block_hash]
        return None

    def reset(self) -> None:
        """重置所有状态（用于多次运行）。"""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.saved_prefill_ms = 0.0
        self.miss_cost_ms = 0.0
        self.gpu_evictions = 0
        self.cpu_evictions = 0
        self.migrate_to_cpu_count = 0
        self.restore_to_gpu_count = 0
        self.migrate_ms_total = 0.0
        self.restore_ms_total = 0.0
        self._clock = 0
