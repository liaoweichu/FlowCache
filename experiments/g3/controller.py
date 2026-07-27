"""
FlowCache-Lossless Controller
==============================
动作空间 A₀：GPU BF16 ↔ CPU BF16 ↔ Evict（无损，无量化）。

决策规则：
  score_b = R_b - λ · hold_cost_b
  - score 高的 block 保留在 GPU
  - score 中等的 block 迁移到 CPU（若 CPU 有空间）
  - score 低的 block 直接淘汰

安全水位：保留 safety_margin（默认 10%）的 GPU 容量作 buffer。
回退机制：controller 异常时回退 SizeCost-LRU。

与 baseline 类接口兼容：access() 方法记录 hits/misses/saved/miss cost。
"""

import logging
from typing import Dict, List, Optional

from cache_manager import LosslessCacheManager
from reuse_estimator import HeuristicReuseEstimator, create_estimator

logger = logging.getLogger(__name__)


class FlowCacheLosslessController:
    """
    FlowCache-Lossless controller (action space A₀).

    Open-loop replay interface compatible with baseline cache classes.
    The controller decides, for each block access:
      1. GPU hit → keep, update last_access
      2. CPU hit → restore to GPU (H2D), then GPU hit
      3. Miss → admit to GPU, possibly evicting/migrating others

    Eviction policy: value-aware (G3′ tiered)
      - Compute R_b for all GPU blocks
      - Migrate blocks with R > migrate_threshold (0.01) to CPU
      - When CPU is full, evict the lowest-R CPU block to make room
      - Evict blocks with R <= migrate_threshold directly from GPU
      - Safety margin: keep 5% GPU capacity free (G3′: reduced from 10%)
    """

    def __init__(self,
                 gpu_capacity_blocks: int,
                 cpu_capacity_blocks: int = -1,
                 block_bytes: int = 917504,
                 cost_model: Optional[Dict] = None,
                 reuse_estimator_config: Optional[Dict] = None,
                 safety_margin: float = 0.05,
                 score_lambda: float = 0.1,
                 fallback: str = "sizecost",
                 migrate_threshold: float = 0.01):
        """
        Args:
            gpu_capacity_blocks: GPU 可容纳的 block 数
            cpu_capacity_blocks: CPU pinned buffer block 数（-1 = 无限制）
            block_bytes: 每个 block 的字节数
            cost_model: 冻结的成本模型（来自 cost-model.json）
            reuse_estimator_config: heuristic 估计器配置
            safety_margin: GPU 安全水位（0.05 = 保留 5% 容量）
            score_lambda: hold_cost 权重
            fallback: 回退策略名称（"sizecost" | "lru"）
            migrate_threshold: GPU→CPU 迁移的 R 阈值（G3′: 0.01，原 G3 误设为 0.1）
        """
        # 有效 GPU 容量 = 总容量 × (1 - safety_margin)
        effective_capacity = max(1, int(gpu_capacity_blocks * (1 - safety_margin)))
        self.effective_gpu_capacity = effective_capacity
        self.total_gpu_capacity = gpu_capacity_blocks
        self.safety_margin = safety_margin
        self.score_lambda = score_lambda
        self.fallback = fallback
        self.block_bytes = block_bytes
        self.migrate_threshold = migrate_threshold

        # 初始化 cache manager
        self.manager = LosslessCacheManager(
            gpu_capacity_blocks=effective_capacity,
            cpu_capacity_blocks=cpu_capacity_blocks,
            block_bytes=block_bytes,
            cost_model=cost_model or {},
        )

        # 初始化 reuse 估计器
        estimator_type = "heuristic"
        if reuse_estimator_config and "type" in reuse_estimator_config:
            estimator_type = reuse_estimator_config["type"]
        self.estimator = create_estimator(
            estimator_type, reuse_estimator_config or {}
        )

        # 回退计数
        self.fallback_count: int = 0
        self._clock: int = 0

    # ------------------------------------------------------------------
    # Core access method (compatible with baseline interface)
    # ------------------------------------------------------------------

    def access(self,
               block_hash: str,
               parent_hash: str = "",
               prefill_ms: float = 0.0,
               block_idx: int = 0,
               share_count: int = 0,
               age: Optional[int] = None) -> bool:
        """
        Access a block. Returns True if hit (GPU or CPU restore).

        Args:
            block_hash: block 哈希
            parent_hash: 父 block 哈希（用于 parent chain）
            prefill_ms: 该 block 的 prefill 成本（用于 saved/miss cost 统计）
            block_idx: 前缀位置（用于 R 估计）
            share_count: 共享 workflow 数（用于 R 估计）
            age: 自上次访问的步数（None = 自动计算）

        Returns:
            True if GPU hit or CPU restore hit; False if miss.
        """
        try:
            return self._access_impl(
                block_hash, parent_hash, prefill_ms,
                block_idx, share_count, age
            )
        except Exception as e:
            logger.warning(f"Controller exception, falling back to {self.fallback}: {e}")
            self.fallback_count += 1
            return self._fallback_access(block_hash, prefill_ms)

    def _access_impl(self,
                     block_hash: str,
                     parent_hash: str,
                     prefill_ms: float,
                     block_idx: int,
                     share_count: int,
                     age: Optional[int]) -> bool:
        """实际访问逻辑。"""
        location = self.manager.lookup(block_hash)

        if location == "gpu":
            # GPU hit
            self.manager.record_hit(prefill_ms)
            self.manager.touch_gpu(block_hash)
            # 更新元数据
            meta = self.manager.gpu_cache[block_hash]
            meta["block_idx"] = block_idx
            meta["share_count"] = share_count
            meta["last_access"] = self._clock
            self._clock += 1
            return True

        if location == "cpu":
            # CPU hit → restore to GPU
            # 需要 GPU 有空间
            self._ensure_gpu_space(1)
            restore_ms = self.manager.restore_to_gpu(block_hash)
            self.manager.record_hit(prefill_ms)
            # 恢复后 prefill 节省 - restore 开销
            # 注意：saved_prefill_ms 记录的是"如果没有缓存需要重新 prefill 的成本"
            # restore 开销单独记录在 restore_ms_total 中
            self.manager.touch_gpu(block_hash)
            meta = self.manager.gpu_cache[block_hash]
            meta["block_idx"] = block_idx
            meta["share_count"] = share_count
            meta["last_access"] = self._clock
            self._clock += 1
            return True

        # Miss
        self.manager.record_miss(prefill_ms)
        self._ensure_gpu_space(1)
        self.manager.admit_gpu(block_hash, {
            "parent_hash": parent_hash,
            "last_access": self._clock,
            "prefill_ms": prefill_ms,
            "block_idx": block_idx,
            "share_count": share_count,
        })
        self._clock += 1
        return False

    # ------------------------------------------------------------------
    # Eviction / migration decision
    # ------------------------------------------------------------------

    def _ensure_gpu_space(self, needed: int) -> None:
        """
        确保 GPU 有足够空间（分层淘汰/迁移）。

        G3′ 分层决策（修复原 G3 的 CPU 层未使用问题）：
          1. 选出 GPU 中 score 最低的 victim
          2. 若 victim_r > migrate_threshold（0.01）：
             - CPU 有空间 → 迁移到 CPU
             - CPU 满 → 先淘汰 CPU 中 R 最低的块腾位，再迁移
          3. 若 victim_r ≤ migrate_threshold → 直接从 GPU 淘汰
        """
        while self.manager.gpu_size() + needed > self.manager.gpu_capacity:
            if not self.manager.gpu_cache:
                break
            # 选择淘汰/迁移目标
            victim = self._select_victim()
            if victim is None:
                break

            # 决策：迁移到 CPU 还是直接淘汰
            victim_meta = self.manager.gpu_cache.get(victim, {})
            victim_age = self._clock - victim_meta.get("last_access", 0)
            victim_share = victim_meta.get("share_count", 0)
            victim_idx = victim_meta.get("block_idx", 0)
            victim_r = self.estimator.estimate(
                age=victim_age,
                share_count=victim_share,
                block_idx=victim_idx,
            )

            # R > 阈值 → 迁移到 CPU（分层：CPU 满时先淘汰 CPU 最低 R 块）
            if victim_r > self.migrate_threshold:
                if self.manager.cpu_full():
                    # CPU 满，淘汰 CPU 中 R 最低的块腾位
                    cpu_victim = self._select_cpu_victim()
                    if cpu_victim is not None:
                        self.manager.evict_from_cpu(cpu_victim)
                # 再次检查（CPU 可能仍满且无法腾位，此时 migrate_to_cpu 内部会直接淘汰）
                if not self.manager.cpu_full():
                    self.manager.migrate_to_cpu(victim)
                else:
                    # CPU 容量为 0 或无法腾位，直接从 GPU 淘汰
                    self.manager.evict_from_gpu(victim)
            else:
                # R 过低，直接淘汰
                self.manager.evict_from_gpu(victim)

    def _select_cpu_victim(self) -> Optional[str]:
        """选择 CPU 池中 R 值最低的 block（用于 CPU 满时腾位）。"""
        if not self.manager.cpu_cache:
            return None

        worst_r = float("inf")
        worst_block = None

        for block_hash, meta in self.manager.cpu_cache.items():
            age = self._clock - meta.get("last_access", 0)
            share = meta.get("share_count", 0)
            idx = meta.get("block_idx", 0)
            r_value = self.estimator.estimate(
                age=age, share_count=share, block_idx=idx
            )
            if r_value < worst_r:
                worst_r = r_value
                worst_block = block_hash

        return worst_block

    def _select_victim(self) -> Optional[str]:
        """
        选择淘汰目标：score 最低的 block。

        score_b = R_b - λ · hold_cost_b
        """
        if not self.manager.gpu_cache:
            return None

        worst_score = float("inf")
        worst_block = None

        for block_hash, meta in self.manager.gpu_cache.items():
            age = self._clock - meta.get("last_access", 0)
            share = meta.get("share_count", 0)
            idx = meta.get("block_idx", 0)
            r_value = self.estimator.estimate(
                age=age, share_count=share, block_idx=idx
            )
            # hold_cost: 估算持有该 block 的机会成本
            hold_cost = self._estimate_hold_cost(meta)
            score = r_value - self.score_lambda * hold_cost

            if score < worst_score:
                worst_score = score
                worst_block = block_hash

        return worst_block

    def _estimate_hold_cost(self, meta: Dict) -> float:
        """估算持有 block 的机会成本（每 step）。"""
        hold_model = self.manager.cost_model.get("hold", {})
        cost_per_byte_step = hold_model.get("cost_per_byte_step", 0.0)
        return cost_per_byte_step * self.block_bytes

    # ------------------------------------------------------------------
    # Fallback (SizeCost-LRU or LRU)
    # ------------------------------------------------------------------

    def _fallback_access(self, block_hash: str, prefill_ms: float) -> bool:
        """回退到简单策略（SizeCost-LRU 或 LRU）。"""
        # 简单 LRU 语义
        if block_hash in self.manager.gpu_cache:
            self.manager.record_hit(prefill_ms)
            self.manager.touch_gpu(block_hash)
            return True

        self.manager.record_miss(prefill_ms)
        while self.manager.gpu_size() >= self.manager.gpu_capacity:
            if self.manager.gpu_cache:
                # 淘汰最久未访问的
                victim = min(self.manager.gpu_cache,
                            key=lambda h: self.manager.gpu_cache[h].get("last_access", 0))
                self.manager.evict_from_gpu(victim)
            else:
                break
        self.manager.admit_gpu(block_hash, {
            "last_access": self._clock,
            "prefill_ms": prefill_ms,
        })
        self._clock += 1
        return False

    # ------------------------------------------------------------------
    # Stats (compatible with baseline interface)
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        return self.manager.hits

    @property
    def misses(self) -> int:
        return self.manager.misses

    @property
    def evictions(self) -> int:
        return self.manager.evictions

    @property
    def saved_prefill_ms(self) -> float:
        return self.manager.saved_prefill_ms

    @property
    def miss_cost_ms(self) -> float:
        return self.manager.miss_cost_ms

    def get_stats(self) -> Dict:
        """返回完整统计（含迁移/恢复开销）。"""
        stats = self.manager.get_stats()
        stats["fallback_count"] = self.fallback_count
        stats["effective_gpu_capacity"] = self.effective_gpu_capacity
        stats["total_gpu_capacity"] = self.total_gpu_capacity
        stats["safety_margin"] = self.safety_margin
        stats["migrate_threshold"] = self.migrate_threshold
        return stats


class NoCacheBaseline:
    """
    No-Cache baseline: every access is a miss.

    Lower bound reference for G3 comparison.
    """

    def __init__(self, capacity: int = 0):
        self.capacity = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0

    def access(self, block_hash: str, prefill_ms: float = 0.0, **kwargs) -> bool:
        """每次访问都是 miss。"""
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        return False

    def get_stats(self) -> Dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "saved_prefill_ms": self.saved_prefill_ms,
            "miss_cost_ms": self.miss_cost_ms,
        }
