"""
Two-Tier Fair Baselines
=======================
公平的 two-tier baselines，用于 §4.1 对照。

所有 baseline 使用与 FlowCache 相同的 GPU + CPU 容量，确保比较公平：
  - GPU miss 时：先查 CPU，CPU hit 则 H2D 恢复到 GPU，CPU miss 则 prefill
  - GPU 满时：按策略选 victim，迁移到 CPU（D2H），CPU 满则淘汰 CPU 中 victim
  - CPU hit 仍计入 hit（因为避免了 prefill），但产生 H2D 开销

与 FlowCache 的区别：
  - FlowCache 用因果 R 值决定迁移哪些块（selective migration）
  - Two-tier LRU/GDSF/SizeCost 用各自的优先级选 victim，但总是迁移（always-migrate）
  - Two-tier Oracle-Cost 用未来访问距离选 victim（离线诊断上界）

统计指标与 FlowCache 对齐：
  hits, misses, evictions, saved_prefill_ms, miss_cost_ms
  migrate_to_cpu_count, restore_to_gpu_count
  migrate_ms_total, restore_ms_total
  last_migrate_ms, last_restore_ms, last_policy_model_ms
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import heapq


class TwoTierLRUCache:
    """Two-tier LRU：GPU LRU + CPU LRU，满时总是迁移到 CPU。

    - GPU hit：更新 LRU 顺序
    - GPU miss + CPU hit：H2D 恢复到 GPU，更新 LRU 顺序
    - GPU miss + CPU miss：prefill，插入 GPU
    - GPU 满时：LRU victim 迁移到 CPU（D2H），CPU 满则淘汰 CPU LRU victim
    """

    def __init__(self,
                 gpu_capacity: int,
                 cpu_capacity: int,
                 d2h_ms: float = 0.0,
                 h2d_ms: float = 0.0):
        self.gpu_capacity = max(1, gpu_capacity)
        self.cpu_capacity = max(0, cpu_capacity)
        self.d2h_ms = d2h_ms
        self.h2d_ms = h2d_ms

        self.gpu_cache: "OrderedDict[str, None]" = OrderedDict()
        self.cpu_cache: "OrderedDict[str, None]" = OrderedDict()

        # 统计
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0
        self.gpu_evictions: int = 0
        self.cpu_evictions: int = 0
        self.migrate_to_cpu_count: int = 0
        self.restore_to_gpu_count: int = 0
        self.migrate_ms_total: float = 0.0
        self.restore_ms_total: float = 0.0

        # 逐访问开销（供 replay_accesses 读取）
        self.last_migrate_ms: float = 0.0
        self.last_restore_ms: float = 0.0
        self.last_policy_model_ms: float = 0.0

    def access(self, block_hash: str, prefill_ms: float = 0.0) -> bool:
        """访问一个 block，返回是否命中（GPU 或 CPU）。"""
        self.last_migrate_ms = 0.0
        self.last_restore_ms = 0.0
        self.last_policy_model_ms = 0.0

        # GPU hit
        if block_hash in self.gpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            self.gpu_cache.move_to_end(block_hash)
            return True

        # CPU hit → restore to GPU
        if block_hash in self.cpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            # 从 CPU 移除
            del self.cpu_cache[block_hash]
            # 确保 GPU 有空间
            self._ensure_gpu_space()
            self.gpu_cache[block_hash] = None
            self.restore_to_gpu_count += 1
            self.restore_ms_total += self.h2d_ms
            self.last_restore_ms = self.h2d_ms
            return True

        # miss
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        self._ensure_gpu_space()
        self.gpu_cache[block_hash] = None
        return False

    def _ensure_gpu_space(self) -> None:
        """确保 GPU 有 1 个空位，满时迁移 LRU victim 到 CPU。"""
        if len(self.gpu_cache) < self.gpu_capacity:
            return
        if not self.gpu_cache:
            return
        # LRU victim = OrderedDict 首元素
        victim = next(iter(self.gpu_cache))
        del self.gpu_cache[victim]
        # 迁移到 CPU
        if self.cpu_capacity > 0:
            if len(self.cpu_cache) >= self.cpu_capacity:
                # CPU 满，淘汰 CPU LRU victim
                cpu_victim = next(iter(self.cpu_cache))
                del self.cpu_cache[cpu_victim]
                self.cpu_evictions += 1
                self.evictions += 1
            self.cpu_cache[victim] = None
            self.migrate_to_cpu_count += 1
            self.migrate_ms_total += self.d2h_ms
            self.last_migrate_ms = self.d2h_ms
        else:
            # 无 CPU 层，直接淘汰
            self.gpu_evictions += 1
            self.evictions += 1


class TwoTierGDSFCache:
    """Two-tier GDSF：GPU GDSF + CPU GDSF，满时总是迁移到 CPU。

    priority = clock + freq / size（uniform size=1，priority = clock + freq）
    """

    def __init__(self,
                 gpu_capacity: int,
                 cpu_capacity: int,
                 d2h_ms: float = 0.0,
                 h2d_ms: float = 0.0):
        self.gpu_capacity = max(1, gpu_capacity)
        self.cpu_capacity = max(0, cpu_capacity)
        self.d2h_ms = d2h_ms
        self.h2d_ms = h2d_ms

        # block_hash -> {freq, priority}
        self.gpu_cache: Dict[str, Dict] = {}
        self.cpu_cache: Dict[str, Dict] = {}
        self._gpu_heap: List[Tuple[float, str]] = []
        self._cpu_heap: List[Tuple[float, str]] = []
        self.gpu_clock: float = 0.0
        self.cpu_clock: float = 0.0

        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0
        self.gpu_evictions: int = 0
        self.cpu_evictions: int = 0
        self.migrate_to_cpu_count: int = 0
        self.restore_to_gpu_count: int = 0
        self.migrate_ms_total: float = 0.0
        self.restore_ms_total: float = 0.0

        self.last_migrate_ms: float = 0.0
        self.last_restore_ms: float = 0.0
        self.last_policy_model_ms: float = 0.0

    def _evict_gpu(self) -> Optional[Tuple[str, Dict]]:
        """从 GPU 淘汰 GDSF victim，返回 (victim hash, entry)。"""
        while self._gpu_heap:
            priority, block_hash = heapq.heappop(self._gpu_heap)
            if block_hash not in self.gpu_cache:
                continue
            current = self.gpu_cache[block_hash]["priority"]
            if abs(current - priority) < 1e-9:
                self.gpu_clock = priority
                entry = self.gpu_cache.pop(block_hash)
                return block_hash, entry
        if self.gpu_cache:
            victim = next(iter(self.gpu_cache))
            self.gpu_clock = self.gpu_cache[victim]["priority"]
            entry = self.gpu_cache.pop(victim)
            return victim, entry
        return None

    def _evict_cpu(self) -> Optional[Tuple[str, Dict]]:
        """从 CPU 淘汰 GDSF victim，返回 (victim hash, entry)。"""
        while self._cpu_heap:
            priority, block_hash = heapq.heappop(self._cpu_heap)
            if block_hash not in self.cpu_cache:
                continue
            current = self.cpu_cache[block_hash]["priority"]
            if abs(current - priority) < 1e-9:
                self.cpu_clock = priority
                entry = self.cpu_cache.pop(block_hash)
                return block_hash, entry
        if self.cpu_cache:
            victim = next(iter(self.cpu_cache))
            self.cpu_clock = self.cpu_cache[victim]["priority"]
            entry = self.cpu_cache.pop(victim)
            return victim, entry
        return None

    def access(self, block_hash: str, prefill_ms: float = 0.0) -> bool:
        self.last_migrate_ms = 0.0
        self.last_restore_ms = 0.0
        self.last_policy_model_ms = 0.0

        # GPU hit
        if block_hash in self.gpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.gpu_cache[block_hash]
            entry["freq"] += 1
            entry["priority"] = self.gpu_clock + entry["freq"]
            heapq.heappush(self._gpu_heap, (entry["priority"], block_hash))
            return True

        # CPU hit → restore to GPU
        if block_hash in self.cpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cpu_cache.pop(block_hash)
            self._ensure_gpu_space()
            # 插入 GPU，freq 保持
            entry["priority"] = self.gpu_clock + entry["freq"]
            self.gpu_cache[block_hash] = entry
            heapq.heappush(self._gpu_heap, (entry["priority"], block_hash))
            self.restore_to_gpu_count += 1
            self.restore_ms_total += self.h2d_ms
            self.last_restore_ms = self.h2d_ms
            return True

        # miss
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        self._ensure_gpu_space()
        freq = 1
        priority = self.gpu_clock + freq
        self.gpu_cache[block_hash] = {"freq": freq, "priority": priority}
        heapq.heappush(self._gpu_heap, (priority, block_hash))
        return False

    def _ensure_gpu_space(self) -> None:
        if len(self.gpu_cache) < self.gpu_capacity:
            return
        result = self._evict_gpu()
        if result is None:
            return
        victim, entry = result
        if self.cpu_capacity > 0:
            if len(self.cpu_cache) >= self.cpu_capacity:
                cpu_result = self._evict_cpu()
                if cpu_result is not None:
                    self.cpu_evictions += 1
                    self.evictions += 1
            # 插入 CPU，freq 保持
            freq = entry.get("freq", 1)
            priority = self.cpu_clock + freq
            self.cpu_cache[victim] = {"freq": freq, "priority": priority}
            heapq.heappush(self._cpu_heap, (priority, victim))
            self.migrate_to_cpu_count += 1
            self.migrate_ms_total += self.d2h_ms
            self.last_migrate_ms = self.d2h_ms
        else:
            self.gpu_evictions += 1
            self.evictions += 1


class TwoTierSizeCostCache:
    """Two-tier SizeCost-LRU：GPU SizeCost + CPU SizeCost。

    priority = clock + size（min-heap，最小 priority 先淘汰 = 大 size 先淘汰）
    """

    def __init__(self,
                 gpu_capacity: int,
                 cpu_capacity: int,
                 d2h_ms: float = 0.0,
                 h2d_ms: float = 0.0):
        self.gpu_capacity = max(1, gpu_capacity)
        self.cpu_capacity = max(0, cpu_capacity)
        self.d2h_ms = d2h_ms
        self.h2d_ms = h2d_ms

        # block_hash -> {size, priority}
        self.gpu_cache: Dict[str, Dict] = {}
        self.cpu_cache: Dict[str, Dict] = {}
        self._gpu_heap: List[Tuple[float, str]] = []
        self._cpu_heap: List[Tuple[float, str]] = []
        self.gpu_clock: float = 0.0
        self.cpu_clock: float = 0.0

        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0
        self.gpu_evictions: int = 0
        self.cpu_evictions: int = 0
        self.migrate_to_cpu_count: int = 0
        self.restore_to_gpu_count: int = 0
        self.migrate_ms_total: float = 0.0
        self.restore_ms_total: float = 0.0

        self.last_migrate_ms: float = 0.0
        self.last_restore_ms: float = 0.0
        self.last_policy_model_ms: float = 0.0

    def _priority(self, size: int, clock: float) -> float:
        """SizeCost priority：小 priority 先淘汰 → 大 size 先淘汰。

        priority = clock + size → 大 size → 大 priority → min-heap 中后弹出
        但我们想淘汰大 size，所以存 -priority 用 min-heap（弹出最小 -priority
        = 最大 priority = 最大 size）。这与单层 SizeCostCache 方向一致。
        """
        return clock + float(size)

    def _evict_gpu(self) -> Optional[Tuple[str, Dict]]:
        while self._gpu_heap:
            neg_pri, block_hash = heapq.heappop(self._gpu_heap)
            if block_hash not in self.gpu_cache:
                continue
            current = self.gpu_cache[block_hash]["priority"]
            if abs(current + neg_pri) < 1e-9:
                self.gpu_clock = -neg_pri
                entry = self.gpu_cache.pop(block_hash)
                return block_hash, entry
        if self.gpu_cache:
            victim = next(iter(self.gpu_cache))
            self.gpu_clock = self.gpu_cache[victim]["priority"]
            entry = self.gpu_cache.pop(victim)
            return victim, entry
        return None

    def _evict_cpu(self) -> Optional[Tuple[str, Dict]]:
        while self._cpu_heap:
            neg_pri, block_hash = heapq.heappop(self._cpu_heap)
            if block_hash not in self.cpu_cache:
                continue
            current = self.cpu_cache[block_hash]["priority"]
            if abs(current + neg_pri) < 1e-9:
                self.cpu_clock = -neg_pri
                entry = self.cpu_cache.pop(block_hash)
                return block_hash, entry
        if self.cpu_cache:
            victim = next(iter(self.cpu_cache))
            self.cpu_clock = self.cpu_cache[victim]["priority"]
            entry = self.cpu_cache.pop(victim)
            return victim, entry
        return None

    def access(self, block_hash: str, prefill_ms: float = 0.0,
               size: int = 16) -> bool:
        self.last_migrate_ms = 0.0
        self.last_restore_ms = 0.0
        self.last_policy_model_ms = 0.0

        # GPU hit
        if block_hash in self.gpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.gpu_cache[block_hash]
            entry["priority"] = self._priority(entry["size"], self.gpu_clock)
            heapq.heappush(self._gpu_heap, (-entry["priority"], block_hash))
            return True

        # CPU hit → restore to GPU
        if block_hash in self.cpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            entry = self.cpu_cache.pop(block_hash)
            self._ensure_gpu_space()
            entry["priority"] = self._priority(entry["size"], self.gpu_clock)
            self.gpu_cache[block_hash] = entry
            heapq.heappush(self._gpu_heap, (-entry["priority"], block_hash))
            self.restore_to_gpu_count += 1
            self.restore_ms_total += self.h2d_ms
            self.last_restore_ms = self.h2d_ms
            return True

        # miss
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        self._ensure_gpu_space()
        priority = self._priority(size, self.gpu_clock)
        self.gpu_cache[block_hash] = {"size": size, "priority": priority}
        heapq.heappush(self._gpu_heap, (-priority, block_hash))
        return False

    def _ensure_gpu_space(self) -> None:
        if len(self.gpu_cache) < self.gpu_capacity:
            return
        result = self._evict_gpu()
        if result is None:
            return
        victim, entry = result
        if self.cpu_capacity > 0:
            if len(self.cpu_cache) >= self.cpu_capacity:
                cpu_result = self._evict_cpu()
                if cpu_result is not None:
                    self.cpu_evictions += 1
                    self.evictions += 1
            # 插入 CPU，size 保持
            size = entry.get("size", 16)
            priority = self._priority(size, self.cpu_clock)
            self.cpu_cache[victim] = {"size": size, "priority": priority}
            heapq.heappush(self._cpu_heap, (-priority, victim))
            self.migrate_to_cpu_count += 1
            self.migrate_ms_total += self.d2h_ms
            self.last_migrate_ms = self.d2h_ms
        else:
            self.gpu_evictions += 1
            self.evictions += 1


class TwoTierOracleCostCache:
    """Two-tier Oracle-Cost：用未来访问距离选 victim，总是迁移到 CPU。

    离线诊断上界：读取 future_accesses，选下次访问最远的 block 淘汰。
    与 FlowCache 的区别：Oracle 知道未来，FlowCache 用因果 R 值估计。
    """

    def __init__(self,
                 gpu_capacity: int,
                 cpu_capacity: int,
                 future_accesses: Dict[str, List[int]],
                 d2h_ms: float = 0.0,
                 h2d_ms: float = 0.0):
        self.gpu_capacity = max(1, gpu_capacity)
        self.cpu_capacity = max(0, cpu_capacity)
        self.future_accesses = future_accesses
        self.d2h_ms = d2h_ms
        self.h2d_ms = h2d_ms

        self.gpu_cache: Dict[str, float] = {}  # block_hash -> next_access_idx
        self.cpu_cache: Dict[str, float] = {}  # block_hash -> next_access_idx

        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_prefill_ms: float = 0.0
        self.miss_cost_ms: float = 0.0
        self.gpu_evictions: int = 0
        self.cpu_evictions: int = 0
        self.migrate_to_cpu_count: int = 0
        self.restore_to_gpu_count: int = 0
        self.migrate_ms_total: float = 0.0
        self.restore_ms_total: float = 0.0

        self.last_migrate_ms: float = 0.0
        self.last_restore_ms: float = 0.0
        self.last_policy_model_ms: float = 0.0

    def _next_access(self, block_hash: str, after_idx: int) -> float:
        """返回 block_hash 在 after_idx 之后的下次访问 idx，无则返回 inf。"""
        accesses = self.future_accesses.get(block_hash, [])
        # 二分查找第一个 > after_idx 的
        lo, hi = 0, len(accesses)
        while lo < hi:
            mid = (lo + hi) // 2
            if accesses[mid] <= after_idx:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(accesses):
            return float(accesses[lo])
        return float("inf")

    def _evict_gpu(self) -> Optional[Tuple[str, float]]:
        """选下次访问最远的 GPU block 淘汰，返回 (victim hash, next_access)。"""
        if not self.gpu_cache:
            return None
        victim = max(self.gpu_cache, key=lambda h: self.gpu_cache[h])
        next_access = self.gpu_cache.pop(victim)
        return victim, next_access

    def _evict_cpu(self) -> Optional[Tuple[str, float]]:
        """选下次访问最远的 CPU block 淘汰，返回 (victim hash, next_access)。"""
        if not self.cpu_cache:
            return None
        victim = max(self.cpu_cache, key=lambda h: self.cpu_cache[h])
        next_access = self.cpu_cache.pop(victim)
        return victim, next_access

    def access(self, block_hash: str, access_idx: int,
               prefill_ms: float = 0.0) -> bool:
        self.last_migrate_ms = 0.0
        self.last_restore_ms = 0.0
        self.last_policy_model_ms = 0.0

        next_idx = self._next_access(block_hash, access_idx)

        # GPU hit
        if block_hash in self.gpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            self.gpu_cache[block_hash] = next_idx
            return True

        # CPU hit → restore to GPU
        if block_hash in self.cpu_cache:
            self.hits += 1
            self.saved_prefill_ms += prefill_ms
            del self.cpu_cache[block_hash]
            self._ensure_gpu_space()
            self.gpu_cache[block_hash] = next_idx
            self.restore_to_gpu_count += 1
            self.restore_ms_total += self.h2d_ms
            self.last_restore_ms = self.h2d_ms
            return True

        # miss
        self.misses += 1
        self.miss_cost_ms += prefill_ms
        self._ensure_gpu_space()
        self.gpu_cache[block_hash] = next_idx
        return False

    def _ensure_gpu_space(self) -> None:
        if len(self.gpu_cache) < self.gpu_capacity:
            return
        result = self._evict_gpu()
        if result is None:
            return
        victim, next_access = result
        if self.cpu_capacity > 0:
            if len(self.cpu_cache) >= self.cpu_capacity:
                cpu_result = self._evict_cpu()
                if cpu_result is not None:
                    self.cpu_evictions += 1
                    self.evictions += 1
            self.cpu_cache[victim] = next_access
            self.migrate_to_cpu_count += 1
            self.migrate_ms_total += self.d2h_ms
            self.last_migrate_ms = self.d2h_ms
        else:
            self.gpu_evictions += 1
            self.evictions += 1
