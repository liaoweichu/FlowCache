"""
FlowCache-Lossless Controller
==============================
动作空间 A₀：GPU BF16 ↔ CPU BF16 ↔ Evict（无损，无量化）。

选择性迁移规则：
  proxy_b = capacity-normalized recency × causal reuse signals
  reuse_value_b = proxy_b · max(prefill_b - restore_b, 0)
  migrate_value_b = reuse_value_b - D2H_b - hold_b
  - migrate_value_b <= threshold：直接驱逐
  - CPU 有空间且 migrate_value_b > threshold：迁移
  - CPU 满：只有 migrate_value_b 超过最低价值 CPU block 才替换并迁移

安全水位：保留 safety_margin（默认 10%）的 GPU 容量作 buffer。
回退机制：controller 异常时回退 SizeCost-LRU。

与 baseline 类接口兼容：access() 方法记录 hits/misses/saved/miss cost。
"""

import heapq
import logging
import math
import time
from typing import Dict, List, Optional

from cache_manager import LosslessCacheManager
from reuse_estimator import create_estimator

logger = logging.getLogger(__name__)


class FlowCacheLosslessController:
    """
    FlowCache-Lossless controller (action space A₀).

    Open-loop replay interface compatible with baseline cache classes.
    The controller decides, for each block access:
      1. GPU hit → keep, update last_access
      2. CPU hit → restore to GPU (H2D), then GPU hit
      3. Miss → admit to GPU, possibly evicting/migrating others

    Selective value variant:
      - GPU victim is exact LRU.
      - On a full-GPU miss, compare the incoming block with the resident
        victim and bypass reusable-cache admission when replacement has
        lower causal cost value.
      - Migrate only when reuse-weighted saved prefill exceeds D2H and hold.
      - When CPU is full, the candidate must also beat its lowest-value block.
      - ``always_migrate`` remains available as a controlled ablation.

    ``proxy_b`` is a bounded, uncalibrated likelihood proxy, not a probability.
    It uses only decision-time information.  The policy deliberately reports
    proxy utility until validation-split probability calibration is available.

    The GPU admission comparison borrows Oracle-Cost's cost-aware ordering,
    but replaces future next-use distance with the causal proxy. It is an
    online heuristic, not an oracle and not an optimality claim. A bypassed
    miss is still computed for the active request; the block is simply not
    retained in the reusable inactive-cache pool.
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
                 migrate_threshold: float = 0.01,
                 migration_policy: str = "selective_value",
                 gpu_admission_policy: str = "always_admit",
                 selective_migration_config: Optional[Dict] = None):
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
            migrate_threshold: deprecated legacy R threshold（仅保留兼容字段）
            migration_policy: "selective_value" | "always_migrate"
            selective_migration_config: net-value admission parameters
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
        policy_aliases = {
            "selective": "selective_value",
            "selective_value": "selective_value",
            "always": "always_migrate",
            "always_migrate": "always_migrate",
            "always_migrate_tiered_lru": "always_migrate",
        }
        if migration_policy not in policy_aliases:
            raise ValueError(
                "migration_policy must be selective_value or always_migrate"
            )
        self.migration_policy = policy_aliases[migration_policy]
        admission_aliases = {
            "always": "always_admit",
            "always_admit": "always_admit",
            "cost_aware_bypass": "oracle_cost_proxy",
            "oracle_cost_proxy": "oracle_cost_proxy",
        }
        if gpu_admission_policy not in admission_aliases:
            raise ValueError(
                "gpu_admission_policy must be oracle_cost_proxy or "
                "always_admit"
            )
        self.gpu_admission_policy = admission_aliases[
            gpu_admission_policy
        ]

        # G3'' 修复：CPU 容量自动设为 GPU 有效容量的 2 倍（-1 时）
        # 避免 CPU 无限增长导致 OOM，同时让 CPU 层触发淘汰
        if cpu_capacity_blocks < 0:
            cpu_capacity_blocks = effective_capacity * 2

        # 初始化 cache manager
        self.manager = LosslessCacheManager(
            gpu_capacity_blocks=effective_capacity,
            cpu_capacity_blocks=cpu_capacity_blocks,
            block_bytes=block_bytes,
            cost_model=cost_model or {},
        )
        self._d2h_ms_per_block = self.manager.estimate_d2h_ms()
        self._h2d_ms_per_block = self.manager.estimate_h2d_ms()

        # 初始化 reuse 估计器
        estimator_type = "heuristic"
        if reuse_estimator_config and "type" in reuse_estimator_config:
            estimator_type = reuse_estimator_config["type"]
        self.estimator = create_estimator(
            estimator_type, reuse_estimator_config or {}
        )
        configured_position_weights = getattr(
            self.estimator, "position_weights", {}
        )
        self._has_position_weights = bool(configured_position_weights)
        early_position = configured_position_weights.get("early", {})
        mid_position = configured_position_weights.get("mid", {})
        late_position = configured_position_weights.get("late", {})
        self._early_position_limit = int(
            early_position.get("block_idx_lt", 10)
        )
        self._mid_position_limit = int(
            mid_position.get("block_idx_lt", 50)
        )
        self._early_position_weight = float(
            early_position.get("weight", 1.5)
        )
        self._mid_position_weight = float(
            mid_position.get("weight", 1.0)
        )
        self._late_position_weight = float(
            late_position.get("weight", 0.7)
        )
        self._max_position_weight = max(
            (
                float(section.get("weight", 0.0))
                for section in configured_position_weights.values()
                if isinstance(section, dict)
            ),
            default=1.0,
        )

        selection_cfg = selective_migration_config or {}
        self.minimum_net_benefit_ms = max(
            0.0, float(selection_cfg.get("minimum_net_benefit_ms", 0.0))
        )
        self.cpu_admission_margin_ms = max(
            0.0, float(selection_cfg.get("cpu_admission_margin_ms", 0.0))
        )
        self.gpu_admission_margin_ms = max(
            0.0, float(selection_cfg.get("gpu_admission_margin_ms", 0.0))
        )
        self.gpu_admission_cold_start_prior = min(
            1.0,
            max(
                0.0,
                float(selection_cfg.get(
                    "gpu_admission_cold_start_prior", 0.05
                )),
            ),
        )
        self.gpu_admission_cold_start_cost_ratio = min(
            1.0,
            max(
                0.0,
                float(selection_cfg.get(
                    "gpu_admission_cold_start_cost_ratio", 0.5
                )),
            ),
        )
        self.gpu_admission_confidence_scale = max(
            1e-6,
            float(selection_cfg.get(
                "gpu_admission_confidence_scale", 1.0
            )),
        )
        self.expected_cpu_residence_steps = max(
            0, int(selection_cfg.get("expected_cpu_residence_steps", 100))
        )
        self.hold_cost_weight = max(
            0.0, float(selection_cfg.get("hold_cost_weight", 1.0))
        )
        self.age_scale_capacity_multiplier = max(
            1e-6,
            float(selection_cfg.get(
                "age_scale_capacity_multiplier", 1.0
            )),
        )
        self.share_count_cap = max(
            1, int(selection_cfg.get("share_count_cap", 8))
        )
        self.reuse_count_scale = max(
            1e-6, float(selection_cfg.get("reuse_count_scale", 2.0))
        )
        signal_weights = selection_cfg.get("signal_weights", {})
        self.share_signal_weight = max(
            0.0, float(signal_weights.get("share", 0.45))
        )
        self.frequency_signal_weight = max(
            0.0, float(signal_weights.get("frequency", 0.35))
        )
        self.position_signal_weight = max(
            0.0, float(signal_weights.get("position", 0.20))
        )
        self.signal_weight_total = (
            self.share_signal_weight
            + self.frequency_signal_weight
            + self.position_signal_weight
        )
        if self.signal_weight_total <= 0:
            raise ValueError("selective migration signal weights sum to zero")
        hold_model = self.manager.cost_model.get("hold", {})
        hold_cost_per_step_ms = max(
            0.0,
            float(hold_model.get("cost_per_byte_step", 0.0) or 0.0)
            * self.block_bytes,
        )
        self._cpu_hold_cost_per_block_ms = (
            hold_cost_per_step_ms
            * self.expected_cpu_residence_steps
            * self.hold_cost_weight
        )

        # 回退计数
        self.fallback_count: int = 0
        self._clock: int = 0
        # Causal reuse history survives physical eviction.  Resetting this
        # count on every miss makes a repeatedly evicted block look forever
        # unseen and creates systematic false negatives.
        self._historical_access_counts: Dict[str, int] = {}

        # Lazy heaps make CPU lowest-R selection amortized O(log N).  Entries
        # are versioned so restore/eviction never requires an O(N) heap delete.
        self._cpu_entry_seq: int = 0
        self._cpu_versions: Dict[str, int] = {}
        self._cpu_value_heap: List[tuple] = []
        self.cpu_heap_compaction_count: int = 0

        # Per-access modeled costs and replay implementation overhead.
        policy_cfg = self.manager.cost_model.get("policy", {})
        self.policy_cost_per_decision_ms = max(
            0.0, float(policy_cfg.get("cost_per_decision_ms", 0.0) or 0.0)
        )
        self.last_migrate_ms: float = 0.0
        self.last_restore_ms: float = 0.0
        self.last_policy_model_ms: float = 0.0
        self.last_replay_wall_ms: float = 0.0
        self.policy_model_ms_total: float = 0.0
        self.replay_wall_ms_total: float = 0.0
        self.policy_decision_count: int = 0
        self.controller_variant = self.migration_policy
        self.policy_stack = (
            f"{self.gpu_admission_policy}+{self.migration_policy}"
        )

        # Selective-migration audit counters.
        self.migration_candidate_count: int = 0
        self.migration_selected_count: int = 0
        self.migration_rejected_count: int = 0
        self.rejected_low_value_count: int = 0
        self.rejected_cpu_competition_count: int = 0
        self.rejected_no_cpu_slot_count: int = 0
        self.cpu_admission_replacement_count: int = 0
        self.candidate_value_index_ms_total: float = 0.0
        self.selected_value_index_ms_total: float = 0.0
        self.rejected_value_index_ms_total: float = 0.0

        # Oracle-Cost-inspired online GPU admission/bypass audit counters.
        # Only full-cache misses are candidates; compulsory warm-up admits are
        # excluded from the admission selection rate.
        self.gpu_admission_candidate_count: int = 0
        self.gpu_admission_selected_count: int = 0
        self.gpu_admission_bypassed_count: int = 0
        self.gpu_admission_candidate_value_index_ms_total: float = 0.0
        self.gpu_admission_incumbent_value_index_ms_total: float = 0.0
        self.gpu_admission_displacement_value_index_ms_total: float = 0.0
        self.gpu_bypassed_prefill_ms_total: float = 0.0

        self.last_migration_candidate_count: int = 0
        self.last_migration_selected_count: int = 0
        self.last_migration_rejected_count: int = 0
        self.last_candidate_value_index_ms: float = 0.0
        self.last_gpu_admission_candidate_count: int = 0
        self.last_gpu_admission_selected_count: int = 0
        self.last_gpu_admission_bypassed_count: int = 0

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
        self.last_migrate_ms = 0.0
        self.last_restore_ms = 0.0
        self.last_policy_model_ms = self.policy_cost_per_decision_ms
        self.last_migration_candidate_count = 0
        self.last_migration_selected_count = 0
        self.last_migration_rejected_count = 0
        self.last_candidate_value_index_ms = 0.0
        self.last_gpu_admission_candidate_count = 0
        self.last_gpu_admission_selected_count = 0
        self.last_gpu_admission_bypassed_count = 0
        started = time.perf_counter()
        try:
            return self._access_impl(
                block_hash, parent_hash, prefill_ms,
                block_idx, share_count, age
            )
        except Exception as e:
            logger.warning(f"Controller exception, falling back to {self.fallback}: {e}")
            self.fallback_count += 1
            return self._fallback_access(block_hash, prefill_ms)
        finally:
            self.last_replay_wall_ms = (time.perf_counter() - started) * 1000.0
            self.replay_wall_ms_total += self.last_replay_wall_ms
            self.policy_model_ms_total += self.last_policy_model_ms
            self.policy_decision_count += 1

    def _access_impl(self,
                     block_hash: str,
                     parent_hash: str,
                     prefill_ms: float,
                     block_idx: int,
                     share_count: int,
                     age: Optional[int]) -> bool:
        """实际访问逻辑。"""
        historical_access_count = (
            self._historical_access_counts.get(block_hash, 0) + 1
        )
        self._historical_access_counts[block_hash] = (
            historical_access_count
        )
        location = self.manager.lookup(block_hash)

        if location == "gpu":
            # GPU hit
            self.manager.record_hit(prefill_ms)
            self.manager.touch_gpu(block_hash)
            # 更新元数据
            meta = self.manager.gpu_cache[block_hash]
            meta["block_idx"] = block_idx
            meta["share_count"] = share_count
            meta["prefill_ms"] = prefill_ms
            meta["access_count"] = historical_access_count
            meta["last_access"] = self._clock
            self._clock += 1
            return True

        if location == "cpu":
            # CPU hit → restore to GPU
            # G3'' Bug 6 修复：_ensure_gpu_space 可能淘汰 CPU 中的目标块
            # 先从 CPU 取出 metadata，腾位后再放回 GPU
            cpu_meta = self.manager.cpu_cache.get(block_hash)
            if cpu_meta is None:
                # 块在 _ensure_gpu_space 中被淘汰了，当作 miss
                self.manager.record_miss(prefill_ms)
                incoming_meta = {
                    "parent_hash": parent_hash,
                    "last_access": self._clock,
                    "prefill_ms": prefill_ms,
                    "block_idx": block_idx,
                    "share_count": share_count,
                    "access_count": historical_access_count,
                }
                self._admit_miss_to_gpu(block_hash, incoming_meta)
                self._clock += 1
                return False

            # 确保 GPU 有空间（保护目标块不被从 CPU 淘汰）
            self.last_migrate_ms += self._ensure_gpu_space(
                1, protect_hash=block_hash
            )
            # 目标块应仍在 CPU（已保护）
            if block_hash in self.manager.cpu_cache:
                self._invalidate_cpu_block(block_hash)
                restore_ms = self.manager.restore_to_gpu(block_hash)
            else:
                # 极端情况：CPU 只有这一个块且满了，无法腾位
                # 用之前保存的 metadata 重建
                restore_ms = 0.0
                self.manager.gpu_cache[block_hash] = cpu_meta.copy()
                self.manager.gpu_cache.move_to_end(block_hash)
            self.last_restore_ms += restore_ms

            self.manager.record_hit(prefill_ms)
            self.manager.touch_gpu(block_hash)
            meta = self.manager.gpu_cache[block_hash]
            meta["block_idx"] = block_idx
            meta["share_count"] = share_count
            meta["prefill_ms"] = prefill_ms
            meta["access_count"] = historical_access_count
            meta["last_access"] = self._clock
            self._clock += 1
            return True

        # Miss
        self.manager.record_miss(prefill_ms)
        incoming_meta = {
            "parent_hash": parent_hash,
            "last_access": self._clock,
            "prefill_ms": prefill_ms,
            "block_idx": block_idx,
            "share_count": share_count,
            "access_count": historical_access_count,
        }
        self._admit_miss_to_gpu(block_hash, incoming_meta)
        self._clock += 1
        return False

    def _admit_miss_to_gpu(self, block_hash: str,
                           incoming_meta: Dict) -> bool:
        """Retain a computed miss only when online cost value justifies it.

        Returning ``False`` means cache bypass, not compute bypass: the active
        request still paid and consumed ``prefill_ms``. Only the reusable
        inactive-cache copy is omitted.
        """
        if not self._should_admit_gpu(incoming_meta):
            self.gpu_bypassed_prefill_ms_total += max(
                0.0, float(incoming_meta.get("prefill_ms", 0.0) or 0.0)
            )
            return False
        self.last_migrate_ms += self._ensure_gpu_space(1)
        self.manager.admit_gpu(block_hash, incoming_meta)
        return True

    # ------------------------------------------------------------------
    # Eviction / migration decision
    # ------------------------------------------------------------------

    def _should_admit_gpu(self, incoming_meta: Dict) -> bool:
        """Compare an incoming miss with the resident LRU victim.

        Oracle-Cost ranks cached objects by future cost per next-use distance.
        The online policy cannot use future accesses, so it substitutes the
        causal reuse proxy and evaluates two choices:

        * keep the incumbent in GPU; or
        * retain the incoming block and preserve as much displaced-victim
          value as the selective CPU policy can recover after transfer costs.

        The miss computation itself is common to both choices and cancels.
        """
        if (
            self.gpu_admission_policy == "always_admit"
            or not self.manager.gpu_full()
            or not self.manager.gpu_cache
        ):
            return True

        victim = next(iter(self.manager.gpu_cache))
        victim_meta = self.manager.gpu_cache[victim]
        incoming_value = self._incoming_gpu_residency_value_index(
            incoming_meta, victim_meta
        )
        incumbent_value = self._gpu_residency_value_index(victim_meta)
        # Selective CPU displacement has non-negative residual value. When the
        # incoming block already beats the incumbent by itself, admission is
        # certain and we can avoid a second O(log N) CPU-heap lookup.
        if (
            self.migration_policy == "selective_value"
            and incoming_value
            > incumbent_value + self.gpu_admission_margin_ms
        ):
            displacement_value = 0.0
        else:
            displacement_value = (
                self._prospective_cpu_displacement_value_index(victim_meta)
            )
        replacement_value = incoming_value + displacement_value

        self.gpu_admission_candidate_count += 1
        self.last_gpu_admission_candidate_count += 1
        self.gpu_admission_candidate_value_index_ms_total += incoming_value
        self.gpu_admission_incumbent_value_index_ms_total += incumbent_value
        self.gpu_admission_displacement_value_index_ms_total += (
            displacement_value
        )

        if (
            replacement_value
            <= incumbent_value + self.gpu_admission_margin_ms
        ):
            self.gpu_admission_bypassed_count += 1
            self.last_gpu_admission_bypassed_count += 1
            return False

        self.gpu_admission_selected_count += 1
        self.last_gpu_admission_selected_count += 1
        return True

    def _prospective_cpu_displacement_value_index(
        self, victim_meta: Dict
    ) -> float:
        """Net value recoverable by moving a displaced GPU victim to CPU."""
        candidate_net = self._migration_net_value_index(victim_meta)

        if self.migration_policy == "selective_value":
            if candidate_net <= self.minimum_net_benefit_ms:
                return 0.0
            if not self.manager.cpu_full():
                return candidate_net
            cpu_victim = self._select_cpu_victim()
            if cpu_victim is None:
                return 0.0
            incumbent_value = self._cpu_residency_value_index(
                self.manager.cpu_cache[cpu_victim]
            )
            if (
                candidate_net
                <= incumbent_value + self.cpu_admission_margin_ms
            ):
                return 0.0
            return candidate_net - incumbent_value

        # Always-migrate is an ablation. If it is combined with cost-aware GPU
        # admission, retain its actual transfer/CPU-replacement cost rather
        # than silently granting the displaced block free CPU residency.
        if not self.manager.cpu_full():
            return candidate_net
        cpu_victim = self._select_cpu_victim()
        if cpu_victim is None:
            return 0.0
        incumbent_value = self._cpu_residency_value_index(
            self.manager.cpu_cache[cpu_victim]
        )
        return candidate_net - incumbent_value

    def _ensure_gpu_space(self, needed: int,
                          protect_hash: Optional[str] = None) -> float:
        """
        Ensure GPU space using selective CPU admission.

        ``selective_value`` compares the candidate's reuse-weighted saved
        prefill against D2H, expected CPU hold cost, and (when CPU is full)
        the incumbent CPU block's value. ``always_migrate`` bypasses the value
        tests but shares the same GPU victim and CPU replacement machinery.

        Args:
            needed: 需要腾出的 block 数
            protect_hash: CPU 淘汰时需保护的 block_hash（避免淘汰即将 restore 的块）
        """
        migrate_ms_total = 0.0
        while self.manager.gpu_size() + needed > self.manager.gpu_capacity:
            if not self.manager.gpu_cache:
                break
            # OrderedDict keeps the exact LRU block first.
            victim = next(iter(self.manager.gpu_cache))
            meta = self.manager.gpu_cache[victim]
            candidate_net = self._migration_net_value_index(meta)
            self.migration_candidate_count += 1
            self.last_migration_candidate_count += 1
            self.candidate_value_index_ms_total += candidate_net
            self.last_candidate_value_index_ms += candidate_net

            cpu_victim = None
            reject_reason = None
            if self.migration_policy == "always_migrate":
                if self.manager.cpu_full():
                    cpu_victim = self._select_cpu_victim(
                        protect_hash=protect_hash
                    )
                    if cpu_victim is None:
                        reject_reason = "no_cpu_slot"
            else:
                should_migrate, cpu_victim, reject_reason = (
                    self._selective_migration_decision(
                        meta,
                        candidate_net=candidate_net,
                        protect_hash=protect_hash,
                    )
                )
                if not should_migrate:
                    cpu_victim = None

            should_migrate = reject_reason is None
            if should_migrate and self.manager.cpu_full():
                if cpu_victim is None:
                    should_migrate = False
                    reject_reason = "no_cpu_slot"
                else:
                    self._invalidate_cpu_block(cpu_victim)
                    self.manager.evict_from_cpu(cpu_victim)
                    self.cpu_admission_replacement_count += 1

            if should_migrate and not self.manager.cpu_full():
                migrate_ms_total += self.manager.migrate_to_cpu(victim)
                if victim in self.manager.cpu_cache:
                    self._register_cpu_block(victim)
                    self.migration_selected_count += 1
                    self.last_migration_selected_count += 1
                    self.selected_value_index_ms_total += candidate_net
                    continue

            # Rejected selective admission (or no usable CPU slot).
            self.manager.evict_from_gpu(victim)
            self.migration_rejected_count += 1
            self.last_migration_rejected_count += 1
            self.rejected_value_index_ms_total += candidate_net
            if reject_reason == "low_value":
                self.rejected_low_value_count += 1
            elif reject_reason == "cpu_competition":
                self.rejected_cpu_competition_count += 1
            else:
                self.rejected_no_cpu_slot_count += 1
        return migrate_ms_total

    def _selective_migration_decision(
        self,
        candidate_meta: Dict,
        candidate_net: Optional[float] = None,
        protect_hash: Optional[str] = None,
    ):
        """Return (migrate, cpu_victim, reject_reason)."""
        if candidate_net is None:
            candidate_net = self._migration_net_value_index(candidate_meta)
        if candidate_net <= self.minimum_net_benefit_ms:
            return False, None, "low_value"

        if not self.manager.cpu_full():
            return True, None, None

        cpu_victim = self._select_cpu_victim(protect_hash=protect_hash)
        if cpu_victim is None:
            return False, None, "no_cpu_slot"
        incumbent_meta = self.manager.cpu_cache[cpu_victim]
        incumbent_value = self._cpu_residency_value_index(incumbent_meta)
        if (
            candidate_net
            <= incumbent_value + self.cpu_admission_margin_ms
        ):
            return False, None, "cpu_competition"
        return True, cpu_victim, None

    def _base_reuse_signal(self, meta: Dict) -> float:
        """Decision-time reuse signal before capacity-normalized recency."""
        share_count = max(0, int(meta.get("share_count", 0)))
        access_count = max(1, int(meta.get("access_count", 1)))
        block_idx = int(meta.get("block_idx", 0))
        cache_key = (share_count, access_count, block_idx)
        if meta.get("_selective_base_signal_key") == cache_key:
            return float(meta["_selective_base_signal"])

        share_signal = min(share_count, self.share_count_cap) / float(
            self.share_count_cap
        )

        reuse_count = access_count - 1
        frequency_signal = 1.0 - math.exp(
            -reuse_count / self.reuse_count_scale
        )

        position_signal = 1.0
        if self._has_position_weights:
            if block_idx < self._early_position_limit:
                raw_position = self._early_position_weight
            elif block_idx < self._mid_position_limit:
                raw_position = self._mid_position_weight
            else:
                raw_position = self._late_position_weight
            raw_position = max(0.0, raw_position)
            position_signal = min(
                1.0,
                raw_position / max(self._max_position_weight, 1e-12),
            )

        weighted = (
            self.share_signal_weight * share_signal
            + self.frequency_signal_weight * frequency_signal
            + self.position_signal_weight * position_signal
        )
        signal = min(
            1.0, max(0.0, weighted / self.signal_weight_total)
        )
        meta["_selective_base_signal_key"] = cache_key
        meta["_selective_base_signal"] = signal
        return signal

    def _reuse_likelihood_proxy(self, meta: Dict) -> float:
        """Bounded reuse proxy with age normalized by effective GPU capacity.

        A fixed hard horizon made every LRU victim look dead whenever GPU
        capacity exceeded that horizon.  Normalizing age by cache capacity
        keeps the signal comparable across the 1/2/4-GiB cells while remaining
        causal and monotonically decreasing with age.
        """
        age = max(0, self._clock - int(meta.get("last_access", 0)))
        age_scale = max(
            1.0,
            self.effective_gpu_capacity
            * self.age_scale_capacity_multiplier,
        )
        recency_signal = math.exp(-age / age_scale)
        return min(
            1.0,
            max(0.0, recency_signal * self._base_reuse_signal(meta)),
        )

    def _recoverable_prefill_ms(self, meta: Dict) -> float:
        """Prefill avoided by a CPU hit after paying H2D restoration."""
        prefill_ms = float(meta.get("prefill_ms", 0.0) or 0.0)
        return max(0.0, prefill_ms - self._h2d_ms_per_block)

    def _reuse_value_index(self, meta: Dict) -> float:
        """Likelihood-proxy-weighted recoverable prefill, in proxy-ms."""
        reuse_proxy = self._reuse_likelihood_proxy(meta)
        recoverable_prefill_ms = max(
            0.0, self._recoverable_prefill_ms(meta)
        )
        return reuse_proxy * recoverable_prefill_ms

    def _gpu_residency_value_index(self, meta: Dict) -> float:
        """Expected-proxy prefill saved by retaining a block on GPU."""
        reuse_proxy = self._reuse_likelihood_proxy(meta)
        prefill_ms = max(0.0, float(meta.get("prefill_ms", 0.0) or 0.0))
        return reuse_proxy * prefill_ms

    def _incoming_gpu_residency_value_index(
        self,
        meta: Dict,
        victim_meta: Optional[Dict] = None,
    ) -> float:
        """GPU value discounted by causally observed reuse evidence.

        A newly computed miss has perfect recency by construction. Without
        this confidence term, almost every first-seen incoming block defeats
        an older incumbent and the admission policy collapses to
        ``always_admit``. Physical bypass does not erase
        ``_historical_access_counts``, so a second observed access raises
        confidence without consulting future labels.

        The cold-start discount is only applied when the incoming prefill
        cost is materially lower than the resident victim *or* the incumbent
        has stronger causally observed reuse evidence. This TinyLFU-style
        doorkeeper guard prevents equal-cost/equal-evidence cyclic workloads
        from trading a large increase in recomputation for a small transfer
        reduction, while still protecting a repeatedly used incumbent from a
        first-seen one-hit candidate.
        """
        access_count = max(1, int(meta.get("access_count", 1)))
        observed_reuses = access_count - 1
        if victim_meta is not None:
            incoming_prefill_ms = max(
                0.0, float(meta.get("prefill_ms", 0.0) or 0.0)
            )
            victim_prefill_ms = max(
                0.0, float(victim_meta.get("prefill_ms", 0.0) or 0.0)
            )
            victim_access_count = max(
                1, int(victim_meta.get("access_count", 1))
            )
            cost_skew_supports_bypass = (
                victim_prefill_ms > 0.0
                and incoming_prefill_ms
                < (
                    victim_prefill_ms
                    * self.gpu_admission_cold_start_cost_ratio
                )
            )
            incumbent_has_stronger_reuse_evidence = (
                victim_access_count > access_count
            )
            if (
                victim_prefill_ms <= 0.0
                or (
                    not cost_skew_supports_bypass
                    and not incumbent_has_stronger_reuse_evidence
                )
            ):
                return self._gpu_residency_value_index(meta)
        evidence = 1.0 - math.exp(
            -observed_reuses / self.gpu_admission_confidence_scale
        )
        confidence = (
            self.gpu_admission_cold_start_prior
            + (1.0 - self.gpu_admission_cold_start_prior) * evidence
        )
        return confidence * self._gpu_residency_value_index(meta)

    def _cpu_hold_cost_ms(self) -> float:
        """Expected CPU opportunity cost for one block admission."""
        return self._cpu_hold_cost_per_block_ms

    def _migration_net_value_index(self, meta: Dict) -> float:
        """Candidate value after D2H placement and expected CPU hold."""
        return (
            self._reuse_value_index(meta)
            - self._d2h_ms_per_block
            - self._cpu_hold_cost_ms()
        )

    def _cpu_residency_value_index(self, meta: Dict) -> float:
        """Remaining value of an already resident CPU block."""
        return self._reuse_value_index(meta) - self._cpu_hold_cost_ms()

    def _score_block(self, meta: Dict) -> float:
        """Compute score for a single block (inlined for speed)."""
        age = self._clock - meta.get("last_access", 0)
        share = meta.get("share_count", 0)
        idx = meta.get("block_idx", 0)
        r_value = self.estimator.estimate(age=age, share_count=share, block_idx=idx)
        hold_cost = self._estimate_hold_cost(meta)
        return r_value - self.score_lambda * hold_cost

    def _select_cpu_victim(self, protect_hash: Optional[str] = None) -> Optional[str]:
        """CPU victim: lowest reuse-value-index block（跳过 protect_hash）。

        Args:
            protect_hash: 需保护的 block_hash（避免淘汰即将 restore 的块）
        """
        if not self.manager.cpu_cache:
            return None

        # Capacity-normalized recency separates into a decision-clock constant
        # and a static per-block key, so the exact minimum remains heapable.
        if self._cpu_value_heap:
            return self._heap_candidate(
                self._cpu_value_heap, protect_hash
            )

        # Compatibility fallback if a block was inserted outside controller
        # primitives and therefore has no heap entry.
        candidates = (
            (h, m) for h, m in self.manager.cpu_cache.items()
            if h != protect_hash
        )
        try:
            return min(
                candidates,
                key=lambda item: self._cpu_residency_value_index(item[1]),
            )[0]
        except ValueError:
            return None

    def _register_cpu_block(self, block_hash: str) -> None:
        """Add/update one CPU block in the lazy victim heaps."""
        meta = self.manager.cpu_cache.get(block_hash)
        if meta is None:
            return
        self._cpu_entry_seq += 1
        seq = self._cpu_entry_seq
        self._cpu_versions[block_hash] = seq
        priority = self._cpu_heap_priority(meta)
        heapq.heappush(
            self._cpu_value_heap, (priority, seq, block_hash)
        )
        self._maybe_compact_cpu_heap()

    def _cpu_heap_priority(self, meta: Dict) -> float:
        """Clock-independent priority preserving current value ordering."""
        last_access = int(meta.get("last_access", 0))
        age_scale = max(
            1.0,
            self.effective_gpu_capacity
            * self.age_scale_capacity_multiplier,
        )
        base_signal = max(1e-12, self._base_reuse_signal(meta))
        recoverable_prefill_ms = max(
            1e-12, self._recoverable_prefill_ms(meta)
        )
        priority = (
            last_access / age_scale
            + math.log(base_signal)
            + math.log(recoverable_prefill_ms)
        )
        return priority

    def _maybe_compact_cpu_heap(self) -> None:
        """Bound stale lazy-heap entries without changing victim semantics."""
        live_count = len(self.manager.cpu_cache)
        maximum_entries = max(4096, 8 * max(1, live_count))
        if len(self._cpu_value_heap) <= maximum_entries:
            return

        compacted = []
        versions = {}
        for block_hash, meta in self.manager.cpu_cache.items():
            self._cpu_entry_seq += 1
            seq = self._cpu_entry_seq
            versions[block_hash] = seq
            compacted.append(
                (self._cpu_heap_priority(meta), seq, block_hash)
            )
        heapq.heapify(compacted)
        self._cpu_versions = versions
        self._cpu_value_heap = compacted
        self.cpu_heap_compaction_count += 1

    def _invalidate_cpu_block(self, block_hash: str) -> None:
        """Invalidate lazy heap entries for a restored/evicted CPU block."""
        self._cpu_versions.pop(block_hash, None)

    def _heap_candidate(self, heap: List[tuple],
                        protect_hash: Optional[str]) -> Optional[str]:
        """Return a live heap minimum without permanently removing it."""
        held = []
        selected = None
        while heap:
            entry = heapq.heappop(heap)
            _, seq, block_hash = entry
            if (
                self._cpu_versions.get(block_hash) != seq
                or block_hash not in self.manager.cpu_cache
            ):
                continue
            held.append(entry)
            if block_hash == protect_hash:
                continue
            selected = block_hash
            break
        for entry in held:
            heapq.heappush(heap, entry)
        return selected

    def _select_victim(self) -> Optional[str]:
        """GPU victim: lowest score block (uses built-in min for speed)."""
        if not self.manager.gpu_cache:
            return None
        return min(self.manager.gpu_cache.items(), key=lambda item: self._score_block(item[1]))[0]

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
                victim = next(iter(self.manager.gpu_cache))
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
        stats["controller_variant"] = self.controller_variant
        stats["gpu_admission_policy"] = self.gpu_admission_policy
        stats["policy_stack"] = self.policy_stack
        stats["online_feature_scope"] = "current_and_past_only"
        stats["future_access_index_used"] = False
        stats["minimum_net_benefit_ms"] = self.minimum_net_benefit_ms
        stats["cpu_admission_margin_ms"] = self.cpu_admission_margin_ms
        stats["gpu_admission_margin_ms"] = self.gpu_admission_margin_ms
        stats["gpu_admission_cold_start_prior"] = (
            self.gpu_admission_cold_start_prior
        )
        stats["gpu_admission_cold_start_cost_ratio"] = (
            self.gpu_admission_cold_start_cost_ratio
        )
        stats["gpu_admission_confidence_scale"] = (
            self.gpu_admission_confidence_scale
        )
        stats["expected_cpu_residence_steps"] = (
            self.expected_cpu_residence_steps
        )
        stats["hold_cost_weight"] = self.hold_cost_weight
        stats["age_scale_capacity_multiplier"] = (
            self.age_scale_capacity_multiplier
        )
        stats["share_count_cap"] = self.share_count_cap
        stats["reuse_count_scale"] = self.reuse_count_scale
        stats["share_signal_weight"] = self.share_signal_weight
        stats["frequency_signal_weight"] = self.frequency_signal_weight
        stats["position_signal_weight"] = self.position_signal_weight
        stats["historical_blocks_tracked"] = len(
            self._historical_access_counts
        )
        stats["cpu_hold_cost_per_block_ms"] = self._cpu_hold_cost_ms()
        stats["migration_candidate_count"] = self.migration_candidate_count
        stats["migration_selected_count"] = self.migration_selected_count
        stats["migration_rejected_count"] = self.migration_rejected_count
        stats["rejected_low_value_count"] = self.rejected_low_value_count
        stats["rejected_cpu_competition_count"] = (
            self.rejected_cpu_competition_count
        )
        stats["rejected_no_cpu_slot_count"] = (
            self.rejected_no_cpu_slot_count
        )
        stats["cpu_admission_replacement_count"] = (
            self.cpu_admission_replacement_count
        )
        stats["cpu_heap_compaction_count"] = (
            self.cpu_heap_compaction_count
        )
        stats["candidate_value_index_ms_total"] = (
            self.candidate_value_index_ms_total
        )
        stats["selected_value_index_ms_total"] = (
            self.selected_value_index_ms_total
        )
        stats["rejected_value_index_ms_total"] = (
            self.rejected_value_index_ms_total
        )
        stats["gpu_admission_candidate_count"] = (
            self.gpu_admission_candidate_count
        )
        stats["gpu_admission_selected_count"] = (
            self.gpu_admission_selected_count
        )
        stats["gpu_admission_bypassed_count"] = (
            self.gpu_admission_bypassed_count
        )
        stats["gpu_admission_candidate_value_index_ms_total"] = (
            self.gpu_admission_candidate_value_index_ms_total
        )
        stats["gpu_admission_incumbent_value_index_ms_total"] = (
            self.gpu_admission_incumbent_value_index_ms_total
        )
        stats["gpu_admission_displacement_value_index_ms_total"] = (
            self.gpu_admission_displacement_value_index_ms_total
        )
        stats["gpu_bypassed_prefill_ms_total"] = (
            self.gpu_bypassed_prefill_ms_total
        )
        stats["gpu_admission_selection_rate"] = (
            self.gpu_admission_selected_count
            / self.gpu_admission_candidate_count
            if self.gpu_admission_candidate_count else 0.0
        )
        stats["gpu_admission_bypass_rate"] = (
            self.gpu_admission_bypassed_count
            / self.gpu_admission_candidate_count
            if self.gpu_admission_candidate_count else 0.0
        )
        stats["migration_selection_rate"] = (
            self.migration_selected_count / self.migration_candidate_count
            if self.migration_candidate_count else 0.0
        )
        stats["restore_per_migration"] = (
            stats.get("restore_to_gpu_count", 0)
            / stats.get("migrate_to_cpu_count", 1)
            if stats.get("migrate_to_cpu_count", 0) else 0.0
        )
        stats["policy_cost_per_decision_ms"] = self.policy_cost_per_decision_ms
        stats["policy_model_ms_total"] = self.policy_model_ms_total
        stats["policy_decision_count"] = self.policy_decision_count
        stats["replay_wall_ms_total"] = self.replay_wall_ms_total
        stats["replay_wall_us_per_access"] = (
            self.replay_wall_ms_total * 1000.0 / self.policy_decision_count
            if self.policy_decision_count else 0.0
        )
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
