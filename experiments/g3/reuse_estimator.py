"""
Heuristic Reuse Estimator
=========================
为 FlowCache-Lossless controller 提供 legacy heuristic/fallback 估计。

不使用 GNN（G5 已删除），采用可解释的 heuristic 公式：

    R_b = exp(-β · age) · (1 + α · share_count) · position_weight(block_idx)

信号：
- age: 自上次访问的步数（越小越可能复用）
- share_count: H 窗口内访问该 block 的不同 workflow 数（越大越可能复用）
- block_idx: 前缀位置（越靠前越可能复用——system prompt 等共享根）

G3-P1 的 selective admission 不直接把该无界 R 当作概率，而是在
controller.py 中用相同的决策时信号构造 [0,1] 的容量归一化 proxy；
固定 horizon 也不再用于 selective victim admission。
"""

import math
from typing import Dict, Optional


class HeuristicReuseEstimator:
    """
    Heuristic reuse-value estimator.

    Estimates R_b for an inactive block at decision time t.
    All signals are observable at decision time (no future info, no Oracle labels).
    """

    def __init__(self,
                 beta: float = 0.005,
                 alpha: float = 0.5,
                 position_weights: Optional[Dict] = None,
                 horizon: int = 1000):
        """
        Args:
            beta: age 衰减率（0.005/step ≈ 200 step 半衰期）
            alpha: share_count 权重
            position_weights: block_idx 位置权重配置
            horizon: H 窗口（step），超出则 R=0
        """
        self.beta = beta
        self.alpha = alpha
        self.horizon = horizon
        # 位置权重默认值（与 config.yaml 一致）
        self.position_weights = position_weights or {
            "early": {"block_idx_lt": 10, "weight": 1.5},
            "mid": {"block_idx_lt": 50, "weight": 1.0},
            "late": {"weight": 0.7},
        }

    def _position_weight(self, block_idx: int) -> float:
        """根据 block_idx 返回位置权重（越靠前权重越高）。"""
        early = self.position_weights.get("early", {})
        mid = self.position_weights.get("mid", {})
        late = self.position_weights.get("late", {})
        if block_idx < early.get("block_idx_lt", 10):
            return early.get("weight", 1.5)
        if block_idx < mid.get("block_idx_lt", 50):
            return mid.get("weight", 1.0)
        return late.get("weight", 0.7)

    def estimate(self,
                 age: int,
                 share_count: int,
                 block_idx: int) -> float:
        """
        估计 block 的复用价值 R_b。

        Args:
            age: 自上次访问的步数（0 = 刚访问）
            share_count: H 窗口内访问该 block 的不同 workflow 数
            block_idx: 前缀位置（0 = 最前）

        Returns:
            R_b ∈ [0, ∞)。age > horizon 时返回 0。
        """
        if age >= self.horizon:
            return 0.0
        if age < 0:
            age = 0
        if share_count < 0:
            share_count = 0

        decay = math.exp(-self.beta * age)
        share_factor = 1.0 + self.alpha * share_count
        pos_w = self._position_weight(block_idx)
        return decay * share_factor * pos_w

    def static_log_priority(self,
                            last_access: int,
                            share_count: int,
                            block_idx: int) -> float:
        """Return the clock-independent ordering key for non-expired blocks.

        For a fixed decision clock ``t``:

            R_b(t) = exp(-beta * t)
                     * exp(beta * last_access_b)
                     * share_factor_b
                     * position_weight_b

        The first factor is shared by every non-expired block.  Therefore the
        lowest-R CPU victim can be maintained in a heap keyed by this log
        priority instead of rescanning the full CPU cache on every migration.
        Expired blocks are handled separately by the controller.
        """
        share_count = max(0, share_count)
        share_factor = max(1e-12, 1.0 + self.alpha * share_count)
        pos_w = max(1e-12, self._position_weight(block_idx))
        return (
            self.beta * last_access
            + math.log(share_factor)
            + math.log(pos_w)
        )

    def estimate_batch(self, blocks: list) -> list:
        """
        批量估计多个 block 的 R 值。

        Args:
            blocks: list of dict，每个 dict 包含 age, share_count, block_idx

        Returns:
            list of float，与 blocks 等长
        """
        return [
            self.estimate(
                age=b.get("age", 0),
                share_count=b.get("share_count", 0),
                block_idx=b.get("block_idx", 0),
            )
            for b in blocks
        ]


class SurvivalReuseEstimator:
    """
    Survival/hazard 模型估计器（W8 后升级用，仍非 GNN）。

    用 Kaplan-Meier 或参数化 hazard 估计 P(reuse within H | features)。
    当前为占位实现，返回与 heuristic 相同的值；W8 冒烟后用真实数据拟合。
    """

    def __init__(self, horizon: int = 1000):
        self.horizon = horizon
        self._fitted = False
        # 占位：拟合后的 hazard 参数
        self._baseline_hazard = 0.005

    def fit(self, reuse_labels: list):
        """
        用 R 标签数据拟合 survival 模型。

        Args:
            reuse_labels: list of dict，每个 dict 包含 age, reused (0/1), features
        """
        # TODO: W8 冒烟后用真实数据实现 Kaplan-Meier 拟合
        # 当前为占位，仅设置 fitted 标志
        self._fitted = True

    def estimate(self, age: int, share_count: int, block_idx: int) -> float:
        """估计 R_b（当前退化为 heuristic）。"""
        if not self._fitted:
            # 未拟合时退化为 heuristic
            est = HeuristicReuseEstimator(horizon=self.horizon)
            return est.estimate(age, share_count, block_idx)
        # 拟合后用 survival 概率（占位实现）
        if age >= self.horizon:
            return 0.0
        survival = math.exp(-self._baseline_hazard * age)
        return survival * (1.0 + 0.5 * share_count)

    def static_log_priority(self,
                            last_access: int,
                            share_count: int,
                            block_idx: int) -> float:
        """Clock-independent ordering key used by the CPU victim heap."""
        if not self._fitted:
            est = HeuristicReuseEstimator(horizon=self.horizon)
            return est.static_log_priority(
                last_access, share_count, block_idx
            )
        share_factor = max(1e-12, 1.0 + 0.5 * max(0, share_count))
        return self._baseline_hazard * last_access + math.log(share_factor)


def create_estimator(estimator_type: str = "heuristic",
                     config: Optional[Dict] = None) -> object:
    """
    工厂函数：根据类型创建估计器。

    Args:
        estimator_type: "heuristic" 或 "survival"
        config: 配置字典（来自 config.yaml 的 flowcache.heuristic）

    Returns:
        ReuseEstimator 实例
    """
    if config is None:
        config = {}
    if estimator_type == "survival":
        return SurvivalReuseEstimator(horizon=config.get("horizon", 1000))
    # 默认 heuristic
    return HeuristicReuseEstimator(
        beta=config.get("beta", 0.005),
        alpha=config.get("alpha", 0.5),
        position_weights=config.get("position_weights"),
        horizon=config.get("horizon", 1000),
    )
