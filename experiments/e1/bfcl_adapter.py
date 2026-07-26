"""
BFCL v3 multi-turn 真实 backend 适配器（G1 实验用）。

⚠️ v0.5（2026-07-26）DISABLED：BFCL 不再作为 FlowCache 数据集。本 adapter 作为
disabled 代码保留以备 rebuttal 时按 IDEA.rewritten.md §6.1 migration 规则恢复使用。
当前实验配置（experiments/e1/config.yaml）已设为 tau-bench only（datasets: ["tau-bench"]），
本 adapter 不会被实例化。详见 experiments/experiment-designs.md v0.5 注。

对接 BFCL v3 (Berkeley Function-Calling Leaderboard v3) 的 multi-turn 子集：
  - 4 子集 × 200 = 800 episodes 全量（multi_turn_base / miss_func / miss_param / long_context）
  - 8 个真实 sim 工具类（VehicleControlAPI / TwitterAPI / GorillaFileSystem 等）
  - scripted user turns（1-7 轮/episode，固定字符串，无 LLM 模拟器）
  - 状态验证（multi_turn_checker 比较 backend 终态）

依赖：
  - pip install bfcl-eval
  - 设置 BFCL_PROJECT_ROOT 环境变量指向 bfcl_eval 源码根目录
  - transformers / torch（FlowCache 自有，用于 agent 推理）

seed 注入：
  BFCL 默认 temperature=0，8 seeds 跑出来相同。
  本 adapter 不接管模型推理（由 record_trajectories.py 负责），
  但提供 seed 配置，record_trajectories.py 据此设 do_sample=True, temperature=0.7。

关键差异（vs τ-bench）：
  - τ-bench 用 LLM user simulator，seed 注入 user model
  - BFCL 用 scripted user turns，seed 注入 agent model decode
  - 二者方法论互补，G1 报告需明确披露

参考：
  - https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
  - bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py
  - bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py
  - bfcl_eval/constants/executable_backend_config.py
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------

BFCL_MULTI_TURN_SUBSETS = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)

# 8 个 sim 工具类（来自 bfcl_eval.constants.executable_backend_config）
BFCL_SIM_CLASSES = (
    "GorillaFileSystem",
    "MathAPI",
    "MessageAPI",
    "TwitterAPI",
    "TicketAPI",
    "TradingBot",
    "TravelAPI",
    "VehicleControlAPI",
)

# 无状态类（不需要 _load_scenario）
BFCL_STATELESS_CLASSES = ("MathAPI",)


# ----------------------------------------------------------------------
# Lazy import helpers
# ----------------------------------------------------------------------

_BFCL_AVAILABLE: Optional[bool] = None


def _check_bfcl_available() -> bool:
    """检测 bfcl_eval 包是否可导入。结果缓存。"""
    global _BFCL_AVAILABLE
    if _BFCL_AVAILABLE is None:
        try:
            import bfcl_eval  # noqa: F401
            _BFCL_AVAILABLE = True
        except ImportError:
            _BFCL_AVAILABLE = False
    return _BFCL_AVAILABLE


def _require_bfcl():
    """导入 bfcl_eval，失败时给出安装提示。"""
    if not _check_bfcl_available():
        raise ImportError(
            "bfcl_eval 包未安装。请安装：\n"
            "  pip install bfcl-eval\n"
            "并设置环境变量 BFCL_PROJECT_ROOT 指向 bfcl_eval 源码根目录。\n"
            "详见 https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard"
        )
    import bfcl_eval  # noqa: F401
    return bfcl_eval


def _ensure_bfcl_root():
    """检查 BFCL_PROJECT_ROOT 环境变量。"""
    if not os.environ.get("BFCL_PROJECT_ROOT"):
        logger.warning(
            "BFCL_PROJECT_ROOT 环境变量未设置。bfcl-eval 的 result/score 会落到包源码深处。"
            "建议：export BFCL_PROJECT_ROOT=/path/to/berkeley-function-call-leaderboard"
        )


# ----------------------------------------------------------------------
# BFCLEpisode：单条 episode 数据结构
# ----------------------------------------------------------------------

class BFCLEpisode:
    """BFCL 单条 episode 的数据结构（用于录制 trace）。"""

    def __init__(
        self,
        entry_id: str,
        subset: str,
        seed: int,
        involved_classes: List[str],
        initial_config: Dict[str, Any],
    ):
        self.entry_id = entry_id
        self.subset = subset
        self.seed = seed
        self.involved_classes = involved_classes
        self.initial_config = initial_config

        # 录制时填充
        self.user_turns: List[str] = []          # scripted user turns (question 字段)
        self.tool_calls: List[List[List[str]]] = []  # [turn][step] of parallel call strings
        self.tool_results: List[List[str]] = []  # [turn][step] of execution results
        self.agent_responses: List[str] = []     # agent 每轮的文本回复
        self.prefill_ms: List[float] = []
        self.decode_ms: List[float] = []
        self.token_ids: List[List[int]] = []     # 每轮 prefill+decode 的 token ids
        self.block_assignments: List[List[Dict]] = []  # 每轮的 block 分配

        # 验证结果
        self.valid: Optional[bool] = None
        self.error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（用于 JSON 落盘）。"""
        return {
            "entry_id": self.entry_id,
            "subset": self.subset,
            "seed": self.seed,
            "involved_classes": self.involved_classes,
            "initial_config": self.initial_config,
            "user_turns": self.user_turns,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "agent_responses": self.agent_responses,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "token_ids": self.token_ids,
            "block_assignments": self.block_assignments,
            "valid": self.valid,
            "error_type": self.error_type,
        }


# ----------------------------------------------------------------------
# BFCLAdapter：对接 BFCL v3 multi-turn backend
# ----------------------------------------------------------------------

class BFCLAdapter:
    """
    BFCL v3 multi-turn 真实 backend 适配器。

    封装 BFCL 的数据加载、sim 类初始化、工具执行、状态验证，为
    record_trajectories.py 提供统一接口。

    用法：
        adapter = BFCLAdapter(subset="multi_turn_base")
        entries = adapter.load_entries()
        for entry, gt in entries:
            episode = adapter.init_episode(entry, gt, seed=42)
            for user_msg in episode.user_turns:
                # ... agent 生成回复 + tool calls ...
                results = adapter.execute_tool_calls(
                    tool_call_strings, episode
                )
                adapter.validate_episode(episode)
    """

    def __init__(self, subset: str = "multi_turn_base"):
        """
        Args:
            subset: BFCL multi-turn 子集名，必须为：
                - "multi_turn_base"
                - "multi_turn_miss_func"
                - "multi_turn_miss_param"
                - "multi_turn_long_context"
        """
        if subset not in BFCL_MULTI_TURN_SUBSETS:
            raise ValueError(
                f"subset must be one of {BFCL_MULTI_TURN_SUBSETS}, got {subset!r}"
            )

        _require_bfcl()
        _ensure_bfcl_root()

        self.subset = subset
        self._is_long_context = (subset == "multi_turn_long_context")
        self._is_miss_func = (subset == "multi_turn_miss_func")
        self._is_miss_param = (subset == "multi_turn_miss_param")

        logger.info("BFCLAdapter(subset=%s) 初始化完成", subset)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_entries(self) -> List[Tuple[Dict, Dict]]:
        """加载 (entry, ground_truth) 对，全量 200 条。

        Returns:
            List of (test_entry, gt_entry) tuples
            test_entry: {id, question, initial_config, involved_classes, path, ...}
            gt_entry: {id, ground_truth: List[List[str]]}
        """
        from bfcl_eval.utils import load_file

        dataset_name = f"BFCL_v3_{self.subset}"
        test_entries = load_file(dataset_name)
        gt_entries = load_file(dataset_name, is_gt=True)

        assert len(test_entries) == len(gt_entries), (
            f"test_entries ({len(test_entries)}) 与 gt_entries ({len(gt_entries)}) 数量不匹配"
        )
        logger.info("Loaded %d entries from %s", len(test_entries), dataset_name)
        return list(zip(test_entries, gt_entries))

    # ------------------------------------------------------------------
    # Episode 初始化
    # ------------------------------------------------------------------

    def init_episode(self, entry: Dict, gt: Dict, seed: int = 0) -> BFCLEpisode:
        """初始化一个 episode，构造 backend 实例。

        Args:
            entry: test entry dict
            gt: ground truth dict
            seed: model decode seed（与 τ-bench 的 user simulator seed 语义不同）

        Returns:
            BFCLEpisode 实例，已填充 user_turns / initial_config / involved_classes
        """
        entry_id = entry.get("id", "unknown")
        involved_classes = entry.get("involved_classes", [])
        initial_config = entry.get("initial_config", {})

        # 提取 scripted user turns
        user_turns = []
        question = entry.get("question", [])
        for turn_messages in question:
            # 每个 turn 是一个 message list，通常只有一条 user message
            for msg in turn_messages:
                if msg.get("role") == "user":
                    user_turns.append(msg.get("content", ""))
                    break

        episode = BFCLEpisode(
            entry_id=entry_id,
            subset=self.subset,
            seed=seed,
            involved_classes=involved_classes,
            initial_config=initial_config,
        )
        episode.user_turns = user_turns
        episode._gt = gt  # 暂存 ground truth 用于后续验证
        episode._entry = entry  # 暂存原始 entry

        # 构造 sim backend 实例（用唯一 model_name 做 globals() 隔离）
        episode._model_name = f"flowcache_{self.subset}_{entry_id}_s{seed}"
        episode._backend_instances = self._init_backend_instances(
            initial_config=initial_config,
            involved_classes=involved_classes,
            model_name=episode._model_name,
        )

        return episode

    def _init_backend_instances(
        self,
        initial_config: Dict[str, Any],
        involved_classes: List[str],
        model_name: str,
    ) -> Dict[str, Any]:
        """初始化 sim 工具类实例。

        参考 bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call
        的初始化逻辑。

        Args:
            initial_config: {class_name: config_dict}
            involved_classes: ["TwitterAPI", "VehicleControlAPI", ...]
            model_name: 唯一标识，用于 globals() 隔离

        Returns:
            {class_name: instance}
        """
        import importlib

        # 类名到模块路径的映射
        # 来自 bfcl_eval.constants.executable_backend_config
        class_to_module = {
            "GorillaFileSystem": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system",
            "MathAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.math_api",
            "MessageAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.message_api",
            "TwitterAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.posting_api",
            "TicketAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.ticket_api",
            "TradingBot": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot",
            "TravelAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.travel_booking",
            "VehicleControlAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.vehicle_control",
        }

        instances: Dict[str, Any] = {}
        for class_name in involved_classes:
            if class_name not in class_to_module:
                logger.warning("未知 sim 类: %s，跳过", class_name)
                continue

            module_path = class_to_module[class_name]
            try:
                module = importlib.import_module(module_path)
                class_ = getattr(module, class_name)
                instance = class_()

                # 有状态类需要 _load_scenario
                if class_name not in BFCL_STATELESS_CLASSES:
                    config = copy.deepcopy(initial_config.get(class_name, {}))
                    if hasattr(instance, "_load_scenario"):
                        instance._load_scenario(
                            config,
                            long_context=self._is_long_context,
                        )

                instances[class_name] = instance
                # 同时塞进 globals()（BFCL 的 eval() 执行需要）
                # 用 model_name 前缀做隔离
                globals()[f"{model_name}_{class_name}_instance"] = instance

            except (ImportError, AttributeError) as e:
                logger.error("初始化 sim 类 %s 失败: %s", class_name, e)
                raise

        return instances

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def execute_tool_calls(
        self,
        tool_call_strings: List[str],
        episode: BFCLEpisode,
    ) -> List[str]:
        """执行一组并行的 tool calls。

        BFCL 的 tool call 是 Python 语法字符串，如 "fillFuelTank(fuelAmount=30.0)"。
        本方法用 execute_multi_turn_func_call 执行（内部用 eval）。

        Args:
            tool_call_strings: ["fillFuelTank(fuelAmount=30.0)", "post_tweet(content='...')"]
            episode: 当前 episode

        Returns:
            List of execution result strings
        """
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        if not tool_call_strings:
            return []

        try:
            # execute_multi_turn_func_call 返回 (results, involved_instances)
            results, _ = execute_multi_turn_func_call(
                func_call_list=tool_call_strings,
                initial_config=episode.initial_config,
                involved_classes=episode.involved_classes,
                model_name=episode._model_name,
                test_entry_id=episode.entry_id,
                long_context=self._is_long_context,
                is_evaL_run=False,
            )
            return results
        except Exception as e:
            logger.error("Tool 执行失败 (entry=%s): %s", episode.entry_id, e)
            episode.error_type = f"execution_error: {type(e).__name__}: {e}"
            return [f"ERROR: {e}"] * len(tool_call_strings)

    # ------------------------------------------------------------------
    # 状态验证
    # ------------------------------------------------------------------

    def validate_episode(self, episode: BFCLEpisode) -> bool:
        """验证 episode 的最终状态是否匹配 ground truth。

        使用 multi_turn_checker 比较 backend 终态。

        Args:
            episode: 已完成录制的 episode

        Returns:
            True 如果状态匹配 ground truth
        """
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
            multi_turn_irrelevance_checker,
        )

        gt = getattr(episode, "_gt", None)
        if gt is None:
            logger.warning("episode %s 无 ground truth，跳过验证", episode.entry_id)
            episode.valid = None
            return False

        # tool_calls 格式：[turn][step] of parallel call strings
        # multi_turn_checker 期望 list[list[list[str]]]（外层=turn，内层=step，最内层=parallel calls）
        model_results = episode.tool_calls if episode.tool_calls else [[]]

        try:
            result = multi_turn_checker(
                multi_turn_model_result_list_decoded=model_results,
                multi_turn_ground_truth_list=gt.get("ground_truth", []),
                test_entry=episode._entry,
                test_category=self.subset,
                model_name=episode._model_name,
            )
            episode.valid = result.get("valid", False)
            if not episode.valid:
                episode.error_type = result.get("error_type", "unknown")
        except Exception as e:
            logger.error("验证失败 (entry=%s): %s", episode.entry_id, e)
            episode.valid = False
            episode.error_type = f"validation_error: {type(e).__name__}: {e}"

        return episode.valid or False

    # ------------------------------------------------------------------
    # Tool schema（供 agent 推理用）
    # ------------------------------------------------------------------

    def get_tool_schema_for_qwen(self, episode: BFCLEpisode) -> str:
        """返回适合 Qwen2.5 chat template 的 tool schema 文本。

        根据 involved_classes 提取可用工具，转 Qwen 兼容格式。

        Args:
            episode: 当前 episode

        Returns:
            tool schema 描述文本
        """
        lines = ["Available tools (use <tool_call> JSON to invoke):"]
        for class_name in episode.involved_classes:
            instance = episode._backend_instances.get(class_name)
            if instance is None:
                continue
            # 遍历实例的公开方法（非 _ 前缀）
            for method_name in dir(instance):
                if method_name.startswith("_"):
                    continue
                method = getattr(instance, method_name, None)
                if not callable(method):
                    continue
                # 尝试获取方法签名
                try:
                    import inspect
                    sig = inspect.signature(method)
                    params = [f"{p}" for p in sig.parameters.values()]
                    lines.append(f"- {class_name}.{method_name}({', '.join(params)})")
                except (ValueError, TypeError):
                    lines.append(f"- {class_name}.{method_name}(...)")

        return "\n".join(lines)

    def get_tool_schema_openai(self, episode: BFCLEpisode) -> List[Dict]:
        """返回 OpenAI function-call 格式的 tool schema。

        Args:
            episode: 当前 episode

        Returns:
            [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        tools = []
        for class_name in episode.involved_classes:
            instance = episode._backend_instances.get(class_name)
            if instance is None:
                continue
            for method_name in dir(instance):
                if method_name.startswith("_"):
                    continue
                method = getattr(instance, method_name, None)
                if not callable(method):
                    continue
                # 简化版 schema，实际需要更精细的参数类型推断
                try:
                    import inspect
                    sig = inspect.signature(method)
                    properties = {}
                    for p in sig.parameters.values():
                        properties[p.name] = {"type": "string"}  # 默认 string
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": f"{class_name}.{method_name}",
                            "description": method.__doc__ or "",
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                            },
                        },
                    })
                except (ValueError, TypeError):
                    continue
        return tools

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close_episode(self, episode: BFCLEpisode):
        """清理 episode 的 backend 实例，从 globals() 移除。"""
        model_name = getattr(episode, "_model_name", None)
        if model_name:
            # 移除 globals() 中的实例引用
            for class_name in episode.involved_classes:
                key = f"{model_name}_{class_name}_instance"
                if key in globals():
                    del globals()[key]
        episode._backend_instances.clear()

    def close(self):
        """清理 adapter 资源。"""
        logger.debug("BFCLAdapter closed")


# ----------------------------------------------------------------------
# 便捷函数：批量加载所有 4 子集
# ----------------------------------------------------------------------

def load_all_bfcl_episodes(
    subsets: Tuple[str, ...] = BFCL_MULTI_TURN_SUBSETS,
) -> List[Tuple[str, Dict, Dict]]:
    """
    加载所有 4 子集的 (entry, gt) 对，共 800 条。

    Args:
        subsets: 要加载的子集元组，默认全量 4 子集

    Returns:
        List of (subset, entry, gt) tuples
    """
    all_entries: List[Tuple[str, Dict, Dict]] = []
    stats: Dict[str, int] = {}

    for subset in subsets:
        adapter = BFCLAdapter(subset=subset)
        try:
            entries = adapter.load_entries()
            stats[subset] = len(entries)
            for entry, gt in entries:
                all_entries.append((subset, entry, gt))
        finally:
            adapter.close()

    stats["total"] = sum(stats.values())
    logger.info("Loaded %d BFCL episodes total: %s", stats["total"], stats)
    return all_entries
