"""
τ-bench 真实 backend 适配器（G1 实验用）。

替换 record_trajectories.py 中的 mock _simulate_tool_result / _simulate_user_response
与 _get_domain_policy 硬编码，对接 tau-bench pip 包（sierra-research/tau-bench）的真实：
  - 165 任务全量加载（115 retail + 50 airline，test split）
  - 真实 system policy（env.wiki，替代硬编码 minimal policy）
  - 真实 tool schema（env.tools_info，OpenAI function-call 格式）
  - 真实 tool 执行（env.step(Action)，操作真实 env.data DB）
  - LLM user simulator（env.step(respond) 触发 llm_user 生成下一句）

依赖：
  - tau-bench 源码安装：pip install -e git+https://github.com/sierra-research/tau-bench.git#egg=tau_bench
  - litellm（tau-bench 依赖）：用于调用 OpenAI/Anthropic 等 API
  - OPENAI_API_KEY 环境变量（默认 user_model=gpt-4o-mini）

seed 注入：
  tau-bench 的 LLMUserSimulationEnv 原生不支持 temperature/seed 参数。
  本 adapter 通过子类化 SeededLLMUser，在 litellm.completion 调用中显式传入
  temperature 与 seed（OpenAI API 支持 extra_body={"seed": ...}）。
  这与 τ-bench 原论文 pass^k（k≤8）的 seed 语义对齐。

参考：
  - https://github.com/sierra-research/tau-bench/blob/main/tau_bench/envs/base.py
  - https://github.com/sierra-research/tau-bench/blob/main/tau_bench/envs/user.py
  - https://github.com/sierra-research/tau-bench/blob/main/tau_bench/types.py
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Lazy import helpers（tau-bench 未发布 PyPI，源码安装，需友好错误提示）
# ----------------------------------------------------------------------

_TAU_BENCH_AVAILABLE: Optional[bool] = None


def _check_tau_bench_available() -> bool:
    """检测 tau_bench 包是否可导入。结果缓存。"""
    global _TAU_BENCH_AVAILABLE
    if _TAU_BENCH_AVAILABLE is None:
        try:
            import tau_bench  # noqa: F401
            _TAU_BENCH_AVAILABLE = True
        except ImportError:
            _TAU_BENCH_AVAILABLE = False
    return _TAU_BENCH_AVAILABLE


def _require_tau_bench():
    """导入 tau_bench，失败时给出安装提示。"""
    if not _check_tau_bench_available():
        raise ImportError(
            "tau_bench 包未安装。请源码安装：\n"
            "  pip install -e git+https://github.com/sierra-research/tau-bench.git#egg=tau_bench\n"
            "tau-bench 未发布到 PyPI，必须源码安装。详见 "
            "https://github.com/sierra-research/tau-bench"
        )
    import tau_bench  # noqa: F401
    return tau_bench


# ----------------------------------------------------------------------
# SeededLLMUser：子类化 LLMUserSimulationEnv 注入 temperature + seed
# ----------------------------------------------------------------------

class SeededLLMUser:
    """
    带 temperature 与 seed 的 LLM user simulator。

    tau-bench 的 LLMUserSimulationEnv.__init__(model, provider) 没有
    temperature/seed 参数，内部 litellm.completion() 调用也未传 seed。
    本类通过 monkey-patch litellm.completion 的方式注入 temperature 与 seed，
    避免重写整个 LLMUserSimulationEnv。

    使用方式：
        user = SeededLLMUser(model="gpt-4o-mini", provider="openai",
                             temperature=0.7, seed=42)
        user.reset(instruction)
        next_msg = user.step(agent_content)

    seed 语义：与 τ-bench 原论文 pass^k 对齐。OpenAI API 的 seed 参数是
    best-effort（不保证完全确定性），但在 temperature>0 时能产生不同变体，
    满足 G1 的 8 seeds 需求。
    """

    def __init__(self, model: str, provider: str,
                 temperature: float = 0.7, seed: int = 0):
        from tau_bench.envs.user import LLMUserSimulationEnv
        import litellm

        self._model = model
        self._provider = provider
        self._temperature = temperature
        self._seed = seed
        self._original_completion = litellm.completion

        # 创建底层 user simulator
        self._user = LLMUserSimulationEnv(model=model, provider=provider)

        # Monkey-patch litellm.completion 注入 temperature + seed
        temperature_val = temperature
        seed_val = seed
        original = self._original_completion

        def _patched_completion(*args, **kwargs):
            # 仅对当前 model 的调用注入，避免影响其他 litellm 调用
            if kwargs.get("model") == model:
                kwargs.setdefault("temperature", temperature_val)
                # OpenAI 透传 seed
                extra_body = kwargs.get("extra_body", {}) or {}
                extra_body.setdefault("seed", seed_val)
                kwargs["extra_body"] = extra_body
            return original(*args, **kwargs)

        litellm.completion = _patched_completion
        self._patched_litellm = litellm

    def reset(self, instruction: Optional[str] = None) -> str:
        return self._user.reset(instruction)

    def step(self, content: str) -> str:
        return self._user.step(content)

    def close(self):
        """恢复 litellm.completion 原始实现。"""
        if self._original_completion is not None:
            self._patched_litellm.completion = self._original_completion


# ----------------------------------------------------------------------
# TauBenchAdapter：对接 tau-bench 真实 backend
# ----------------------------------------------------------------------

class TauBenchAdapter:
    """
    τ-bench 真实 backend 适配器。

    封装 tau-bench 的 Env / Task / LLMUserSimulationEnv，为
    record_trajectories.py 提供统一接口：
      - list_tasks()：165 任务全量
      - get_system_policy()：env.wiki（真实 markdown policy）
      - get_tools_info()：env.tools_info（OpenAI function schema）
      - reset(task_index)：初始化 episode，返回初始 user 消息
      - step_tool(action)：执行工具调用，返回 EnvResponse
      - step_respond(content)：触发 user simulator 生成下一句
      - close()：清理资源

    用法：
        adapter = TauBenchAdapter(domain="retail", seed=42,
                                  user_model="gpt-4o-mini")
        tasks = adapter.list_tasks()
        for i in range(len(tasks)):
            obs = adapter.reset(i)
            # ... agent 生成回复 ...
            resp = adapter.step_tool(action)
            # 或 resp = adapter.step_respond(agent_text)
            if resp.done:
                break
    """

    def __init__(
        self,
        domain: str,
        seed: int = 0,
        user_model: str = "gpt-4o-mini",
        user_provider: str = "openai",
        user_temperature: float = 0.7,
        task_split: str = "test",
    ):
        """
        Args:
            domain: "retail" 或 "airline"
            seed: user simulator 的 seed（与 τ-bench pass^k 对齐）
            user_model: user simulator 用的 LLM（默认 gpt-4o-mini，需 OPENAI_API_KEY）
            user_provider: litellm provider，如 "openai"
            user_temperature: user simulator 采样温度（>0 才能产生 seed 变体）
            task_split: "test"（165 任务）/ "train" / "dev"

        Note:
            4090D 24GB 显存不足以同时跑本地 Qwen2.5-7B agent + 本地 user model，
            故 user simulator 默认走 OpenAI API。如需本地 user model，需另起
            vLLM 服务并设 OPENAI_API_BASE=http://localhost:8000/v1。
        """
        if domain not in ("retail", "airline"):
            raise ValueError(f"domain must be 'retail' or 'airline', got {domain!r}")

        _require_tau_bench()
        from tau_bench.envs import get_env
        from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: F401

        self.domain = domain
        self.seed = seed
        self.task_split = task_split
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_temperature = user_temperature

        # Action 常量与类型
        self._RESPOND_ACTION_NAME = RESPOND_ACTION_NAME
        self._Action = Action

        # 构造 SeededLLMUser（注入 temperature + seed）
        self._seeded_user = SeededLLMUser(
            model=user_model,
            provider=user_provider,
            temperature=user_temperature,
            seed=seed,
        )

        # 构造 env（user_strategy="llm" 使用 LLMUserSimulationEnv）
        # get_env 内部会用 self._seeded_user._user 作为 user simulator
        # 注意：get_env 不直接接受 user 实例，需通过 user_strategy + user_model 构造
        # 这里我们构造 env 后，手动替换 env.user_sim 为 seeded_user
        self._env = get_env(
            env_name=domain,
            user_strategy="llm",
            user_model=user_model,
            user_provider=user_provider,
            task_split=task_split,
        )
        # 替换 env 内部的 user simulator 为 seeded 版本
        # tau_bench.envs.base.Env 有 self.user_simulator 属性
        if hasattr(self._env, "user_simulator"):
            self._env.user_simulator = self._seeded_user._user
        elif hasattr(self._env, "user_sim"):
            self._env.user_sim = self._seeded_user._user
        else:
            logger.warning(
                "无法替换 env 的 user simulator，seed 注入可能失效。"
                "tau-bench 版本可能已变更 user simulator 属性名。"
            )

        self._current_task_index: Optional[int] = None
        logger.info(
            "TauBenchAdapter(domain=%s, seed=%d, user_model=%s, task_split=%s) 初始化完成",
            domain, seed, user_model, task_split,
        )

    # ------------------------------------------------------------------
    # Task 加载
    # ------------------------------------------------------------------

    def list_tasks(self) -> List[Any]:
        """返回 165 任务全量（test split: 115 retail + 50 airline）。"""
        tasks = self._env.tasks
        logger.info("Loaded %d tasks from %s/%s", len(tasks), self.domain, self.task_split)
        return tasks

    def get_task(self, task_index: int) -> Any:
        """获取指定 index 的任务。"""
        return self._env.tasks[task_index]

    # ------------------------------------------------------------------
    # System policy & tools
    # ------------------------------------------------------------------

    def get_system_policy(self) -> str:
        """返回真实 system policy（env.wiki，markdown 格式）。

        替换 record_trajectories.py 的 _get_domain_policy 硬编码。
        """
        return self._env.wiki

    def get_rules(self) -> List[str]:
        """返回 domain rules（env.rules）。"""
        return self._env.rules

    def get_tools_info(self) -> List[Dict]:
        """返回 OpenAI function-call 格式的 tool schema（env.tools_info）。

        可直接注入 Qwen 的 system prompt 或用于 function-calling。
        """
        return self._env.tools_info

    def get_tools_schema_for_qwen(self) -> str:
        """返回适合 Qwen2.5 chat template 的 tool schema 文本。

        Qwen2.5 支持 <tool_call> 格式，需要把 OpenAI tools_info 转为
        Qwen 兼容的文本描述。
        """
        tools = self._env.tools_info
        lines = ["Available tools (use <tool_call> JSON to invoke):"]
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                fn = tool["function"]
                name = fn.get("name", "unknown")
                desc = fn.get("description", "")
                params = fn.get("parameters", {}).get("properties", {})
                param_strs = [f"{k}: {v.get('type', 'any')}" for k, v in params.items()]
                lines.append(f"- {name}({', '.join(param_strs)}): {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Episode 控制
    # ------------------------------------------------------------------

    def reset(self, task_index: int) -> Dict[str, Any]:
        """重置 env 到指定 task，返回初始观察。

        Returns:
            {"observation": str, "info": dict, "task": Task}
        """
        self._current_task_index = task_index
        resp = self._env.reset(task_index=task_index)
        return {
            "observation": resp.observation,
            "info": resp.info.model_dump() if hasattr(resp.info, "model_dump") else vars(resp.info),
            "task": self._env.tasks[task_index],
        }

    def step_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """执行一次工具调用。

        Args:
            tool_name: 工具名（如 "get_order_details"）
            kwargs: 工具参数

        Returns:
            {"observation": str, "reward": float, "done": bool, "info": dict}
        """
        action = self._Action(name=tool_name, kwargs=kwargs)
        return self._step(action)

    def step_respond(self, content: str) -> Dict[str, Any]:
        """让 agent 回复用户，触发 user simulator 生成下一句。

        Args:
            content: agent 回复内容

        Returns:
            {"observation": str, "reward": float, "done": bool, "info": dict}
            observation 是 user simulator 生成的下一句用户消息，
            或 "###STOP###" 表示会话结束。
        """
        action = self._Action(name=self._RESPOND_ACTION_NAME, kwargs={"content": content})
        return self._step(action)

    def _step(self, action) -> Dict[str, Any]:
        """内部 step 封装，统一返回 dict。"""
        resp = self._env.step(action)
        info_dict = resp.info.model_dump() if hasattr(resp.info, "model_dump") else vars(resp.info)
        return {
            "observation": resp.observation,
            "reward": resp.reward,
            "done": resp.done,
            "info": info_dict,
        }

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close(self):
        """清理资源，恢复 litellm 原始实现。"""
        self._seeded_user.close()
        logger.debug("TauBenchAdapter closed")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ----------------------------------------------------------------------
# 便捷函数：批量加载所有 165 任务
# ----------------------------------------------------------------------

def load_all_tau_bench_tasks(
    user_model: str = "gpt-4o-mini",
    user_provider: str = "openai",
    user_temperature: float = 0.7,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int, Any]], Dict[str, int]]:
    """
    加载 retail + airline 两域的 165 任务全量。

    Args:
        user_model/user_provider/user_temperature/seed: 见 TauBenchAdapter

    Returns:
        (tasks, stats)
        tasks: List of (domain, task_index, task_obj)
        stats: {"retail": 115, "airline": 50, "total": 165}
    """
    all_tasks: List[Tuple[str, int, Any]] = []
    stats: Dict[str, int] = {}

    for domain in ("retail", "airline"):
        adapter = TauBenchAdapter(
            domain=domain,
            seed=seed,
            user_model=user_model,
            user_provider=user_provider,
            user_temperature=user_temperature,
        )
        try:
            tasks = adapter.list_tasks()
            stats[domain] = len(tasks)
            for i, task in enumerate(tasks):
                all_tasks.append((domain, i, task))
        finally:
            adapter.close()

    stats["total"] = sum(stats.values())
    logger.info("Loaded %d tasks total: %s", stats["total"], stats)
    return all_tasks, stats
