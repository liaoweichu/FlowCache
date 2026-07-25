"""
G0 后端：HuggingFace transformers 后端，带 KV cache 拦截能力。

负责：
- 加载 Qwen2.5-7B-Instruct 模型与 tokenizer
- 提供 forward_with_kv() 执行前向传播并返回 past_key_values
- 提供 slice_kv_into_blocks() 将 past_key_values 按 block_size 切片
- 提供 restore_kv_from_blocks() 将 block 列表重组为 past_key_values
- 提供 get_model_info() 收集模型 revision、tokenizer、config 等元信息

兼容 transformers >= 4.36 的 DynamicCache 与旧版 tuple 两种 past_key_values 格式。
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Backend:
    """HuggingFace transformers backend with KV cache interception."""

    def __init__(self, model_name: str, dtype=torch.bfloat16, device_map="auto"):
        self.model_name = model_name
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载 tokenizer 与模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        # 缓存基本结构信息，避免每次 forward 都去读 config
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads

    def forward_with_kv(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
    ) -> Tuple[torch.Tensor, object]:
        """执行 forward pass，返回 logits 和 past_key_values。"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return outputs.logits, outputs.past_key_values

    @staticmethod
    def _pkv_format(past_key_values) -> str:
        """检测 past_key_values 的格式：'dynamic_new' / 'dynamic_old' / 'tuple'。"""
        if hasattr(past_key_values, "key_cache"):
            return "dynamic_old"
        if hasattr(past_key_values, "layers"):
            return "dynamic_new"
        return "tuple"

    @staticmethod
    def _pkv_layer_kv(past_key_values, layer_idx: int):
        """从任意格式的 past_key_values 中提取第 layer_idx 层的 (key, value) 张量。"""
        fmt = Backend._pkv_format(past_key_values)
        if fmt == "dynamic_old":
            return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
        elif fmt == "dynamic_new":
            layer = past_key_values.layers[layer_idx]
            return layer.keys, layer.values
        else:
            return past_key_values[layer_idx][0], past_key_values[layer_idx][1]

    @staticmethod
    def _pkv_seq_len(past_key_values) -> int:
        """从任意格式的 past_key_values 中获取序列长度。"""
        fmt = Backend._pkv_format(past_key_values)
        if fmt == "dynamic_old":
            return past_key_values.key_cache[0].shape[2]
        elif fmt == "dynamic_new":
            return past_key_values.layers[0].keys.shape[2]
        else:
            return past_key_values[0][0].shape[2]

    def slice_kv_into_blocks(
        self, past_key_values, block_size: int = 16
    ) -> List[Dict]:
        """
        将 past_key_values 按 block_size 切分为 block 列表。

        每个 block 是一个 dict，包含：
        - 'layer_k': 每层 key 张量列表 [batch, heads, block_size, head_dim]
        - 'layer_v': 每层 value 张量列表
        - 'block_idx': 该 block 在序列中的索引
        - 'token_range': (start, end) token 位置区间

        兼容 DynamicCache（transformers >= 4.36, >= 5.x）和 legacy tuple 格式。
        最后一个 block 可能不满 block_size。
        """
        fmt = self._pkv_format(past_key_values)
        seq_len = self._pkv_seq_len(past_key_values)

        blocks: List[Dict] = []
        num_full_blocks = seq_len // block_size

        # 处理完整 block
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            block = {
                "block_idx": block_idx,
                "token_range": (start, end),
                "layer_k": [],
                "layer_v": [],
            }
            for layer_idx in range(self.num_layers):
                k_full, v_full = self._pkv_layer_kv(past_key_values, layer_idx)
                k = k_full[:, :, start:end, :].clone()
                v = v_full[:, :, start:end, :].clone()
                block["layer_k"].append(k)
                block["layer_v"].append(v)
            blocks.append(block)

        # 处理剩余 token（最后一个不满 block_size 的 block）
        remainder = seq_len % block_size
        if remainder > 0:
            start = num_full_blocks * block_size
            end = seq_len
            block = {
                "block_idx": num_full_blocks,
                "token_range": (start, end),
                "layer_k": [],
                "layer_v": [],
            }
            for layer_idx in range(self.num_layers):
                k_full, v_full = self._pkv_layer_kv(past_key_values, layer_idx)
                k = k_full[:, :, start:end, :].clone()
                v = v_full[:, :, start:end, :].clone()
                block["layer_k"].append(k)
                block["layer_v"].append(v)
            blocks.append(block)

        return blocks

    def restore_kv_from_blocks(self, blocks: List[Dict]):
        """
        将 block 列表重组为 past_key_values。

        返回 DynamicCache（兼容新旧两版 transformers）。如果环境不支持
        DynamicCache，则回退为 legacy tuple 格式。
        """
        try:
            from transformers.cache_utils import DynamicCache, DynamicLayer
            has_dynamic = True
        except ImportError:
            has_dynamic = False

        if has_dynamic:
            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                k_chunks = [block["layer_k"][layer_idx] for block in blocks]
                v_chunks = [block["layer_v"][layer_idx] for block in blocks]
                k_full = torch.cat(k_chunks, dim=2)
                v_full = torch.cat(v_chunks, dim=2)
                # 新版 DynamicCache (>=5.x): layers 中是 DynamicLayer 对象
                # 旧版 DynamicCache (>=4.36, <5.x): 直接有 key_cache/value_cache
                if hasattr(cache, "key_cache"):
                    cache.key_cache.append(k_full)
                    cache.value_cache.append(v_full)
                else:
                    layer = DynamicLayer()
                    layer.keys = k_full
                    layer.values = v_full
                    layer.is_initialized = True
                    cache.layers.append(layer)
            return cache
        else:
            # legacy tuple 回退路径
            legacy = []
            for layer_idx in range(self.num_layers):
                k_chunks = [block["layer_k"][layer_idx] for block in blocks]
                v_chunks = [block["layer_v"][layer_idx] for block in blocks]
                k_full = torch.cat(k_chunks, dim=2)
                v_full = torch.cat(v_chunks, dim=2)
                legacy.append((k_full, v_full))
            # transformers 旧版期望 tuple of tuple
            return tuple(tuple(pair) for pair in legacy)

    def get_model_info(self) -> Dict:
        """获取模型 revision、tokenizer info、config，用于 freeze record。"""
        import transformers

        info = {
            "model_name": self.model_name,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": str(self.dtype),
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
        }

        # 从 HuggingFace hub 获取模型 revision
        try:
            from huggingface_hub import model_info

            mi = model_info(self.model_name)
            info["model_sha"] = mi.sha
            info["model_revision"] = mi.last_modified
        except Exception:
            info["model_sha"] = "unknown"
            info["model_revision"] = "unknown"

        # tokenizer 信息
        info["tokenizer_pad_token"] = self.tokenizer.pad_token
        info["tokenizer_eos_token"] = self.tokenizer.eos_token
        info["tokenizer_chat_template"] = (
            "custom" if self.tokenizer.chat_template else "default"
        )

        # CUDA 信息
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_total_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )

        return info

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize 文本，返回 input_ids 张量。"""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def tokenize_chat(self, messages: List[Dict]) -> torch.Tensor:
        """使用模型 chat template 对 chat message 列表进行 tokenize。"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def greedy_decode(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        max_new_tokens: int = 1,
    ) -> List[int]:
        """贪心解码一个或多个 token。"""
        if past_key_values is None:
            logits, past_key_values = self.forward_with_kv(input_ids)
        else:
            logits, past_key_values = self.forward_with_kv(
                input_ids, past_key_values
            )

        tokens: List[int] = []
        for _ in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_token.item())
            logits, past_key_values = self.forward_with_kv(
                next_token.unsqueeze(0), past_key_values
            )
        return tokens

    # =========================================================================
    # 显存监控工具方法
    # =========================================================================

    def get_memory_status(self) -> Dict:
        """
        返回当前 GPU 显存状态（GB）。

        Returns:
            {
                'available': bool,         # CUDA 是否可用
                'allocated_gb': float,     # 当前已分配显存
                'reserved_gb': float,      # 当前 reserved 显存
                'free_gb': float,          # 估算空闲显存（total - allocated）
                'total_gb': float,         # GPU 总显存
                'peak_allocated_gb': float,# 峰值分配显存
            }
            CUDA 不可用时返回 available=False，其余字段为 0.0。
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0,
                "total_gb": 0.0,
                "peak_allocated_gb": 0.0,
            }
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": max(total - allocated, 0.0),
            "total_gb": total,
            "peak_allocated_gb": peak,
        }

    def print_memory_status(self, tag: str = "") -> None:
        """打印当前显存状态，用于调试和实验间监控。"""
        s = self.get_memory_status()
        if not s["available"]:
            print(f"  [mem{(' '+tag) if tag else ''}] CUDA unavailable")
            return
        print(
            f"  [mem{(' '+tag) if tag else ''}] "
            f"allocated={s['allocated_gb']:.2f}GB, "
            f"reserved={s['reserved_gb']:.2f}GB, "
            f"free={s['free_gb']:.2f}GB, "
            f"total={s['total_gb']:.2f}GB, "
            f"peak={s['peak_allocated_gb']:.2f}GB"
        )

    def assert_memory_available(
        self, required_gb: float, tag: str = ""
    ) -> bool:
        """
        检查是否有足够显存可用于下一批操作。

        Args:
            required_gb: 需要的显存下限（GB）。
            tag: 用于日志的标签。

        Returns:
            True 如果空闲显存 >= required_gb；False 否则（同时打印警告）。
        """
        s = self.get_memory_status()
        if not s["available"]:
            return True  # CPU 模式不检查
        if s["free_gb"] < required_gb:
            print(
                f"  [mem WARN{(' '+tag) if tag else ''}] "
                f"free={s['free_gb']:.2f}GB < required={required_gb:.2f}GB, "
                f"triggering empty_cache()"
            )
            torch.cuda.empty_cache()
            s = self.get_memory_status()
            if s["free_gb"] < required_gb:
                print(
                    f"  [mem WARN{(' '+tag) if tag else ''}] "
                    f"after cleanup free={s['free_gb']:.2f}GB still < "
                    f"required={required_gb:.2f}GB"
                )
                return False
        return True

    def safe_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        required_gb: Optional[float] = None,
        tag: str = "",
    ) -> Tuple[Optional[torch.Tensor], Optional[object]]:
        """
        带 OOM 防护的 forward 调用。

        若 CUDA OOM，会尝试 empty_cache 后重试一次；仍失败则返回 (None, None)，
        由调用方决定降级策略（跳过该样本/标记 OOM）。

        Args:
            input_ids: 输入 token 张量。
            past_key_values: 可选的 past KV cache。
            required_gb: 可选，调用前检查的显存下限。
            tag: 日志标签。

        Returns:
            (logits, past_key_values)；OOM 时为 (None, None)。
        """
        if required_gb is not None and not self.assert_memory_available(
            required_gb, tag=tag
        ):
            return None, None
        try:
            return self.forward_with_kv(input_ids, past_key_values)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM{(' '+tag) if tag else ''}] {e}")
            torch.cuda.empty_cache()
            try:
                return self.forward_with_kv(input_ids, past_key_values)
            except torch.cuda.OutOfMemoryError as e2:
                print(f"  [OOM{(' '+tag) if tag else ''}] retry failed: {e2}")
                torch.cuda.empty_cache()
                return None, None
