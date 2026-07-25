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

        兼容 DynamicCache（transformers >= 4.36）和 legacy tuple 两种格式。
        最后一个 block 可能不满 block_size。
        """
        # 判断是否为 DynamicCache 格式
        is_dynamic = hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        )

        # 从第一层获取 seq_len
        if is_dynamic:
            seq_len = past_key_values.key_cache[0].shape[2]
        else:
            # legacy 格式：tuple of tuples，每层 (key, value)
            seq_len = past_key_values[0][0].shape[2]

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
                if is_dynamic:
                    k = past_key_values.key_cache[layer_idx][:, :, start:end, :]
                    v = past_key_values.value_cache[layer_idx][:, :, start:end, :]
                else:
                    k = past_key_values[layer_idx][0][:, :, start:end, :]
                    v = past_key_values[layer_idx][1][:, :, start:end, :]
                # clone 避免 view 被后续 forward 覆盖
                block["layer_k"].append(k.clone())
                block["layer_v"].append(v.clone())
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
                if is_dynamic:
                    k = past_key_values.key_cache[layer_idx][:, :, start:end, :]
                    v = past_key_values.value_cache[layer_idx][:, :, start:end, :]
                else:
                    k = past_key_values[layer_idx][0][:, :, start:end, :]
                    v = past_key_values[layer_idx][1][:, :, start:end, :]
                block["layer_k"].append(k.clone())
                block["layer_v"].append(v.clone())
            blocks.append(block)

        return blocks

    def restore_kv_from_blocks(self, blocks: List[Dict]):
        """
        将 block 列表重组为 past_key_values。

        返回 DynamicCache（transformers >= 4.36）。如果环境不支持
        DynamicCache，则回退为 legacy tuple 格式。
        """
        try:
            from transformers.cache_utils import DynamicCache
            has_dynamic = True
        except ImportError:
            has_dynamic = False

        if has_dynamic:
            cache = DynamicCache()
            # 沿序列维度拼接每个 block
            for layer_idx in range(self.num_layers):
                k_chunks = [block["layer_k"][layer_idx] for block in blocks]
                v_chunks = [block["layer_v"][layer_idx] for block in blocks]
                k_full = torch.cat(k_chunks, dim=2)
                v_full = torch.cat(v_chunks, dim=2)
                # 直接写入 cache 的内部列表
                cache.key_cache.append(k_full)
                cache.value_cache.append(v_full)
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
