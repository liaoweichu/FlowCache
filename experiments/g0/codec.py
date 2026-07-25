"""
Q8/Q4 KV cache 量化 codec (Task 6).

- Q8: per-tensor int8 量化（scale + int8 tensor）
- Q4: per-tensor int4 量化（scale + int4 tensor，存储为 int8）
- Lineage 隔离：量化 block 获得与 canonical BF16 不同的 lineage 标签

量化后的 block 在 identity 哈希层面与 canonical BF16 block 不同，
从而实现 approximate lineage 隔离（量化祖先的 child 不与 canonical lineage 别名）。
"""

import time
from typing import Dict

import torch


def encode_q8(tensor: torch.Tensor) -> Dict:
    """
    Per-tensor int8 量化。

    将 BF16/FP16 张量按绝对值最大值归一化到 [-127, 127] 范围，
    四舍五入后转为 int8（内部先转为 float32 以兼容所有 dtype）。

    Args:
        tensor: 待量化的 BF16/FP16 张量。

    Returns:
        {'data': int8_tensor, 'scale': float, 'lineage': 'approximate_q8'}
    """
    tensor_f32 = tensor.float()
    scale = tensor_f32.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    int8_data = torch.round(tensor_f32 / scale).clamp(-128, 127).to(torch.int8)
    return {"data": int8_data, "scale": scale, "lineage": "approximate_q8"}


def decode_q8(encoded: Dict, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Q8 反量化回 BF16。

    Args:
        encoded: encode_q8 返回的字典。
        dtype: 反量化后的目标 dtype（默认 bfloat16）。

    Returns:
        反量化后的 BF16 张量。
    """
    return encoded["data"].to(dtype) * encoded["scale"]


def encode_q4(tensor: torch.Tensor) -> Dict:
    """
    Per-tensor int4 量化。

    将 BF16/FP16 张量按绝对值最大值归一化到 [-7, 7] 范围，
    四舍五入后转为 int4（以 int8 张量存储，内部先转为 float32 以兼容所有 dtype）。

    Args:
        tensor: 待量化的 BF16/FP16 张量。

    Returns:
        {'data': int8_tensor (存储 int4 值), 'scale': float, 'lineage': 'approximate_q4'}
    """
    tensor_f32 = tensor.float()
    scale = tensor_f32.abs().max().item() / 7.0
    if scale == 0:
        scale = 1.0
    int4_data = torch.round(tensor_f32 / scale).clamp(-8, 7).to(torch.int8)
    return {"data": int4_data, "scale": scale, "lineage": "approximate_q4"}


def decode_q4(encoded: Dict, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Q4 反量化回 BF16。

    Args:
        encoded: encode_q4 返回的字典。
        dtype: 反量化后的目标 dtype（默认 bfloat16）。

    Returns:
        反量化后的 BF16 张量。
    """
    return encoded["data"].to(dtype) * encoded["scale"]


def check_lineage_isolation(bf16_hash: str, q8_hash: str, q4_hash: str) -> bool:
    """
    验证量化 block 的 lineage 与 canonical BF16 不同。

    量化后的 block（Q8/Q4）应具有与 canonical BF16 不同的 identity 哈希，
    且 Q8 与 Q4 之间也应不同。这确保 approximate lineage 不会与 canonical
    lineage 别名。

    Args:
        bf16_hash: canonical BF16 block 的哈希。
        q8_hash: Q8 量化 block 的哈希。
        q4_hash: Q4 量化 block 的哈希。

    Returns:
        True 如果三者两两不同（lineage 隔离成功）。
    """
    return bf16_hash != q8_hash and bf16_hash != q4_hash and q8_hash != q4_hash


def measure_encode_time(tensor: torch.Tensor, precision: str, repeats: int = 10) -> float:
    """
    测量编码延迟（多次取平均，单位 ms）。

    Args:
        tensor: 待量化的张量。
        precision: "q8" 或 "q4"。
        repeats: 重复次数。

    Returns:
        平均编码延迟（ms）。
    """
    encode_fn = encode_q8 if precision == "q8" else encode_q4
    # 预热
    encode_fn(tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()
    for _ in range(repeats):
        encode_fn(tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.time() - t0) / repeats * 1000  # ms
    return elapsed


def measure_decode_time(encoded: Dict, repeats: int = 10) -> float:
    """
    测量解码延迟（多次取平均，单位 ms）。

    Args:
        encoded: encode_q8 或 encode_q4 返回的字典。
        repeats: 重复次数。

    Returns:
        平均解码延迟（ms）。
    """
    # 预热
    if encoded["lineage"] == "approximate_q8":
        decode_q8(encoded)
    else:
        decode_q4(encoded)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    decode_fn = decode_q8 if encoded["lineage"] == "approximate_q8" else decode_q4
    t0 = time.time()
    for _ in range(repeats):
        decode_fn(encoded)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.time() - t0) / repeats * 1000  # ms
    return elapsed
