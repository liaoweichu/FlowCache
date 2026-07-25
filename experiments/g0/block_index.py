"""
Block identity 哈希、父链连续性校验、invalidation 检查。

实现 IDEA Section 1.2 的 block identity：
    I_b = (m, r, tau, c, a, h_parent, tokenIds, positions)

其中：
- m: model_id
- r: revision
- tau: template_hash（chat template 哈希）
- c: config_hash（模型 config 哈希）
- a: adapter_id（adapter 标识，无 adapter 时为空字符串）
- h_parent: 父 block 的哈希
- tokenIds: 本 block 内的 token id 列表
- positions: token 位置（由 block_idx * block_size 推导）

本模块扩展自 experiments/e1/trace_utils.py 的 compute_block_hash，
在原有 (token_ids, parent_hash, block_idx, block_size) 基础上加入
model_id/revision/template_hash/config_hash/adapter_id 字段，使 block
identity 在跨模型、跨 template、跨 adapter 场景下都能正确区分。
"""

import hashlib
import json
from typing import List, Dict, Tuple, Optional


def compute_block_hash(
    token_ids: List[int],
    parent_hash: str,
    block_idx: int,
    block_size: int = 16,
    model_id: str = "",
    revision: str = "",
    template_hash: str = "",
    config_hash: str = "",
    adapter_id: str = "",
) -> str:
    """
    计算 block identity 哈希 I_b。

    I_b = (m, r, tau, c, a, h_parent, tokenIds, positions)
    - m: model_id
    - r: revision
    - tau: template_hash
    - c: config_hash
    - a: adapter_id
    - h_parent: 父 block 哈希
    - tokenIds: 本 block 内 token id 列表
    - positions: token 位置（由 block_idx * block_size 推导）

    使用 SHA-256 截断为 16 hex 字符。

    Args:
        token_ids: 本 block 内的 token id 列表。
        parent_hash: 父 block 哈希（根 block 传空字符串）。
        block_idx: 本 block 在序列中的索引。
        block_size: 每个 block 的 token 数（默认 16）。
        model_id: 模型标识（如 "Qwen/Qwen2.5-7B-Instruct"）。
        revision: 模型 revision（HuggingFace commit sha 或 last_modified）。
        template_hash: chat template 的哈希。
        config_hash: 模型 config 的哈希。
        adapter_id: adapter 标识（无 adapter 时为空字符串）。

    Returns:
        16 字符 hex 哈希字符串，唯一标识该 block。
    """
    content = json.dumps(
        {
            "m": model_id,
            "r": revision,
            "tau": template_hash,
            "c": config_hash,
            "a": adapter_id,
            "h_parent": parent_hash,
            "token_ids": token_ids,
            "block_idx": block_idx,
            "block_size": block_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return full_hash[:16]


def verify_parent_chain(blocks: List[Dict]) -> Tuple[bool, List[str]]:
    """
    校验父链连续性。

    每个非根 block 的 parent_hash 必须等于前一个 block 的 hash；
    根 block 的 parent_hash 必须为空字符串。

    Args:
        blocks: block dict 列表，每个至少包含 'hash' 和 'parent_hash'，
                按从根到叶的顺序排列。

    Returns:
        (is_valid, errors)：is_valid 表示整条链是否合法，
        errors 为错误信息列表（空列表表示无错误）。
    """
    errors: List[str] = []
    if not blocks:
        return True, []

    for i, block in enumerate(blocks):
        block_hash = block.get("hash", "")
        parent_hash = block.get("parent_hash", "")

        if i == 0:
            if parent_hash != "":
                errors.append(
                    f"Block 0 has non-empty parent_hash: {parent_hash}"
                )
        else:
            prev_hash = blocks[i - 1].get("hash", "")
            if parent_hash != prev_hash:
                errors.append(
                    f"Block {i} parent_hash ({parent_hash}) != "
                    f"block {i - 1} hash ({prev_hash})"
                )

    return len(errors) == 0, errors


def check_invalidation(
    blocks_a: List[Dict],
    blocks_b: List[Dict],
    change_point: int,
) -> Dict:
    """
    检查 change_point 之前的 block 哈希相同、之后的 block 哈希不同。

    用于 template/identifier 变化场景（category ③④）：变化点之前的 prefix
    应当共享同一哈希，变化点及之后的 block 哈希应当不同。

    Args:
        blocks_a: 变化前的 block 列表。
        blocks_b: 变化后的 block 列表（template/identifier 已修改）。
        change_point: 变化发生的 block 索引，此索引之前的 block 应当匹配，
                      此索引及之后应当不同。

    Returns:
        {
            'pre_change_match': bool,   # change_point 之前的 block 是否全部匹配
            'post_change_differ': bool, # change_point 及之后的 block 是否全部不同
            'details': list,            # 每个 block 的对比详情
        }
    """
    details: List[Dict] = []
    pre_match = True
    post_differ = True

    min_len = min(len(blocks_a), len(blocks_b))

    for i in range(min_len):
        hash_a = blocks_a[i].get("hash", "")
        hash_b = blocks_b[i].get("hash", "")

        if i < change_point:
            # 应当匹配
            matches = hash_a == hash_b
            if not matches:
                pre_match = False
            details.append(
                {
                    "block_idx": i,
                    "hash_a": hash_a,
                    "hash_b": hash_b,
                    "expected": "match",
                    "actual": "match" if matches else "differ",
                }
            )
        else:
            # 应当不同
            differs = hash_a != hash_b
            if not differs:
                post_differ = False
            details.append(
                {
                    "block_idx": i,
                    "hash_a": hash_a,
                    "hash_b": hash_b,
                    "expected": "differ",
                    "actual": "differ" if differs else "match",
                }
            )

    return {
        "pre_change_match": pre_match,
        "post_change_differ": post_differ,
        "details": details,
    }


def compute_template_hash(template_str: str) -> str:
    """计算 chat template 字符串的哈希。"""
    return hashlib.sha256(template_str.encode("utf-8")).hexdigest()[:16]


def compute_config_hash(config_dict: Dict) -> str:
    """计算模型 config dict 的哈希。"""
    content = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
