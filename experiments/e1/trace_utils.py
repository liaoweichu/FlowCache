"""
Utility functions for E1 trajectory processing.
Provides block identity hashing, parent chain computation,
and cross-workflow prefix deduplication.
"""

import hashlib
import json
import os
import glob
from typing import List, Tuple, Dict, Optional


def compute_block_hash(token_ids: List[int],
                       parent_hash: str,
                       block_idx: int,
                       block_size: int = 16) -> str:
    """
    Compute block identity hash I_b = (token_ids, parent_hash).

    Uses SHA-256 truncated to 16 hex chars.

    This implements IDEA Section 1.2's block identity:
        I_b = (m, r, tau, c, a, h_parent, tokenIds, positions)

    For E1, we simplify: all workflows use the same model/config, so
    we only need token_ids + parent_hash for identity. The block_idx
    and block_size are included as additional distinguishing context.

    Args:
        token_ids: List of token IDs in this block.
        parent_hash: Hash string of the parent block (empty string for root).
        block_idx: Index of this block in the sequence.
        block_size: Number of tokens per block (default 16).

    Returns:
        A 16-character hex hash string uniquely identifying this block.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")

    content = json.dumps({
        "token_ids": token_ids,
        "parent_hash": parent_hash,
        "block_idx": block_idx,
        "block_size": block_size
    }, sort_keys=True, separators=(",", ":"))

    full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return full_hash[:16]


def compute_parent_chain(blocks: List[Dict]) -> List[str]:
    """
    Given a list of blocks (each with 'hash' and 'parent_hash'),
    verify and return the chain of hashes from root to leaf.

    Each block's parent_hash must match the hash of the preceding block.
    The first block's parent_hash must be an empty string.

    Args:
        blocks: List of dicts, each with 'hash' and 'parent_hash' keys,
                ordered from root to leaf.

    Returns:
        List of block hashes in order from root to leaf if the chain is
        valid. Returns an empty list if the chain is broken.
    """
    if not blocks:
        return []

    chain = []
    for i, block in enumerate(blocks):
        block_hash = block.get("hash", "")
        parent_hash = block.get("parent_hash", "")

        if i == 0:
            if parent_hash != "":
                return []  # Root block must have empty parent_hash
        else:
            if parent_hash != chain[-1]:
                return []  # Parent hash must match the previous block's hash

        chain.append(block_hash)

    return chain


def deduplicate_blocks(all_blocks: List[Dict]) -> Dict[str, List[str]]:
    """
    Given blocks from all workflows, group by block hash.

    Args:
        all_blocks: List of block dicts, each must contain at least
                    'block_hash' and 'workflow_id'.

    Returns:
        A dict mapping block_hash -> list of workflow_ids that share
        this exact prefix block. Used to compute share_count (how many
        workflows share this exact prefix).
    """
    hash_to_workflows: Dict[str, List[str]] = {}

    for block in all_blocks:
        block_hash = block.get("block_hash", "")
        workflow_id = block.get("workflow_id", "unknown")

        if not block_hash:
            continue

        if block_hash not in hash_to_workflows:
            hash_to_workflows[block_hash] = []

        if workflow_id not in hash_to_workflows[block_hash]:
            hash_to_workflows[block_hash].append(workflow_id)

    return hash_to_workflows


def load_trajectory(trace_path: str) -> Dict:
    """
    Load a saved trajectory JSON file.

    Args:
        trace_path: Path to a trajectory .json file.

    Returns:
        Parsed trajectory dict, or empty dict if the file does not exist
        or is malformed.
    """
    if not os.path.isfile(trace_path):
        return {}

    with open(trace_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}


def load_all_trajectories(trace_dir: str) -> List[Dict]:
    """
    Load all trajectory JSON files from a directory.

    Args:
        trace_dir: Directory containing trajectory .json files.

    Returns:
        List of parsed trajectory dicts. Skips files that fail to parse.
    """
    trajectories = []
    pattern = os.path.join(trace_dir, "*.json")

    for filepath in sorted(glob.glob(pattern)):
        traj = load_trajectory(filepath)
        if traj:
            trajectories.append(traj)

    return trajectories
