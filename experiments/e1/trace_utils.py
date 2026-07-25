"""
Utility functions for E1 trajectory processing.
Provides block identity hashing, parent chain computation,
and cross-workflow prefix deduplication.

Block identity (compute_block_hash, verify_parent_chain,
compute_template_hash, compute_config_hash) is unified to the G0
8-tuple version (m, r, tau, c, a, h_parent, tokenIds, positions)
defined in experiments/g0/block_index.py. The previous 4-tuple
simplified version is deprecated in favor of cross-experiment
consistency.
"""

import json
import os
import glob
import sys
from pathlib import Path
from typing import List, Dict

# Re-export G0 8-tuple block identity implementation.
# experiments/g0/block_index.py must be importable as `block_index`.
_G0_DIR = Path(__file__).resolve().parent.parent / "g0"
if str(_G0_DIR) not in sys.path:
    sys.path.insert(0, str(_G0_DIR))

from block_index import (  # noqa: F401  (re-exported)
    compute_block_hash,
    verify_parent_chain,
    compute_template_hash,
    compute_config_hash,
)


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
