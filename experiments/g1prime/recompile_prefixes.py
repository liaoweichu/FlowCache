"""
G1' Physical Prefix Recompiler
==============================
Recompiles physical KV-prefix blocks for τ-bench trajectories.

Existing E1 traces (``experiments/e1/traces/bf16/tau_bench/*.json``) store
``block_hash`` based on the *per-message raw token_ids* — i.e. the tokens of
each individual message, NOT the chat-template-wrapped complete prompt.
FlowCache's real object of study is physical KV-prefix reuse of the
*complete* prompt, so the block identities must be recompiled.

For every assistant step in every trajectory this script:
  1. Collects the full message history (system+user+assistant+tool) that
     preceded the assistant step, in ``step_id`` order.
  2. Renders the complete prompt with
     ``tokenizer.apply_chat_template(messages, add_generation_prompt=True)``
     — the trailing ``<|im_start|>assistant\\n`` header is part of the
     prefill prefix the model actually received.
  3. Tokenizes the prompt with ``tokenizer(prompt, add_special_tokens=False)``
     to obtain the true prefill token_ids (no double special tokens).
  4. Slices the token stream into ``block_size``-token blocks *continuously*
     across message boundaries (no per-message fragmentation); the final
     block may be partial.
  5. Computes ``block_hash`` via the G0 8-tuple block identity
     ``(m, r, tau, c, a, h_parent, tokenIds, positions)`` implemented in
     ``experiments/g0/block_index.compute_block_hash`` (SHA-256 truncated to
     16 hex characters). ``parent_hash`` chains every block to its
     predecessor (root block has ``parent_hash == ""``).

Block-identity fields (``model_id``, ``revision``, ``template_hash``,
``config_hash``, ``adapter_id``) are taken from each trace's G0-frozen
``meta`` — NOT from the tokenizer load path — so the recompiled hashes stay
consistent with the G0 spec and with the existing E1 traces regardless of
which local/remote path the tokenizer was loaded from.

Output: ``experiments/g1prime/physical_traces/request_prefixes.jsonl``
(one JSON object per assistant step / request event, one per line).

Read-only w.r.t. ``experiments/e1/``.

Usage:
    python experiments/g1prime/recompile_prefixes.py --max-episodes 3
    python experiments/g1prime/recompile_prefixes.py --model-id Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments" / "e1" / "config.yaml"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "experiments" / "e1" / "traces" / "bf16" / "tau_bench"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "physical_traces" / "request_prefixes.jsonl"

# G0 block_index import (mirror experiments/e1/trace_utils.py).
_G0_DIR = PROJECT_ROOT / "experiments" / "g0"
if str(_G0_DIR) not in sys.path:
    sys.path.insert(0, str(_G0_DIR))

try:
    from block_index import compute_block_hash  # noqa: E402
except ImportError as exc:  # pragma: no cover - import-time guard
    sys.stderr.write(
        f"ERROR: cannot import compute_block_hash from experiments/g0/block_index.py "
        f"({exc}). Ensure the G0 module is present.\n"
    )
    raise


# ---------------------------------------------------------------------------
# Config / tokenizer loading
# ---------------------------------------------------------------------------
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load config.yaml. Prefers PyYAML; returns {} on failure."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        sys.stderr.write(f"WARNING: failed to load config {path}: {exc}\n")
        return {}


def resolve_model_id_for_loading(config: Dict[str, Any], override: Optional[str]) -> str:
    """Resolve the tokenizer load path: CLI override > config model.name."""
    if override:
        return override
    model_cfg = config.get("model", {}) if config else {}
    name = model_cfg.get("name")
    if name:
        return name
    return "Qwen/Qwen2.5-7B-Instruct"


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    """
    Load the G0-frozen Qwen tokenizer via ``transformers.AutoTokenizer``.

    Aborts with a clear message if ``transformers`` is missing or the
    tokenizer cannot be loaded (e.g. Windows machine without the model and
    no network access to HuggingFace Hub).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.stderr.write(
            "ERROR: the `transformers` package is not installed.\n"
            "       Install it with:  pip install transformers tokenizers\n"
            "       (tokenizer-only use does not require torch for Qwen2.5).\n"
        )
        sys.exit(2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        sys.stderr.write(
            f"ERROR: failed to load tokenizer from '{model_id}'.\n"
            f"       Underlying error: {exc}\n"
            f"       On Windows without a local Qwen model, pass a HuggingFace "
            f"id instead, e.g.:\n"
            f"         python experiments/g1prime/recompile_prefixes.py "
            f"--model-id Qwen/Qwen2.5-7B-Instruct\n"
        )
        sys.exit(2)

    if not getattr(tokenizer, "chat_template", None):
        sys.stderr.write(
            f"ERROR: tokenizer loaded from '{model_id}' has no chat_template. "
            f"apply_chat_template cannot be used. Aborting.\n"
        )
        sys.exit(2)

    return tokenizer


# ---------------------------------------------------------------------------
# Trace discovery / loading
# ---------------------------------------------------------------------------
def iter_trace_files(trace_dir: Path) -> List[Path]:
    """Sorted list of trajectory JSON files (skip checkpoints / reports)."""
    if not trace_dir.exists():
        return []
    files = []
    for p in trace_dir.glob("*.json"):
        if p.name.startswith("_"):
            continue
        files.append(p)
    files.sort()
    return files


def load_trace(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt reconstruction & tokenization
# ---------------------------------------------------------------------------
def rebuild_message_history(steps: List[Dict[str, Any]], assistant_idx: int) -> List[Dict[str, str]]:
    """
    Return the message history (list of {role, content}) preceding the
    assistant step at ``assistant_idx``, in step_id order.
    """
    history: List[Dict[str, str]] = []
    for st in steps[:assistant_idx]:
        role = st.get("role")
        content = st.get("content", "")
        if role is None:
            continue
        history.append({"role": role, "content": content})
    return history


def tokenize_full_prompt(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    """
    Render the full prompt via ``apply_chat_template`` (with generation
    prompt) and tokenize it. Returns the token-id list of the true prefill
    prefix (history + trailing ``<|im_start|>assistant\\n`` header).

    ``add_special_tokens=False`` avoids re-adding special tokens already
    emitted by the chat template.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        prompt_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    token_ids = enc["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(t) for t in token_ids]


# ---------------------------------------------------------------------------
# Blockification
# ---------------------------------------------------------------------------
def blockify(
    token_ids: List[int],
    block_size: int,
    g0_identity: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Slice ``token_ids`` into continuous ``block_size``-token blocks.

    Cross-message fragmentation does not occur because the whole prompt is
    tokenized as one stream. The final block may be partial.

    Each block record:
        {block_idx, token_range_start, token_range_end, block_hash, parent_hash}

    ``block_hash`` uses the G0 8-tuple identity via ``compute_block_hash``.
    ``parent_hash`` chains to the previous block (root == "").
    """
    blocks: List[Dict[str, Any]] = []
    parent_hash = ""
    n = len(token_ids)
    if n == 0:
        return blocks

    block_idx = 0
    start = 0
    while start < n:
        end = min(start + block_size, n)
        chunk = token_ids[start:end]
        bh = compute_block_hash(
            token_ids=chunk,
            parent_hash=parent_hash,
            block_idx=block_idx,
            block_size=block_size,
            model_id=g0_identity["model_id"],
            revision=g0_identity["revision"],
            template_hash=g0_identity["template_hash"],
            config_hash=g0_identity["config_hash"],
            adapter_id=g0_identity["adapter_id"],
        )
        blocks.append(
            {
                "block_idx": block_idx,
                "token_range_start": start,
                "token_range_end": end,
                "block_hash": bh,
                "parent_hash": parent_hash,
            }
        )
        parent_hash = bh
        block_idx += 1
        start = end

    return blocks


def build_g0_identity(meta: Dict[str, Any]) -> Dict[str, str]:
    """Extract the G0-frozen identity fields used in block hashing."""
    return {
        "model_id": str(meta.get("model_id", "")),
        "revision": str(meta.get("revision", "")),
        "template_hash": str(meta.get("template_hash", "")),
        "config_hash": str(meta.get("config_hash", "")),
        "adapter_id": str(meta.get("adapter_id", "")),
    }


# ---------------------------------------------------------------------------
# Per-trace processing
# ---------------------------------------------------------------------------
def process_trace(
    trace: Dict[str, Any],
    tokenizer,
) -> List[Dict[str, Any]]:
    """
    Produce one request event per assistant step in the trace.
    Returns the list of events (caller writes them out).
    """
    meta = trace.get("meta", {})
    steps = trace.get("steps", [])
    block_size = int(meta.get("block_size", 16))
    g0_identity = build_g0_identity(meta)

    workflow_id = meta.get("workflow_id", "")
    task_id = meta.get("task_id", "")
    seed = meta.get("seed")
    domain = meta.get("domain", "")

    events: List[Dict[str, Any]] = []
    for idx, st in enumerate(steps):
        if st.get("role") != "assistant":
            continue

        messages = rebuild_message_history(steps, idx)
        if not messages:
            # No history (should not happen for τ-bench); skip defensively.
            continue

        token_ids = tokenize_full_prompt(tokenizer, messages)
        blocks = blockify(token_ids, block_size, g0_identity)

        event = {
            "request_id": f"{workflow_id}__s{st.get('step_id', idx)}",
            "workflow_id": workflow_id,
            "task_id": task_id,
            "seed": seed,
            "domain": domain,
            "step_id": st.get("step_id", idx),
            "arrival_time_ms": st.get("arrival_time_ms", 0.0),
            "prefill_ms": st.get("prefill_ms", 0.0),
            "num_prefix_tokens": len(token_ids),
            "num_blocks": len(blocks),
            # G0-frozen identity (self-describing; load-bearing for block_hash).
            "model_id": g0_identity["model_id"],
            "revision": g0_identity["revision"],
            "template_hash": g0_identity["template_hash"],
            "config_hash": g0_identity["config_hash"],
            "adapter_id": g0_identity["adapter_id"],
            "block_size": block_size,
            "blocks": blocks,
        }
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G1' physical prefix recompiler for τ-bench trajectories.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to E1 config.yaml (default: {DEFAULT_CONFIG_PATH}).",
    )
    p.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIR,
        help=f"Directory of trajectory JSON files (default: {DEFAULT_TRACE_DIR}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Tokenizer load path / HF id. Overrides config model.name. "
            "Use this on machines where the config's local model path does "
            "not exist (e.g. Qwen/Qwen2.5-7B-Instruct)."
        ),
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only the first N trajectories (0 or unset = all).",
    )
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading the tokenizer.",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()

    # --- Load config & tokenizer ---
    config = load_yaml_config(args.config)
    trust_remote_code = bool(
        config.get("model", {}).get("trust_remote_code", True)
    ) and not args.no_trust_remote_code
    load_model_id = resolve_model_id_for_loading(config, args.model_id)

    print("=" * 78)
    print("G1' Physical Prefix Recompiler")
    print("=" * 78)
    print(f"Config       : {args.config}")
    print(f"Trace dir    : {args.trace_dir}")
    print(f"Output       : {args.output}")
    print(f"Tokenizer    : {load_model_id}  (trust_remote_code={trust_remote_code})")
    print()

    trace_files = iter_trace_files(args.trace_dir)
    print(f"[1] Found {len(trace_files)} trajectory files.")
    if not trace_files:
        sys.stderr.write("ERROR: no trajectory files found. Aborting.\n")
        return 1

    if args.max_episodes is not None and args.max_episodes > 0:
        trace_files = trace_files[: args.max_episodes]
        print(f"    --max-episodes {args.max_episodes}: processing "
              f"{len(trace_files)} traces.")

    print()
    print("[2] Loading tokenizer ...")
    t0 = time.time()
    tokenizer = load_tokenizer(load_model_id, trust_remote_code=trust_remote_code)
    print(f"    Loaded in {time.time() - t0:.1f}s. "
          f"vocab_size={getattr(tokenizer, 'vocab_size', '?')}, "
          f"chat_template present={bool(getattr(tokenizer, 'chat_template', None))}")

    # --- Output dir ---
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Process traces ---
    print()
    print("[3] Recompiling physical prefixes ...")
    t_start = time.time()
    total_traces = 0
    total_requests = 0
    total_blocks = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, tf in enumerate(trace_files):
            try:
                trace = load_trace(tf)
            except Exception as exc:
                sys.stderr.write(f"    WARNING: skip {tf.name} ({exc})\n")
                continue

            events = process_trace(trace, tokenizer)
            for ev in events:
                out_f.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")))
                out_f.write("\n")

            total_traces += 1
            total_requests += len(events)
            total_blocks += sum(ev["num_blocks"] for ev in events)

            if (i + 1) % 100 == 0 or (i + 1) == len(trace_files):
                elapsed = time.time() - t_start
                print(
                    f"    [{i + 1}/{len(trace_files)}] "
                    f"traces={total_traces} requests={total_requests} "
                    f"blocks={total_blocks} elapsed={elapsed:.1f}s"
                )

    # --- Summary ---
    elapsed = time.time() - t_start
    print()
    print("[4] Done.")
    print(f"    traces processed   : {total_traces}")
    print(f"    request events     : {total_requests}")
    print(f"    total blocks       : {total_blocks}")
    print(f"    output file        : {args.output}")
    print(f"    elapsed            : {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
