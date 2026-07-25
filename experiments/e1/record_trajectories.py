"""
E1 Trajectory Recording Script
================================
Records BF16 trajectories for tau-bench workflows using Qwen3-8B-Instruct.
For each workflow, records:
- Token IDs per step (system prompt, user message, assistant response, tool call, tool result)
- Tool call parameters and results
- Block assignments (token range -> block hash mapping)
- Prefill/decode timing

Output: One JSON file per workflow in experiments/e1/traces/bf16/

Design notes:
- The simplified workflow loop uses direct model calls with message-based
  conversation management. Real tau-bench tool execution is pending W3 integration.
- Block identity uses SHA-256 (16 hex chars) over (token_ids, parent_hash, block_idx).
- Timing uses time.perf_counter() for wall-clock measurement on the critical path.
"""

import json
import logging
import os
import re
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch

from trace_utils import compute_block_hash

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16          # tokens per block
MAX_NEW_TOKENS = 512     # max generated tokens per assistant turn
MAX_WORKFLOW_TURNS = 30  # safety limit on conversation turns

# Regex for tau-bench style tool call parsing (<function_call>...</function_call>)
_TOOL_CALL_RE = re.compile(
    r"<function_call>\s*(.*?)\s*</function_call>", re.DOTALL
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "record_trajectories.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic / fallback workflow definitions
# Used when g2-pilot-subset.json is not available.
# Each entry defines a minimal tau-bench-style scenario with system policy,
# tools schema, and initial user instruction.
# ---------------------------------------------------------------------------

_SYNTHETIC_WORKFLOWS = [
    {
        "task_id": "retail-1",
        "domain": "retail",
        "system_policy": (
            "You are a helpful customer service assistant for an online retail store. "
            "You can help customers with order tracking, returns, product inquiries, "
            "and refunds. Always be polite and professional.\n\n"
            "Available tools:\n"
            "- lookup_order(order_id: str): Look up order details by ID.\n"
            "- check_return_eligibility(order_id: str, item_id: str): Check if an item can be returned.\n"
            "- initiate_refund(order_id: str, item_id: str, reason: str): Start a refund for a returned item.\n"
            "- search_products(query: str): Search the product catalog.\n\n"
            "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
            "with a JSON object inside."
        ),
        "user_instruction": (
            "I ordered a pair of running shoes last week (order #ORD-12345) and they arrived "
            "damaged. The left shoe has a tear in the sole. Can you help me return them and get a refund?"
        ),
    },
    {
        "task_id": "retail-2",
        "domain": "retail",
        "system_policy": (
            "You are a helpful customer service assistant for an online retail store. "
            "You can help customers with order tracking, returns, product inquiries, "
            "and refunds. Always be polite and professional.\n\n"
            "Available tools:\n"
            "- lookup_order(order_id: str): Look up order details by ID.\n"
            "- check_return_eligibility(order_id: str, item_id: str): Check if an item can be returned.\n"
            "- initiate_refund(order_id: str, item_id: str, reason: str): Start a refund for a returned item.\n"
            "- search_products(query: str): Search the product catalog.\n\n"
            "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
            "with a JSON object inside."
        ),
        "user_instruction": (
            "Hi, I want to buy a birthday gift for my daughter. She's 8 years old "
            "and loves arts and crafts. Can you recommend something suitable from your store?"
        ),
    },
    {
        "task_id": "airline-1",
        "domain": "airline",
        "system_policy": (
            "You are a helpful customer service assistant for an airline. "
            "You can assist with flight booking, cancellations, seat selection, "
            "baggage inquiries, and flight status checks. Always be polite and professional.\n\n"
            "Available tools:\n"
            "- search_flights(origin: str, destination: str, date: str): Search available flights.\n"
            "- book_flight(flight_id: str, passenger_name: str, seat_preference: str): Book a flight.\n"
            "- check_flight_status(flight_id: str): Get current flight status.\n"
            "- cancel_booking(booking_ref: str): Cancel an existing booking.\n\n"
            "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
            "with a JSON object inside."
        ),
        "user_instruction": (
            "I need to fly from New York to San Francisco next Monday. "
            "I prefer a morning flight if possible. Can you help me find one?"
        ),
    },
    {
        "task_id": "airline-2",
        "domain": "airline",
        "system_policy": (
            "You are a helpful customer service assistant for an airline. "
            "You can assist with flight booking, cancellations, seat selection, "
            "baggage inquiries, and flight status checks. Always be polite and professional.\n\n"
            "Available tools:\n"
            "- search_flights(origin: str, destination: str, date: str): Search available flights.\n"
            "- book_flight(flight_id: str, passenger_name: str, seat_preference: str): Book a flight.\n"
            "- check_flight_status(flight_id: str): Get current flight status.\n"
            "- cancel_booking(booking_ref: str): Cancel an existing booking.\n\n"
            "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
            "with a JSON object inside."
        ),
        "user_instruction": (
            "My flight yesterday (booking ref ABC-XYZ-789) was cancelled due to weather. "
            "I need to rebook for tomorrow and I'd like a window seat. Can you help?"
        ),
    },
]

# ---------------------------------------------------------------------------
# Helper: tool call parsing
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> Optional[Dict]:
    """
    Extract the first <function_call>...</function_call> block from model output
    and parse it as JSON.

    Returns a dict with {'name': ..., 'arguments': {...}} or None if no tool call found.
    """
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
        # Handle both {name, arguments} and flat argument dicts
        if "name" in parsed and "arguments" in parsed:
            return {"name": str(parsed["name"]), "arguments": parsed["arguments"]}
        else:
            return {"name": "unknown_tool", "arguments": parsed}
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call JSON: %s", raw[:120])
        return None


def _simulate_tool_result(tool_call: Dict) -> str:
    """
    Produce a plausible simulated tool result for a given tool call.
    This is a placeholder for the real tau-bench backend integration (W3).

    TODO(W3): Replace with actual tau-bench backend simulator calls.
    """
    name = tool_call.get("name", "unknown_tool")
    args = tool_call.get("arguments", {})

    if name == "lookup_order":
        order_id = args.get("order_id", "unknown")
        return json.dumps({
            "order_id": order_id,
            "status": "delivered",
            "items": [{"item_id": "ITEM-001", "name": "Running Shoes", "price": 89.99}],
            "delivery_date": "2026-07-20",
        })

    elif name == "check_return_eligibility":
        order_id = args.get("order_id", "unknown")
        item_id = args.get("item_id", "unknown")
        return json.dumps({
            "order_id": order_id,
            "item_id": item_id,
            "eligible": True,
            "return_window_days": 30,
            "reason_required": True,
        })

    elif name == "initiate_refund":
        order_id = args.get("order_id", "unknown")
        return json.dumps({
            "order_id": order_id,
            "refund_id": f"REF-{abs(hash(order_id)) % 100000:05d}",
            "amount": 89.99,
            "status": "processing",
            "eta_days": 5,
        })

    elif name == "search_products":
        query = args.get("query", "")
        return json.dumps({
            "query": query,
            "results": [
                {"id": "P-101", "name": "Deluxe Art Set", "price": 34.99},
                {"id": "P-102", "name": "Craft Beads Kit", "price": 19.99},
            ],
        })

    elif name == "search_flights":
        return json.dumps({
            "flights": [
                {"flight_id": "FL-1001", "departure": "08:30", "arrival": "14:00", "seats": 45},
                {"flight_id": "FL-1002", "departure": "12:00", "arrival": "17:30", "seats": 120},
            ],
        })

    elif name == "book_flight":
        return json.dumps({
            "booking_ref": "BOOK-A1B2C3",
            "flight_id": args.get("flight_id", "unknown"),
            "passenger": args.get("passenger_name", "unknown"),
            "status": "confirmed",
        })

    elif name == "check_flight_status":
        return json.dumps({
            "flight_id": args.get("flight_id", "unknown"),
            "status": "on_time",
            "departure_gate": "B12",
        })

    elif name == "cancel_booking":
        return json.dumps({
            "booking_ref": args.get("booking_ref", "unknown"),
            "status": "cancelled",
            "refund_amount": 350.00,
        })

    else:
        return json.dumps({"result": "ok", "tool": name})


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """
    Records full BF16 trajectories for tau-bench workflows using Qwen2.5-7B-Instruct.

    For each workflow we:
      1. Build a conversation from (system, user, assistant, tool) messages.
      2. For each new message, tokenize with block tracking.
      3. On assistant turns, call model.generate to get the next response.
      4. Parse tool calls; simulate tool results.
      5. Record timings, token IDs, and block assignments per step.
      6. Save the complete trajectory as JSON.
    """

    def __init__(self, config_path: str = "experiments/e1/config.yaml"):
        self._config_path = config_path
        self._config = self._load_config(config_path)
        self._block_size = int(self._config.get("cache", {}).get("block_size", BLOCK_SIZE))

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self._device)

        self._model, self._tokenizer = self._init_model()

        # Global block index: block_hash -> {token_start, token_end, parent_hash, workflow_id, ...}
        self._global_block_index: Dict[str, Dict] = {}

        self._output_dir = Path(
            self._config.get("output", {}).get("trace_dir", "traces/bf16")
        )
        if not self._output_dir.is_absolute():
            self._output_dir = Path(os.path.dirname(__file__)) / self._output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML config. Resolves relative to this script if needed."""
        path = Path(config_path)
        if not path.is_absolute():
            # Try relative to the project root (two levels up from this file)
            script_dir = Path(__file__).resolve().parent
            candidates = [
                path.resolve(),
                (script_dir / path).resolve(),
                (script_dir.parent.parent / path).resolve(),
            ]
            for p in candidates:
                if p.exists():
                    path = p
                    break

        if not path.exists():
            logger.warning("Config file %s not found; using defaults.", config_path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
        return cfg

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_model(self):
        """Load Qwen2.5-7B-Instruct in BF16 with device_map='auto'."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self._config.get("model", {})
        model_name = model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct")
        dtype_str = model_cfg.get("dtype", "bfloat16")
        trust_remote = model_cfg.get("trust_remote_code", True)
        device_map = model_cfg.get("device_map", "auto")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote
        )
        # Qwen3 uses <|im_start|> / <|im_end|> chat template; ensure pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, dtype_str, device_map)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote,
        )
        model.eval()
        logger.info("Model loaded. Parameters: %s",
                     sum(p.numel() for p in model.parameters()))

        return model, tokenizer

    # ------------------------------------------------------------------
    # Workflow loading
    # ------------------------------------------------------------------

    def load_workflow_subset(self, subset_path: str = "") -> List[Dict]:
        """
        Load tau-bench workflow subset.

        Priority:
          1. Explicit subset_path (if provided and exists).
          2. Config's workload.subset_file (e.g., ../g2-pilot-subset.json).
          3. Synthetic fallback workflows (built-in).

        Returns a list of workflow dicts with keys:
          task_id, domain, system_policy, user_instruction
        """
        # --- Try explicit path ---
        if subset_path and os.path.isfile(subset_path):
            return self._load_subset_json(subset_path)

        # --- Try config path ---
        cfg_subset = self._config.get("workload", {}).get("subset_file", "")
        if cfg_subset:
            subset_full = Path(cfg_subset)
            if not subset_full.is_absolute():
                subset_full = Path(__file__).resolve().parent.parent / cfg_subset
            if subset_full.exists():
                return self._load_subset_json(str(subset_full))

        # --- Try tau-bench data directory ---
        tau_bench_dirs = [
            Path("tau-bench/data"),
            Path("sierra-research/tau-bench/data"),
            Path.home() / "tau-bench" / "data",
        ]
        for tb_dir in tau_bench_dirs:
            if tb_dir.exists():
                logger.info("tau-bench data found at %s", tb_dir)
                return self._load_tau_bench_workflows(str(tb_dir))

        # --- Fallback: synthetic workflows ---
        logger.warning(
            "No workflow subset file or tau-bench data found. "
            "Using built-in synthetic workflows (%d total).",
            len(_SYNTHETIC_WORKFLOWS),
        )
        max_wf = self._config.get("workload", {}).get("max_workflows", len(_SYNTHETIC_WORKFLOWS))
        return _SYNTHETIC_WORKFLOWS[:max_wf]

    def _load_subset_json(self, path: str) -> List[Dict]:
        """Load workflows from a g2-pilot-subset.json or similar file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both list and {workflows: [...]} formats
        if isinstance(data, list):
            workflows = data
        elif isinstance(data, dict):
            workflows = data.get("workflows", data.get("tasks", []))
        else:
            workflows = []
        logger.info("Loaded %d workflows from %s", len(workflows), path)
        return workflows

    def _load_tau_bench_workflows(self, tb_data_dir: str) -> List[Dict]:
        """
        Load workflows directly from tau-bench data directory.
        tau-bench organizes tasks by domain: retail/tasks/, airline/tasks/, etc.
        Each task JSON file has: id, description, user_scenario, initial_state, ...
        We adapt these into our workflow dict format.
        """
        workflows = []
        domains = ["retail", "airline"]
        for domain in domains:
            domain_tasks_dir = Path(tb_data_dir) / domain / "tasks"
            if not domain_tasks_dir.is_dir():
                continue
            for task_file in sorted(domain_tasks_dir.glob("*.json")):
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        task = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue
                workflow = self._tau_bench_task_to_workflow(task, domain)
                if workflow:
                    workflows.append(workflow)

        max_wf = self._config.get("workload", {}).get("max_workflows", 80)
        workflows = workflows[:max_wf]
        logger.info("Loaded %d workflows from tau-bench data at %s", len(workflows), tb_data_dir)
        return workflows

    def _tau_bench_task_to_workflow(self, task: Dict, domain: str) -> Optional[Dict]:
        """Convert a raw tau-bench task dict into our standard workflow dict."""
        task_id = task.get("id", task.get("task_id", ""))
        if not task_id:
            return None

        user_instruction = task.get("user_scenario", task.get("description", ""))
        # tau-bench domain policies are typically loaded from domain config,
        # not per-task. We'll construct a minimal policy here.
        system_policy = self._get_domain_policy(domain)

        return {
            "task_id": task_id,
            "domain": domain,
            "system_policy": system_policy,
            "user_instruction": user_instruction,
            "raw_task": task,  # keep original for later use
        }

    @staticmethod
    def _get_domain_policy(domain: str) -> str:
        """Return a minimal domain policy prompt for tau-bench domains."""
        policies = {
            "retail": (
                "You are a helpful customer service assistant for an online retail store. "
                "You can help customers with order tracking, returns, product inquiries, "
                "and refunds. Always be polite and professional.\n\n"
                "Available tools:\n"
                "- lookup_order(order_id: str): Look up order details by ID.\n"
                "- check_return_eligibility(order_id: str, item_id: str): Check if an item can be returned.\n"
                "- initiate_refund(order_id: str, item_id: str, reason: str): Start a refund for a returned item.\n"
                "- search_products(query: str): Search the product catalog.\n"
                "- check_stock(item_id: str): Check item availability.\n"
                "- modify_order(order_id: str, changes: dict): Modify an existing order.\n\n"
                "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
                "with a JSON object inside containing 'name' and 'arguments'."
            ),
            "airline": (
                "You are a helpful customer service assistant for an airline. "
                "You can assist with flight booking, cancellations, seat selection, "
                "baggage inquiries, and flight status checks. Always be polite and professional.\n\n"
                "Available tools:\n"
                "- search_flights(origin: str, destination: str, date: str): Search available flights.\n"
                "- book_flight(flight_id: str, passenger_name: str, seat_preference: str): Book a flight.\n"
                "- check_flight_status(flight_id: str): Get current flight status.\n"
                "- cancel_booking(booking_ref: str): Cancel an existing booking.\n"
                "- get_booking_details(booking_ref: str): Get full booking information.\n"
                "- change_seat(booking_ref: str, seat_preference: str): Change seat assignment.\n"
                "- lookup_baggage(booking_ref: str): Check baggage allowance and status.\n\n"
                "When you need to use a tool, wrap the call in <function_call>...</function_call> tags "
                "with a JSON object inside containing 'name' and 'arguments'."
            ),
        }
        return policies.get(domain, policies["retail"])

    # ------------------------------------------------------------------
    # Tokenization with block tracking
    # ------------------------------------------------------------------

    def tokenize_with_block_tracking(
        self, text: str, parent_hash: str = ""
    ) -> Tuple[List[int], List[Dict]]:
        """
        Tokenize *text* and split token_ids into blocks of BLOCK_SIZE.
        Each block receives an identity hash over (token_ids, parent_hash, block_idx).

        Args:
            text: The raw text to tokenize (may include special chat tokens).
            parent_hash: Hash of the immediately preceding block ("" for first block).

        Returns:
            token_ids: Full list of token IDs for this text.
            blocks: List of dicts:
                {block_idx, token_range_start, token_range_end, block_hash, parent_hash}
        """
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return [], []

        blocks = []
        for i in range(0, len(token_ids), self._block_size):
            chunk = token_ids[i:i + self._block_size]
            block_idx = len(blocks)
            # If the chunk is shorter than block_size, pad tracking but still hash
            block_hash = compute_block_hash(
                chunk, parent_hash, block_idx, self._block_size
            )
            block_info = {
                "block_idx": block_idx,
                "token_range_start": i,
                "token_range_end": min(i + self._block_size, len(token_ids)),
                "block_hash": block_hash,
                "parent_hash": parent_hash,
            }
            blocks.append(block_info)
            parent_hash = block_hash  # chain to next block

        return token_ids, blocks

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def _build_chat_messages(
        self, system_policy: str, conversation_so_far: List[Dict]
    ) -> List[Dict]:
        """
        Build the full message list for the model's chat template.

        Qwen3 uses:
          <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>...
        """
        messages = [{"role": "system", "content": system_policy}]
        messages.extend(conversation_so_far)
        return messages

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply the tokenizer's chat template to a list of message dicts."""
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: manual construction for Qwen2.5-style format
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(parts) + "\n<|im_start|>assistant\n"

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_response(
        self, messages: List[Dict]
    ) -> Tuple[str, int, float, float]:
        """
        Generate assistant response from the current conversation.

        Returns:
            generated_text: Decoded assistant output.
            num_prefill_tokens: Number of tokens processed in the prefill pass.
            prefill_ms: Wall-clock time for the prefill forward pass.
            decode_ms: Wall-clock time for the decode (generation) phase.
        """
        prompt_text = self._apply_chat_template(messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        num_prefill_tokens = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy decode
            temperature=1.0,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # Separate prefill and decode timing using a forward-only pass
        prefill_ms = self._measure_prefill(inputs)

        generated_ids = outputs[0][num_prefill_tokens:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        decode_ms = max(0.0, total_ms - prefill_ms)

        return generated_text, num_prefill_tokens, prefill_ms, decode_ms

    @torch.no_grad()
    def _measure_prefill(self, inputs) -> float:
        """
        Estimate prefill latency by running a single forward pass
        (no generation). This isolates the KV computation time for the prefix.
        """
        t0 = time.perf_counter()
        _ = self._model(**inputs, use_cache=True)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def run_workflow(self, workflow: Dict) -> Dict:
        """
        Run a single tau-bench workflow and record the full trajectory.

        The conversation loop:
          step 0: system prompt (recorded but not generated)
          step 1: user initial instruction
          step 2: assistant response (generated)
          step 3: tool call (parsed from assistant)
          step 4: tool result (simulated)
          step 5: user simulator message (U2)
          step 6: assistant response (generated)
          ... loop ...
          step N: final assistant response (no tool call / task complete)

        Each step records: step_id, role, content, token_ids, block_assignments,
                          prefill_ms, decode_ms, and optionally tool_call / tool_result.
        """
        task_id = workflow.get("task_id", "unknown")
        domain = workflow.get("domain", "unknown")
        system_policy = workflow.get("system_policy", "")
        user_instruction = workflow.get("user_instruction", "")

        logger.info("=== Running workflow: %s (domain=%s) ===", task_id, domain)

        steps: List[Dict] = []
        global_token_offset = 0  # tracks absolute position across steps
        parent_hash = ""          # root block's parent is empty
        step_id = 0
        total_prefill_ms = 0.0
        total_decode_ms = 0.0
        total_tokens = 0

        # ---- Step 0: system prompt ----
        sys_tokens, sys_blocks = self.tokenize_with_block_tracking(
            system_policy, parent_hash=parent_hash
        )
        if sys_blocks:
            parent_hash = sys_blocks[-1]["block_hash"]

        _register_blocks(self._global_block_index, sys_blocks, task_id, global_token_offset)
        global_token_offset += len(sys_tokens)
        total_tokens += len(sys_tokens)

        steps.append({
            "step_id": step_id,
            "role": "system",
            "content": system_policy,
            "token_ids": sys_tokens,
            "token_count": len(sys_tokens),
            "block_assignments": sys_blocks,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "tool_call": None,
            "tool_result": None,
        })
        step_id += 1

        # Conversation history (role/content pairs for the chat template)
        conversation: List[Dict] = []

        # ---- Step 1: user initial instruction ----
        user_msg = {"role": "user", "content": user_instruction}
        conversation.append(user_msg)

        usr_tokens, usr_blocks = self.tokenize_with_block_tracking(
            self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system_policy}, user_msg],
                tokenize=False,
                add_generation_prompt=False,
            ) if hasattr(self._tokenizer, "apply_chat_template") else
            f"<|im_start|>system\n{system_policy}<|im_end|>\n<|im_start|>user\n{user_instruction}<|im_end|>",
            parent_hash=parent_hash,
        )

        # For user message, we only track NEW tokens added beyond the system prefix
        new_user_tokens, new_user_blocks = self.tokenize_with_block_tracking(
            user_instruction, parent_hash=parent_hash
        )
        if new_user_blocks:
            parent_hash = new_user_blocks[-1]["block_hash"]

        _register_blocks(self._global_block_index, new_user_blocks, task_id, global_token_offset)
        global_token_offset += len(new_user_tokens)
        total_tokens += len(new_user_tokens)

        steps.append({
            "step_id": step_id,
            "role": "user",
            "content": user_instruction,
            "token_ids": new_user_tokens,
            "token_count": len(new_user_tokens),
            "block_assignments": new_user_blocks,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "tool_call": None,
            "tool_result": None,
        })
        step_id += 1

        # ---- Main conversation loop ----
        tools_called_this_workflow = 0

        for turn in range(MAX_WORKFLOW_TURNS):
            # --- Assistant generation ---
            full_messages = self._build_chat_messages(system_policy, conversation)
            try:
                assistant_text, num_prefill, prefill_ms, decode_ms = self._generate_response(
                    full_messages
                )
            except Exception as exc:
                logger.error("Generation failed for %s at turn %d: %s", task_id, turn, exc)
                break

            total_prefill_ms += prefill_ms
            total_decode_ms += decode_ms

            # Trim potential trailing/im_start tokens
            assistant_text = assistant_text.strip()

            # Tokenize and block-track assistant response
            asst_tokens, asst_blocks = self.tokenize_with_block_tracking(
                assistant_text, parent_hash=parent_hash
            )
            if asst_blocks:
                parent_hash = asst_blocks[-1]["block_hash"]

            _register_blocks(self._global_block_index, asst_blocks, task_id, global_token_offset)
            global_token_offset += len(asst_tokens)
            total_tokens += len(asst_tokens)

            assistant_msg = {"role": "assistant", "content": assistant_text}
            conversation.append(assistant_msg)

            # Try to parse a tool call
            tool_call = parse_tool_call(assistant_text)

            steps.append({
                "step_id": step_id,
                "role": "assistant",
                "content": assistant_text,
                "token_ids": asst_tokens,
                "token_count": len(asst_tokens),
                "block_assignments": asst_blocks,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "tool_call": tool_call,
                "tool_result": None,
            })
            step_id += 1

            # --- If no tool call, workflow is complete ---
            if not tool_call:
                logger.info("Workflow %s complete after %d turns (no tool call).", task_id, turn + 1)
                break

            tools_called_this_workflow += 1

            # --- Tool execution ---
            # TODO(W3): Integrate tau-bench backend simulator here.
            # The current implementation uses _simulate_tool_result() which returns
            # plausible mock responses. Replace with actual tau-bench tool execution
            # when the backend is integrated in W3-W4.
            tool_result_text = _simulate_tool_result(tool_call)

            tool_tokens, tool_blocks = self.tokenize_with_block_tracking(
                tool_result_text, parent_hash=parent_hash
            )
            if tool_blocks:
                parent_hash = tool_blocks[-1]["block_hash"]

            _register_blocks(self._global_block_index, tool_blocks, task_id, global_token_offset)
            global_token_offset += len(tool_tokens)
            total_tokens += len(tool_tokens)

            tool_msg = {"role": "tool", "content": tool_result_text}
            conversation.append(tool_msg)

            steps.append({
                "step_id": step_id,
                "role": "tool",
                "content": tool_result_text,
                "token_ids": tool_tokens,
                "token_count": len(tool_tokens),
                "block_assignments": tool_blocks,
                "prefill_ms": 0.0,
                "decode_ms": 0.0,
                "tool_call": None,
                "tool_result": tool_result_text,
            })
            step_id += 1

            # --- User simulator (next user message) ---
            # TODO(W3): Replace with actual tau-bench user simulator.
            # Currently generates a minimal acknowledgment/continuation.
            user_sim_text = _simulate_user_response(conversation, tool_result_text)

            usr2_tokens, usr2_blocks = self.tokenize_with_block_tracking(
                user_sim_text, parent_hash=parent_hash
            )
            if usr2_blocks:
                parent_hash = usr2_blocks[-1]["block_hash"]

            _register_blocks(self._global_block_index, usr2_blocks, task_id, global_token_offset)
            global_token_offset += len(usr2_tokens)
            total_tokens += len(usr2_tokens)

            user_sim_msg = {"role": "user", "content": user_sim_text}
            conversation.append(user_sim_msg)

            steps.append({
                "step_id": step_id,
                "role": "user",
                "content": user_sim_text,
                "token_ids": usr2_tokens,
                "token_count": len(usr2_tokens),
                "block_assignments": usr2_blocks,
                "prefill_ms": 0.0,
                "decode_ms": 0.0,
                "tool_call": None,
                "tool_result": None,
            })
            step_id += 1

        # ---- Handle zero tool calls ----
        if tools_called_this_workflow == 0 and turn >= MAX_WORKFLOW_TURNS - 1:
            logger.error(
                "Workflow %s produced zero tool calls across %d turns. "
                "Recording as-is; check model output format.",
                task_id, turn + 1,
            )

        # ---- Assemble trajectory ----
        trajectory = {
            "meta": {
                "workflow_id": task_id,
                "domain": domain,
                "model": self._config.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct"),
                "block_size": self._block_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_steps": len(steps),
                "total_tokens": total_tokens,
                "total_prefill_ms": round(total_prefill_ms, 2),
                "total_decode_ms": round(total_decode_ms, 2),
                "num_tool_calls": tools_called_this_workflow,
                "num_turns": turn + 1,
            },
            "steps": steps,
            "global_block_index": self._global_block_index,
        }

        logger.info(
            "Workflow %s done: %d steps, %d tokens, %.0f ms prefill, %.0f ms decode",
            task_id, len(steps), total_tokens, total_prefill_ms, total_decode_ms,
        )
        return trajectory

    # ------------------------------------------------------------------
    # Batch recording
    # ------------------------------------------------------------------

    def record_all(self, subset_path: str = "") -> List[str]:
        """
        Record trajectories for all workflows in the subset.

        Saves each trajectory to experiments/e1/traces/bf16/{workflow_id}.json

        Returns:
            List of saved file paths.
        """
        workflows = self.load_workflow_subset(subset_path)
        if not workflows:
            logger.warning("No workflows to record.")
            return []

        max_wf = self._config.get("workload", {}).get("max_workflows", len(workflows))
        workflows = workflows[:max_wf]

        saved_paths = []
        for i, workflow in enumerate(workflows):
            task_id = workflow.get("task_id", f"wf-{i:03d}")
            logger.info("[%d/%d] Recording workflow: %s", i + 1, len(workflows), task_id)

            try:
                trajectory = self.run_workflow(workflow)
            except Exception as exc:
                logger.exception("Failed to record workflow %s: %s", task_id, exc)
                continue

            # Save to file
            safe_id = _safe_filename(task_id)
            out_path = self._output_dir / f"{safe_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False, default=str)

            saved_paths.append(str(out_path))
            logger.info("Saved trajectory to %s", out_path)

        logger.info("Recorded %d/%d trajectories to %s",
                     len(saved_paths), len(workflows), self._output_dir)
        return saved_paths


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _register_blocks(
    global_index: Dict[str, Dict],
    blocks: List[Dict],
    workflow_id: str,
    offset: int,
) -> None:
    """Register each block in the global block index."""
    for blk in blocks:
        block_hash = blk.get("block_hash", "")
        if not block_hash:
            continue
        if block_hash not in global_index:
            global_index[block_hash] = {
                "token_start": offset + blk["token_range_start"],
                "token_end": offset + blk["token_range_end"],
                "parent_hash": blk.get("parent_hash", ""),
                "workflow_ids": [],
            }
        wf_ids = global_index[block_hash].setdefault("workflow_ids", [])
        if workflow_id not in wf_ids:
            wf_ids.append(workflow_id)


def _simulate_user_response(conversation: List[Dict], last_tool_result: str) -> str:
    """
    Produce a plausible user simulator response given conversation history
    and the latest tool result.

    TODO(W3): Replace with actual tau-bench user simulator.
    """
    # Simple: acknowledge the tool result and continue the task narrative
    try:
        result_data = json.loads(last_tool_result)
    except (json.JSONDecodeError, TypeError):
        return "Thank you. Please continue."

    # Try to generate a context-appropriate follow-up
    if "refund_id" in result_data or "refund" in str(result_data).lower():
        return "Thank you for processing the refund. How long will it take to appear on my card?"
    elif "results" in result_data:
        return "Those look good. Can you tell me more about the first option?"
    elif "flights" in result_data:
        return "Great, the morning flight at 8:30 AM works for me. Please book that one."
    elif "booking_ref" in result_data or "status" in result_data:
        return "Thank you for the information. That covers everything I needed."
    elif "eligible" in result_data:
        return "Yes, I'd like to proceed with the return."
    else:
        return "Thank you. Please continue."


def _safe_filename(name: str) -> str:
    """Convert a workflow ID into a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser():
    """Build the CLI argument parser for record_trajectories.

    Returns:
        argparse.ArgumentParser with G1 flags:
        --config / --dataset / --bfcl-subset / --seed / --max-episodes /
        --resume / --no-resume / --output-dir / --subset (legacy).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Record BF16 trajectories for G1 experiments.",
    )
    parser.add_argument(
        "--config", default="experiments/e1/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "tau-bench", "bfcl_v3"],
        help="Which dataset to record. 'all' = both (default: all)",
    )
    parser.add_argument(
        "--bfcl-subset", default=None,
        choices=["multi_turn_base", "multi_turn_miss_func",
                 "multi_turn_miss_param", "multi_turn_long_context"],
        help="BFCL subset (only valid with --dataset bfcl_v3). Default: all 4 subsets.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed to record. Default: all seeds from config.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap on episodes per (dataset, seed). For smoke tests.",
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Skip existing trace files (default: true)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-record even if trace file exists.",
    )
    # Legacy / compatibility flags (kept so old invocations still work).
    parser.add_argument(
        "--subset", default="",
        help="Path to workflow subset JSON (overrides config). Legacy.",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Override output directory for trajectory files.",
    )
    return parser


def main():
    """Entry point for E1 trajectory recording."""
    args = _build_arg_parser().parse_args()

    recorder = TrajectoryRecorder(config_path=args.config)

    if args.output_dir:
        recorder._output_dir = Path(args.output_dir)
        recorder._output_dir.mkdir(parents=True, exist_ok=True)

    # Note: args.seed / args.dataset / args.bfcl_subset / args.max_episodes
    # are consumed by the multi-seed multi-dataset recording loop that will
    # be implemented in Task 4-7. For now we delegate to the existing
    # record_all path so legacy behavior is preserved.
    saved_paths = recorder.record_all(subset_path=args.subset)

    report = {
        "experiment": "E1",
        "description": "Workload characterization trajectory recording",
        "total_workflows_recorded": len(saved_paths),
        "output_directory": str(recorder._output_dir),
        "files": saved_paths,
    }
    report_path = recorder._output_dir / "_recording_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"E1 Recording complete.")
    print(f"Workflows recorded: {len(saved_paths)}")
    print(f"Output directory:   {recorder._output_dir}")
    print(f"Report:             {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
