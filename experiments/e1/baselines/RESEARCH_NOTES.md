# Closest Baseline Research Notes

Research date: 2026-07-26
Researcher: sub-agent (GLM-5.2) delegated by parent agent
Scope: Verify official code availability for two candidate closest-baseline papers and assess feasibility of running them on τ-bench traces with `block_assignments` schema (`block_hash`, `parent_hash`, `token_range_start/end`) on a single RTX 4090D (24 GB VRAM).

---

## PBKV (arXiv 2605.06472)

### Search results
- WebSearch `"PBKV KV cache github arxiv 2605.06472 code"` — top hit is the arXiv abstract page itself (no code link in abstract). No third-party GitHub repo found.
- WebFetch `https://arxiv.org/abs/2605.06472` — abstract page only links to PDF, HTML, TeX source, and DOI. No "Code" link next to the title.
- WebFetch `https://arxiv.org/html/2605.06472v1` — full HTML paper read. No mention of a code release URL anywhere in the body, conclusion, or appendix. The paper says "PBKV is built on SGLang" and uses GraphSAGE as the predictor backbone, but does not point to any released artifact.
- WebSearch `"PBKV" "Prediction-Based KV-Cache" github code repository` — only third-party blogs summarizing the paper (st-hakky.com, arxiv.deeppaper.ai); none link to a code repo.
- WebSearch `"Haoyu Zheng" PBKV workflow github code` — returned an unrelated GitHub user `11Haoyu` (economist profile, no PBKV repo) and a different "Haoyu Zheng" at Zhejiang University (PILOT paper, ACL 2026). The PBKV first author Haoyu Zheng is at Wuhan University — no public repo found under his name.
- WebSearch `"PBKV" "code is available" OR "github.com" Haoyu Zheng Wuhan University` — no hits. (A neighboring paper `vibe-serve` from UW-SyFi does release code, but PBKV itself does not.)
- Conclusion of search: no public code artifact for PBKV exists as of 2026-07-26.

Paper metadata confirmed:
- Title: "Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management"
- Authors: Haoyu Zheng (Wuhan Univ.), Fangcheng Fu (SJTU), Jia Wu (Macquarie), Binhang Yuan (HKUST), Yongqiang Zhang (Dameng DB), Hao Wang, Yuanyuan Zhu, Xiao Yan, Jiawei Jiang (corresponding: jiawei.jiang@whu.edu.cn)
- Submitted: 7 May 2026 (v1) — very recent preprint (~2.5 months old at research date), no venue acceptance announced yet.
- License: CC BY 4.0 (paper only).

### Official code availability
- **Status**: UNAVAILABLE
- **Repo URL**: not found
- **Last commit**: N/A
- **Trace input format**: unknown (no code to inspect). From the paper: PBKV is built on SGLang's Radix Tree + HiCache two-tier store, and the predictor consumes (i) the global call graph G (agent transition patterns) and (ii) the per-request LLM prefill embedding x. This implies the input is the SGLang request stream + an agent call graph, NOT a τ-bench-style block-assignment trace.
- **τ-bench compatibility**: INCOMPATIBLE (without code). The paper's input abstractions (agent call graph + prefill embeddings) are richer than our `block_assignments` trace; even if code existed, an adapter would be required to map `block_hash`/`parent_hash`/`token_range_*` onto SGLang radix-tree nodes and to synthesize a call graph from per-step records.
- **Dependencies** (inferred from paper, not from repo):
  - SGLang (base inference engine, with Radix Tree + HiCache)
  - PyTorch (GraphSAGE backbone training/inference)
  - PyTorch Geometric (PyG) or DGL (for GraphSAGE; paper cites Hamilton et al. GraphSAGE)
  - vLLM is NOT used (paper explicitly says PBKV is on SGLang, and contrasts with vLLM)
  - HuggingFace Transformers (model weights)
- **4090D 24GB feasibility**: UNKNOWN without code. Paper reports experiments on three workflow benchmarks with models up to Llama-3.1-8B-class. A 7B–8B model in fp16 (~15 GB weights) plus KV cache + GraphSAGE predictor should fit in 24 GB at small batch sizes, but the paper does not disclose exact GPU memory usage. Feasible in principle, unverifiable in practice without the implementation.

### Recommendation
PBKV has no public code as of 2026-07-26 and the paper is a fresh preprint with no announced venue. Do not attempt faithful reproduction — instead build an **inspired variant** that captures PBKV's two core ideas (multi-step lookahead reuse score + hierarchical eviction that reclaims retired-workflow private cache first), reusing our existing τ-bench trace adapter and a lightweight GraphSAGE predictor trained on the call graph derived from `block_hash`/`parent_hash` chains.

---

## KVFlow (arXiv 2507.07400)

### Search results
- WebSearch `"KVFlow github arxiv 2507.07400 Agent Step Graph code"` — top hit is the official OpenReview page (NeurIPS 2025 poster, accepted) plus the GitHub repo `https://github.com/PanZaifeng/KVFlow`.
- WebFetch `https://arxiv.org/abs/2507.07400` — abstract page; links to PDF, HTML, TeX source, DOI. (arXiv abstract pages do not surface code links, but the OpenReview page and README confirm the repo.)
- WebFetch `https://github.com/PanZaifeng/KVFlow` — confirmed official repo. README explicitly states: "This repository contains the source code for [NeurIPS'25] KVFlow". Maintained by first author Zaifeng Pan (UCSD, corresponding author Yufei Ding's group).
- WebFetch `https://github.com/PanZaifeng/KVFlow/tree/main/SScheduler` — README for `PFEngine` (the SScheduler layer): a lightweight Python scheduling/timestep engine, NOT a trace-replay harness. Public API is `Scheduler`, `PlanManager`, `SpaceManager` — agent simulations register managers and call `update_agent_timestep(...)`. No JSON-trace loader.
- WebFetch `https://github.com/PanZaifeng/KVFlow/tree/main/benchmark` — folder is a fork of SGLang's benchmark suite (gsm8k, mmlu, hellaswag, mtbench, react, multi_turn_chat, lora, hicache, etc.). No τ-bench directory. No trace-replay benchmark.
- WebFetch `https://github.com/PanZaifeng/KVFlow/tree/main/python/sglang` — confirms the serving engine is a modified SGLang (configs/, eval/, lang/, srt/, test/), launched via `python -m sglang.launch_server --config ./python/sglang/configs/example.yaml`. CLI knobs include `--load_ahead_step`, `--evict_pri_level`, `--enable_holding`, `--enable_interrupt`, `--disable_prefetch`, `--disable_kv_pf`.

Repo activity:
- Created: Feb 18, 2026 (initial commit `0db29c6`).
- Last commit: Mar 13, 2026 (`7ef897e` — "Update README").
- Total commits: 2 (very low activity; essentially a one-shot release).
- Branches: 1, Tags: 0, Issues: 1 open.
- Languages: Python 75.4%, Rust 10.5%, C++ 6.7%, CUDA 6.7%, Shell 0.3%.
- License: Apache-2.0.

### Official code availability
- **Status**: AVAILABLE
- **Repo URL**: https://github.com/PanZaifeng/KVFlow
- **Last commit**: 2026-03-13 (only 2 commits total since 2026-02-18)
- **Trace input format**: KVFlow does NOT consume a τ-bench-style block-assignment trace. Its two input channels are: (1) `SScheduler/` (PFEngine) — a Python API where the user instantiates `PlanManager`/`SpaceManager` and calls `update_agent_timestep(...)` to feed agent-step metadata at runtime; (2) `python/sglang/` — a modified SGLang server configured via YAML/JSON, launched with `python -m sglang.launch_server --config <yaml>`. The Agent Step Graph (ASG) is constructed in code via the SScheduler API, not loaded from a JSON file.
- **τ-bench compatibility**: NEEDS_ADAPTER. Our τ-bench traces (each step carries `block_assignments` with `block_hash`, `parent_hash`, `token_range_start`, `token_range_end`) cannot be fed directly. We would need to: (a) translate the per-step `block_assignments` into SGLang prefix-tree requests issued against the running server, and (b) drive the SScheduler `PlanManager` with step metadata derived from the trace (mapping `parent_hash` chains to ASG edges and `token_range_*` to KV-cache node sizes). The ASG abstraction in KVFlow is a natural fit for our `block_hash`/`parent_hash` DAG, but the integration requires a custom adapter and a running SGLang server.
- **Dependencies** (from README and repo layout):
  - Modified SGLang (vendored at `python/sglang/` — this is the serving engine)
  - PyTorch (SGLang dependency)
  - CUDA / `sgl-kernel` (vendored C++/CUDA kernels for attention)
  - Rust components (`sgl-router/` — 10.5% of code)
  - vLLM is NOT used; KVFlow is SGLang-based (paper explicitly compares against SGLang hierarchical radix cache baseline)
  - GraphSAGE / PyG / DGL: NOT required by KVFlow (KVFlow uses a deterministic steps-to-execution computation over ASG, no learned predictor)
  - HuggingFace Transformers (model weights via `--model-path`)
- **4090D 24GB feasibility**: YES (with caveats). The paper reports experiments on Qwen2.5-7B and Llama-3.1-8B at `--max-total-tokens 100000` with `--enable-hierarchical-cache`. A 7B model in fp16 (~14–15 GB weights) leaves ~9–10 GB for KV cache and CUDA graphs on a 4090D — tight but workable for single-workflow latency experiments. The README's example invocation (`--max-total-tokens 100000 --hicache-size 20`) suggests this is the intended configuration class. Caveats: (1) high-concurrency multi-workflow scenarios from the paper likely exceed 24 GB; (2) Rust toolchain + CUDA build needs a Linux environment (Windows native support is uncertain — AutoDL Linux used as of 2026-07-26).
- **Linux feasibility (AutoDL)**: YES (active, 2026-07-26 upgrade). AutoDL Linux platform provides root + CUDA toolkit + sufficient disk (~20GB for SGLang + Rust build). KVFlow `kvflow_faithful` entry in `experiments/e1/config.yaml` upgraded from `enabled: false / deferred` to `enabled: true / active`. τ-bench adapter engineering (clone → compile → adapter → run) is a follow-up independent task.

### Recommendation
KVFlow has an official, publicly available, Apache-2.0-licensed codebase built on a modified SGLang. It is the more practical choice for a faithful reproduction. As of 2026-07-26, the AutoDL Linux environment is available (root + CUDA + Rust toolchain), so the original WSL2/Linux constraint no longer applies. Remaining work: (i) a custom τ-bench trace adapter that translates `block_assignments` into SGLang prefix-tree requests and ASG step metadata via the SScheduler API, and (ii) a 7B-class model that fits in 24 GB. Plan for ~1–2 weeks of adapter engineering before any baseline number can be reproduced. Status: **active faithful reproduction** (config.yaml `enabled: true`); adapter implementation in progress.

---

## ThunderAgent (arXiv 2602.13692, ICML 2026 Spotlight)

### Search results
- WebSearch `"ThunderAgent github arxiv 2602.13692 program-aware KV code"` — top hit is the official repo `https://github.com/ThunderAgent-org/ThunderAgent`. ICML 2026 virtual page links to the same repo.
- WebFetch `https://arxiv.org/abs/2602.13692` — abstract page; the paper is "ThunderAgent: Program-aware KV Cache Management for Agent Inference" (ICML 2026 Spotlight).
- WebFetch `https://github.com/ThunderAgent-org/ThunderAgent` — confirmed official repo. README explicitly states: "ThunderAgent itself does not require a GPU" — it is a FastAPI proxy that routes OpenAI-compatible requests to vLLM / SGLang / SkyRL backends with program-aware capacity scheduling. The only API change required from the client side is adding `program_id` to `extra_body`.
- Repo activity:
  - Created: 2026-02-10.
  - Last commit: 2026-06-06 (144 commits total — active maintenance).
  - Branches: 1, Tags: 0, Issues: 3 open.
  - Languages: Python 100% (pure Python, no Rust/CUDA compilation).
  - License: MIT.
  - Integrations: NVIDIA Dynamo 2.0, SkyRL.

### Official code availability
- **Status**: AVAILABLE
- **Repo URL**: https://github.com/ThunderAgent-org/ThunderAgent
- **Last commit**: 2026-06-06
- **Trace input format**: OpenAI-compatible API calls; the only augmentation is `program_id` in `extra_body`. CLI: `thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000`. The agent runtime is FastAPI; it does not consume a JSON block-assignment trace.
- **τ-bench compatibility**: NEEDS_ADAPTER (inspired variant). ThunderAgent is an API-level proxy, not a block-level cache policy. Its core mechanisms — `--use-acting-token-decay` (priority ∝ 2^{-t} where t is the time since the workflow's last activity), `--router tr` (program-aware capacity scheduling), and `--gpu-memory-pressure` (online feedback) — operate at the request/program level, not on individual KV blocks. To run on our τ-bench traces (`block_hash`/`parent_hash`/`token_range_*`), we would need to (a) map each `program_id` to a `workflow_id` carried in the trace, (b) translate the API-level time-decay policy into a block-level priority score, and (c) drop the GPU-memory-pressure feedback loop (no live backend in open-loop replay).
- **Windows feasibility**: YES (native). Pure Python 100%, `pip install -e .` with no Rust/CUDA build. Core deps: fastapi, httpx, uvicorn. ThunderAgent itself is CPU-bound (it proxies to a separate vLLM/SGLang backend), so it runs natively on Windows.
- **4090D 24GB feasibility**: N/A for the proxy itself (CPU-only). Backend inference (vLLM/SGLang with a 7B model) requires GPU but is decoupled from the proxy. For our inspired variant on the open-loop trace replay, no GPU is needed at all — it is a pure Python cache policy.

### Recommendation
ThunderAgent has public, MIT-licensed, pure-Python code and is ICML 2026 Spotlight with industrial adoption (NVIDIA Dynamo 2.0, SkyRL). However, ThunderAgent is an API-level proxy, not a block-level cache policy — a faithful reproduction would require a live vLLM/SGLang backend and a τ-bench adapter that translates `block_assignments` into OpenAI-compatible requests with `program_id`. Instead, we capture ThunderAgent's three distinguishing ideas at the block-cache level as an **inspired variant**:
1. **Program-aware (workflow-aware) grouping** — blocks belonging to the same `workflow_id` are managed as a group; per-workflow last-activity timestamp is tracked.
2. **Time decay** — priority score includes a `2^{-(now - workflow_last_activity) * decay_rate}` factor; blocks from paused workflows decay exponentially.
3. **Capacity scheduling across workflows** — when evicting, prefer blocks from the most-paused workflow (the block with the minimum composite priority score is evicted, plus its prefix-chain descendants).

The inspired variant likely UNDERESTIMATES ThunderAgent's true performance (the paper's online decay-rate tuning + GPU-pressure feedback would adapt better than our hand-tuned rate). If this inspired variant already shows non-trivial improvement over simple heuristics, the faithful ThunderAgent would likely show even larger improvement.

Implemented in `experiments/e1/baselines/thunderagent_inspired.py` (258 lines), with 9 unit tests in `test_thunderagent_inspired.py` (all passing). Integrated into `experiments/e1/compare_oracle.py` as the `thunderagent_inspired` baseline alongside `pbkv_inspired`.

---

## Overall Recommendation
- **Chosen closest baselines**: KVFlow (faithful reproduction via official repo, **AutoDL Linux environment available as of 2026-07-26, adapter implementation in progress**) + PBKV-inspired variant + ThunderAgent-inspired variant (two lightweight reimplementations on top of our existing τ-bench harness).
- **Reasoning**:
  - **KVFlow** is the only candidate with public official code that is NeurIPS-2025-accepted, Apache-2.0, and built on SGLang (same engine family PBKV targets). It is the faithful-reproduction anchor; its ASG abstraction maps cleanly onto our `block_hash`/`parent_hash` traces. As of 2026-07-26, AutoDL Linux environment is available (root + CUDA + Rust toolchain), so the original WSL2/Linux constraint no longer applies. `kvflow_faithful` upgraded to `enabled: true` in config.yaml; adapter implementation in progress.
  - **PBKV-inspired** captures PBKV's two distinguishing ideas (GraphSAGE-style multi-step lookahead reuse score + chain-aware hierarchical eviction) because PBKV has no public code (2.5-month-old preprint, no venue). Faithful reproduction is impossible.
  - **ThunderAgent-inspired** captures ThunderAgent's three distinguishing ideas (program-aware workflow grouping + 2^{-t} time decay + cross-workflow capacity scheduling) because ThunderAgent is an API-level proxy, not a block-level cache policy. ICML 2026 Spotlight + NVIDIA Dynamo 2.0 adoption gives this variant the strongest venue/industrial-validation backing among the three.
  - The two inspired variants are **complementary**: PBKV-inspired focuses on *reuse prediction* (GraphSAGE-style features + multi-step lookahead), while ThunderAgent-inspired focuses on *workflow-aware time decay* (program-level scheduling + exponential decay). Together they cover the two main axes of the closest-baseline landscape that simple heuristics (LRU/GDSF/SizeCost/APC-LRU) miss.
- **Implementation plan**:
  1. (Done) Implement PBKV-inspired variant: `baselines/pbkv_inspired.py` + `test_pbkv_inspired.py`, integrated into `compare_oracle.py`.
  2. (Done) Implement ThunderAgent-inspired variant: `baselines/thunderagent_inspired.py` + `test_thunderagent_inspired.py` (9 unit tests, all passing), integrated into `compare_oracle.py` as `thunderagent_inspired`.
  3. (Active, 2026-07-26 upgrade) KVFlow faithful reproduction: clone `https://github.com/PanZaifeng/KVFlow`, build modified SGLang + sgl-kernel + Rust on **AutoDL Linux** with CUDA, smoke-test the example YAML config with a 7B model on the 4090D. Write a τ-bench adapter that converts each `block_assignments` step into (a) an SGLang prefix-cache request and (b) an SScheduler `PlanManager.update_agent_timestep(...)` call, constructing the ASG from `parent_hash` chains. Status: `kvflow_faithful.enabled` upgraded to `true` in config.yaml; adapter engineering (~1–2 weeks) is the next independent task. The two inspired variants already provide immediate closest-baseline coverage for G1 while the adapter is being built.
  4. (Done) Run all baselines on τ-bench traces via `compare_oracle.py`; the output JSON now contains `pbkv_inspired` and `thunderagent_inspired` alongside the 6 heuristic/oracle baselines (LRU, GDSF, SizeCost, APC-LRU, Belady, Oracle-Cost).
