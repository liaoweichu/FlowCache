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
- **4090D 24GB feasibility**: YES (with caveats). The paper reports experiments on Qwen2.5-7B and Llama-3.1-8B at `--max-total-tokens 100000` with `--enable-hierarchical-cache`. A 7B model in fp16 (~14–15 GB weights) leaves ~9–10 GB for KV cache and CUDA graphs on a 4090D — tight but workable for single-workflow latency experiments. The README's example invocation (`--max-total-tokens 100000 --hicache-size 20`) suggests this is the intended configuration class. Caveats: (1) high-concurrency multi-workflow scenarios from the paper likely exceed 24 GB; (2) Rust toolchain + CUDA build may need a Linux environment (Windows native support is uncertain — likely needs WSL2).

### Recommendation
KVFlow has an official, publicly available, Apache-2.0-licensed codebase built on a modified SGLang. It is the more practical choice for a faithful reproduction, but it requires (i) a Linux/WSL2 build environment with CUDA + Rust toolchain, (ii) a custom τ-bench trace adapter that translates `block_assignments` into SGLang prefix-tree requests and ASG step metadata via the SScheduler API, and (iii) a 7B-class model that fits in 24 GB. Plan for ~1–2 weeks of adapter engineering before any baseline number can be reproduced.

---

## Overall Recommendation
- **Chosen closest baseline**: KVFlow (faithful reproduction via official repo) + PBKV-inspired variant (lightweight reimplementation on top of our existing τ-bench harness).
- **Reasoning**: KVFlow is the only one of the two with public official code, is NeurIPS-2025-accepted, is Apache-2.0, and is built on SGLang which is the same engine family PBKV targets — so KVFlow's repo can serve as the faithful-reproduction anchor and its ASG abstraction maps cleanly onto our `block_hash`/`parent_hash` traces. PBKV has no public code (paper is a 2.5-month-old preprint with no venue), so faithful reproduction is impossible; instead we capture PBKV's two distinguishing ideas (multi-step lookahead reuse score + hierarchical eviction of retired-workflow private cache first) as an inspired variant layered on top of either the KVFlow codebase or our own simulator.
- **Implementation plan**: (1) Clone `https://github.com/PanZaifeng/KVFlow`, build SGLang + sgl-kernel under WSL2 with CUDA, smoke-test the example YAML config with a 7B model on the 4090D. (2) Write a τ-bench adapter that converts each `block_assignments` step into (a) an SGLang prefix-cache request and (b) an SScheduler `PlanManager.update_agent_timestep(...)` call, constructing the ASG from `parent_hash` chains. (3) Once KVFlow baseline numbers are reproduced on a subset of `experiments/e1/traces/bf16/tau_bench/*.json`, implement the PBKV-inspired variant (GraphSAGE multi-step lookahead + hierarchical eviction) on top of the same harness for head-to-head comparison.
