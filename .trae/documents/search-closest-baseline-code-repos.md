# 搜索同领域论文的开源代码作为 G1 Closest Baseline

## Summary

当前 G1 实验已有 6 个纯代码 baseline（LRU/GDSF/SizeCost/APC-LRU/Belady/Oracle-Cost）+ 1 个 PBKV-inspired variant。但 closest baseline 条件仅靠 inspired variant 满足，属于"弱满足"——PBKV 无官方代码、KVFlow 有官方代码但需 WSL2+Rust/CUDA 无法在当前 Windows 环境复现。

本计划的目标：**系统搜索 §3.1 表中其他 closest baseline 候选论文的开源代码仓库**，筛选出**纯 Python、可在 Windows + RTX 4090D 直接运行、可适配 τ-bench trace 格式**的实现，作为 G1 实验的补充 closest baseline（faithful reproduction 优于 inspired variant）。

## Current State Analysis

### 已实现 baseline（[experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py)）
| # | Baseline | 角色 | 类型 |
|---|---|---|---|
| 1-4 | LRU / GDSF / SizeCost / APC-LRU | 简单启发式 | 纯代码 |
| 5-6 | Belady / Oracle-Cost | Oracle 上界 | 纯代码 |
| 7 | PBKV-Inspired | closest baseline | **inspired variant**（PBKV 无官方代码）|

### 已调研但未实现的 closest baseline（[experiments/e1/baselines/RESEARCH_NOTES.md](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/RESEARCH_NOTES.md)）
- **KVFlow**（arXiv 2507.07400, NeurIPS 2025）：官方代码 https://github.com/PanZaifeng/KVFlow 可用，但 Rust 10.5% + CUDA 13.4% + 魔改 SGLang，**需 WSL2/Linux 构建，当前 Windows 环境无法复现** → 排除

### §3.1 表中尚未调研代码可用性的 closest baseline 候选（13 项）
依据 [IDEA.rewritten.md §3.1](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) L222-240：

| # | 工作 | arXiv/Venue | 是否已调研代码 |
|---|---|---|---|
| 1 | vLLM APC | docs.vllm.ai | ✓（已实现为 APCLRUCache）|
| 2 | KVFlow | 2507.07400 | ✓（需 WSL2，排除）|
| 3 | PBKV | 2605.06472 | ✓（无代码，已做 inspired）|
| 4 | **CacheWise** | 2606.16824 | ✗ 待调研 |
| 5 | **Learned Prefix Caching** | NeurIPS 2025 | ✗ 待调研 |
| 6 | **InferCept** | PMLR v235 | ✗ 待调研 |
| 7 | **Continuum** | 2511.02230 | ✗ 待调研 |
| 8 | **ThunderAgent** | 2602.13692 | ✗ 待调研 |
| 9 | **TokenCake** | 2510.18586 | ✗ 待调研 |
| 10 | **Helium** | 2603.16104 | ✗ 待调研 |
| 11 | **ARKV** | 2603.08727 | ✗ 待调研 |
| 12 | **QKVShare** | 2605.03884 | ✗ 待调研 |
| 13 | **GraphFlow** | 2605.22566 (ICML 2026) | ✗ 待调研 |
| 14 | **Agent Memory** | 2603.04428 | ✗ 待调研 |
| 15 | **HybridFlow** | 2512.22137 | ✗ 待调研 |

### G1 通过条件（[IDEA.rewritten.md §7 G1](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) L548-557）
1. **headroom ≥ 10%**：已满足（Oracle-Cost 12.8% − max simple 0% = 12.8% @ budget 0.10）
2. **closest baseline 可比性**：当前仅靠 PBKV-inspired 满足，需补强

## Proposed Changes

### Step 1：系统调研 12 项候选论文的开源代码

对 §3.1 表中尚未调研代码可用性的 12 项 closest baseline 候选，逐一搜索开源仓库。

**调研方法**（每篇论文执行）：
1. **WebSearch** 搜索 `"<论文标题>" github code` / `"<论文简称>" github repository` / `"<第一作者> <关键词>" github`
2. **WebFetch** arXiv abs 页面，查找 "Code" 链接
3. **WebFetch** OpenReview / PMLR / ACM DL 页面（若有），查找 code release 链接
4. 若找到 repo，**WebFetch repo README**，提取：
   - 编程语言占比（Python / C++ / Rust / CUDA）
   - 依赖列表（是否需编译 Rust/CUDA 内核）
   - 是否依赖特定推理引擎（SGLang / vLLM / DeepSpeed）
   - 输入格式（trace 格式 / API 调用 / benchmark 数据集）
   - License

**调研候选清单**（12 项，按 §3.1 表顺序）：
1. CacheWise (2606.16824)
2. Learned Prefix Caching (NeurIPS 2025, 414f642a)
3. InferCept (PMLR v235, abhyankar24a)
4. Continuum (2511.02230)
5. ThunderAgent (2602.13692)
6. TokenCake (2510.18586)
7. Helium (2603.16104)
8. ARKV (2603.08727)
9. QKVShare (2605.03884)
10. GraphFlow (2605.22566, ICML 2026)
11. Agent Memory (2603.04428)
12. HybridFlow (2512.22137)

**输出**：每篇论文填入下表，写入 `reviews/closest-baseline-code-search.md`：

| 工作 | arXiv | Repo URL | 语言/构建 | 依赖引擎 | 输入格式 | License | Windows 可行性 | τ-bench 兼容性 | 与 FlowCache 接近度 | 推荐度 |

### Step 2：筛选与排序

对每篇有开源代码的论文，按以下 4 个维度评分（每维 0-3 分，总分 0-12）：

| 维度 | 0 分 | 1 分 | 2 分 | 3 分 |
|---|---|---|---|---|
| Windows 可行性 | 需 WSL2+Rust/CUDA | 需 WSL2 但纯 Python | 原生 Windows + 少量适配 | 原生 Windows + pip install |
| τ-bench 兼容性 | 需重写输入层 | 需大幅适配 | 需中等适配 | 需轻量 adapter |
| 与 FlowCache 接近度 | 不同问题域 | 部分相关 | 工作流感知 KV 管理 | closest baseline（§3.1 表）|
| 代码质量/可维护性 | 无文档/无法运行 | 有文档但难运行 | 可运行但需调试 | 文档完善可直接运行 |

**筛选标准**：
- **优先实现**：总分 ≥ 9 且 Windows 可行性 ≥ 2
- **备选**：总分 6-8，记录但暂不实现
- **排除**：总分 < 6 或 Windows 可行性 = 0

### Step 3：输出调研报告

创建 `reviews/closest-baseline-code-search.md`，包含：
1. 调研方法与范围
2. 12 项候选论文逐项调研结果（含 repo URL、语言、依赖、Windows 可行性、τ-bench 兼容性）
3. 评分表与排序
4. Top 1-3 推荐候选（若有）
5. 若无合格候选：明确结论 + 建议（如"全部 closest baseline 候选均无法在 Windows 原生运行，建议接受 PBKV-inspired 作为 closest baseline 并在论文中明确标注"）

### Step 4（条件性）：实现推荐的 closest baseline

**仅当 Step 3 筛选出 ≥1 个合格候选时执行**：
1. 在 `experiments/e1/baselines/` 下创建 `<paper_name>_adapter.py` 或 `<paper_name>_inspired.py`
2. 文件 docstring 顶部明确标注 "FAITHFUL REPRODUCTION" 或 "INSPIRED VARIANT"（参考 [pbkv_inspired.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/pbkv_inspired.py) 的标注格式）
3. 实现缓存类，接口与现有 7 个 baseline 一致：
   - `__init__(self, capacity: int, ...)`
   - `access(self, block_hash: str, parent_hash: str = "", prefill_ms: float = 0.0) -> bool`
   - 计数器：`hits` / `misses` / `evictions` / `saved_prefill_ms` / `miss_cost_ms`
4. 在 [experiments/e1/compare_oracle.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/compare_oracle.py) 的 `main()` 中集成新 baseline
5. 更新 `_print_summary` 表格追加新列
6. 创建 `experiments/e1/baselines/test_<paper_name>.py` 单元测试
7. 重跑 `python experiments/e1/compare_oracle.py` 验证
8. 更新 [experiments/e1/baselines/RESEARCH_NOTES.md](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/baselines/RESEARCH_NOTES.md) 追加新 baseline 的调研结论

### Step 5（条件性）：更新 spec 与文档

**仅当 Step 4 执行时**：
1. 更新 [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) G1.4.1 可比性检查清单，填入新 baseline 的判定结果
2. 更新 [IDEA.rewritten.md §8 Ch.4](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) 主表对照行（若有变更）
3. 在 `.trae/specs/complete-g1-baselines/` 下追加新的 spec delta（若实现工作量较大）

## Assumptions & Decisions

### 假设
1. §3.1 表中的 12 项候选论文覆盖了 FlowCache 的全部 closest baseline 空间（已排除 KVFlow/PBKV/vLLM APC）
2. 纯 Python 实现意味着：无 Rust/CUDA 源码编译、无 C++ 扩展构建、pip install 后可直接 import
3. τ-bench trace 格式（`block_assignments` 含 `block_hash` / `parent_hash` / `token_range_start` / `token_range_end`）是 FlowCache 的唯一输入格式，候选 baseline 需能适配此格式或其调度策略可在此 trace 上 replay

### 决策
1. **范围限定**：仅搜索 §3.1 表中的 closest baseline 候选（用户明确选择"Closest baseline 优先"），不扩展到量化/residency 论文（G4 范围）
2. **环境约束**：仅接受纯 Python 可在 Windows 原生运行的实现（用户明确选择"纯 Python 可直接运行"），排除需 WSL2/Rust/CUDA 的 repo
3. **不替换 PBKV-inspired**：新 baseline 作为**补充**而非替换，PBKV-inspired 保留以供对比
4. **faithful 优先**：若某论文有官方代码且可忠实运行，优先于 inspired variant
5. **若无可行候选**：接受现状（PBKV-inspired + KVFlow 待 WSL2），在调研报告中明确记录并结束

### 不做的事
- 不修改现有 7 个 baseline 的实现
- 不删除 PBKV-inspired variant
- 不实现需 WSL2 的 KVFlow adapter（用户已确认环境约束）
- 不扩展到 G4 量化 baseline 搜索（超出当前范围）

## Verification

### 调研报告完整性验证
- [ ] `reviews/closest-baseline-code-search.md` 已创建
- [ ] 12 项候选论文逐一调研，每项含 repo URL（或"无代码"结论）
- [ ] 评分表完整（4 维度 × 12 候选）
- [ ] Top 推荐（或"无合格候选"结论）明确

### 条件性实现验证（若执行 Step 4）
- [ ] 新 baseline 文件已创建，docstring 标注 FAITHFUL/INSPIRED
- [ ] `compare_oracle.py` 集成新 baseline，results JSON 含新字段
- [ ] `_print_summary` 表格含新列
- [ ] 单元测试 `test_<paper_name>.py` 通过
- [ ] `python experiments/e1/compare_oracle.py` 成功运行
- [ ] 新 baseline 的 hit% 介于最佳简单策略与 Oracle 之间（合理性检查）
- [ ] `RESEARCH_NOTES.md` 已追加新 baseline 调研结论

### 文档同步验证（若执行 Step 5）
- [ ] `experiment-designs.md` G1.4.1 表已更新
- [ ] `IDEA.rewritten.md` §8 Ch.4 已更新（若有变更）

## 执行顺序

1. **Step 1-3**（调研与报告）：并行调研 12 项候选，输出 `reviews/closest-baseline-code-search.md`
2. **判断点**：若 Step 3 筛选出 ≥1 个合格候选 → 执行 Step 4-5；否则结束
3. **Step 4-5**（条件性实现）：实现推荐的 closest baseline + 更新文档
