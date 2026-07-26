# FlowCache 项目介绍与进展

> 本文件为 FlowCache 项目的高层概览，整合 idea、执行计划、Gate 状态与代码结构。
> 最后更新：2026-07-26

---

## 1. 项目简介

| 项 | 内容 |
|---|---|
| 项目名称 | **FlowCache**: Decoupling Reuse Value and Fidelity Risk for Prefix Caching in Memory-Constrained Agent Workflows |
| 中文题目 | 面向内存受限 Agent 工作流的复用价值–保真风险解耦前缀缓存 |
| 主投方向 | IEEE ICWS 2027（CFP TBD，截稿时间待官方发布） |
| 备选 venue | IEEE EDGE 2026/2027；具备真实双节点部署后考虑 IEEE TSC |
| 硬件约束 | 单卡 NVIDIA RTX 4090D 24GB |
| 结论边界 | memory-constrained GPU emulation，不等同于真实移动/边缘设备 |
| 主模型 | Qwen2.5-7B-Instruct（BF16 ~15GB，GQA + RoPE，原生 tool calling） |
| 项目阶段 | Conditional Go——G0 已通过，G1 进行中，路线 A 推进中 |

---

## 2. 研究问题

多工作流 Agent 服务在推理、工具等待、分支切换、会话恢复之间反复暂停与继续。暂停后的 KV 前缀若全部留在 GPU 显存吃紧，全部卸载或驱逐则恢复时需重传或重新 prefill。已有工作分别优化"未来是否复用"和"缓存精度"，但通常未联合决策。

**核心研究问题**：在固定 GPU 缓存预算和任务质量非劣约束下，分别估计 exact-prefix KV block 的未来复用价值与量化敏感度，并联合决定其精度和驻留位置，能否比"预测式驱逐 + 统一量化"等解耦方案获得更好的 TTFT、尾延迟和 SLO goodput？

### 2.1 四个核心假设

1. 真实 Agent workload 中存在非平凡的 exact-prefix 再访问，且离线 oracle 明显优于 LRU/简单启发式。
2. 未来复用价值与量化敏感度并非高度一致，单一分数会产生系统性错误分配。
3. 联合控制在扣除预测、迁移和量化开销后，优于最强的同后端解耦基线。
4. 质量损失能够被预先设定的非劣区间约束。

任一假设失败则按风险表收缩或转路线 B。

### 2.2 核心修正（与早期 idea 的差异）

- **缓存兼容性由确定性规则判断**（模型/version/tokenizer/template/adapter/父链哈希/lineage），不由工作流语义判断。
- **DAG 只预测未来访问**，不决定 KV 可复用性。
- **只管理 inactive prefix cache**，不修改 active decode KV。
- **复用价值与保真风险解耦**：高复用不等于必须高精度，低复用不等于可低精度近似。
- **主论文不含异构边云路由**，作为后续 TSC 扩展。

---

## 3. 方法蓝图

```text
Agent / Tool Runtime
        │ partial workflow state + canonical events
        ▼
Prefix-Stable Workflow Compiler
        │ token ids + parent block hashes + invalidation events
        ▼
Exact-Prefix Cache Index
        │ candidate inactive blocks
        ├──────────────► Reuse-Value Estimator R
        └──────────────► Fidelity-Risk Estimator D
                               │
                               ▼
                 Joint Residency Controller
                    │       │        │
                    ▼       ▼        ▼
                 GPU KV   CPU KV   Evicted
                 BF16/Q*  BF16/Q*  Recompute
```

`Q*`（量化存储）仅在 G4 量化门槛通过后启用。

---

## 4. Gate 与路线体系

### 4.1 三条路线

| 路线 | 定位 | 状态 |
|---|---|---|
| **A** | Exact-prefix reuse value + fidelity risk + joint precision/residency（推荐主线） | active |
| B | When Does Workflow Structure Create Physical KV Reuse?（保守回退） | fallback |
| C | Model-Scoped Shadow Frontiers（异构端云扩展） | deferred（TSC 扩展） |

### 4.2 关键路径

```
G0 ──► G1 ──► G3 ──► G4 ──► G2
```

- **G0** Exactness & Loadability：BF16 缓存恢复与重算一致，block identity/父链/lineage 无误。
- **G1** Opportunity：exact-prefix overlap、next-use distance、oracle headroom ≥ 10%。
- **G3** Lossless Residency：GPU BF16 / CPU BF16 / evict 控制器优于 size-aware LRU/GDSF。
- **G4** Quantization：后端支持目标模型 KV 量化与恢复，端到端不破坏延迟。
- **G2** Two-Axis Necessity：joint policy 在质量约束下胜过最强解耦组合。

G5（Learning）已删除，GNN 不再作为 gate，简单可解释 controller 是默认设计。

---

## 5. 当前进展

### 5.1 Gate 状态快照（2026-07-26）

| Gate | 状态 | 周次 | 关键证据 |
|---|---|---|---|
| **G0** | **passed** | W1–W2 | KV bit-identical 220/220，identity 100/100，codec 100 block lineage 100/100，8×4k=20.2GB < 24GB |
| G1 | not_started | W6–W7 | 待 Ch.1 画像数据复用判定 |
| G2 | not_started | W7–W8 | 待 Ch.2 Pilot + Ch.4 主表 |
| G3 | not_started | W8, W10–W11 | 待 W8 冒烟 + Ch.4 主表 |
| G4 | not_started | W9 | 待 Ch.3 fidelity 侧数据 |

### 5.2 已完成工作

**G0 阶段（已完成，2026-07-25 通过）**

- 冻结主模型 Qwen2.5-7B-Instruct（从原 Qwen3-8B 变更，因 Qwen3 在 modelscope 不可用）。
- 实现 block identity / compute lineage / 父链哈希 / fail-closed invalidation。
- 跑通 inactive Q-storage → active BF16 的编码/解码/staging/precision-lineage 隔离。
- KV bit-identical 验证 220/220 通过，identity 100/100，codec 100 block lineage 100/100。
- 8×4k 上下文显存峰值 20.2GB < 24GB，留出 KV pool 与安全水位。

**G1 基础设施（Phase 1 已完成，8 commits）**

- 统一 `compute_block_hash` 到 G0 8-tuple 版本（model_id / revision / template_hash / config_hash / adapter_id 等）。
- 扩展 `experiments/e1/config.yaml` 支持多数据集 + 8 seeds + resume。
- 新增 CLI 参数 `--seed` / `--dataset` / `--bfcl-subset` / `--max-episodes` / `--resume`。
- 实现 `TauBenchAdapter` 与 `BFCLAdapter` 适配器模式 dispatch。
- 实现 τ-bench 与 BFCL 的多 seed/多数据集录制循环，支持 checkpoint/resume。
- 修复 BFCL metadata propagation bug（必须通过 `_tokenize` closure 传播 G0 8-tuple）。

**质量与内存安全修复（commit `ebfee1e`）**

- 3 处 idea faithfulness 修正：BFCL 样本量 800 → 6400（8 decode seeds）；`_measure_prefill` 调整到 `generate` 之前以保证 miss_cost 计时准确；docstring Qwen3-8B → Qwen2.5-7B-Instruct。
- 6 处内存泄漏修复：移除 trace 文件中的 `global_block_index`（防止 O(n²) 磁盘占用 ~300GB）；BFCL `close_episode` 用 try/finally 包裹防 globals() 泄漏；`_generate_response` 与 `_measure_prefill` 中 `del + empty_cache` 释放 GPU tensor；episode 循环中加 `empty_cache`；模型加载设置 `max_memory` 75% 模型 / 25% KV。
- 新增 2 个回归测试验证 trace JSON 不含 `global_block_index`。

**测试状态**：43 passed / 3 skipped，无回归。

### 5.3 数据集体系（v0.3 精简后）

| Workload | 样本 | 角色 |
|---|---|---|
| τ-bench | 495（165 任务 × 3 seeds，实际录制 1320 episodes 对齐 pass^k） | 主表 workload 1 |
| BFCL v3 multi-turn | 6400（800 任务 × 8 decode seeds） | 主表 workload 2 |
| StableToolBench | 500 | Ch.5 family-out 证据 |
| LongBench | 1,000 | Ch.3 fidelity 质量面 |
| GSM8K | 100 | Ch.3 accuracy sanity |
| BurstGPT 窗口 | — | Ch.4 到达结构 replay 参数（不计样本量） |
| LMSYS-Chat-1M | 500 | Ch.1 负对照附注 |

**已删除**：SWE 轨迹、Toolathlon、MuSiQue、2WikiMultihopQA、合成 DAG（详见 `trim-dataset-portfolio/spec.md`）。

核心样本总量约 3,300，相对原方案 ~8,800 节省约 4.5 GPU 小时。

---

## 6. 代码结构

```
Prefix Caching/
├── IDEA.rewritten.md            # 项目 idea 主文档（v0.3 重写）
├── execution-plan.md            # 14 周执行计划与 Stage Gate 管理
├── ccfa.yaml                    # Gate/Experiment 状态追踪
├── PROJECT_OVERVIEW.md          # 本文件
│
├── experiments/
│   ├── experiment-designs.md    # 实验设计单一事实来源
│   ├── g2-pilot-design.md       # G2 Pilot 设计
│   │
│   ├── g0/                      # G0 exactness/loadability 测试
│   │   ├── backend.py           # 推理后端封装
│   │   ├── block_index.py       # block identity 与父链
│   │   ├── codec.py             # Q-storage 编解码
│   │   ├── codec_spike.py       # Q-storage spike 验证
│   │   ├── exactness_test.py    # KV bit-identical 测试
│   │   ├── freeze_record.py     # 模型/后端 revision 冻结
│   │   ├── memory_test.py       # 显存峰值测量
│   │   ├── structure_cases.py   # block 结构 case
│   │   ├── verdict.py           # G0 判定
│   │   ├── run_g0.py            # G0 主入口
│   │   └── outputs/             # G0 报告与结果
│   │
│   └── e1/                      # G1/Ch.1 录制与画像
│       ├── config.yaml          # 多数据集 + 8 seeds 配置
│       ├── trace_utils.py       # trace 工具与 block hash
│       ├── taubench_adapter.py  # τ-bench 适配器
│       ├── bfcl_adapter.py      # BFCL v3 适配器
│       ├── record_trajectories.py  # 多 seed/dataset 录制循环
│       ├── characterize_workload.py  # workload 画像
│       ├── compare_oracle.py    # oracle vs LRU/heuristic 对比
│       ├── plot_characterization.py  # 画像绘图
│       ├── tests/               # 7 个测试文件
│       ├── traces/              # 已录制 trace（部分 smoke test）
│       └── outputs/             # 画像报告与图
│
├── manuscript/
│   ├── main.tex
│   └── references.bib
│
├── reviews/
│   ├── prior-art-verification.md
│   └── revision-ledger.md
│
└── submission/
    └── checks.md
```

---

## 7. 下一步计划（W3–W7）

依据 `execution-plan.md` 与近期 topics：

### W3–W5：轨迹录制（关键路径）

- **目标**：录制 τ-bench 495 + BFCL v3 6400 的可重放 trace（实际 episode 总量约 7,720）。
- **环境**：AutoDL 云 GPU（RTX 4090D），模型路径 `/autodl-pub/models/Qwen2.5-7B-Instruct`。
- **预算**：约 50 GPU 小时。
- **关键检查**：trace JSON 不含 `global_block_index`；BFCL trace 的 G0 8-tuple metadata 完整传播；`_measure_prefill` 在 `generate` 之前执行以保证 miss_cost 计时。
- **失败动作**：trace 不可重放则阻塞 G1，间接触发 G1 失败动作（转路线 B）。

### W6–W7：Ch.1 画像 + G1 判定

- 复用 W3–W5 录制的 trace 计算：exact-prefix overlap、LCP tokens、next-use distance、working-set size、KV/总显存占比。
- 比较 LRU / GDSF / 同引擎 APC / 离线 oracle。
- 至少保证 PBKV 或 KVFlow 中一个 closest baseline 可公平运行。
- **G1 通过条件**：oracle 相对最佳简单策略存在约 10% 的 miss-cost 或 p95 TTFT 改进空间。

### W7–W8：Ch.2 Pilot + Ch.3 reuse 侧（并行）

- Ch.2 R–D 错位 Pilot：τ-bench 80 子集，计算 Spearman ρ + 四象限分析。
- Ch.3 reuse 侧：heuristic + survival/hazard estimator（GNN 已删除）。
- G2 存在性判定（R–D 错位），最终判定延后到 Ch.4 主表。

### W8：G3 冒烟

- 主 cell × 4 个无损对照（No-Cache / APC-LRU / GDSF / Reuse-Only）× 约 100 workflow 子集。
- 防止无损驻留不成立时白做量化。

---

## 8. 工程约定

- **block_size** = 16；KV 预算档 10/25/50/100%；β=0.005/step；H=1000 step。
- **统计单位** = workflow（paired bootstrap 95% CI，1000 次）。
- **open-loop 与 closed-loop 结果绝不混表**。
- **真实数据集禁令**：禁止自建/合成数据集；数据集选择须对齐同类论文实践。
- **预注册字段**（ε/δ/N、SLO 阈值、净收益阈值）在对应 pilot 后冻结并回写文档。
- **commit 规范**：遵循 `.trae/rules/git-commit-message.md`。
- **GPU 资源**：`_generate_response` 与 `_measure_prefill` 必须 `del + empty_cache`；BFCL `close_episode` 必须 try/finally。

---

## 9. 关键文档索引

| 文档 | 用途 |
|---|---|
| [IDEA.rewritten.md](file:///d:/00MyProject/Prefix%20Caching/IDEA.rewritten.md) | 项目 idea 主文档（v0.3） |
| [execution-plan.md](file:///d:/00MyProject/Prefix%20Caching/execution-plan.md) | 14 周执行计划与 Stage Gate |
| [ccfa.yaml](file:///d:/00MyProject/Prefix%20Caching/ccfa.yaml) | Gate/Experiment 状态追踪 |
| [experiments/experiment-designs.md](file:///d:/00MyProject/Prefix%20Caching/experiments/experiment-designs.md) | 实验设计单一事实来源 |
| [experiments/g2-pilot-design.md](file:///d:/00MyProject/Prefix%20Caching/experiments/g2-pilot-design.md) | G2 Pilot 设计 |
| [.trae/specs/experiment-scope-redesign/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/experiment-scope-redesign/spec.md) | v0.3 精简方案 spec |
| [.trae/specs/trim-dataset-portfolio/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/trim-dataset-portfolio/spec.md) | 数据集精简 spec |
| [reviews/prior-art-verification.md](file:///d:/00MyProject/Prefix%20Caching/reviews/prior-art-verification.md) | 公开工作边界核验 |

---

## 10. 写作就绪条件

进入论文写作前必须满足（IDEA Section 14）：

1. G0–G3 通过；
2. G1 证明真实决策空间存在；
3. 最强 close baseline 已能公平运行或清楚解释不兼容；
4. G2 和 G4 均通过，joint policy 在质量约束下胜过最强解耦组合；
5. 主 claim 有一张相同后端的 Pareto 主图和 workflow-level 置信区间；
6. 所有贡献均对应真实结果，不使用预期数字填充结论。

**当前状态**：条件 1 部分满足（G0 通过，G1/G3 待判定）；条件 2–6 均未启动。
