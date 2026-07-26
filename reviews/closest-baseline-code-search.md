# Closest Baseline 开源代码调研报告

**调研日期**：2026-07-26
**调研范围**：FlowCache IDEA.rewritten.md §3.1 表中 12 项尚未调研代码可用性的 closest baseline 候选论文
**调研目的**：筛选出纯 Python、可在 Windows + RTX 4090D 直接运行、可适配 τ-bench trace 格式的开源实现，作为 G1 实验的补充 closest baseline
**环境约束**：原生 Windows + RTX 4090D 24GB + 纯 Python（无 Rust/CUDA 编译）

---

## 0.1 Linux 可行性重评（2026-07-26 更新）

**触发条件**：用户指出云端实验环境为 **AutoDL Linux 平台**（有 GPU + root + CUDA toolkit），可编译 Rust/CUDA/fork 后端。原调研以"原生 Windows + 纯 Python"为筛选硬约束，导致多个强候选被 EXCLUDE 或降权。本节重新评估这些候选在 Linux 下的可行性。

### 重评结论

| Baseline | 原状态 | Linux 可行性 | venue | 与 G1 相关性 | 重评决策 |
|---|---|---|---|---|---|
| **KVFlow** | deferred（WSL2/CUDA/Rust） | **YES**（AutoDL Linux 可编译） | NeurIPS 2025 | 高（agent workflow KV） | **新增 faithful reproduction**（`enabled: true`） |
| InferCept | EXCLUDE（Windows=0） | YES 但风险高（Issue #2 复现困难 + 1 年未更新） | ICML 2024 | 中（interruption-aware） | 不新增（复现风险高于收益） |
| Continuum | EXCLUDE（Windows=0） | YES 但 preview version（9 commits，不含论文估计逻辑） | arXiv 2511.02230 | 中（TTL+reload） | 不新增（无法忠实复现论文逻辑） |
| Helium | BACKUP（Windows=1） | YES（vLLM 依赖） | SIGMOD 2026 | 中（workflow-as-query-plan） | 不新增（问题域重叠有限） |
| LPC | BACKUP（Windows=1） | YES（vLLM+CUDA） | NeurIPS 2025 | 低（非 workflow-aware） | 不新增（与 FlowCache 距离较远） |
| Agent Memory | EXCLUDE（Apple Silicon） | NO（硬依赖 MLX） | arXiv 2603.04428 | 低 | 仍 EXCLUDE |

### 决策依据

1. **只新增 KVFlow faithful**：G1 第二项条件（G1.8）只需 ≥1 个 faithful closest baseline。KVFlow 是最佳候选——venue 最强（NeurIPS 2025）、官方代码完整（含魔改 SGLang + SScheduler）、ASG 抽象与 τ-bench `block_hash`/`parent_hash` DAG 天然契合（G1.4.1 已判定 "faithful，待 adapter 实现"）。
2. **不新增 InferCept**：虽 ICML 2024 venue 强，但 GitHub Issue #2 显示已有用户因 torch/vllm 版本问题无法复现，且仓库 1 年+ 未更新，复现风险高于收益。
3. **不新增 Continuum**：仅 9 commits 的 "preview version"，README 明确说不含论文估计逻辑，无法忠实复现。
4. **不新增 Helium/LPC**：Helium 的 workflow-as-query-plan 与 FlowCache exact-prefix block 问题域重叠有限；LPC 非 workflow-aware，与 FlowCache 距离较远。
5. **不新增 CacheWise**：venue 不明（arXiv preprint，无正式 proceedings），coding agent traces 不直接适用 τ-bench，留作后续备选。

### KVFlow faithful reproduction 状态

- **config.yaml**：`kvflow_faithful` 从 `enabled: false / deferred` 升级为 `enabled: true / active`
- **G1.4.1**：判定从 "faithful（待 adapter 实现）" 改为 "faithful（AutoDL Linux adapter 实现中）"
- **adapter 工程**：clone KVFlow repo → 编译魔改 SGLang + Rust + CUDA → 写 τ-bench adapter（block_assignments → SGLang prefix-tree + PlanManager.update_agent_timestep）→ 运行 faithful baseline。这是后续独立任务，不在本文档范围内。

---

## 1. 调研方法

### 1.1 纳入标准
- 来源：IDEA.rewritten.md §3.1 "最接近的公开工作" 表
- 排除已调研项：vLLM APC（已实现为 APCLRUCache）、KVFlow（需 WSL2+Rust/CUDA，已排除）、PBKV（无官方代码，已做 inspired variant）

### 1.2 调研步骤（每篇论文）
1. WebSearch 论文标题 + "github code" / "github repository"
2. WebSearch 第一作者 + 关键词 + "github"
3. WebFetch arXiv abs 页面查找 "Code" 链接
4. WebFetch OpenReview / PMLR / ICML virtual 页面查找 code release
5. 若找到 repo，WebFetch README 提取：语言、依赖、引擎、输入格式、License、活跃度、Windows 兼容性

### 1.3 评分维度（每维 0-3 分，总分 0-12）
| 维度 | 0 分 | 1 分 | 2 分 | 3 分 |
|---|---|---|---|---|
| Windows 可行性 | 需 WSL2+Rust/CUDA | 需 WSL2 但纯 Python | 原生 Windows + 少量适配 | 原生 Windows + pip install |
| τ-bench 兼容性 | 需重写输入层 | 需大幅适配 | 需中等适配 | 需轻量 adapter |
| 与 FlowCache 接近度 | 不同问题域 | 部分相关 | 工作流感知 KV 管理 | closest baseline（§3.1 表）|
| 代码质量 | 无文档/无法运行 | 有文档但难运行 | 可运行但需调试 | 文档完善可直接运行 |

### 1.4 筛选标准
- **PRIORITY_IMPLEMENT**：总分 ≥ 9 且 Windows 可行性 ≥ 2
- **BACKUP**：总分 6-8
- **EXCLUDE**：总分 < 6 或 Windows 可行性 = 0

---

## 2. 调研结果总览表

| # | 工作 | Repo URL | 语言 | Windows | τ-bench | FlowCache | 质量 | 总分 | 推荐 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **CacheWise** | [cachewise-project/cachewise-coding-traces](https://github.com/cachewise-project/cachewise-coding-traces) + [vllm fork](https://github.com/cachewise-project/vllm) | Python 100% (traces) / vLLM fork | 2 | 3 | 3 | 2 | **10** | **PRIORITY_IMPLEMENT** |
| 2 | Learned Prefix Caching | [yangdsh/LPC](https://github.com/yangdsh/LPC) | Python 85% + CUDA 10% | 1 | 1 | 2 | 2 | 6 | BACKUP |
| 3 | InferCept | [WukLab/InferCept](https://github.com/WukLab/InferCept) | Py 40% + CUDA 6% + NB 44% | 0 | 1 | 3 | 1 | 5 | EXCLUDE |
| 4 | Continuum | [Hanchenli/vllm-continuum](https://github.com/Hanchenli/vllm-continuum) | Py 87% + CUDA 8% | 0 | 1 | 3 | 2 | 6 | EXCLUDE (Windows=0) |
| 5 | **ThunderAgent** | [ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) | Python 100% | 3 | 2 | 3 | 3 | **11** | **PRIORITY_IMPLEMENT** |
| 6 | TokenCake | NOT FOUND | N/A | 0 | 0 | 2 | 0 | 2 | EXCLUDE |
| 7 | Helium | [mlsys-io/helium_demo](https://github.com/mlsys-io/helium_demo) | Python 99.6% | 1 | 1 | 3 | 2 | 7 | BACKUP |
| 8 | ARKV | [LSC/ARKV](https://github.com/Large-scale-Sustainable-Computing-LSC/ARKV) | Jupyter 90% + Python 10% | 2 | 1 | 1 | 2 | 6 | BACKUP |
| 9 | QKVShare | NOT FOUND | N/A | 0 | 0 | 2 | 0 | 2 | EXCLUDE |
| 10 | GraphFlow | NOT FOUND | N/A | 0 | 0 | 3 | 0 | 3 | EXCLUDE |
| 11 | Agent Memory | [yshk-mxim/agent-memory](https://github.com/yshk-mxim/agent-memory) | Python 99% (MLX) | 0 | 2 | 2 | 3 | 7 | EXCLUDE (Apple Silicon only) |
| 12 | HybridFlow | [WanyuGroup/ICML2026_HybridFlow](https://github.com/WanyuGroup/ICML2026_HybridFlow) | Python 99% | 3 | 1 | 1 | 3 | 8 | BACKUP |

**统计**：
- 代码可用：10/12（83%）
- PRIORITY_IMPLEMENT：2 项（ThunderAgent、CacheWise）
- BACKUP：4 项
- EXCLUDE：6 项（无代码 3 项，Windows 不可行 3 项）

---

## 3. PRIORITY_IMPLEMENT 候选详细分析

### 3.1 ThunderAgent（总分 11，首选实现）

- **arXiv/Venue**：arXiv:2602.13692 / **ICML 2026 Spotlight**
- **Repo URL**：https://github.com/ThunderAgent-org/ThunderAgent
- **第一作者**：Hao Kang（Georgia Tech）
- **语言**：Python 100%
- **构建依赖**：`pip install -e .`，纯 Python，无 Rust/CUDA 编译。核心依赖：fastapi、httpx、uvicorn
- **推理引擎**：ThunderAgent 自身是 CPU-bound 代理（无需 GPU），后端对接 vLLM / SGLang / SkyRL。README 明确说 "ThunderAgent itself does not require a GPU"
- **输入格式**：OpenAI 兼容 API 调用，唯一改动是在 `extra_body` 中加 `program_id`。CLI 启动：`thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000`
- **License**：MIT
- **Last commit**：2026-06-06（144 commits，活跃度高；已集成进 NVIDIA Dynamo 2.0 和 SkyRL）
- **核心 idea**：
  - Program-aware pause/restore：以 `program_id` 标识的工作流为单位管理 KV 生命周期
  - Time decay：`--use-acting-token-decay` 实现 2^{-t} 衰减，暂停时间越长优先级越低
  - 资源生命周期管理：`--router tr` 程序感知容量调度
- **评分**：Windows=3, τ-bench=2, FlowCache=3, 质量=3 → 总分 **11**
- **推荐理由**：
  1. 纯 Python 100%，pip install 无编译，原生 Windows 可跑
  2. OpenAI 兼容 API + `program_id` 设计，τ-bench trace 适配成本低
  3. ICML 2026 Spotlight，已集成进 NVIDIA Dynamo 2.0，工业验证充分
  4. program-aware + time decay 是现有 baseline 未覆盖的维度
- **实现路径**：实现 ThunderAgent-inspired 块级缓存类，提取 time decay (2^{-t}) + program-aware 优先级，适配到 `compare_oracle.py` 的 block-level 接口

### 3.2 CacheWise（总分 10，备选实现）

- **arXiv/Venue**：arXiv:2606.16824（2026-06-15 提交）
- **Repo URL**：
  - 跟踪数据集：https://github.com/cachewise-project/cachewise-coding-traces（纯 Python）
  - 算法实现：https://github.com/cachewise-project/vllm（vLLM fork）
- **语言**：跟踪数据集 100% Python；算法仓库继承 vLLM（Python + C++ + CUDA）
- **构建依赖**：跟踪数据集 `pip install -r requirements.txt`（极简）；算法仓库需 vLLM 环境（可用预编译 wheel）
- **推理引擎**：跟踪分析为独立脚本；算法基于 vLLM fork
- **输入格式**：JSON 跟踪格式，含真实 Claude Code 用户的 sanitized traces（LLM 调用 + 工具调用序列）
- **License**：Apache-2.0
- **Last commit**：2026-06-14
- **核心 idea**：
  - 工具元数据预测 reuse order
  - Prefix-aware scheduling 与 reuse-aware eviction
- **评分**：Windows=2, τ-bench=3, FlowCache=3, 质量=2 → 总分 **10**
- **推荐理由**：
  1. 真实 coding agent 跟踪数据集（Claude Code traces）公开可得
  2. JSON 跟踪格式与 τ-bench 工具调用序列高度相似，轻量适配即可
  3. 直接面向 coding agent 工作流的 KVCache 管理，是 §3.1 表中最接近的基线
- **实现路径**：跟踪数据集可在 Windows 原生运行；算法需从 vLLM fork 提取 eviction 逻辑

---

## 4. BACKUP 候选简要分析

### 4.1 HybridFlow（总分 8）
- Repo: https://github.com/WanyuGroup/ICML2026_HybridFlow（ICML 2026, MIT, Python 99%）
- Windows=3, τ-bench=1, FlowCache=1, 质量=3
- 核心问题：edge-cloud 路由（0-1 背包 + Lagrangian），非 KV cache 管理
- 排除原因：与 FlowCache 问题域重叠有限

### 4.2 Helium（总分 7）
- Repo: https://github.com/mlsys-io/helium_demo（SIGMOD 2026, Apache-2.0, Python 99.6%）
- Windows=1, τ-bench=1, FlowCache=3, 质量=2
- 核心问题：workflow-as-query-plan + proactive caching
- 排除原因：vLLM 依赖在 Windows 上需 WSL2（Windows=1 未达 ≥2 要求）

### 4.3 Agent Memory（总分 7）
- Repo: https://github.com/yshk-mxim/agent-memory（MIT, Python 99%）
- Windows=0, τ-bench=2, FlowCache=2, 质量=3
- 核心问题：边缘多 agent 持久化 Q4 KV cache
- 排除原因：**硬依赖 Apple MLX 框架**，仅支持 Apple Silicon（M1/M2/M3/M4），不支持 Windows + NVIDIA GPU

### 4.4 Learned Prefix Caching（总分 6）
- Repo: https://github.com/yangdsh/LPC（NeurIPS 2025, License 未指定, Python 85% + CUDA 10%）
- Windows=1, τ-bench=1, FlowCache=2, 质量=2
- 核心问题：从对话内容预测 continuation probability 指导前缀驱逐
- 排除原因：预编译 vLLM wheel 仅 Linux，Windows 需 WSL2

### 4.5 ARKV（总分 6）
- Repo: https://github.com/Large-scale-Sustainable-Computing-LSC/ARKV（CCGRID 2025, MIT, Jupyter 90% + Python 10%）
- Windows=2, τ-bench=1, FlowCache=1, 质量=2
- 核心问题：attention statistics 驱动 Original/Quantized/Evicted 三态（单模型长上下文）
- 排除原因：非 workflow-aware，与 FlowCache 距离较远

---

## 5. EXCLUDE 候选简要分析

| 工作 | 排除原因 | Linux 重评 |
|---|---|---|
| InferCept (ICML 2024) | Windows=0：必须 nvcc 编译 CUDA 内核；GitHub Issue #2 显示已有用户因 torch/vllm 版本问题无法复现；已 1 年+ 未更新 | Linux 下可编译，但复现风险高（Issue #2 + 1 年未更新），不新增 |
| Continuum (arXiv 2511.02230) | Windows=0：vLLM fork 需 CUDA 编译；仅 9 commits 的 "preview version"（不含论文估计逻辑） | Linux 下可编译，但 preview version 不含论文估计逻辑，无法忠实复现，不新增 |
| TokenCake (arXiv 2510.18586) | 无代码：4 次搜索均未找到，论文全文无代码链接，作者 GitHub 亦无相关仓库 | N/A（无代码） |
| QKVShare (arXiv 2605.03884) | 无代码：论文提及 "current repository" 但未给 URL；6 次搜索未找到 | N/A（无代码） |
| GraphFlow (ICML 2026) | 无代码：7 次搜索未找到；ICML virtual poster 页面无代码链接；GitHub 上同名仓库均不相关 | N/A（无代码） |
| Agent Memory (arXiv 2603.04428) | Windows=0：硬依赖 Apple MLX 框架，仅支持 Apple Silicon，不支持 Windows + NVIDIA GPU | Linux 下也不可行（硬依赖 MLX），仍 EXCLUDE |

---

## 6. 实现建议

### 6.1 首选实现：ThunderAgent-inspired

**理由**：
1. 总分最高（11），Windows 可行性满分（3），纯 Python 100%
2. ICML 2026 Spotlight + NVIDIA Dynamo 2.0 集成，venue 与工业验证最强
3. Time decay (2^{-t}) + program-aware 调度是现有 7 个 baseline 未覆盖的维度
4. 与 PBKV-inspired（reuse prediction）互补，共同构成 closest baseline 对照

**实现方式**：Inspired variant（ThunderAgent 是 API 代理非块级缓存策略，需提取核心 idea 适配到 block-level 接口）

**核心 idea 提取**：
- **Time decay**：块的优先级随暂停时间 t 衰减：priority ∝ 2^{-t}
- **Program-aware**：同属一个 workflow（program_id）的块作为一组管理，整组暂停/恢复
- **Capacity scheduling**：跨 workflow 按容量优先级分配缓存配额

### 6.2 次选实现：KVFlow faithful reproduction（AutoDL Linux，2026-07-26 升级）

**理由**：
1. NeurIPS 2025，venue 最强（唯一 NeurIPS/ICML/SIGMOD 级别且有官方代码的 closest baseline）
2. 官方代码完整：魔改 SGLang + SScheduler PFEngine + ASG（Agent Step Graph）抽象
3. ASG 与 τ-bench `block_hash`/`parent_hash` DAG 天然契合，G1.4.1 已判定 "faithful，待 adapter 实现"
4. AutoDL Linux 平台有 GPU + root + CUDA，可编译 Rust/CUDA/fork 后端，原 Windows 约束不再适用
5. 是 G1 第二项条件（"≥1 个 closest baseline 忠实运行"）的最强证据——比 inspired variant 更有说服力

**实现方式**：Faithful reproduction（官方代码 + τ-bench adapter）

**adapter 工程步骤**（后续独立任务）：
1. 在 AutoDL Linux 上 clone https://github.com/PanZaifeng/KVFlow
2. 编译魔改 SGLang + Rust + CUDA（需 ~20GB 磁盘 + 1-2 小时编译时间）
3. 写 τ-bench adapter：将 `block_assignments` 翻译为 SGLang prefix-tree 请求 + `PlanManager.update_agent_timestep(...)` 调用，构造 ASG
4. 用 7B 模型（Qwen2.5-7B / Llama-3.1-8B）在 τ-bench trace 上运行 faithful baseline
5. 对比 KVFlow faithful vs PBKV-inspired vs ThunderAgent-inspired vs 简单启发式

**状态**：`config.yaml` 中 `kvflow_faithful.enabled` 已从 `false` 升级为 `true`；adapter 工程待启动。

### 6.3 备选实现：CacheWise-inspired（若 ThunderAgent/KVFlow 效果不佳）

**理由**：
1. 总分 10，τ-bench 兼容性满分（3），跟踪格式与 τ-bench 高度相似
2. 真实 coding agent traces 公开可得，可作为补充数据集
3. Reuse-order prediction 与 FlowCache 的 reuse-value estimation 直接对齐

**实现方式**：Inspired variant（算法在 vLLM fork 中，需提取 eviction 逻辑）

---

## 7. 结论

在 §3.1 表 12 项 closest baseline 候选中：
- **2 项 PRIORITY_IMPLEMENT**：ThunderAgent（11 分）、CacheWise（10 分）
- **4 项 BACKUP**：HybridFlow（8）、Helium（7）、Agent Memory（7）、LPC/ARKV（6）
- **6 项 EXCLUDE**：无代码 3 项、Windows 不可行 3 项

**推荐实现 ThunderAgent-inspired** 作为 G1 实验的补充 closest baseline，与现有 PBKV-inspired 共同满足 G1 第 2 项条件（closest baseline 可比性）。ThunderAgent 的 time decay + program-aware 调度提供了与 PBKV（reuse prediction）不同的 closest baseline 视角，增强 G1 对照的说服力。
