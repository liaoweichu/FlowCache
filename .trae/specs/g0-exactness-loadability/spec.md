# G0: Exactness & Loadability Code Spec

## Why

G0 是所有后续实验的正确性基础（W1–W2）。需要可执行代码验证：模型能加载、KV cache 可拦截/恢复、BF16 缓存恢复与重算一致、block identity 正确、Q8/Q4 codec 跑通、显存可承载。用户将在云端 GPU 上执行。

## What Changes

- 新建 `experiments/g0/` 目录，包含完整 G0 实验代码
- 后端选用 HuggingFace `transformers`（直接访问 `past_key_values`，满足 KV 拦截/恢复需求）
- 复用 `experiments/e1/trace_utils.py` 的 `compute_block_hash` / `compute_parent_chain`
- 产出 6 个 artifact 文件（freeze-record.json, real-structure-cases.json, exactness-report.md, codec-spike-report.md, memory-report.md, g0-verdict.md）

## Impact

- Affected code: 新建 `experiments/g0/` 目录（~10 个 Python 文件 + 1 个 config.yaml）
- 不影响：e1/ 已有代码、experiment-designs.md 设计文档
- 依赖：transformers>=4.46, torch, datasets (HuggingFace), τ-bench (pip install tau-bench)

## ADDED Requirements

### Requirement: G0 实验代码可执行

系统 SHALL 提供一个入口脚本 `experiments/g0/run_g0.py`，通过 `python run_g0.py --step all` 执行全部 G0 步骤，或 `--step <N>` 执行单步。

#### Scenario: 云端 GPU 执行
- **WHEN** 用户在云端 GPU 实例上执行 `python experiments/g0/run_g0.py --step all`
- **THEN** 系统依次执行 step 0-7，产出全部 artifact 到 `experiments/g0/outputs/`，并生成 `g0-verdict.md` 判定报告

### Requirement: Step 0 - 后端冻结与 freeze-record.json

系统 SHALL 加载 Qwen2.5-7B-Instruct，记录模型 revision、tokenizer、chat template、transformers 版本、CUDA 版本、GPU 型号到 `freeze-record.json`。

### Requirement: Step 1 - 显存峰值测量

系统 SHALL 测量以下配置的 allocated/reserved 峰值（各 5 次重复取 max）：
- 仅加载模型
- 并发 4 × 4K context
- 并发 8 × 8K context

判定：权重 + active cache + staging + 安全水位 ≤ 24GB。

### Requirement: Step 2 - 真实结构用例集

系统 SHALL 从 τ-bench（HuggingFace `sihu/tau-bench` 或 pip `tau-bench`）生成 6 类结构用例：
- ① 同域任务对（共享 system prompt）：retail 15 对 + airline 15 对
- ② 分支历史：同一任务不同 user seed 的轨迹对，共享前缀
- ③ chat template 变化：同一会话用 2 种 template 渲染
- ④ 模型标识变化：元数据字段受控变换
- ⑤ 纯追加长会话：多轮对话逐轮递增
- ⑥ 无共享对照：跨域任务对

产出 `real-structure-cases.json`，含每个用例的类别、来源 ID、共享结构先验标注。

### Requirement: Step 3 - block identity 模块

系统 SHALL 实现 block identity 哈希 `I_b = (m, r, tau, c, a, h_parent, tokenIds, positions)`：
- 复用 e1/trace_utils.py 的 `compute_block_hash`（扩展加入 model_id/revision 字段）
- 实现父链连续性校验
- 实现 invalidation 逻辑（template/标识变化时 block 应失效）

### Requirement: Step 4 - BF16 exactness 测试

系统 SHALL 对 6 类结构用例的全部 block 验证：
- KV 张量逐元素 bit-identical（缓存恢复路径 vs 重算路径）
- logits max abs diff ≤ 1e-3
- greedy decode top-1 token 一致率 100%

技术路径：用 `model()` forward 获取 `past_key_values`（重算路径），将其切片为 block 后恢复（缓存路径），对比两条路径的 KV 张量和 logits。

### Requirement: Step 5 - Q8/Q4 codec spike

系统 SHALL 实现 KV cache 量化 codec：
- Q8：per-tensor int8 量化（scale + int8 tensor）
- Q4：per-tensor int4 量化（scale + int4 tensor）
- 对 100 个 block 做 roundtrip（BF16 → Q8/Q4 → BF16），记录 MSE、max abs err、logit KL、编解码延迟
- 验证 approximate lineage 隔离：量化祖先的 child 不与 canonical BF16 lineage 别名

### Requirement: Step 6 - 判定报告

系统 SHALL 汇总 6 个判定条件的通过/失败状态到 `g0-verdict.md`：
1. BF16 缓存恢复与重算一致
2. block identity/父链/invalidation 无错误
3. freeze-record 完整
4. codec/staging/lineage spike 跑通
5. 后端能拦截/恢复 KV
6. 显存可承载

任一失败则标记 G0 = FAILED，提示失败动作。
