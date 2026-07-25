# G0 Verdict Report

G0: Exactness & Loadability 判定报告。

## 判定条件汇总

| # | 条件 | 状态 | 证据 |
|---|------|------|------|
| 1 | BF16 缓存恢复与重算一致 | ✓ PASS | KV bit-identical: 220/220, logits diff ≤ 1e-3: 220/220, top-1 match: 220/220 |
| 2 | block identity/父链/invalidation 无错误 | ✓ PASS | identity: 100/100 (非 cat5: 90/90, cat5: 10/10), parent chain: 100/100, invalidation: 20/20 |
| 3 | freeze-record 完整 | ✓ PASS | 校验通过: True |
| 4 | codec/staging/lineage spike 跑通 | ✓ PASS | 测试 block 数: 100, Q8 MSE: 1.60e-02, Q4 MSE: 8.92e-01, lineage 隔离: True |
| 5 | 后端能拦截/恢复 KV | ✓ PASS | slice + restore 成功执行: True, KV bit-identical: 220/220 |
| 6 | 显存可承载 | ✓ PASS | 最大 reserved 峰值: 20.238 GB, 含安全水位: 22.261 GB, 上限: 24 GB, 判定: PASS |

## 总体判定

**G0 = PASSED** ✅

所有 6 个判定条件均通过，G0 实验完成，可进入后续实验。

## 汇总

- 通过条件数: 6/6
- 总体判定: PASSED

## 正向发现

G0 在验证后端正确性的同时，产出了两项对 prefix caching 研究有直接价值的发现：

### 发现 1：Tokenizer 非前缀稳定现象（cat5 实证）

- **现象**：Qwen2.5 BPE tokenizer 在 chat-template 边界（`\n` 与 `\nI`、`<|im_start|>assistant\n` 与紧接其后的回复首字符）产生跨边界合并 token。
- **影响**：纯追加多轮会话用 apply_chat_template 重新渲染时，token id 序列不以前缀 N 为严格前缀，block hash 从追加边界起分叉。
- **研究意义**：朴素按 token id 前缀匹配会丢失复用机会，为 IDEA 中 C2 联合控制器的 boundary-aware 决策、C3 "reuse value 与 fidelity 风险错位" 主张提供实证依据。
- **处置**：写入 cat5 用例的 expected_incremental_sharing=False，实测 False 则 identity_check PASS，作为 G0 正向输出。

### 发现 2：Q4 per-tensor 量化损失过大

- **数据**：Q8 MSE=1.6e-2, KL mean=0.97；Q4 MSE=0.89, KL mean=5.2, KL max=19.2，max_abs_err 高达 29.6。
- **结论**：Q4 per-tensor 量化对 Qwen2.5-7B 的 KV cache 不可用，后续 G2+ 显存压缩方案应只用 Q8 或改用 per-channel/per-group 量化。
