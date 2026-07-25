# G0 Verdict Report

G0: Exactness & Loadability 判定报告。

## 判定条件汇总

| # | 条件 | 状态 | 证据 |
|---|------|------|------|
| 1 | BF16 缓存恢复与重算一致 | ✓ PASS | KV bit-identical: 220/220, logits diff ≤ 1e-3: 220/220, top-1 match: 220/220 |
| 2 | block identity/父链/invalidation 无错误 | ✗ FAIL | identity: 90/100, parent chain: 100/100, invalidation: 20/20 |
| 3 | freeze-record 完整 | ✓ PASS | 校验通过: True |
| 4 | codec/staging/lineage spike 跑通 | ✓ PASS | 测试 block 数: 100, Q8 MSE: 1.60e-02, Q4 MSE: 8.92e-01, lineage 隔离: True |
| 5 | 后端能拦截/恢复 KV | ✓ PASS | slice + restore 成功执行: True, KV bit-identical: 220/220 |
| 6 | 显存可承载 | ✗ FAIL | 最大 reserved 峰值: 18.979 GB, 含安全水位: 20.877 GB, 上限: 24 GB, 判定: FAIL |

## 总体判定

**G0 = FAILED** ❌

存在未通过的判定条件，需根据以下失败动作进行修复：

### 条件 2: block identity/父链/invalidation 无错误

- 证据: identity: 90/100, parent chain: 100/100, invalidation: 20/20
- 失败动作: 检查 compute_block_hash 的元数据字段（model_id/revision/template_hash/config_hash/adapter_id）是否正确传入；检查 verify_parent_chain 的父链连续性逻辑；检查 check_invalidation 的 change_point 设置。

### 条件 6: 显存可承载

- 证据: 最大 reserved 峰值: 18.979 GB, 含安全水位: 20.877 GB, 上限: 24 GB, 判定: FAIL
- 失败动作: 减小并发数或上下文长度；启用 Q8/Q4 量化减少 KV cache 显存占用；检查 GPU 显存是否足够（需 ≥ 24GB）。


## 汇总

- 通过条件数: 4/6
- 总体判定: FAILED
