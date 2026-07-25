# G0 Verdict Report

G0: Exactness & Loadability 判定报告。

## 判定条件汇总

| # | 条件 | 状态 | 证据 |
|---|------|------|------|
| 1 | BF16 缓存恢复与重算一致 | ✗ FAIL | exactness-results.json 中无数值结果 |
| 2 | block identity/父链/invalidation 无错误 | ✗ FAIL | exactness-results.json 中无 identity 结果 |
| 3 | freeze-record 完整 | ✗ FAIL | freeze-record.json 不存在或为空 |
| 4 | codec/staging/lineage spike 跑通 | ✗ FAIL | codec-results.json 不存在或为空 |
| 5 | 后端能拦截/恢复 KV | ✗ FAIL | 无 exactness 结果，无法验证后端拦截/恢复 |
| 6 | 显存可承载 | ✗ FAIL | memory-results.json 不存在或为空 |

## 总体判定

**G0 = FAILED** ❌

存在未通过的判定条件，需根据以下失败动作进行修复：

### 条件 1: BF16 缓存恢复与重算一致

- 证据: exactness-results.json 中无数值结果
- 失败动作: 检查 backend.slice_kv_into_blocks / restore_kv_from_blocks 的张量拷贝逻辑，确认 clone() 调用正确；检查 forward_with_kv 是否正确传递 past_key_values。

### 条件 2: block identity/父链/invalidation 无错误

- 证据: exactness-results.json 中无 identity 结果
- 失败动作: 检查 compute_block_hash 的元数据字段（model_id/revision/template_hash/config_hash/adapter_id）是否正确传入；检查 verify_parent_chain 的父链连续性逻辑；检查 check_invalidation 的 change_point 设置。

### 条件 3: freeze-record 完整

- 证据: freeze-record.json 不存在或为空
- 失败动作: 检查 freeze_record.py 的 generate_freeze_record 是否收集了全部必填字段；确认模型已正确加载且 get_model_info() 返回完整信息。

### 条件 4: codec/staging/lineage spike 跑通

- 证据: codec-results.json 不存在或为空
- 失败动作: 检查 codec.py 的 Q8/Q4 编解码逻辑；确认 codec_spike.py 能正确收集 block 并执行 roundtrip；检查 lineage 隔离的 adapter_id 设置。

### 条件 5: 后端能拦截/恢复 KV

- 证据: 无 exactness 结果，无法验证后端拦截/恢复
- 失败动作: 检查 Backend 类的 KV cache 拦截能力（slice_kv_into_blocks / restore_kv_from_blocks）；确认 DynamicCache 格式兼容性；检查张量 clone 是否到位。

### 条件 6: 显存可承载

- 证据: memory-results.json 不存在或为空
- 失败动作: 减小并发数或上下文长度；启用 Q8/Q4 量化减少 KV cache 显存占用；检查 GPU 显存是否足够（需 ≥ 24GB）。


## 汇总

- 通过条件数: 0/6
- 总体判定: FAILED
