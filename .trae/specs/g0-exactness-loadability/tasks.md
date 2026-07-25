# Tasks

- [ ] Task 1: 创建 experiments/g0/ 目录结构和 config.yaml
  - [ ] SubTask 1.1: 创建 config.yaml（模型名、dtype、block_size=16、显存配置、输出路径）
  - [ ] SubTask 1.2: 创建 outputs/ 子目录

- [ ] Task 2: 实现 backend.py - 模型加载与 KV cache 拦截
  - [ ] SubTask 2.1: 加载 Qwen2.5-7B-Instruct（BF16, device_map=auto）
  - [ ] SubTask 2.2: 实现 `forward_with_kv()` - 执行 forward pass，返回 logits + past_key_values
  - [ ] SubTask 2.3: 实现 `slice_kv_into_blocks()` - 将 past_key_values 按 block_size 切片为 block 列表
  - [ ] SubTask 2.4: 实现 `restore_kv_from_blocks()` - 将 block 列表重组为 past_key_values
  - [ ] SubTask 2.5: 实现 `get_model_info()` - 获取模型 revision、tokenizer info、config

- [ ] Task 3: 实现 block_index.py - block identity 与父链
  - [ ] SubTask 3.1: 扩展 compute_block_hash 加入 model_id/revision 字段（I_b 完整定义）
  - [ ] SubTask 3.2: 实现 verify_parent_chain() - 校验父链连续性
  - [ ] SubTask 3.3: 实现 check_invalidation() - template/标识变化时验证 block 失效

- [ ] Task 4: 实现 structure_cases.py - 6 类真实结构用例生成
  - [ ] SubTask 4.1: 从 τ-bench 下载 retail/airline 任务定义
  - [ ] SubTask 4.2: 生成 ① 同域任务对（retail 15 对 + airline 15 对）
  - [ ] SubTask 4.3: 生成 ② 分支历史对（同任务不同 seed，共享前缀）
  - [ ] SubTask 4.4: 生成 ③ chat template 变化（同一会话 × 2 版本 template）
  - [ ] SubTask 4.5: 生成 ④ 模型标识变化（元数据字段变换）
  - [ ] SubTask 4.6: 生成 ⑤ 纯追加长会话（多轮对话）
  - [ ] SubTask 4.7: 生成 ⑥ 无共享对照（跨域任务对）
  - [ ] SubTask 4.8: 输出 real-structure-cases.json

- [ ] Task 5: 实现 exactness_test.py - BF16 缓存恢复 vs 重算一致性
  - [ ] SubTask 5.1: 对每个结构用例，执行重算路径（完整 forward）
  - [ ] SubTask 5.2: 执行缓存路径（切片 block → 恢复 → 续算）
  - [ ] SubTask 5.3: 对比 KV 张量（bit-identical 比例）
  - [ ] SubTask 5.4: 对比 logits（max abs diff, mean abs diff, cosine sim）
  - [ ] SubTask 5.5: 对比 greedy decode top-1 token 一致率
  - [ ] SubTask 5.6: 输出 exactness-report.md

- [ ] Task 6: 实现 codec.py - Q8/Q4 量化 codec
  - [ ] SubTask 6.1: 实现 Q8 编码/解码（per-tensor int8）
  - [ ] SubTask 6.2: 实现 Q4 编码/解码（per-tensor int4）
  - [ ] SubTask 6.3: 实现 lineage 隔离检查（approximate vs canonical）

- [ ] Task 7: 实现 codec_spike.py - 100 block roundtrip 测试
  - [ ] SubTask 7.1: 从 exactness 测试的 block 中抽取 100 个 unique block
  - [ ] SubTask 7.2: 对每个 block 执行 Q8/Q4 roundtrip
  - [ ] SubTask 7.3: 记录 MSE、max abs err、logit KL、编解码延迟、staging 峰值字节
  - [ ] SubTask 7.4: 验证 lineage 隔离正确性
  - [ ] SubTask 7.5: 输出 codec-spike-report.md

- [ ] Task 8: 实现 memory_test.py - 显存峰值测量
  - [ ] SubTask 8.1: 测量仅加载模型的 allocated/reserved 峰值（5 次重复）
  - [ ] SubTask 8.2: 测量并发 4 × 4K context 的峰值
  - [ ] SubTask 8.3: 测量并发 8 × 8K context 的峰值
  - [ ] SubTask 8.4: 输出 memory-report.md

- [ ] Task 9: 实现 freeze_record.py - 冻结记录生成
  - [ ] SubTask 9.1: 收集模型 revision、tokenizer、chat template、transformers 版本
  - [ ] SubTask 9.2: 收集 CUDA/driver/GPU 型号
  - [ ] SubTask 9.3: 输出 freeze-record.json

- [ ] Task 10: 实现 verdict.py - G0 判定报告
  - [ ] SubTask 10.1: 汇总 6 个判定条件通过/失败状态
  - [ ] SubTask 10.2: 生成 g0-verdict.md（含表 G0-1 到 G0-4）

- [ ] Task 11: 实现 run_g0.py - 主入口
  - [ ] SubTask 11.1: argparse 支持 --step all/0/1/2/3/4/5/6
  - [ ] SubTask 11.2: 按顺序调用各模块
  - [ ] SubTask 11.3: 打印进度和最终 verdict

# Task Dependencies

- Task 2 (backend) 是所有其他 Task 的基础
- Task 3 (block_index) 依赖 Task 2
- Task 4 (structure_cases) 独立，可与 Task 3 并行
- Task 5 (exactness) 依赖 Task 2 + 3 + 4
- Task 6 (codec) 依赖 Task 3
- Task 7 (codec_spike) 依赖 Task 5 + 6
- Task 8 (memory) 依赖 Task 2
- Task 9 (freeze_record) 依赖 Task 2
- Task 10 (verdict) 依赖 Task 5 + 7 + 8 + 9
- Task 11 (run_g0) 依赖全部
