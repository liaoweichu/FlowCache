# Checklist

## experiments/e1/record_trajectories.py — `_record_all_g1`
- [x] `_record_all_g1` 函数签名不再含 `bfcl_subset_filter` 参数
- [x] `dataset_filter` 默认值为 `"tau-bench"`（不再是 `"all"`）
- [x] 删除了 `all_bfcl_subsets` / `bfcl_subsets` 变量解析
- [x] 删除了 `elif dataset == "bfcl_v3":` 分发分支
- [x] report dict 不再含 `"bfcl_subset_filter"` 字段
- [x] docstring 已更新，无 BFCL 描述

## experiments/e1/record_trajectories.py — `_record_bfcl_g1`
- [x] `_record_bfcl_g1` 方法已完全删除
- [x] grep `_record_bfcl_g1` 在 record_trajectories.py 中无匹配

## experiments/e1/record_trajectories.py — CLI
- [x] `--dataset` 参数 `choices=["tau-bench"]`，`default="tau-bench"`
- [x] `--bfcl-subset` 参数已删除
- [x] `main()` 中不再传 `bfcl_subset_filter=args.bfcl_subset`
- [x] `_build_arg_parser` docstring 已更新

## experiments/e1/record_trajectories.py — 保留项
- [x] `_run_episode_bfcl` 方法仍保留（grep 验证：第 14/377/691/1351 行）
- [x] `_parse_bfcl_tool_calls` 函数仍保留（grep 验证：第 186/1365/1475 行）
- [x] `_init_adapter` 中的 `elif dataset == "bfcl_v3":` 分支仍保留（第 401 行）
- [x] `bfcl_adapter.py` 文件未变更

## experiments/e1/tests/test_config_load.py
- [x] `test_config_has_bfcl_subsets` 已删除
- [x] `test_config_has_workload_datasets_list` 断言 `cfg["workload"]["datasets"] == ["tau-bench"]`
- [x] `test_config_trace_subdirs_present` 断言 `set(subdirs) == {"tau_bench"}`
- [x] 文件顶部 docstring 已更新（1320 episodes，无 BFCL 描述）

## experiments/e1/tests/test_record_cli_args.py
- [x] `test_argparse_accepts_bfcl_subset` 已删除
- [x] `test_argparse_default_bfcl_subset_is_none` 已删除
- [x] `test_argparse_default_dataset_is_tau_bench` 断言 `args.dataset == "tau-bench"`
- [x] `test_argparse_invalid_dataset_rejected` 用 `"bfcl_v3"` 验证拒绝
- [x] `test_argparse_accepts_seed_and_dataset` 仍通过
- [x] 文件顶部 docstring 已更新

## experiments/e1/tests/test_record_all_g1.py
- [x] `test_record_all_g1_bfcl_naming` 已删除
- [x] `test_record_all_g1_dataset_filter_bfcl_only` 已删除
- [x] `test_trace_output_excludes_global_block_index_bfcl` 已删除
- [x] `_MockBFCLAdapter` 类已删除
- [x] `_make_recorder` 默认 config 不含 `bfcl_v3` 字段
- [x] 文件顶部 docstring 已更新（1320 episodes，非 7720）

## experiments/e1/tests/test_episode_loops.py（保留）
- [x] `test_run_episode_bfcl_produces_trace` 仍存在并通过
- [x] 文件无修改（保留 BFCL 单 episode 测试）

## experiments/e1/tests/test_meta_fields.py（保留）
- [x] `test_run_episode_bfcl_propagates_model_id_into_blocks` 仍存在并通过
- [x] `test_run_episode_bfcl_meta_includes_metadata_fields` 仍存在并通过
- [x] 文件无修改（保留 BFCL 元数据传播测试）

## experiments/e1/tests/test_adapter_dispatch.py（保留）
- [x] `test_init_adapter_returns_bfcl_when_available` 仍存在并 skip（bfcl_eval 未安装）
- [x] `test_init_adapter_bfcl_default_subset_is_multi_turn_base` 仍存在并 skip
- [x] 文件无修改（保留 `_init_adapter` BFCL 分支测试）

## 验证
- [x] `pytest experiments/e1/tests/` 全部通过（37 passed, 3 skipped）
- [x] grep `bfcl_subset_filter|_record_bfcl_g1` 在 record_trajectories.py 中无匹配
- [x] grep `--bfcl-subset` 在 record_trajectories.py 中无匹配
- [x] grep `_run_episode_bfcl|_parse_bfcl_tool_calls` 在 record_trajectories.py 中仍有匹配（保留项）
- [x] config.yaml 中 `workload.datasets == ["tau-bench"]`，`output.trace_subdirs == ["tau_bench"]`
- [x] bfcl_adapter.py 文件未变更
