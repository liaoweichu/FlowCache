# Tasks

- [x] Task 1: 修改 `experiments/e1/record_trajectories.py` 的 `_record_all_g1` 方法
  - [x] SubTask 1.1: 删除 `bfcl_subset_filter` 参数（函数签名）
  - [x] SubTask 1.2: 将 `dataset_filter` 默认值从 `"all"` 改为 `"tau-bench"`
  - [x] SubTask 1.3: 删除 `all_bfcl_subsets` 与 `bfcl_subsets` 解析逻辑
  - [x] SubTask 1.4: 删除 `elif dataset == "bfcl_v3":` 分发分支，仅保留 `if dataset == "tau-bench":` 分支
  - [x] SubTask 1.5: 删除 report 中的 `"bfcl_subset_filter": bfcl_subset_filter` 字段
  - [x] SubTask 1.6: 更新 `_record_all_g1` 顶部的 docstring，移除 BFCL 文档

- [x] Task 2: 删除 `experiments/e1/record_trajectories.py` 中的 `_record_bfcl_g1` 方法
  - [x] SubTask 2.1: 删除整个 `_record_bfcl_g1` 方法

- [x] Task 3: 修改 `experiments/e1/record_trajectories.py` 的 CLI 参数
  - [x] SubTask 3.1: 将 `--dataset` 参数 `choices` 改为 `["tau-bench"]`，默认值改为 `"tau-bench"`
  - [x] SubTask 3.2: 删除 `--bfcl-subset` 参数整段
  - [x] SubTask 3.3: 在 `main()` 中删除 `bfcl_subset_filter=args.bfcl_subset` 传参
  - [x] SubTask 3.4: 更新 `_build_arg_parser` 顶部 docstring，移除 `--bfcl-subset` 文档

- [x] Task 4: 修改 `experiments/e1/tests/test_config_load.py`
  - [x] SubTask 4.1: 删除 `test_config_has_bfcl_subsets` 函数
  - [x] SubTask 4.2: 修改 `test_config_has_workload_datasets_list`，断言改为 `== ["tau-bench"]`
  - [x] SubTask 4.3: 修改 `test_config_trace_subdirs_present`，断言改为 `== {"tau_bench"}`
  - [x] SubTask 4.4: 更新文件顶部 docstring，移除 BFCL 7720 episodes 描述

- [x] Task 5: 修改 `experiments/e1/tests/test_record_cli_args.py`
  - [x] SubTask 5.1: 删除 `test_argparse_accepts_bfcl_subset` 函数
  - [x] SubTask 5.2: 删除 `test_argparse_default_bfcl_subset_is_none` 函数
  - [x] SubTask 5.3: 重命名为 `test_argparse_default_dataset_is_tau_bench`，断言 `args.dataset == "tau-bench"`
  - [x] SubTask 5.4: 修改 `test_argparse_invalid_dataset_rejected`，用 `"bfcl_v3"` 验证拒绝
  - [x] SubTask 5.5: `test_argparse_accepts_seed_and_dataset` 仍通过
  - [x] SubTask 5.6: 更新文件顶部 docstring

- [x] Task 6: 修改 `experiments/e1/tests/test_record_all_g1.py`
  - [x] SubTask 6.1: 删除 `test_record_all_g1_bfcl_naming` 函数
  - [x] SubTask 6.2: 删除 `test_record_all_g1_dataset_filter_bfcl_only` 函数
  - [x] SubTask 6.3: 删除 `test_trace_output_excludes_global_block_index_bfcl` 函数
  - [x] SubTask 6.4: 删除 `_MockBFCLAdapter` 类
  - [x] SubTask 6.5: 修改 `_make_recorder` 默认 config，删除 `bfcl_v3` 字段
  - [x] SubTask 6.6: 更新文件顶部 docstring（1320 episodes）

- [x] Task 7: 验证测试套件可通过
  - [x] SubTask 7.1: 运行 `pytest experiments/e1/tests/` 验证所有测试通过（37 passed, 3 skipped）
  - [x] SubTask 7.2: 在 `record_trajectories.py` 中 grep `bfcl_subset_filter|_record_bfcl_g1` 确认无残留
  - [x] SubTask 7.3: 在 `record_trajectories.py` 中 grep `--bfcl-subset` 确认 CLI 已清理
  - [x] SubTask 7.4: 确认 `_run_episode_bfcl` 与 `_parse_bfcl_tool_calls` 仍保留（grep 验证）
  - [x] SubTask 7.5: 确认 `bfcl_adapter.py` 文件未变更

# Task Dependencies

- Task 1, 2, 3 必须串行（同一文件 record_trajectories.py，避免合并冲突）
- Task 4, 5, 6 可并行（不同测试文件）
- Task 7 依赖 Task 1-6 全部完成
