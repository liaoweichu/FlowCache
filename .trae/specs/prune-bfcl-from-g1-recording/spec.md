# Prune BFCL from G1 Recording Code Spec

## Why

[single-dataset-taubench-only spec](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/single-dataset-taubench-only/spec.md) 已将 `experiments/e1/config.yaml` 改为单数据集 τ-bench 1,320 episodes，但 [experiments/e1/record_trajectories.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/record_trajectories.py) 仍保留 BFCL 主流程调用：`_record_all_g1` 仍会按 `workload.datasets` 列表分发到 `_record_bfcl_g1`，CLI 仍暴露 `--dataset bfcl_v3` / `--bfcl-subset` 选项，相关测试仍断言 config 必须包含 `bfcl_v3` 字段。代码与 config 不一致会导致：(1) `_record_all_g1` 读取 `cfg["workload"]["bfcl_v3"]` 时 KeyError；(2) CLI 仍允许用户指定 `--dataset bfcl_v3` 但 config 无该配置；(3) `test_config_load.py` 测试失败。

## What Changes

- **BREAKING**: 从 `record_trajectories.py` 的 `_record_all_g1` 中删除 BFCL 分发分支（`elif dataset == "bfcl_v3":` 调用 `_record_bfcl_g1`）
- **BREAKING**: 删除 `_record_bfcl_g1` 方法（不再被任何路径调用；BFCL 录制逻辑保留在 `_run_episode_bfcl` 中供 rebuttal 复用）
- **BREAKING**: 删除 `_record_all_g1` 中对 `bfcl_subset_filter` 参数的解析与传递逻辑
- **BREAKING**: CLI `--dataset` 选项的 `choices` 从 `["all", "tau-bench", "bfcl_v3"]` 改为 `["tau-bench"]`（移除 `"all"` 与 `"bfcl_v3"`）
- **BREAKING**: 删除 CLI `--bfcl-subset` 参数
- **BREAKING**: `main()` 中删除 `bfcl_subset_filter=args.bfcl_subset` 传参
- 保留 `_run_episode_bfcl` 方法（仅供未来 rebuttal 单独调用，不再被 `_record_all_g1` 调用）
- 保留 `_parse_bfcl_tool_calls` 函数（被 `_run_episode_bfcl` 使用）
- 保留 `_init_adapter` 中的 `elif dataset == "bfcl_v3":` 分支（允许 standalone 调用，但不在主流程触发）
- 保留 [experiments/e1/bfcl_adapter.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/bfcl_adapter.py) 不变（rebuttal 备用）
- 更新 [experiments/e1/tests/test_config_load.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_config_load.py)：删除 `test_config_has_bfcl_subsets`、`test_config_has_workload_datasets_list` 中的 bfcl_v3 断言、`test_config_trace_subdirs_present` 中的 bfcl_v3 断言
- 更新 [experiments/e1/tests/test_record_cli_args.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_record_cli_args.py)：删除 `test_argparse_accepts_bfcl_subset`、`test_argparse_default_bfcl_subset_is_none`，更新 `test_argparse_default_dataset_is_all` → `test_argparse_default_dataset_is_tau_bench`
- 更新 [experiments/e1/tests/test_record_all_g1.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_record_all_g1.py)：删除 `test_record_all_g1_bfcl_naming`、`test_record_all_g1_dataset_filter_bfcl_only`、`test_trace_output_excludes_global_block_index_bfcl` 三个 BFCL-only 测试；将 `_make_recorder` 默认 config 中删除 `bfcl_v3` 字段
- 保留 [experiments/e1/tests/test_episode_loops.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_episode_loops.py) 中的 `test_run_episode_bfcl_produces_trace`（仍测试保留的 `_run_episode_bfcl` 方法）
- 保留 [experiments/e1/tests/test_meta_fields.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_meta_fields.py) 中的 BFCL 元数据传播测试（仍测试保留的 `_run_episode_bfcl` 方法）
- 保留 [experiments/e1/tests/test_adapter_dispatch.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_adapter_dispatch.py) 中的 BFCL 分发测试（仍测试保留的 `_init_adapter` BFCL 分支）

## Impact

- Affected specs:
  - [single-dataset-taubench-only/spec.md](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/single-dataset-taubench-only/spec.md): 本 spec 是其代码侧补丁，完成 "Affected code" 中 record_trajectories.py 的修改
- Affected code:
  - [experiments/e1/record_trajectories.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/record_trajectories.py): 删除 `_record_bfcl_g1` 方法、`_record_all_g1` 中 BFCL 分发与 bfcl_subset_filter 参数、CLI `--dataset` 选项 BFCL 选项、`--bfcl-subset` 参数
  - [experiments/e1/tests/test_config_load.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_config_load.py): 删除 BFCL 相关断言
  - [experiments/e1/tests/test_record_cli_args.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_record_cli_args.py): 删除 `--bfcl-subset` 测试，更新默认 dataset 测试
  - [experiments/e1/tests/test_record_all_g1.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/tests/test_record_all_g1.py): 删除 3 个 BFCL-only 测试，清理默认 config
  - [experiments/e1/bfcl_adapter.py](file:///d:/00MyProject/Prefix%20Caching/experiments/e1/bfcl_adapter.py): 无变更（保留供 rebuttal）
- 受益：
  - 消除 config 与代码的不一致：`_record_all_g1` 不再尝试读取已删除的 `cfg["workload"]["bfcl_v3"]`
  - CLI 不再暴露已废弃的 `--dataset bfcl_v3` / `--bfcl-subset` 选项，避免用户误用
  - 测试套件不再断言 config 必须包含 BFCL 字段，与 single-dataset config 对齐
  - BFCL 录制能力完整保留在 `_run_episode_bfcl` + `bfcl_adapter.py`，rebuttal 时可手动调用
- 风险：
  - 删除 `_record_bfcl_g1` 后，若 rebuttal 需要批量录制 BFCL，需重新实现该方法（但单数据集 spec 已接受此风险）
  - CLI 不再支持 `--dataset all`，旧脚本若用该值会报错（spec 已 BREAKING，可接受）

## ADDED Requirements

### Requirement: G1 主录制循环仅分发 τ-bench

`_record_all_g1` 仅根据 `cfg["workload"]["datasets"]` 列表中的 `"tau-bench"` 项分发到 `_record_tau_bench_g1`；不再识别 `"bfcl_v3"` 项；不再解析 `bfcl_subset_filter` 参数。

#### Scenario: 单数据集 τ-bench 录制
- **WHEN** 调用 `_record_all_g1()` 且 config 中 `workload.datasets = ["tau-bench"]`
- **THEN** 仅调用 `_record_tau_bench_g1(seeds, resume, max_episodes)`
- **AND** 不调用 `_record_bfcl_g1`
- **AND** 不读取 `cfg["workload"]["bfcl_v3"]`（避免 KeyError）

#### Scenario: 未知 dataset 项被跳过
- **WHEN** config 中 `workload.datasets` 含有 `"bfcl_v3"` 或其他未知项
- **THEN** `_record_all_g1` 输出 warning 并跳过该项
- **AND** 不抛出异常

### Requirement: CLI 仅暴露 τ-bench 选项

`record_trajectories.py` 的 `--dataset` 参数 `choices` 为 `["tau-bench"]`，默认值为 `"tau-bench"`；不再提供 `--bfcl-subset` 参数。

#### Scenario: 默认 CLI 调用
- **WHEN** 用户运行 `python record_trajectories.py`（无参数）
- **THEN** `args.dataset == "tau-bench"`
- **AND** `args.bfcl_subset` 属性不存在（AttributeError）

#### Scenario: 用户尝试指定 BFCL
- **WHEN** 用户运行 `python record_trajectories.py --dataset bfcl_v3`
- **THEN** argparse 拒绝该值并退出（SystemExit）

#### Scenario: 用户尝试指定 bfcl-subset
- **WHEN** 用户运行 `python record_trajectories.py --bfcl-subset multi_turn_base`
- **THEN** argparse 拒绝该参数并退出（SystemExit）

## MODIFIED Requirements

### Requirement: `_record_all_g1` 函数签名

**原（v0.3）**：
```python
def _record_all_g1(
    self,
    dataset_filter: str = "all",
    seed_filter: Optional[int] = None,
    bfcl_subset_filter: Optional[str] = None,
    max_episodes: Optional[int] = None,
    resume: bool = True,
) -> int:
```

**现（本 spec）**：
```python
def _record_all_g1(
    self,
    dataset_filter: str = "tau-bench",
    seed_filter: Optional[int] = None,
    max_episodes: Optional[int] = None,
    resume: bool = True,
) -> int:
```

`bfcl_subset_filter` 参数被移除；`dataset_filter` 默认值从 `"all"` 改为 `"tau-bench"`。

### Requirement: CLI 参数集

**原（v0.3）**：`--config / --dataset (choices=[all, tau-bench, bfcl_v3], default=all) / --bfcl-subset / --seed / --max-episodes / --resume / --no-resume / --subset / --output-dir`

**现（本 spec）**：`--config / --dataset (choices=[tau-bench], default=tau-bench) / --seed / --max-episodes / --resume / --no-resume / --subset / --output-dir`

### Requirement: 测试套件 BFCL 覆盖范围

**原（v0.3）**：测试套件覆盖 `_record_bfcl_g1` / `_run_episode_bfcl` / `--bfcl-subset` CLI / config `bfcl_v3.subsets` / trace_subdirs `bfcl_v3` 等所有 BFCL 路径。

**现（本 spec）**：测试套件仅覆盖 `_run_episode_bfcl` / `_init_adapter` BFCL 分支（保留方法），删除 `_record_bfcl_g1` / `--bfcl-subset` / config `bfcl_v3` / trace_subdirs `bfcl_v3` 相关测试（已删除路径）。

## REMOVED Requirements

### Requirement: `_record_bfcl_g1` 方法

**Reason**: `_record_all_g1` 不再分发到 BFCL，该方法成为死代码。BFCL 单 episode 录制逻辑保留在 `_run_episode_bfcl` 中，rebuttal 时可手动调用。

**Migration**: 若 rebuttal 需要批量录制 BFCL，可从 git 历史恢复 `_record_bfcl_g1` 或重新实现基于 `_run_episode_bfcl` 的批量循环。

### Requirement: CLI `--bfcl-subset` 参数

**Reason**: BFCL 已从主流程移除，`--bfcl-subset` 不再有任何效果。

**Migration**: 若 rebuttal 需要单 subset 录制，可直接调用 `_run_episode_bfcl` + `BFCLAdapter(subset=...)`。

### Requirement: CLI `--dataset all` 与 `--dataset bfcl_v3` 选项

**Reason**: 单数据集 spec 已删除 BFCL，`all` 与 `bfcl_v3` 不再有意义。

**Migration**: 无；用户应使用 `--dataset tau-bench` 或不指定（默认 tau-bench）。

### Requirement: config `bfcl_v3.subsets` 字段（测试断言）

**Reason**: config 已在 [single-dataset-taubench-only spec](file:///d:/00MyProject/Prefix%20Caching/.trae/specs/single-dataset-taubench-only/spec.md) 中删除 `bfcl_v3` 配置块，测试不应再断言其存在。

**Migration**: 无；测试已删除相关断言。
