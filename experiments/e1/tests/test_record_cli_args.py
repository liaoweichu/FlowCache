"""
Test CLI argument parser for record_trajectories.py.

Background: G1 needs --seed, --dataset, --bfcl-subset, --max-episodes,
--resume/--no-resume flags to drive multi-dataset × multi-seed recording.
This test verifies the parser is exposed via `_build_arg_parser()` and
accepts all required flags with correct defaults.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "g0"))

import record_trajectories as rt


def test_argparse_accepts_seed_and_dataset():
    parser = rt._build_arg_parser()
    args = parser.parse_args(["--seed", "42", "--dataset", "tau-bench"])
    assert args.seed == 42
    assert args.dataset == "tau-bench"


def test_argparse_accepts_bfcl_subset():
    parser = rt._build_arg_parser()
    args = parser.parse_args([
        "--dataset", "bfcl_v3",
        "--bfcl-subset", "multi_turn_base",
        "--seed", "123",
    ])
    assert args.bfcl_subset == "multi_turn_base"


def test_argparse_default_dataset_is_all():
    parser = rt._build_arg_parser()
    args = parser.parse_args([])
    assert args.dataset == "all"  # 默认录制两个数据集
    assert args.seed is None  # None 表示用 config 中的全部 seeds


def test_argparse_accepts_resume_flag():
    parser = rt._build_arg_parser()
    args = parser.parse_args(["--no-resume"])
    assert args.resume is False


def test_argparse_default_resume_is_true():
    parser = rt._build_arg_parser()
    args = parser.parse_args([])
    assert args.resume is True


def test_argparse_accepts_max_episodes():
    parser = rt._build_arg_parser()
    args = parser.parse_args(["--max-episodes", "5"])
    assert args.max_episodes == 5


def test_argparse_default_bfcl_subset_is_none():
    parser = rt._build_arg_parser()
    args = parser.parse_args(["--dataset", "bfcl_v3"])
    assert args.bfcl_subset is None  # None = all 4 subsets


def test_argparse_invalid_dataset_rejected():
    parser = rt._build_arg_parser()
    import pytest
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "swe-bench"])  # not in choices
