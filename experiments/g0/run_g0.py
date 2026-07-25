"""
G0: Exactness & Loadability - 主入口脚本 (Task 11)。

用法：
    python run_g0.py --step all     # 执行全部步骤
    python run_g0.py --step 0       # Step 0: 后端冻结 + freeze-record
    python run_g0.py --step 1       # Step 1: 显存测试
    python run_g0.py --step 2       # Step 2: 结构用例生成
    python run_g0.py --step 3       # Step 3: Block identity 测试
    python run_g0.py --step 4       # Step 4: Exactness 测试
    python run_g0.py --step 5       # Step 5: Codec spike
    python run_g0.py --step 6       # Step 6: 判定报告
"""
import argparse
import json
import os
import sys

# 添加当前目录到 path，支持相对导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml


def load_config(config_path: str = None) -> dict:
    """加载 config.yaml 配置文件。"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_output_path(config: dict, key: str) -> str:
    """根据 config 中的 output 配置拼接完整输出路径。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, config["output"][key])


def _load_cases(config: dict) -> dict:
    """加载结构用例 JSON 文件。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, config["output"]["structure_cases"])
    if not os.path.exists(cases_path):
        print(f"  [ERROR] 结构用例文件不存在: {cases_path}")
        print(f"  请先运行: python run_g0.py --step 2")
        sys.exit(1)
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 各步骤实现
# =============================================================================

def step0_freeze(config: dict, backend) -> bool:
    """Step 0: 后端冻结 + freeze-record.json"""
    from freeze_record import generate_freeze_record, save_freeze_record, validate_freeze_record

    print("  生成 freeze-record...")
    record = generate_freeze_record(backend)
    output_path = _get_output_path(config, "freeze_record")
    save_freeze_record(record, output_path)

    valid, missing = validate_freeze_record(record)
    print(f"  freeze-record valid: {valid}")
    if missing:
        print(f"  missing fields: {missing}")
    return valid


def step1_memory(config: dict, backend) -> dict:
    """Step 1: 显存峰值测量"""
    from memory_test import run_memory_test

    output_path = _get_output_path(config, "memory_report")
    return run_memory_test(backend, config, output_path)


def step2_cases(config: dict) -> dict:
    """Step 2: 生成结构用例"""
    from structure_cases import generate_all_cases, save_cases

    cases = generate_all_cases()
    output_path = _get_output_path(config, "structure_cases")
    save_cases(cases, output_path)
    return cases


def step3_block_identity(config: dict, backend) -> dict:
    """Step 3: Block identity 测试"""
    from exactness_test import test_block_identity

    cases = _load_cases(config)
    print("  运行 block identity 测试...")
    results = test_block_identity(backend, cases, config["cache"]["block_size"])

    # 保存结果
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, config["output"]["dir"], "block-identity-results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to {output_path}")

    # 打印摘要
    summary = results["summary"]
    print(
        f"  Identity: {summary['identity_pass']}/{summary['total']} passed, "
        f"Parent chain: {summary['parent_chain_pass']}/{summary['total']} passed"
    )
    return results


def step4_exactness(config: dict, backend) -> dict:
    """Step 4: BF16 exactness 测试"""
    from exactness_test import run_exactness_test

    cases = _load_cases(config)
    output_path = _get_output_path(config, "exactness_report")
    return run_exactness_test(backend, cases, config["cache"]["block_size"], output_path)


def step5_codec(config: dict, backend) -> dict:
    """Step 5: Q8/Q4 codec spike 测试"""
    from codec_spike import run_codec_spike

    cases = _load_cases(config)
    output_path = _get_output_path(config, "codec_report")
    return run_codec_spike(
        backend, cases, config["cache"]["block_size"],
        config["codec"]["num_blocks"], output_path,
    )


def step6_verdict(config: dict) -> dict:
    """Step 6: G0 判定报告"""
    from verdict import generate_verdict

    output_path = _get_output_path(config, "verdict")
    result = generate_verdict(config, output_path)

    # 打印最终判定
    print(f"\n  G0 Verdict: {result['overall']}")
    for c in result["conditions"]:
        status = "✓" if c["passed"] else "✗"
        print(f"    {status} 条件 {c['id']}: {c['name']}")

    return result


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="G0: Exactness & Loadability")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "0", "1", "2", "3", "4", "5", "6"],
        help="执行的步骤 (默认: all)",
    )
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"G0: Exactness & Loadability")
    print(f"{'='*60}")
    print(f"Model: {config['model']['name']}")
    print(f"Block size: {config['cache']['block_size']}")
    print(f"Step: {args.step}\n")

    # 初始化 backend（大多数步骤需要）
    backend = None
    if args.step in ["all", "0", "1", "3", "4", "5"]:
        from backend import Backend
        import torch

        print("加载模型...")
        backend = Backend(
            model_name=config["model"]["name"],
            dtype=getattr(torch, config["model"]["dtype"]),
            device_map=config["model"].get("device_map", "auto"),
        )
        print(f"Backend loaded: {backend.model_name}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    steps = {
        "0": ("Step 0: Backend Freeze", lambda: step0_freeze(config, backend)),
        "1": ("Step 1: Memory Test", lambda: step1_memory(config, backend)),
        "2": ("Step 2: Structure Cases", lambda: step2_cases(config)),
        "3": ("Step 3: Block Identity", lambda: step3_block_identity(config, backend)),
        "4": ("Step 4: Exactness Test", lambda: step4_exactness(config, backend)),
        "5": ("Step 5: Codec Spike", lambda: step5_codec(config, backend)),
        "6": ("Step 6: Verdict", lambda: step6_verdict(config)),
    }

    if args.step == "all":
        for step_id in ["0", "1", "2", "3", "4", "5", "6"]:
            name, func = steps[step_id]
            print(f"\n--- {name} ---")
            try:
                func()
            except Exception as e:
                print(f"  [ERROR] {name} failed: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行后续步骤（verdict 会报告失败）
                if step_id == "6":
                    continue
    else:
        name, func = steps[args.step]
        print(f"\n--- {name} ---")
        func()

    print(f"\n{'='*60}")
    print(f"G0 complete. Check outputs/ for artifacts.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
