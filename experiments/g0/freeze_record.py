"""
生成 G0 的 freeze-record.json。

记录模型 revision、tokenizer、chat template、backend、CUDA/driver 版本，
用于在后续实验中复现 G0 的运行环境与模型快照。
"""

import json
import os
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple


def generate_freeze_record(backend) -> Dict:
    """
    从 Backend 实例生成 freeze record。

    Args:
        backend: Backend 实例（已加载模型与 tokenizer）。

    Returns:
        包含全部 freeze record 字段的 dict。
    """
    model_info = backend.get_model_info()

    record = {
        "generated_at": datetime.now().isoformat(),
        "model": model_info,
        "tokenizer": {
            "name": backend.model_name,
            "pad_token": backend.tokenizer.pad_token,
            "eos_token": backend.tokenizer.eos_token,
            "bos_token": backend.tokenizer.bos_token,
            "chat_template_present": backend.tokenizer.chat_template is not None,
            "chat_template_length": (
                len(backend.tokenizer.chat_template)
                if backend.tokenizer.chat_template
                else 0
            ),
        },
        "backend": {
            "library": "transformers",
            "version": model_info.get("transformers_version", "unknown"),
            "device_map": "auto",
        },
        "system": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "cuda": {
            "version": model_info.get("cuda_version", "unknown"),
            "gpu_name": model_info.get("gpu_name", "unknown"),
            "gpu_memory_total_gb": model_info.get("gpu_memory_total_gb", 0),
        },
    }

    # 尝试获取 driver 版本
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            record["cuda"]["driver_version"] = result.stdout.strip()
        else:
            record["cuda"]["driver_version"] = "unknown"
    except Exception:
        record["cuda"]["driver_version"] = "unknown"

    return record


def save_freeze_record(record: Dict, output_path: str):
    """将 freeze record 保存为 JSON 文件。"""
    # 兼容相对路径：如果 output_path 没有目录部分，跳过 makedirs
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[freeze_record] Saved to {output_path}")


def validate_freeze_record(record: Dict) -> Tuple[bool, List[str]]:
    """
    校验 freeze record 是否包含全部必填字段。

    Returns:
        (is_valid, missing_fields)：is_valid 表示是否通过校验，
        missing_fields 为缺失字段列表（空列表表示无缺失）。
    """
    required_fields = [
        "generated_at",
        "model.model_name",
        "model.transformers_version",
        "model.dtype",
        "model.num_layers",
        "model.model_sha",
        "tokenizer.pad_token",
        "tokenizer.chat_template_present",
        "backend.library",
        "backend.version",
        "cuda.version",
        "cuda.gpu_name",
    ]

    missing: List[str] = []
    for field in required_fields:
        parts = field.split(".")
        val = record
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                missing.append(field)
                break

    return len(missing) == 0, missing
