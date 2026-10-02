from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def cpu_feature_flags() -> set[str]:
    """Return CPU features enabled by both hardware and the current OS."""
    try:
        import numpy as np
        raw = np._core._multiarray_umath.__cpu_features__
    except (ImportError, AttributeError):
        return set()
    aliases = {
        "AVX": "avx", "AVX2": "avx2", "AVX512F": "avx512f",
        "AVX512VNNI": "avx512_vnni", "FMA3": "fma3", "F16C": "f16c",
    }
    return {alias for source, alias in aliases.items() if raw.get(source, False)}


def _run(command: list[str]) -> str | None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_machine() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd().anchor))
    cpu_flags = None
    if sys.platform == "win32":
        cpu_info = _run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"])
        virtualization = _run(["powershell", "-NoProfile", "-Command",
                               "(Get-CimInstance Win32_ComputerSystem).Model"])
    else:
        cpu_info = _run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"])
        cpu_flags = _run(["sh", "-c", "grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2-"])
        virtualization = _run(["systemd-detect-virt"])
    ort = None
    try:
        import onnxruntime
        ort = {"version": onnxruntime.__version__, "providers": onnxruntime.get_available_providers()}
    except ImportError:
        pass
    torch_info = None
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                      "threads": torch.get_num_threads()}
    except ImportError:
        pass
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "machine": platform.machine(),
        "cpu": cpu_info or platform.processor(), "cpu_flags": cpu_flags,
        "runtime_cpu_features": sorted(cpu_feature_flags()),
        "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
        "virtualization_model": virtualization,
        "memory": {"total_bytes": memory.total, "available_bytes": memory.available},
        "disk": {"root": str(Path.cwd().anchor), "total_bytes": disk.total, "free_bytes": disk.free},
        "python": {"version": sys.version, "executable": sys.executable},
        "onnxruntime": ort, "torch": torch_info,
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }

