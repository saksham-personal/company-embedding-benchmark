from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embedding_bench.config import canonical_digest


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_digest(project_root: Path) -> str:
    digest = hashlib.sha256()
    files = [*project_root.glob("src/**/*.py"), *project_root.glob("scripts/*.py"),
             *project_root.glob("configs/*.json"), project_root / "pyproject.toml", project_root / "uv.lock"]
    for path in sorted(files):
        if path.is_file():
            digest.update(path.relative_to(project_root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(*, project_root: Path, machine_role: str, command: list[str],
               catalog_digest: str | None = None, dataset_manifest: dict[str, Any] | None = None,
               seed: int = 20261002) -> dict[str, Any]:
    from embedding_bench.hardware import inspect_machine
    hardware = inspect_machine()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine_role": machine_role,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit(project_root),
        "git_dirty": bool(dirty.stdout.strip()),
        "source_digest": source_digest(project_root),
        "hardware": hardware,
        "hardware_id": canonical_digest({key: hardware.get(key) for key in
            ("os", "machine", "cpu", "physical_cores", "logical_cores", "runtime_cpu_features", "virtualization_model")}),
        "command": command,
        "seed": seed,
        "catalog_digest": catalog_digest,
        "dataset_id": dataset_manifest.get("dataset_id") if dataset_manifest else None,
        "dataset_digest": canonical_digest(dataset_manifest) if dataset_manifest else None,
        "thread_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def write_result(path: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

