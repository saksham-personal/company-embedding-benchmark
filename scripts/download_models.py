from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from embedding_bench.config import load_json
from embedding_bench.downloads import DownloadError, download_verified, safe_extract_zip, verify_payload


def select(records, artifact_ids, model_ids):
    selected = []
    for record in records:
        if record.get("status") not in {"ready", "available"}:
            continue
        if artifact_ids and record["artifact_id"] not in artifact_ids:
            continue
        if model_ids and record["model_id"] not in model_ids:
            continue
        selected.append(record)
    unknown_artifacts = set(artifact_ids or ()) - {row["artifact_id"] for row in records}
    unknown_models = set(model_ids or ()) - {row["model_id"] for row in records}
    if unknown_artifacts or unknown_models:
        raise DownloadError(f"unknown selection: artifacts={sorted(unknown_artifacts)}, models={sorted(unknown_models)}")
    return selected


def install(record, model_root: Path, work_dir: Path, *, force: bool) -> dict[str, object]:
    artifact_id = record["artifact_id"]
    destination = (model_root / artifact_id).resolve()
    model_root = model_root.resolve()
    if model_root not in destination.parents:
        raise DownloadError(f"invalid artifact_id path: {artifact_id}")
    if destination.is_dir() and not force:
        try:
            verify_payload(destination, record["files"])
            return {"artifact_id": artifact_id, "status": "already_verified", "path": str(destination)}
        except DownloadError:
            pass
    github = record["github"]
    url = f"https://github.com/{github['repository']}/releases/download/{github['release']}/{github['asset']}"
    archive = download_verified(url, work_dir / "downloads" / github["asset"], record["archive_sha256"])
    with tempfile.TemporaryDirectory(dir=model_root, prefix=f".{artifact_id}-") as temporary:
        staging = Path(temporary) / "payload"
        safe_extract_zip(archive, staging)
        verify_payload(staging, record["files"])
        if destination.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = destination.with_name(destination.name + f".replaced-{stamp}")
            destination.replace(backup)
        staging.replace(destination)
    verify_payload(destination, record["files"])
    return {"artifact_id": artifact_id, "status": "installed", "path": str(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download checksummed model packages from GitHub Releases")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path,
                        help="Archive cache; defaults to the model root parent")
    parser.add_argument("--artifact", action="append", help="Install one artifact ID; repeatable")
    parser.add_argument("--model", action="append", help="Install all artifacts for a model ID; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall after verification; the previous directory is retained with a timestamp")
    args = parser.parse_args()
    document = load_json(args.project_root / "configs" / "artifacts.json")
    records = select(document["artifacts"], args.artifact, args.model)
    if not records:
        raise SystemExit("no ready artifacts matched the selection")
    args.model_root.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or args.model_root.parent).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for record in records:
        try:
            result = install(record, args.model_root, work_dir, force=args.force)
        except Exception as exc:
            failed = True
            result = {"artifact_id": record["artifact_id"], "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = work_dir / "model-installation.json"
    summary.write_text(json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(summary)
    return int(failed)


raise SystemExit(main())

