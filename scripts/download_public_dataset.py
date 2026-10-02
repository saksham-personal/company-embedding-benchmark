from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from embedding_bench.datasets.public import import_beir
from embedding_bench.downloads import download_verified, safe_extract_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a pinned public dataset using GitHub only")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.project_root / "configs" / "public_datasets.json").read_text(encoding="utf-8"))
    records = {row["dataset_id"]: row for row in manifest["datasets"]}
    if args.dataset_id not in records:
        raise SystemExit(f"unknown public dataset: {args.dataset_id}")
    record = records[args.dataset_id]
    github = record["github"]
    url = f"https://github.com/{github['repository']}/releases/download/{github['release']}/{github['asset']}"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir, prefix="dataset-") as temporary:
        temporary = Path(temporary)
        archive = download_verified(url, temporary / github["asset"], record["archive_sha256"])
        extracted = safe_extract_zip(archive, temporary / "extracted")
        source = extracted / record.get("extracted_subdir", "")
        license_info = record["license"]
        license_name = json.dumps(license_info, sort_keys=True) if isinstance(license_info, dict) else str(license_info)
        import_beir(source, args.output, dataset_id=record["dataset_id"],
                    license_name=license_name, source_revision=record["source"]["revision"])
        from embedding_bench.datasets import load_dataset
        installed = load_dataset(args.output)
        if len(installed.companies) != record["expected_corpus_count"] or \
                len(installed.queries) != record["expected_query_count"]:
            raise SystemExit("installed public dataset count does not match pinned manifest")
    print(args.output)
    return 0


raise SystemExit(main())

