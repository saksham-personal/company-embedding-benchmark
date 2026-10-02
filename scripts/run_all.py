from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from embedding_bench.benchmark.provenance import provenance, write_result
from embedding_bench.config import Catalog
from embedding_bench.datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and run variants serially in fresh CPU processes")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--machine-role", choices=("development", "vdi"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--variant", action="append", help="Repeat to select specific variants")
    parser.add_argument("--representation", action="append", choices=("description_only", "description_keywords",
                        "one_vector_fields_rich", "fields_max", "fields_weighted"))
    parser.add_argument("--batch-size", type=int, default=8, help="Quality/validation batch size")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-file", type=Path, help="Reuse an existing matching validation receipt")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Shorter performance measurements; quality uses the entire supplied dataset")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root / "results"
    output.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(root / "configs")
    dataset = load_dataset(args.dataset)
    selected = args.variant or [key for key, row in catalog.variants.items() if row.get("enabled", True)]
    if set(selected) - set(catalog.variants):
        raise SystemExit("unknown variants in selection")
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                        "OPENBLAS_NUM_THREADS": str(args.threads),
                        "HF_HOME": str(args.model_root.parent / "cache" / "huggingface"),
                        "HF_MODULES_CACHE": str(args.model_root.parent / "cache" / "modules")})
    receipt_path = args.validation_file or output / "validation" / "artifact-validation.json"
    if not args.validation_file:
        command = [sys.executable, str(root / "scripts" / "validate_artifacts.py"),
                   "--project-root", str(root), "--model-root", str(args.model_root), "--dataset", str(args.dataset),
                   "--output-dir", str(receipt_path.parent), "--threads", str(args.threads),
                   "--batch-size", str(args.batch_size), "--machine-role", args.machine_role]
        for variant in selected:
            command.extend(["--variant", variant])
        if args.llama_server:
            command.extend(["--llama-server", str(args.llama_server)])
        subprocess.run(command, cwd=root, env=environment, check=False)
    if not receipt_path.is_file():
        raise SystemExit("reference-validation receipt was not created")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validated = {row["variant_id"]: row for row in receipt["variants"]}
    outcomes = []
    for variant_id in selected:
        check = validated.get(variant_id, {"status": "failed", "error": "no validation record"})
        if check["status"] != "passed":
            skipped = {"variant_id": variant_id, "status": check["status"], "kind": "benchmark_not_run",
                       "error": check.get("error"), "reason": check.get("reason", "failed_validation")}
            write_result(output / "raw" / f"{variant_id}__not_run.json", skipped)
            outcomes.append(skipped)
            continue
        common = ["--project-root", str(root), "--model-root", str(args.model_root), "--variant", variant_id,
                  "--dataset", str(args.dataset), "--machine-role", args.machine_role,
                  "--threads", str(args.threads), "--validation-file", str(receipt_path)]
        if args.llama_server:
            common.extend(["--llama-server", str(args.llama_server)])
        representations = args.representation or (("description_keywords",) if args.smoke else (
            "description_only", "description_keywords", "one_vector_fields_rich", "fields_max", "fields_weighted"))
        if dataset.manifest.get("track") == "public_retrieval":
            representations = ("description_only",)
        for representation in representations:
            path = output / "quality" / f"{variant_id}__{dataset.manifest['dataset_id']}__{representation}.json"
            command = [sys.executable, "-m", "embedding_bench.cli", "quality", *common,
                       "--representation", representation, "--batch-size", str(args.batch_size), "--output", str(path)]
            completed = subprocess.run(command, cwd=root, env=environment, check=False)
            outcomes.append({"variant_id": variant_id, "experiment": "quality", "representation": representation,
                             "status": "measured" if completed.returncode == 0 else "failed", "path": str(path)})
        if not args.quality_only:
            path = output / "performance" / f"{variant_id}__{dataset.manifest['dataset_id']}__{args.threads}threads.json"
            command = [sys.executable, "-m", "embedding_bench.cli", "speed", *common, "--output", str(path),
                       "--rounds", "3" if args.smoke else "20"]
            completed = subprocess.run(command, cwd=root, env=environment, check=False)
            outcomes.append({"variant_id": variant_id, "experiment": "performance",
                             "status": "measured" if completed.returncode == 0 else "failed", "path": str(path)})
    summary = {"status": "measured", "kind": "run_summary", "smoke": args.smoke,
               "selected_all_enabled": set(selected) == {key for key, row in catalog.variants.items() if row.get("enabled", True)},
               "all_selected_measured": all(row["status"] == "measured" for row in outcomes),
               "outcomes": outcomes, "provenance": provenance(project_root=root, machine_role=args.machine_role,
                   command=sys.argv, catalog_digest=catalog.digest, dataset_manifest=dataset.manifest)}
    write_result(output / "raw" / f"run_summary__{dataset.manifest['dataset_id']}.json", summary)
    subprocess.run([sys.executable, "-m", "embedding_bench.cli", "report", "--results", str(output),
                    "--reports", str(root / "reports")], cwd=root, env=environment, check=True)
    return int(any(row["status"] == "failed" for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())

