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
    parser = argparse.ArgumentParser(description="Paired artifact validation and retrieval deltas in separate CPU processes")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--representation", choices=("description_only", "description_keywords", "one_vector_fields_rich",
                        "fields_max", "fields_weighted"), default="description_keywords")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--machine-role", choices=("development", "vdi"), default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    catalog = Catalog.load(root / "configs")
    dataset = load_dataset(args.dataset)
    ref, cand = catalog.variants[args.reference], catalog.variants[args.candidate]
    if ref["model_id"] != cand["model_id"] or ref["dimensions"] != cand["dimensions"]:
        raise SystemExit("reference and candidate must have the same family and output dimensions")
    if ref["backend"] != "transformers" or ref["weight_quantization"] != "none":
        raise SystemExit("reference must be the native unquantized Transformers row")
    output = args.output or root / "results" / "comparisons" / f"{args.candidate}__quantization.json"
    receipt_path = args.validation_file or output.parent / (args.candidate + "__validation") / "artifact-validation.json"
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                        "OPENBLAS_NUM_THREADS": str(args.threads)})
    common = ["--project-root", str(root), "--model-root", str(args.model_root), "--dataset", str(args.dataset),
              "--threads", str(args.threads), "--machine-role", args.machine_role]
    if args.llama_server:
        common.extend(["--llama-server", str(args.llama_server)])
    if not args.validation_file:
        receipt_path.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(root / "scripts" / "validate_artifacts.py"), *common,
            "--variant", args.reference, "--variant", args.candidate, "--output-dir", str(receipt_path.parent),
            "--batch-size", str(args.batch_size)], env=environment, check=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {row["variant_id"]: row for row in receipt["variants"]}
    check = checks.get(args.candidate, {"status": "failed", "error": "no validation record"})
    if check["status"] == "passed" and (checks.get(args.reference, {}).get("status") != "passed"
                                        or check.get("reference") != args.reference):
        raise SystemExit("receipt does not contain the requested passed native reference pair")
    result = {"status": "measured" if check["status"] == "passed" else check["status"],
              "experiment": "weight_quantization", "reference": args.reference, "candidate": args.candidate,
              "representation": args.representation, "error": check.get("error"),
              "embedding_validation": {"passed": check["status"] == "passed",
                  "cosine": check.get("document_similarity", {}).get("cosine", {}),
                  "query_cosine": check.get("query_similarity", {}).get("cosine", {})}}
    if result["status"] == "measured":
        measured = {}
        for variant in (args.reference, args.candidate):
            path = output.parent / f"{variant}__paired_quality__{args.representation}.json"
            command = [sys.executable, "-m", "embedding_bench.cli", "quality", *common,
                "--variant", variant, "--representation", args.representation, "--batch-size", str(args.batch_size),
                "--validation-file", str(receipt_path), "--output", str(path)]
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode != 0:
                result.update(status="failed", error=f"quality command failed for {variant}")
                break
            measured[variant] = json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro"]
        if result["status"] == "measured":
            keys = ("recall@25", "recall@50", "recall@100", "ndcg@20", "mrr@10")
            delta = {key: measured[args.candidate][key] - measured[args.reference][key] for key in keys}
            result["quality"] = {"reference": measured[args.reference], "candidate": measured[args.candidate],
                                 "candidate_minus_reference": delta}
            result["quality_delta_gate"] = all(delta[key] >= -.01 for key in ("recall@50", "recall@100", "ndcg@20"))
            result["effectively_lossless"] = None
            result["interpretation"] = "Quality deltas only; a measured speed/storage benefit is also required."
    result["provenance"] = provenance(project_root=root, machine_role=args.machine_role, command=sys.argv,
        catalog_digest=catalog.digest, dataset_manifest=dataset.manifest)
    write_result(output, result)
    print(json.dumps({"output": str(output), "status": result["status"]}), flush=True)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

