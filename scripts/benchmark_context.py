from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from embedding_bench.adapters import load_adapter
from embedding_bench.benchmark.provenance import provenance, write_result
from embedding_bench.benchmark.quality import run_quality
from embedding_bench.config import Catalog, canonical_digest
from embedding_bench.datasets import load_dataset
from embedding_bench.downloads import verify_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare whole record and field retrieval across context datasets")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        help="Repeat for short, medium, long, and very-long dataset views")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--machine-role", choices=("development", "vdi"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--llama-server", type=Path)
    args = parser.parse_args()
    catalog = Catalog.load(args.project_root / "configs")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.threads)
    variant = catalog.variants[args.variant]
    artifact = catalog.artifacts[variant["artifact_id"]]
    verify_payload(args.model_root / variant["artifact_id"], artifact["files"])
    if args.llama_server:
        os.environ["LLAMA_SERVER"] = str(args.llama_server.resolve())
    adapter = load_adapter(catalog, args.variant, args.model_root, args.threads)
    rows = []
    for dataset_path in args.dataset:
        dataset = load_dataset(dataset_path)
        for representation in ("one_vector", "fields_max"):
            measured = run_quality(adapter, dataset, representation=representation, batch_size=args.batch_size)
            rows.append({"dataset_id": dataset.manifest["dataset_id"],
                         "representation": representation,
                         "dataset_digest": canonical_digest(dataset.manifest),
                         "metrics": measured["metrics"]["macro"]})
    result = {"status": "measured", "experiment": "context", "variant_id": args.variant,
              "validation": {"status": "not_run", "scope": "exploratory context stress"},
              "runtime": getattr(adapter, "runtime", {}),
              "rows": rows, "provenance": provenance(
                  project_root=args.project_root, machine_role=args.machine_role,
                  command=sys.argv, catalog_digest=catalog.digest),
              "limitations": ["padding appends neutral text after the relevant facts; this tests compute/truncation costs but not evidence buried late in a document"]}
    output = args.output or args.project_root / "results" / "comparisons" / f"{args.variant}__context.json"
    write_result(output, result)
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))
    if hasattr(adapter, "close"):
        adapter.close()
    return 0


raise SystemExit(main())

