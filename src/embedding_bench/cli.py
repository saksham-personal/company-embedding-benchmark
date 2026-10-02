from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from embedding_bench.adapters import load_adapter
from embedding_bench.benchmark.provenance import provenance, write_result
from embedding_bench.benchmark.quality import instruction_for_model, run_quality
from embedding_bench.benchmark.speed import run_speed
from embedding_bench.benchmark.validation import check_receipt
from embedding_bench.config import Catalog
from embedding_bench.datasets import build_controlled_dataset, load_dataset
from embedding_bench.datasets.public import import_beir
from embedding_bench.downloads import verify_payload
from embedding_bench.hardware import inspect_machine
from embedding_bench.reporting import generate_report


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _common_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--llama-server", type=Path,
                        help="Pinned official llama-server executable for GGUF variants")
    parser.add_argument("--machine-role", choices=("development", "vdi"), default="development")
    parser.add_argument("--validation-file", type=Path, help="Receipt from scripts/validate_artifacts.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedding-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    machine = sub.add_parser("inspect-machine")
    machine.add_argument("--output", type=Path, default=Path("results/raw/hardware.json"))
    controlled = sub.add_parser("build-controlled")
    controlled.add_argument("--output", type=Path, default=Path("data/processed/controlled-v1"))
    controlled.add_argument("--companies", type=int, default=2400)
    controlled.add_argument("--queries-per-profile", type=int, default=10)
    controlled.add_argument("--seed", type=int, default=20261002)
    controlled.add_argument("--description-words", type=int,
                            help="Pad descriptions to a deterministic approximate word length")
    public = sub.add_parser("import-beir")
    public.add_argument("--source", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    public.add_argument("--dataset-id", required=True)
    public.add_argument("--license", required=True)
    public.add_argument("--source-revision", required=True)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("dataset", type=Path)
    quality = sub.add_parser("quality")
    _common_runtime(quality)
    quality.add_argument("--dataset", type=Path, required=True)
    quality.add_argument("--representation", choices=("description_only", "description_keywords", "one_vector", "one_vector_fields_rich", "fields_max", "fields_weighted"), default="description_keywords")
    quality.add_argument("--batch-size", type=int, default=8, help="Document batch size; queries always use 1")
    quality.add_argument("--instruction")
    quality.add_argument("--output", type=Path)
    speed = sub.add_parser("speed")
    _common_runtime(speed)
    speed.add_argument("--dataset", type=Path, required=True)
    speed.add_argument("--rounds", type=int, default=20)
    speed.add_argument("--batch-size", type=int, action="append", help="Document batch sizes; repeatable")
    speed.add_argument("--instruction")
    speed.add_argument("--output", type=Path)
    report = sub.add_parser("report")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--reports", type=Path, default=Path("reports"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = getattr(args, "project_root", project_root()).resolve()
    if args.command == "inspect-machine":
        output = args.output if args.output.is_absolute() else root / args.output
        write_result(output, inspect_machine())
        print(output)
        return 0
    if args.command == "build-controlled":
        output = args.output if args.output.is_absolute() else root / args.output
        print(build_controlled_dataset(output, company_count=args.companies,
                                       queries_per_profile=args.queries_per_profile, seed=args.seed,
                                       description_words=args.description_words))
        return 0
    if args.command == "import-beir":
        print(import_beir(args.source, args.output, dataset_id=args.dataset_id,
                          license_name=args.license, source_revision=args.source_revision))
        return 0
    if args.command == "validate-dataset":
        dataset = load_dataset(args.dataset)
        print(json.dumps({"dataset_id": dataset.manifest["dataset_id"],
                          "companies": len(dataset.companies), "queries": len(dataset.queries)}, indent=2))
        return 0
    if args.command in {"quality", "speed"}:
        if args.threads < 1:
            raise SystemExit("threads must be positive")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(args.threads)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_HOME", str(args.model_root.parent / "cache" / "huggingface"))
        os.environ.setdefault("HF_MODULES_CACHE", str(args.model_root.parent / "cache" / "modules"))
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.threads)
        if args.llama_server:
            os.environ["LLAMA_SERVER"] = str(args.llama_server.resolve())
        catalog = Catalog.load(root / "configs")
        dataset = load_dataset(args.dataset)
        variant = catalog.variants[args.variant]
        instruction = instruction_for_model(catalog.models[variant["model_id"]], dataset, args.instruction)
        run_provenance = provenance(project_root=root, machine_role=args.machine_role, command=sys.argv,
                                    catalog_digest=catalog.digest, dataset_manifest=dataset.manifest)
        validation = None
        if args.validation_file:
            receipt = json.loads(args.validation_file.read_text(encoding="utf-8"))
            validation = check_receipt(receipt, catalog=catalog, project_root=root, dataset=dataset,
                machine_role=args.machine_role, hardware_id=run_provenance["hardware_id"], threads=args.threads,
                variant_id=args.variant, instruction=instruction,
                document_batch_size=args.batch_size if args.command == "quality" else None)
        artifact = catalog.artifacts[variant["artifact_id"]]
        verify_payload(args.model_root / variant["artifact_id"], artifact["files"])
        load_started = time.perf_counter()
        adapter = load_adapter(catalog, args.variant, args.model_root, args.threads)
        cold_load_seconds = time.perf_counter() - load_started
        if args.command == "quality":
            result = run_quality(adapter, dataset, representation=args.representation,
                                 batch_size=args.batch_size, instruction=instruction)
            default = root / "results" / "quality" / f"{args.variant}__{dataset.manifest['dataset_id']}__{args.representation}.json"
        else:
            texts = [row["description"] for row in dataset.companies[:256]]
            query_texts = [row["text"] for row in dataset.queries[:64]]
            result = run_speed(adapter, texts, query_texts,
                               batch_sizes=tuple(args.batch_size or (1, 8, 32, 64)),
                               measurement_rounds=args.rounds, cold_load_seconds=cold_load_seconds,
                               instruction=instruction)
            default = root / "results" / "performance" / f"{args.variant}.json"
        result["provenance"] = run_provenance
        result["validation"] = validation or {"status": "not_run", "scope": "exploratory row; reference conformance not established"}
        output = args.output or default
        write_result(output, result)
        print(output)
        if hasattr(adapter, "close"):
            adapter.close()
        return 0
    if args.command == "report":
        markdown, html_path = generate_report(args.results, args.reports)
        print(markdown)
        print(html_path)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

