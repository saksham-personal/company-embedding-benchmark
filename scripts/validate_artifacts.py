"""Validate local artifacts against same-family references in isolated CPU processes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from embedding_bench.adapters import UnsupportedHardware, load_adapter
from embedding_bench.benchmark.provenance import provenance, source_digest, write_result
from embedding_bench.benchmark.quality import compare_embeddings, effective_instruction, instruction_for_model
from embedding_bench.config import Catalog, canonical_digest
from embedding_bench.datasets import load_dataset
from embedding_bench.downloads import verify_payload

SENTINEL = "@@ARTIFACT_VALIDATION@@"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--machine-role", choices=("development", "vdi"), default="development")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def reference_for(catalog, variant_id):
    variant = catalog.variants[variant_id]
    matches = [key for key, row in catalog.variants.items()
               if row["model_id"] == variant["model_id"] and row["dimensions"] == variant["dimensions"]
               and row["backend"] == "transformers" and row["weight_quantization"] == "none"]
    if not matches:
        raise ValueError(f"no same-family, same-dimension reference for {variant_id}")
    return matches[0]


def export_one(args, catalog, dataset, variant_id, fingerprint):
    adapter = None
    try:
        variant = catalog.variants[variant_id]
        artifact = catalog.artifacts[variant["artifact_id"]]
        # Hash payloads before executing any packaged custom model code.
        verify_payload(args.model_root / variant["artifact_id"], artifact["files"])
        adapter = load_adapter(catalog, variant_id, args.model_root, args.threads)
        indices = np.linspace(0, len(dataset.companies) - 1, min(100, len(dataset.companies)), dtype=int)
        documents = [dataset.companies[index]["description"] for index in indices]
        queries = [row["text"] for row in dataset.queries[:100]]
        instruction = effective_instruction(adapter, dataset, args.instruction)
        doc_vectors = adapter.encode_documents(documents, batch_size=args.batch_size)
        query_vectors = adapter.encode_queries(queries, batch_size=1, instruction=instruction)
        doc_repeat = adapter.encode_documents(documents[:args.batch_size], batch_size=args.batch_size)
        query_repeat = adapter.encode_queries(queries[:args.batch_size], batch_size=1, instruction=instruction)
        deterministic = bool(np.allclose(doc_vectors[:len(doc_repeat)], doc_repeat, atol=1e-6, rtol=1e-5)
                             and np.allclose(query_vectors[:len(query_repeat)], query_repeat, atol=1e-6, rtol=1e-5))
        doc_small_batch = adapter.encode_documents(documents[:2], batch_size=2)
        query_small_batch = adapter.encode_queries(queries[:2], batch_size=2, instruction=instruction)
        padding_cosine_min = float(min(np.sum(doc_vectors[:2] * doc_small_batch, axis=1).min(),
                                       np.sum(query_vectors[:2] * query_small_batch, axis=1).min()))
        if not deterministic:
            raise ValueError("same-batch embedding determinism failed")
        quantized = variant["weight_quantization"] != "none"
        if not quantized and padding_cosine_min < .999:
            raise ValueError(f"batch-padding consistency cosine below .999: {padding_cosine_min}")
        np.savez(args.output_dir / f"{variant_id}.npz", documents=doc_vectors, queries=query_vectors)
        return {"variant_id": variant_id, "status": "passed", "fingerprint": fingerprint,
                "deterministic": deterministic, "document_count": len(documents), "query_count": len(queries),
                "padding_consistency_cosine_min": padding_cosine_min,
                "padding_gate": "diagnostic_only_for_quantized" if quantized else "cosine_at_least_0.999",
                "batch_variation_warning": bool(quantized and padding_cosine_min < .999),
                "document_batch_size": args.batch_size, "query_batch_size": 1,
                "dimension": adapter.dimension, "max_tokens": adapter.max_tokens,
                "instruction": instruction, "runtime": getattr(adapter, "runtime", {})}
    except UnsupportedHardware as exc:
        return {"variant_id": variant_id, "status": "skipped", "reason": "unsupported_hardware", "error": str(exc)}
    except Exception as exc:
        return {"variant_id": variant_id, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(args.project_root / "configs")
    dataset = load_dataset(args.dataset)
    run_provenance = provenance(project_root=args.project_root, machine_role=args.machine_role,
        command=sys.argv, catalog_digest=catalog.digest, dataset_manifest=dataset.manifest)
    fingerprint = canonical_digest({"catalog": catalog.digest, "dataset": dataset.manifest,
        "source": source_digest(args.project_root), "threads": args.threads,
        "document_batch_size": args.batch_size, "query_batch_size": 1,
        "machine_role": args.machine_role, "hardware_id": run_provenance["hardware_id"],
        "instructions": {key: instruction_for_model(model, dataset, args.instruction)
                         for key, model in catalog.models.items()}})
    requested = args.variant or [key for key, row in catalog.variants.items() if row.get("enabled", True)]
    if set(requested) - set(catalog.variants):
        raise SystemExit("unknown variant selection")
    if args.child:
        if len(requested) != 1:
            raise SystemExit("child mode requires exactly one variant")
        if args.llama_server:
            os.environ["LLAMA_SERVER"] = str(args.llama_server.resolve())
        result = export_one(args, catalog, dataset, requested[0], fingerprint)
        print(SENTINEL + json.dumps(result), flush=True)
        return int(result["status"] == "failed")
    references = {variant_id: reference_for(catalog, variant_id) for variant_id in requested}
    needed = list(dict.fromkeys([*references.values(), *requested]))
    exports = {}
    for variant_id in needed:
        cached = args.output_dir / f"{variant_id}.export.json"
        if cached.exists() and not args.force:
            row = json.loads(cached.read_text(encoding="utf-8"))
            if row.get("status") == "passed" and row.get("fingerprint") == fingerprint:
                if (args.output_dir / f"{variant_id}.npz").is_file():
                    artifact = catalog.artifacts[catalog.variants[variant_id]["artifact_id"]]
                    verify_payload(args.model_root / artifact["artifact_id"], artifact["files"])
                    exports[variant_id] = row
                    print(json.dumps({"variant_id": variant_id, "status": "cached_pass"}), flush=True)
                    continue
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--variant", variant_id,
                   "--project-root", str(args.project_root), "--model-root", str(args.model_root),
                   "--dataset", str(args.dataset), "--output-dir", str(args.output_dir),
                   "--threads", str(args.threads), "--batch-size", str(args.batch_size),
                   "--machine-role", args.machine_role]
        if args.llama_server:
            command += ["--llama-server", str(args.llama_server)]
        if args.instruction:
            command += ["--instruction", args.instruction]
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                            "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                            "OPENBLAS_NUM_THREADS": str(args.threads),
                            "HF_HOME": str(args.model_root.parent / "cache" / "huggingface"),
                            "HF_MODULES_CACHE": str(args.model_root.parent / "cache" / "modules")})
        print(f"Validating {variant_id}", flush=True)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=1800)
            line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(SENTINEL)), None)
            row = json.loads(line[len(SENTINEL):]) if line else {
                "variant_id": variant_id, "status": "failed", "error_type": "ChildProcessError",
                "exit_code": completed.returncode, "error": (completed.stderr or completed.stdout)[-2000:]}
        except subprocess.TimeoutExpired:
            row = {"variant_id": variant_id, "status": "failed", "error_type": "Timeout", "error": "validation exceeded 1800 seconds"}
        exports[variant_id] = row
        write_result(cached, row)
        print(json.dumps(row), flush=True)
    rows = []
    for variant_id in requested:
        reference_id = references[variant_id]
        row = dict(exports[variant_id], reference=reference_id)
        if row["status"] == "passed" and exports[reference_id]["status"] != "passed":
            row.update(status="failed", error="reference did not pass; comparison unavailable")
        elif row["status"] == "passed":
            with np.load(args.output_dir / f"{variant_id}.npz") as candidate, \
                    np.load(args.output_dir / f"{reference_id}.npz") as reference:
                is_fp32 = catalog.variants[variant_id]["weight_quantization"] == "none"
                limits = {"p5_threshold": .99 if is_fp32 else .90, "minimum_threshold": .98 if is_fp32 else .80}
                row["document_similarity"] = compare_embeddings(reference["documents"], candidate["documents"], **limits)
                row["query_similarity"] = compare_embeddings(reference["queries"], candidate["queries"], **limits)
                if not row["document_similarity"]["passed"] or not row["query_similarity"]["passed"]:
                    row.update(status="failed", error="reference similarity gate failed")
        row["scope"] = "sampled short descriptions and screening queries; full-length behavior requires context experiments"
        rows.append(row)
    output = args.output_dir / "artifact-validation.json"
    write_result(output, {"status": "measured", "kind": "artifact_validation", "fingerprint": fingerprint,
                          "variants": rows, "provenance": run_provenance})
    print(output, flush=True)
    return int(any(row["status"] == "failed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
