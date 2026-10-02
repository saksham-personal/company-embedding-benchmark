from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from embedding_bench.adapters import UnsupportedHardware, load_adapter
from embedding_bench.benchmark.provenance import write_result
from embedding_bench.config import Catalog
from embedding_bench.downloads import verify_payload

SENTINEL = "@@EMBEDDING_BENCH_RESULT@@"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Offline model smoke validation in isolated processes")
    value.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--variant", action="append")
    value.add_argument("--threads", type=int, default=8)
    value.add_argument("--llama-server", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return value


def verify_one(args: argparse.Namespace, variant_id: str) -> dict[str, object]:
    if args.llama_server:
        os.environ["LLAMA_SERVER"] = str(args.llama_server.resolve())
    catalog = Catalog.load(args.project_root / "configs")
    try:
        artifact = catalog.artifacts[catalog.variants[variant_id]["artifact_id"]]
        verify_payload(args.model_root / artifact["artifact_id"], artifact["files"])
        adapter = load_adapter(catalog, variant_id, args.model_root, args.threads)
        texts = ["A manufacturer of HVAC-grade steel sheet.", "Fraud monitoring software for banks."]
        queries = adapter.encode_queries(texts, batch_size=2)
        documents = adapter.encode_documents(texts, batch_size=2)
        if queries.shape != documents.shape or queries.shape[1] != adapter.dimension:
            raise ValueError(f"unexpected shapes {queries.shape}, {documents.shape}")
        result: dict[str, object] = {"variant_id": variant_id, "status": "passed",
                                    "shape": list(queries.shape)}
        if hasattr(adapter, "close"):
            adapter.close()
        return result
    except UnsupportedHardware as exc:
        return {"variant_id": variant_id, "status": "skipped",
                "reason": "unsupported_hardware", "error": str(exc)}
    except Exception as exc:
        return {"variant_id": variant_id, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc)}


def main() -> int:
    args = parser().parse_args()
    catalog = Catalog.load(args.project_root / "configs")
    variants = args.variant or [key for key, row in catalog.variants.items() if row.get("enabled", True)]
    if args.child:
        if len(variants) != 1:
            raise SystemExit("child mode requires one variant")
        result = verify_one(args, variants[0])
        print(SENTINEL + json.dumps(result, separators=(",", ":")))
        return int(result["status"] == "failed")
    results = []
    for variant_id in variants:
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--project-root", str(args.project_root), "--model-root", str(args.model_root),
                   "--variant", variant_id, "--threads", str(args.threads), "--child"]
        if args.llama_server:
            command.extend(["--llama-server", str(args.llama_server)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(SENTINEL)), None)
        if line:
            result = json.loads(line[len(SENTINEL):])
        else:
            result = {"variant_id": variant_id, "status": "failed",
                      "error_type": "ChildProcessError",
                      "error": (completed.stderr or completed.stdout)[-2000:]}
        results.append(result)
        print(json.dumps(result), flush=True)
    output = args.output or args.project_root / "results" / "raw" / "model_verification.json"
    write_result(output, {"status": "measured", "kind": "model_verification", "variants": results})
    print(output)
    return int(any(row["status"] == "failed" for row in results))


raise SystemExit(main())

