from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") in {"measured", "skipped", "failed"}:
            value["_path"] = str(path)
            rows.append(value)
    return rows


def _quality_rows(results):
    rows = []
    for result in results:
        macro = result.get("metrics", {}).get("macro")
        if result.get("status") != "measured" or not macro:
            continue
        prov = result.get("provenance", {})
        rows.append({
            "variant_id": result.get("variant_id"), "dataset_id": result.get("dataset_id"),
            "dataset_track": result.get("dataset_track"), "representation": result.get("representation"),
            "dimension": result.get("dimension"), "machine_role": prov.get("machine_role"),
            "validation": result.get("validation", {}).get("status", "not_run"),
            "recall_kind": ("strict known-positive" if result.get("dataset_track") == "source_backed"
                            else result["metrics"]["metric_semantics"].get("recall_name", "recall")
                            if "metric_semantics" in result["metrics"] else "recall"),
            "recall@25": macro.get("recall@25"), "recall@50": macro.get("recall@50"),
            "recall@100": macro.get("recall@100"), "candidate_recall@50": macro.get("candidate_recall@50"),
            "ndcg@10": macro.get("ndcg@10"), "ndcg@20": macro.get("ndcg@20"), "mrr@10": macro.get("mrr@10"),
            "hard_negative_share@25": macro.get("hard_negative_share@25"),
            "recall_query_count": result["metrics"].get("metric_query_counts", {}).get("recall@50"),
            "query_count": result.get("query_count"), "company_count": result.get("company_count"),
            "dataset_digest": prov.get("dataset_digest"), "hardware_id": prov.get("hardware_id"),
            "catalog_digest": prov.get("catalog_digest"), "source_digest": prov.get("source_digest"),
            "threads": result.get("runtime", {}).get("threads"),
        })
    return rows


def _performance_rows(results):
    rows = []
    for result in results:
        if result.get("status") != "measured" or "document_batches" not in result:
            continue
        prov = result.get("provenance", {})
        query = (result.get("query_latency") or [{}])[0]
        for batch in result["document_batches"]:
            rows.append({
                "variant_id": result.get("variant_id"), "dataset_id": prov.get("dataset_id"),
                "machine_role": prov.get("machine_role"), "validation": result.get("validation", {}).get("status", "not_run"),
                "bucket": batch.get("bucket"), "threads": result.get("runtime", {}).get("threads"),
                "batch_size": batch["batch_size"], "rounds": batch["rounds"],
                "docs_per_second": batch["docs_per_second"], "batch_p50_ms": batch["latency_ms"]["p50"],
                "batch_p95_ms": batch["latency_ms"]["p95"], "query_p50_ms": query.get("latency_ms", {}).get("p50"),
                "query_p95_ms": query.get("latency_ms", {}).get("p95"), "cold_load_seconds": result.get("cold_load_seconds"),
                "sampled_peak_rss_bytes": batch.get("sampled_peak_rss_bytes", batch.get("rss_bytes_after")),
                "hardware_id": prov.get("hardware_id"), "dataset_digest": prov.get("dataset_digest"),
            })
    return rows


def _quantization_rows(results):
    rows = []
    for result in results:
        if result.get("status") != "measured" or result.get("experiment") != "weight_quantization" or "quality" not in result:
            continue
        delta = result["quality"]["candidate_minus_reference"]
        cosine = result["embedding_validation"].get("cosine", {})
        rows.append({
            "reference": result["reference"], "candidate": result["candidate"],
            "dataset_id": result.get("provenance", {}).get("dataset_id"),
            "machine_role": result.get("provenance", {}).get("machine_role"),
            "representation": result.get("representation"),
            "recall@50_delta": delta.get("recall@50"), "recall@100_delta": delta.get("recall@100"),
            "ndcg@20_delta": delta.get("ndcg@20"), "embedding_cosine_mean": cosine.get("mean"),
            "embedding_cosine_p5": cosine.get("p5"), "quality_delta_gate": result.get("quality_delta_gate"),
        })
    return rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(lines, rows, columns):
    lines.append("| " + " | ".join(label for _, label in columns) + " |\n")
    lines.append("|" + "|".join("---" for _ in columns) + "|\n")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value) if value is not None else "")
        lines.append("| " + " | ".join(value.replace("|", "/").replace("\n", " ") for value in values) + " |\n")


def generate_report(results_dir: str | Path, reports_dir: str | Path) -> tuple[Path, Path]:
    results_dir, reports_dir = Path(results_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(results_dir)
    quality, performance, quantization = _quality_rows(results), _performance_rows(results), _quantization_rows(results)
    _write_csv(results_dir / "quality" / "company_retrieval.csv", quality)
    _write_csv(results_dir / "performance" / "all_models.csv", performance)
    _write_csv(results_dir / "comparisons" / "quantization.csv", quantization)
    lines = ["# Embedding Benchmark Results\n\n",
             "> Generated only from measured result JSON. Tables retain machine, dataset and validation status.\n\n",
             "## Decision status\n\n",
             "A production model recommendation is deferred. An individual VDI row cannot establish a winner. "
             "Select a deployment only after complete, comparable VDI quality and performance runs, passed reference gates, "
             "and representative company judgements. This public company pilot alone is insufficient.\n\n"]
    lines.append("## Company and public retrieval quality\n\n")
    _table(lines, quality, [("variant_id", "Variant"), ("dataset_id", "Dataset"), ("representation", "Representation"),
        ("dimension", "Dim"), ("machine_role", "Machine"), ("validation", "Validation"), ("recall_kind", "Recall meaning"),
        ("recall@25", "R@25"), ("recall@50", "R@50"), ("recall@100", "R@100"),
        ("candidate_recall@50", "Candidate R@50"), ("ndcg@10", "nDCG@10"), ("ndcg@20", "nDCG@20")])
    lines.append("\nSource-backed R@K uses direct relevance-2 positives on eligible queries; candidate recall also includes "
                 "partial relevance-1 matches. Exclusion probes without strict positives are omitted from strict recall denominators. "
                 "Recall@100 on the 130-company pilot is especially weak evidence.\n\n")
    lines.append("## CPU performance\n\n")
    _table(lines, performance, [("variant_id", "Variant"), ("machine_role", "Machine"), ("bucket", "Token bucket"),
        ("threads", "Threads"), ("batch_size", "Batch"), ("rounds", "Rounds"), ("docs_per_second", "Docs/s"),
        ("query_p50_ms", "Query p50 ms"), ("query_p95_ms", "Query p95 ms"), ("sampled_peak_rss_bytes", "Sampled peak RSS")])
    lines.append("\nFresh-process load times do not flush the filesystem cache. Memory sums parent and server RSS at 20 ms "
                 "intervals. GGUF requests are serialized; batch rows measure groups of sequential requests. Short development "
                 "runs cannot establish stable target-VDI latency percentiles.\n\n")
    lines.append("## Paired artifact comparisons\n\n")
    _table(lines, quantization, [("reference", "Reference"), ("candidate", "Candidate"), ("dataset_id", "Dataset"),
        ("representation", "Representation"), ("recall@50_delta", "R@50 delta"), ("recall@100_delta", "R@100 delta"),
        ("ndcg@20_delta", "nDCG@20 delta"), ("embedding_cosine_p5", "Cosine p5"), ("quality_delta_gate", "Quality delta gate")])
    lines.append("\nThe one-point quality delta gate does not alone imply effectively lossless quantization. "
                 "It must also have a measured CPU/storage benefit and hold on representative data.\n\n")
    failures = []
    for result in results:
        if result["status"] in {"failed", "skipped"}:
            failures.append({"variant_id": result.get("variant_id", result.get("candidate")),
                             "status": result["status"], "error": result.get("error", result.get("reason"))})
        if result.get("kind") in {"artifact_validation", "model_verification"}:
            failures.extend(row for row in result.get("variants", []) if row.get("status") in {"failed", "skipped"})
    lines.append("## Failed or skipped checks\n\n")
    _table(lines, failures, [("variant_id", "Variant"), ("status", "Status"), ("error", "Reason")])
    lines.extend(["\n## Interpretation and reproduction\n\n",
        "Controlled data is synthetic and exhaustively labelled. Public company qrels are incomplete: missing pairs "
        "remain unknown; Precision/MAP are suppressed. nDCG follows pooled-qrel scoring with unjudged gains of zero. "
        "Only rows with matching dataset, representation, hardware, thread budget, catalog and source digests should be compared. "
        "Rows without a passed validation receipt are exploratory. Full raw JSON and CSV retain those identities.\n\n",
        "Follow [VDI_AGENT_GUIDE.md](VDI_AGENT_GUIDE.md) for the complete commands. From the repository run "
        "\u0060python scripts/run_all.py --model-root <MODEL_DIR> --dataset <DATASET_DIR> --machine-role vdi "
        "--threads 8 --llama-server <LLAMA_SERVER_EXE> --output-dir <RESULTS_DIR>\u0060, then "
        "\u0060python scripts/generate_report.py --results <RESULTS_DIR> --reports reports\u0060.\n"])
    markdown = reports_dir / "FINAL_BENCHMARK.md"
    markdown.write_text("".join(lines), encoding="utf-8")
    html_path = reports_dir / "FINAL_BENCHMARK.html"
    html_path.write_text("<!doctype html><meta charset='utf-8'><title>Embedding Benchmark</title>"
        "<style>body{font:15px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap}</style>"
        f"<pre>{html.escape(markdown.read_text(encoding='utf-8'))}</pre>", encoding="utf-8")
    return markdown, html_path

