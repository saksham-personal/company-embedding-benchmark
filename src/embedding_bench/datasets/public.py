from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import load_dataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_beir(source: str | Path, output: str | Path, *, dataset_id: str,
                license_name: str, source_revision: str) -> Path:
    """Convert an unpacked BEIR dataset into the benchmark schema.

    The source directory must contain corpus.jsonl, queries.jsonl, and
    qrels/test.tsv. No network calls occur here.
    """
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    companies = []
    for row in _read_jsonl(source / "corpus.jsonl"):
        doc_id = str(row["_id"])
        title, body = str(row.get("title", "")).strip(), str(row.get("text", "")).strip()
        description = " ".join(value for value in (title, body) if value)
        companies.append({
            "company_id": doc_id, "track": "public_retrieval", "name": title or doc_id,
            "description": description, "keywords": [],
            "fields": {"core_business": description, "products_services": description,
                       "customers_end_markets": "", "capabilities": ""},
            "attributes": {}, "source_refs": [],
        })
    qrels_source = source / "qrels" / "test.tsv"
    qrel_rows = []
    with qrels_source.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle, delimiter="\t")
        for row in reader:
            qrel_rows.append({
                "query_id": str(row.get("query-id") or row.get("query_id")),
                "company_id": str(row.get("corpus-id") or row.get("corpus_id")),
                "relevance": str(row.get("score") or row.get("relevance")),
            })
    evaluated_query_ids = {row["query_id"] for row in qrel_rows}
    queries = [{
        "query_id": str(row["_id"]), "text": str(row["text"]),
        "category": "public_retrieval", "constraints": {}, "source_refs": [],
        "judgement_scope": "official_qrels"
    } for row in _read_jsonl(source / "queries.jsonl") if str(row["_id"]) in evaluated_query_ids]
    (output / "companies.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in companies), encoding="utf-8")
    (output / "queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queries), encoding="utf-8")
    with (output / "qrels.tsv").open("w", encoding="utf-8", newline="") as output_handle:
        output_handle.write("query_id\tcompany_id\trelevance\tjudgement_source\trationale_id\n")
        for row in qrel_rows:
            output_handle.write(
                f"{row['query_id']}\t{row['company_id']}\t{row['relevance']}\tofficial_qrels\t\n"
            )
    (output / "hard_negatives.jsonl").write_text("", encoding="utf-8")
    (output / "sources.jsonl").write_text("", encoding="utf-8")
    names = ["companies.jsonl", "queries.jsonl", "qrels.tsv", "hard_negatives.jsonl", "sources.jsonl"]
    manifest = {
        "schema_version": 1, "dataset_id": dataset_id, "track": "public_retrieval",
        "license": license_name, "source_revision": source_revision,
        "judgement_completeness": "official_qrels",
        "company_count": len(companies), "query_count": len(queries),
        "files": {name: _hash(output / name) for name in names},
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_dataset(output)
    return output

