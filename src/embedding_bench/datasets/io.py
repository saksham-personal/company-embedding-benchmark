from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkDataset:
    manifest: dict[str, Any]
    companies: tuple[dict[str, Any], ...]
    queries: tuple[dict[str, Any], ...]
    qrels: dict[str, dict[str, int]]
    hard_negatives: dict[str, frozenset[str]]
    root: Path

    @property
    def is_complete(self) -> bool:
        return self.manifest.get("judgement_completeness") in {"exhaustive", "complete"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(root: str | Path, *, verify_hashes: bool = True) -> BenchmarkDataset:
    root = Path(root)
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise DatasetError("dataset.json must use schema_version 1")
    if verify_hashes:
        hashes = manifest.get("files") or manifest.get("file_hashes") or {}
        for name, expected in hashes.items():
            actual = _sha256(root / name)
            if actual != expected:
                raise DatasetError(f"checksum mismatch for {name}: {actual} != {expected}")
    companies = _jsonl(root / "companies.jsonl")
    queries = _jsonl(root / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query_id", "company_id", "relevance"}
        if not required.issubset(reader.fieldnames or []):
            raise DatasetError(f"qrels.tsv needs columns {sorted(required)}")
        for row in reader:
            qrels.setdefault(row["query_id"], {})[row["company_id"]] = int(row["relevance"])
    hard: dict[str, set[str]] = {}
    hard_path = root / "hard_negatives.jsonl"
    if hard_path.exists():
        for row in _jsonl(hard_path):
            hard.setdefault(row["query_id"], set()).add(row["company_id"])
    dataset = BenchmarkDataset(
        manifest, tuple(companies), tuple(queries), qrels,
        {key: frozenset(value) for key, value in hard.items()}, root
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: BenchmarkDataset) -> None:
    company_ids = [row.get("company_id") for row in dataset.companies]
    query_ids = [row.get("query_id") for row in dataset.queries]
    if None in company_ids or len(company_ids) != len(set(company_ids)):
        raise DatasetError("company_id values must be present and unique")
    if None in query_ids or len(query_ids) != len(set(query_ids)):
        raise DatasetError("query_id values must be present and unique")
    company_set, query_set = set(company_ids), set(query_ids)
    for row in dataset.companies:
        if not str(row.get("description", "")).strip():
            raise DatasetError(f"company {row.get('company_id')} has no description")
        if not isinstance(row.get("fields"), dict):
            raise DatasetError(f"company {row.get('company_id')} has no fields")
    for row in dataset.queries:
        if not str(row.get("text", "")).strip():
            raise DatasetError(f"query {row.get('query_id')} has no text")
    for query_id, labels in dataset.qrels.items():
        if query_id not in query_set:
            raise DatasetError(f"qrels references unknown query {query_id}")
        if not labels or not any(value > 0 for value in labels.values()):
            raise DatasetError(f"query {query_id} has no positive judgement")
        unknown = set(labels) - company_set
        if unknown:
            raise DatasetError(f"qrels references unknown companies: {sorted(unknown)[:3]}")
        if any(value not in {0, 1, 2} for value in labels.values()):
            raise DatasetError(f"query {query_id} has invalid relevance")
    missing_qrels = query_set - set(dataset.qrels)
    if missing_qrels:
        raise DatasetError(f"queries without qrels: {sorted(missing_qrels)[:3]}")
    if dataset.manifest.get("track") == "source_backed":
        query_by_id = {row["query_id"]: row for row in dataset.queries}
        without_direct = [query_id for query_id, labels in dataset.qrels.items()
                          if query_by_id[query_id].get("strict_direct_match_recall_eligible", True)
                          and not any(value >= 2 for value in labels.values())]
        if without_direct:
            raise DatasetError(f"source-backed queries without strict direct matches: {without_direct[:3]}")
    for query_id, ids in dataset.hard_negatives.items():
        if query_id not in query_set or not ids.issubset(company_set):
            raise DatasetError(f"invalid hard-negative references for {query_id}")
        labelled_positive = {cid for cid, rel in dataset.qrels.get(query_id, {}).items() if rel > 0}
        if ids & labelled_positive:
            raise DatasetError(f"hard negative labelled positive for {query_id}")
    if dataset.manifest.get("track") == "controlled":
        companies = {row["company_id"]: row for row in dataset.companies}
        for query in dataset.queries:
            query_id = query["query_id"]
            for company_id, relevance in dataset.qrels[query_id].items():
                matches = satisfies_constraints(companies[company_id], query.get("constraints", {}))
                if (relevance > 0) != matches:
                    raise DatasetError(
                        f"controlled qrel disagrees with constraints: {query_id}/{company_id}"
                    )


def satisfies_constraints(company: dict[str, Any], constraints: dict[str, Any]) -> bool:
    attributes = company.get("attributes", {})
    for expected, should_match in ((constraints.get("must", []), True),
                                   (constraints.get("must_not", []), False)):
        for rule in expected:
            value = rule.get("value")
            actual = attributes.get(rule.get("field"))
            if rule.get("op") != "contains":
                raise DatasetError(f"unsupported controlled constraint operator: {rule.get('op')}")
            values = actual if isinstance(actual, list) else [actual]
            matched = value in values
            if matched != should_match:
                return False
    return True


def render_company(company: dict[str, Any], representation: str = "one_vector") -> list[tuple[str, str]]:
    """Return (field name, text) pairs for a company representation."""
    if representation == "description_only":
        return [("description", company["description"])]
    if representation == "description_keywords":
        keywords = ", ".join(str(value) for value in company.get("keywords", []) if value)
        text = company["description"] + (f" Keywords: {keywords}." if keywords else "")
        return [("description_keywords", text)]
    if representation in {"one_vector", "one_vector_fields_rich"}:
        fields = company["fields"]
        parts = [company["description"]]
        for key in ("products_services", "customers_end_markets", "capabilities"):
            value = fields.get(key)
            if value and value not in parts:
                parts.append(value)
        return [("company", " ".join(parts))]
    if representation in {"fields_max", "fields_weighted"}:
        order = ("core_business", "products_services", "customers_end_markets", "capabilities")
        return [(name, company["fields"][name]) for name in order if company["fields"].get(name)]
    raise DatasetError(f"unknown representation: {representation}")

