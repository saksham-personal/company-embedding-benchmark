from __future__ import annotations

from pathlib import Path
from typing import Any

from embedding_bench.datasets.io import BenchmarkDataset, render_company
from embedding_bench.evaluation.metrics import evaluate_rankings
from embedding_bench.evaluation.search import rank_dataset


COMPANY_INSTRUCTION = ("Given a company screening criterion, retrieve company descriptions that match the "
                       "specified products, services, customer segments, and exclusions.")


def effective_instruction(adapter: Any, dataset: BenchmarkDataset, instruction: str | None) -> str | None:
    convention = adapter.model["text_convention"]["query"]
    if not convention.get("instruction_template"):
        return None
    if instruction:
        return instruction
    if dataset.manifest.get("track") in {"controlled", "source_backed", "private_company"}:
        return COMPANY_INSTRUCTION
    return convention.get("default_instruction")


def run_quality(adapter: Any, dataset: BenchmarkDataset, *, representation: str = "one_vector",
                batch_size: int = 32, top_k: int = 100,
                instruction: str | None = None) -> dict[str, Any]:
    query_ids = [row["query_id"] for row in dataset.queries]
    query_texts = [row["text"] for row in dataset.queries]
    field_texts: list[str] = []
    field_company_ids: list[str] = []
    field_names: list[str] = []
    for company in dataset.companies:
        for field_name, text in render_company(company, representation):
            field_texts.append(text)
            field_company_ids.append(company["company_id"])
            field_names.append(field_name)
    instruction = effective_instruction(adapter, dataset, instruction)
    query_vectors = adapter.encode_queries(query_texts, instruction=instruction, batch_size=batch_size)
    document_vectors = adapter.encode_documents(field_texts, batch_size=batch_size)
    aggregation = "weighted" if representation == "fields_weighted" else "max"
    rankings = rank_dataset(
        query_vectors, document_vectors, field_company_ids, field_names,
        top_k=top_k, aggregation=aggregation
    )
    metrics = evaluate_rankings(
        query_ids, rankings, dataset.qrels, complete_judgements=dataset.is_complete,
        hard_negatives=dataset.hard_negatives,
        categories={row["query_id"]: row.get("category", "uncategorized") for row in dataset.queries},
        dataset_track=dataset.manifest.get("track"),
        strict_eligible={row["query_id"]: row.get("strict_direct_match_recall_eligible", True)
                         for row in dataset.queries}
    )
    result = {
        "status": "measured", "variant_id": adapter.variant["variant_id"],
        "dataset_id": dataset.manifest["dataset_id"], "dataset_track": dataset.manifest.get("track"),
        "representation": representation,
        "instruction": instruction or adapter.model["text_convention"]["query"].get("default_instruction"),
        "runtime": getattr(adapter, "runtime", {}),
        "dimension": adapter.dimension, "max_tokens": adapter.max_tokens,
        "query_count": len(query_ids), "company_count": len(dataset.companies),
        "representation_inputs": {
            "description_only": "sparse description only",
            "description_keywords": "sparse description plus supplied keywords",
            "one_vector": "fields-rich description, products/services, customers/end markets and capabilities",
            "one_vector_fields_rich": "fields-rich description, products/services, customers/end markets and capabilities",
            "fields_max": "separate enriched semantic fields, max aggregation",
            "fields_weighted": "separate enriched semantic fields, fixed weighted aggregation",
        }[representation],
        "metrics": metrics,
        "rankings": {
            query_id: [{"company_id": company_id, "score": score} for company_id, score in ranking]
            for query_id, ranking in zip(query_ids, rankings, strict=True)
        },
    }
    if dataset.manifest.get("track") == "source_backed" and len(dataset.companies) < 500:
        result["limitations"] = [
            "small source-backed pilot; Recall@100 is not decision-grade and this track cannot support a production recommendation"
        ]
    return result


def embedding_pair_validation(reference: Any, candidate: Any, texts: list[str], *,
                              batch_size: int = 16) -> dict[str, Any]:
    import numpy as np
    reference_vectors = reference.encode_documents(texts, batch_size=batch_size)
    candidate_vectors = candidate.encode_documents(texts, batch_size=batch_size)
    if reference_vectors.shape != candidate_vectors.shape:
        return {"passed": False, "reason": "shape_mismatch",
                "reference_shape": list(reference_vectors.shape),
                "candidate_shape": list(candidate_vectors.shape)}
    return compare_embeddings(reference_vectors, candidate_vectors)


def compare_embeddings(reference_vectors: Any, candidate_vectors: Any, *,
                       p5_threshold: float = .90, minimum_threshold: float = .80) -> dict[str, Any]:
    import numpy as np
    if reference_vectors.shape != candidate_vectors.shape:
        return {"passed": False, "reason": "shape_mismatch",
                "reference_shape": list(reference_vectors.shape),
                "candidate_shape": list(candidate_vectors.shape)}
    cosines = np.sum(reference_vectors * candidate_vectors, axis=1)
    stats = {
        "mean": float(np.mean(cosines)), "median": float(np.median(cosines)),
        "p5": float(np.percentile(cosines, 5)), "minimum": float(np.min(cosines)),
    }
    return {"passed": bool(stats["p5"] >= p5_threshold and stats["minimum"] >= minimum_threshold),
            "thresholds": {"p5": p5_threshold, "minimum": minimum_threshold}, "cosine": stats,
            "sample_count": len(cosines)}

