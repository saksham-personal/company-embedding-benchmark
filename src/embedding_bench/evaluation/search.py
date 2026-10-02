from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from embedding_bench.adapters.base import normalize_float32


def exact_rank(query_vectors: np.ndarray, document_vectors: np.ndarray,
               document_ids: Sequence[str], *, top_k: int) -> list[list[tuple[str, float]]]:
    """Exact normalized float32 dot-product ranking with deterministic ties."""
    queries = normalize_float32(query_vectors)
    documents = normalize_float32(document_vectors)
    if documents.shape[0] != len(document_ids):
        raise ValueError("document_ids length does not match document vectors")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query and document dimensions differ")
    ids = np.asarray(document_ids, dtype=str)
    scores = queries @ documents.T
    rankings: list[list[tuple[str, float]]] = []
    limit = min(top_k, len(ids))
    for row in scores:
        order = np.lexsort((ids, -row))[:limit]
        rankings.append([(str(ids[index]), float(row[index])) for index in order])
    return rankings


def rank_dataset(query_vectors: np.ndarray, field_vectors: np.ndarray,
                 field_company_ids: Sequence[str], field_names: Sequence[str], *,
                 top_k: int, aggregation: str = "max",
                 field_weights: dict[str, float] | None = None) -> list[list[tuple[str, float]]]:
    queries = normalize_float32(query_vectors)
    fields = normalize_float32(field_vectors)
    if len(field_company_ids) != len(fields) or len(field_names) != len(fields):
        raise ValueError("field metadata length mismatch")
    scores = queries @ fields.T
    company_ids = sorted(set(field_company_ids))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, company_id in enumerate(field_company_ids):
        grouped[company_id].append(index)
    results: list[list[tuple[str, float]]] = []
    weights = field_weights or {
        "core_business": .35, "products_services": .30,
        "customers_end_markets": .20, "capabilities": .15,
        "company": 1.0,
    }
    for query_scores in scores:
        company_scores = []
        for company_id in company_ids:
            indices = grouped[company_id]
            values = query_scores[indices]
            if aggregation == "max":
                score = float(values.max())
            elif aggregation == "weighted":
                raw_weights = np.asarray([weights.get(field_names[index], 0.0) for index in indices], dtype=np.float32)
                if float(raw_weights.sum()) <= 0:
                    raise ValueError(f"no positive field weights for {company_id}")
                score = float(np.dot(values, raw_weights / raw_weights.sum()))
            else:
                raise ValueError(f"unknown aggregation: {aggregation}")
            company_scores.append((company_id, score))
        company_scores.sort(key=lambda item: (-item[1], item[0]))
        results.append(company_scores[: min(top_k, len(company_scores))])
    return results

