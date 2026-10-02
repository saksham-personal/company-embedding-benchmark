from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def _dcg(relevances: Iterable[int]) -> float:
    return sum((2 ** relevance - 1) / math.log2(rank + 2)
               for rank, relevance in enumerate(relevances))


def evaluate_rankings(
    query_ids: list[str],
    rankings: list[list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    cutoffs: tuple[int, ...] = (10, 25, 50, 100),
    secondary_cutoffs: tuple[int, ...] = (10, 20, 50),
    complete_judgements: bool,
    hard_negatives: dict[str, frozenset[str]] | None = None,
    categories: dict[str, str] | None = None,
    dataset_track: str | None = None,
    strict_eligible: dict[str, bool] | None = None,
) -> dict[str, object]:
    if len(query_ids) != len(rankings):
        raise ValueError("query_ids and rankings length mismatch")
    hard_negatives = hard_negatives or {}
    categories = categories or {}
    strict_eligible = strict_eligible or {}
    per_query: list[dict[str, float | str]] = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        labels = qrels.get(query_id)
        if not labels:
            raise ValueError(f"missing qrels for {query_id}")
        candidate_positives = {doc_id for doc_id, relevance in labels.items() if relevance > 0}
        strict_positives = ({doc_id for doc_id, relevance in labels.items() if relevance >= 2}
                            if dataset_track == "source_backed" else candidate_positives)
        if not candidate_positives:
            raise ValueError(f"query {query_id} has no positives")
        use_strict = dataset_track != "source_backed" or strict_eligible.get(query_id, bool(strict_positives))
        positives = strict_positives if use_strict else candidate_positives
        if use_strict and not strict_positives:
            raise ValueError(f"strict-eligible query {query_id} has no direct relevance-2 positives")
        ranked_ids = [doc_id for doc_id, _ in ranking]
        row: dict[str, float | str] = {"query_id": query_id}
        for k in cutoffs:
            retrieved = set(ranked_ids[:k])
            hits = len(retrieved & positives)
            if use_strict:
                row[f"recall@{k}"] = hits / len(positives)
                row[f"hit_rate@{k}"] = float(hits > 0)
            if dataset_track == "source_backed":
                if use_strict:
                    row[f"strict_recall@{k}"] = hits / len(strict_positives)
                row[f"candidate_recall@{k}"] = len(retrieved & candidate_positives) / len(candidate_positives)
            annotated_hard = hard_negatives.get(query_id, frozenset())
            hard_hits = len(retrieved & annotated_hard)
            row[f"hard_negative_share@{k}"] = hard_hits / max(1, min(k, len(ranked_ids)))
            row[f"hard_negative_capture@{k}"] = hard_hits / max(1, len(annotated_hard))
        for k in secondary_cutoffs:
            relevance = [labels.get(doc_id, 0) for doc_id in ranked_ids[:k]]
            ideal = sorted((value for value in labels.values() if value > 0), reverse=True)[:k]
            ideal_dcg = _dcg(ideal)
            row[f"ndcg@{k}"] = _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0
            if complete_judgements:
                row[f"precision@{k}"] = sum(value > 0 for value in relevance) / max(1, k)
        top_ten = ranked_ids[:10]
        reciprocal = 0.0
        precision_sum = 0.0
        hits = 0
        for rank, doc_id in enumerate(top_ten, 1):
            if labels.get(doc_id, 0) > 0:
                hits += 1
                if reciprocal == 0:
                    reciprocal = 1 / rank
                precision_sum += hits / rank
        row["mrr@10"] = reciprocal
        if complete_judgements:
            row["map@10"] = precision_sum / min(len(positives), 10)
        per_query.append(row)
    keys = sorted({key for row in per_query for key in row if key != "query_id"})
    macro = {key: sum(float(row[key]) for row in per_query if key in row) /
                  sum(1 for row in per_query if key in row) for key in keys}
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in per_query:
        grouped[categories.get(str(row["query_id"]), "uncategorized")].append(row)
    by_category: dict[str, dict[str, float]] = {}
    for category, rows in grouped.items():
        by_category[category] = {
            key: sum(float(row[key]) for row in rows if key in row) /
                 sum(1 for row in rows if key in row)
            for key in keys if any(key in row for row in rows)
        }
    hard_micro = {}
    for k in cutoffs:
        retrieved_count = sum(min(k, len(ranking)) for ranking in rankings)
        annotated_count = sum(len(hard_negatives.get(query_id, ())) for query_id in query_ids)
        hard_hits = sum(
            len({doc_id for doc_id, _ in ranking[:k]} & hard_negatives.get(query_id, frozenset()))
            for query_id, ranking in zip(query_ids, rankings, strict=True)
        )
        hard_micro[f"hard_negative_share@{k}"] = hard_hits / max(1, retrieved_count)
        hard_micro[f"hard_negative_capture@{k}"] = hard_hits / max(1, annotated_count)
    return {
        "metric_semantics": {
            "judgements": "complete" if complete_judgements else "pooled_incomplete",
            "unjudged": "negative" if complete_judgements else "unknown",
            "recall_name": "recall" if complete_judgements else "known_positive_recall",
            "source_backed_relevance": (
                "strict rel>=2 and candidate rel>=1 reported separately"
                if dataset_track == "source_backed" else None
            ),
        },
        "macro": macro,
        "metric_query_counts": {key: sum(key in row for row in per_query) for key in keys},
        "per_query": per_query,
        "by_category": by_category,
        "hard_negative_micro": hard_micro,
    }

