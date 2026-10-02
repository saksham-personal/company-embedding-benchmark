import numpy as np

from embedding_bench.evaluation.metrics import evaluate_rankings
from embedding_bench.evaluation.search import exact_rank, rank_dataset


def test_exact_search_uses_cosine_and_stable_id_ties():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    docs = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    ranking = exact_rank(queries, docs, ["b", "a", "c"], top_k=3)[0]
    assert [item[0] for item in ranking] == ["a", "b", "c"]
    assert ranking[0][1] == 1.0


def test_field_aggregation_returns_company_level_results():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    fields = np.array([[1.0, 0.0], [0.0, 1.0], [.8, .2]], dtype=np.float32)
    results = rank_dataset(queries, fields, ["a", "a", "b"],
                           ["core_business", "capabilities", "core_business"],
                           top_k=2, aggregation="max")
    assert [item[0] for item in results[0]] == ["a", "b"]


def test_metrics_match_hand_calculation_and_hide_precision_for_partial_qrels():
    ranking = [[("n", .9), ("p1", .8), ("p2", .7), ("x", .6)]]
    qrels = {"q": {"p1": 2, "p2": 1, "n": 0}}
    hard = {"q": frozenset({"n"})}
    complete = evaluate_rankings(["q"], ranking, qrels, complete_judgements=True,
                                 hard_negatives=hard, cutoffs=(2, 4), secondary_cutoffs=(2,))
    macro = complete["macro"]
    assert macro["recall@2"] == .5
    assert macro["recall@4"] == 1.0
    assert macro["precision@2"] == .5
    assert macro["mrr@10"] == .5
    assert macro["hard_negative_capture@2"] == 1.0
    partial = evaluate_rankings(["q"], ranking, qrels, complete_judgements=False,
                                cutoffs=(2,), secondary_cutoffs=(2,))
    assert "precision@2" not in partial["macro"]
    assert "map@10" not in partial["macro"]
    assert partial["metric_semantics"]["unjudged"] == "unknown"


def test_source_backed_strict_and_candidate_recall_are_separate():
    ranking = [[("partial", .9), ("direct", .8)]]
    result = evaluate_rankings(
        ["q"], ranking, {"q": {"direct": 2, "partial": 1}},
        complete_judgements=False, cutoffs=(1, 2), secondary_cutoffs=(2,),
        dataset_track="source_backed"
    )
    assert result["macro"]["strict_recall@1"] == 0.0
    assert result["macro"]["candidate_recall@1"] == .5
    assert result["macro"]["strict_recall@2"] == 1.0


def test_source_exclusion_probe_without_direct_match_is_candidate_only():
    result = evaluate_rankings(
        ["q"], [[("partial", .9)]], {"q": {"partial": 1}},
        complete_judgements=False, cutoffs=(1,), secondary_cutoffs=(1,),
        dataset_track="source_backed", strict_eligible={"q": False}
    )
    assert "strict_recall@1" not in result["macro"]
    assert result["macro"]["candidate_recall@1"] == 1.0


def test_strict_recall_macro_excludes_queries_without_direct_positives():
    result = evaluate_rankings(["direct", "probe"], [[("d", .9)], [("p", .8)]],
        {"direct": {"d": 2}, "probe": {"p": 1}}, complete_judgements=False,
        cutoffs=(1,), secondary_cutoffs=(1,), dataset_track="source_backed",
        strict_eligible={"direct": True, "probe": False})
    assert result["macro"]["strict_recall@1"] == 1.0
    assert result["metric_query_counts"]["strict_recall@1"] == 1
    assert result["metric_query_counts"]["candidate_recall@1"] == 2

