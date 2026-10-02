import json

import pytest

from embedding_bench.config import Catalog, ConfigError
from embedding_bench.reporting import generate_report


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_catalog_rejects_cross_model_artifact(tmp_path):
    write(tmp_path / "models.json", {"schema_version": 1, "models": [
        {"model_id": "a", "architecture": {"supported_dimensions": [2]}}
    ]})
    write(tmp_path / "artifacts.json", {"schema_version": 1, "artifacts": [
        {"artifact_id": "b-art", "model_id": "b"}
    ]})
    write(tmp_path / "variants.json", {"schema_version": 1, "variants": []})
    with pytest.raises(ConfigError, match="unknown model"):
        Catalog.load(tmp_path)


def test_report_defers_recommendation_without_vdi_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    value = {
        "status": "measured", "variant_id": "model", "dataset_id": "data",
        "representation": "one_vector", "dimension": 4,
        "metrics": {"macro": {"recall@25": .5, "recall@50": .6, "recall@100": .8,
                                "ndcg@20": .4, "mrr@10": .3}},
        "provenance": {"machine_role": "development"}
    }
    (results / "result.json").write_text(json.dumps(value), encoding="utf-8")
    markdown, html = generate_report(results, tmp_path / "reports")
    text = markdown.read_text(encoding="utf-8")
    assert "recommendation is deferred" in text
    assert html.exists()


def test_report_cannot_name_a_winner_from_one_vdi_row(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write(results / "row.json", {"status": "measured", "variant_id": "one-model", "dataset_id": "controlled",
        "dataset_track": "controlled", "representation": "description_only", "dimension": 4,
        "metrics": {"macro": {"recall@50": 1.0, "recall@100": 1.0, "ndcg@20": 1.0}},
        "provenance": {"machine_role": "vdi"}})
    markdown, _ = generate_report(results, tmp_path / "reports")
    text = markdown.read_text(encoding="utf-8")
    assert "recommendation is deferred" in text
    assert "Current VDI quality leader" not in text


def test_failed_result_metrics_are_not_added_to_quality_table(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write(results / "row.json", {"status": "failed", "variant_id": "broken-model",
        "metrics": {"macro": {"recall@50": 1.0}}})
    generate_report(results, tmp_path / "reports")
    assert (results / "quality" / "company_retrieval.csv").read_text() == ""


def test_report_preserves_all_candidate_recall_cutoffs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write(results / "row.json", {"status": "measured", "variant_id": "model", "dataset_track": "source_backed",
        "metrics": {"macro": {"candidate_recall@25": .1, "candidate_recall@50": .2, "candidate_recall@100": .3}}})
    markdown, _ = generate_report(results, tmp_path / "reports")
    text = markdown.read_text(encoding="utf-8")
    assert all(f"Candidate R@{k}" in text for k in (25, 50, 100))

