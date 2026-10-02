import hashlib
from pathlib import Path

import pytest

from embedding_bench.datasets import build_controlled_dataset, load_dataset
from embedding_bench.datasets.io import DatasetError, satisfies_constraints
from embedding_bench.datasets.profiles import PROFILES


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_controlled_dataset_is_deterministic_and_exhaustive(tmp_path):
    first = build_controlled_dataset(tmp_path / "a", company_count=len(PROFILES) * 16,
                                     queries_per_profile=2, seed=7)
    second = build_controlled_dataset(tmp_path / "b", company_count=len(PROFILES) * 16,
                                      queries_per_profile=2, seed=7)
    assert digest_tree(first) == digest_tree(second)
    dataset = load_dataset(first)
    assert dataset.is_complete
    assert len(dataset.queries) == len(PROFILES) * 2
    for query in dataset.queries:
        labels = dataset.qrels[query["query_id"]]
        assert sum(value > 0 for value in labels.values()) == 8
        assert len(dataset.hard_negatives[query["query_id"]]) == 8


def test_hash_verification_detects_mutation(tmp_path):
    root = build_controlled_dataset(tmp_path / "data", company_count=len(PROFILES) * 16,
                                    queries_per_profile=1, seed=8)
    with (root / "companies.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(DatasetError, match="checksum mismatch"):
        load_dataset(root)


def test_constraint_evaluator_applies_exclusions():
    company = {"attributes": {"offering": ["software"], "customer_type": ["banks"]}}
    constraints = {"must": [{"field": "offering", "op": "contains", "value": "software"}],
                   "must_not": [{"field": "customer_type", "op": "contains", "value": "contractors"}]}
    assert satisfies_constraints(company, constraints)
    constraints["must_not"][0]["value"] = "banks"
    assert not satisfies_constraints(company, constraints)

