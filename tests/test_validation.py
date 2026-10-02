import copy
from types import SimpleNamespace

import numpy as np
import pytest

from embedding_bench.benchmark.provenance import source_digest
from embedding_bench.benchmark.quality import run_quality
from embedding_bench.benchmark.validation import check_receipt
from embedding_bench.config import canonical_digest
from embedding_bench.datasets.io import BenchmarkDataset


@pytest.mark.parametrize("changed", ["dataset_digest", "machine_role", "hardware_id", "threads", "instruction", "batch"])
def test_receipt_rejects_different_execution_scope(tmp_path, changed):
    dataset = SimpleNamespace(manifest={"dataset_id": "data", "track": "controlled"})
    catalog = SimpleNamespace(digest="catalog")
    receipt = {"provenance": {"catalog_digest": "catalog", "source_digest": source_digest(tmp_path),
        "dataset_digest": canonical_digest(dataset.manifest), "machine_role": "vdi", "hardware_id": "machine"},
        "variants": [{"variant_id": "row", "status": "passed", "runtime": {"threads": 8},
                      "instruction": "find companies", "query_batch_size": 1, "document_batch_size": 8}]}
    options = dict(catalog=catalog, project_root=tmp_path, dataset=dataset, machine_role="vdi",
                   hardware_id="machine", threads=8, variant_id="row", instruction="find companies", document_batch_size=8)
    assert check_receipt(receipt, **options)["status"] == "passed"
    bad = copy.deepcopy(receipt)
    if changed in ("dataset_digest", "machine_role", "hardware_id"):
        bad["provenance"][changed] = "different"
    elif changed == "threads":
        bad["variants"][0]["runtime"]["threads"] = 4
    elif changed == "instruction":
        bad["variants"][0]["instruction"] = "other task"
    else:
        bad["variants"][0]["document_batch_size"] = 32
    with pytest.raises(ValueError, match="validation receipt"):
        check_receipt(bad, **options)


def test_quality_queries_are_independent_of_document_batch(tmp_path):
    calls = []
    class Adapter:
        model = {"text_convention": {"query": {}}}
        variant = {"variant_id": "row"}
        dimension, max_tokens, runtime = 2, 512, {}
        def encode_queries(self, texts, *, instruction, batch_size):
            calls.append(("query", batch_size))
            return np.tile([[1., 0.]], (len(texts), 1))
        def encode_documents(self, texts, *, batch_size):
            calls.append(("document", batch_size))
            return np.tile([[1., 0.]], (len(texts), 1))
    data = BenchmarkDataset({"dataset_id": "x", "track": "controlled", "judgement_completeness": "exhaustive"},
        ({"company_id": "c", "description": "steel casting"},),
        ({"query_id": "q", "text": "steel companies"},), {"q": {"c": 2}}, {}, tmp_path)
    result = run_quality(Adapter(), data, representation="description_only", batch_size=8)
    assert calls == [("query", 1), ("document", 8)]
    assert result["query_batch_size"] == 1 and result["document_batch_size"] == 8


def test_cli_hashes_payload_before_loading_custom_code(tmp_path, monkeypatch):
    from embedding_bench import cli
    artifact_dir = tmp_path / "models" / "artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.py").write_text("mutated code", encoding="utf-8")
    catalog = SimpleNamespace(digest="catalog", variants={"row": {"artifact_id": "artifact", "model_id": "model"}},
        models={"model": {"text_convention": {"query": {}}}},
        artifacts={"artifact": {"files": [{"path": "model.py", "size": 12, "sha256": "0" * 64}]}})
    monkeypatch.setattr(cli.Catalog, "load", lambda _: catalog)
    monkeypatch.setattr(cli, "load_dataset", lambda _: SimpleNamespace(manifest={}))
    monkeypatch.setattr(cli, "provenance", lambda **_: {"hardware_id": "machine"})
    loaded = []
    monkeypatch.setattr(cli, "load_adapter", lambda *args: loaded.append(args))
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        cli.main(["quality", "--project-root", str(tmp_path), "--model-root", str(tmp_path / "models"),
                  "--dataset", str(tmp_path), "--variant", "row"])
    assert loaded == []


def test_runner_removes_stale_receipt_and_stops_on_validation_failure(tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("test_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "results"
    receipt = output / "validation" / "artifact-validation.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"stale":true}')
    monkeypatch.setattr(runner.Catalog, "load", lambda _: SimpleNamespace(variants={"row": {"enabled": True}}))
    monkeypatch.setattr(runner, "load_dataset", lambda _: SimpleNamespace(manifest={}))
    calls = []
    def failing_validation(command, **kwargs):
        calls.append(command)
        assert not receipt.exists()
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(runner.subprocess, "run", failing_validation)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--project-root", str(tmp_path), "--model-root", str(tmp_path),
        "--dataset", str(tmp_path), "--machine-role", "vdi", "--output-dir", str(output)])
    with pytest.raises(subprocess.CalledProcessError):
        runner.main()
    assert len(calls) == 1
