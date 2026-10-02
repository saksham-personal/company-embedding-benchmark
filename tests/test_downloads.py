import hashlib
import zipfile

import pytest

from embedding_bench.downloads import DownloadError, safe_extract_zip, verify_payload


def test_safe_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(DownloadError, match="unsafe zip"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_zip_extracts_regular_files(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/corpus.jsonl", "{}\n")
    output = safe_extract_zip(archive, tmp_path / "out")
    assert (output / "dataset" / "corpus.jsonl").read_text() == "{}\n"


def test_payload_verifies_size_and_hash(tmp_path):
    value = b"model"
    (tmp_path / "model.bin").write_bytes(value)
    verify_payload(tmp_path, [{"path": "model.bin", "size": len(value),
                               "sha256": hashlib.sha256(value).hexdigest()}])
    with pytest.raises(DownloadError, match="size mismatch"):
        verify_payload(tmp_path, [{"path": "model.bin", "size": 1,
                                   "sha256": hashlib.sha256(value).hexdigest()}])

