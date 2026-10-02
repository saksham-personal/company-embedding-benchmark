from __future__ import annotations

from pathlib import Path
from typing import Any

from embedding_bench.benchmark.provenance import source_digest
from embedding_bench.config import canonical_digest


def check_receipt(receipt: dict[str, Any], *, catalog: Any, project_root: Path, dataset: Any,
                  machine_role: str, hardware_id: str, threads: int, variant_id: str,
                  instruction: str | None, document_batch_size: int | None = None) -> dict[str, Any]:
    """Bind a passed gate to the exact data, code, machine and encoding policy."""
    expected = {"catalog_digest": catalog.digest, "source_digest": source_digest(project_root),
                "dataset_digest": canonical_digest(dataset.manifest), "machine_role": machine_role,
                "hardware_id": hardware_id}
    for name, value in expected.items():
        if receipt.get("provenance", {}).get(name) != value:
            raise ValueError(f"validation receipt does not match current {name}")
    row = next((row for row in receipt.get("variants", []) if row["variant_id"] == variant_id), None)
    if not row or row.get("status") != "passed":
        raise ValueError(f"variant has no passed reference-validation receipt: {variant_id}")
    if row.get("runtime", {}).get("threads") != threads:
        raise ValueError("validation receipt does not match current thread budget")
    if row.get("instruction") != instruction or row.get("query_batch_size") != 1:
        raise ValueError("validation receipt does not match current query encoding policy")
    if document_batch_size is not None and row.get("document_batch_size") != document_batch_size:
        raise ValueError("validation receipt does not match current document batch size")
    return row
