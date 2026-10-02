from __future__ import annotations

import gc
import statistics
import threading
import time
from typing import Any

import numpy as np
import psutil


def _percentiles(values: list[float]) -> dict[str, float]:
    return {name: float(np.percentile(values, value)) for name, value in
            (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))}


def _resources(process):
    rss, cpu = 0, 0.0
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.Error:
        processes = [process]
    for item in processes:
        try:
            rss += item.memory_info().rss
            times = item.cpu_times()
            cpu += times.user + times.system
        except psutil.Error:
            pass
    return rss, cpu


def _measure(adapter: Any, texts: list[str], *, kind: str, batch_sizes: tuple[int, ...],
             warmup_rounds: int, measurement_rounds: int, bucket: str,
             instruction: str | None = None) -> list[dict[str, Any]]:
    encode = adapter.encode_queries if kind == "query" else adapter.encode_documents
    options = {"instruction": instruction} if kind == "query" else {}
    process = psutil.Process()
    rows = []
    for batch_size in batch_sizes:
        def batch(round_index):
            return [texts[(round_index * batch_size + index) % len(texts)] for index in range(batch_size)]
        for index in range(warmup_rounds):
            encode(batch(index), batch_size=batch_size, **options)
        gc.collect()
        rss_start, cpu_start = _resources(process)
        peak = [rss_start]
        stopped = threading.Event()
        def sample():
            while not stopped.wait(.02):
                peak[0] = max(peak[0], _resources(process)[0])
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        latencies_ms = []
        start_all = time.perf_counter()
        try:
            for index in range(measurement_rounds):
                start = time.perf_counter()
                encode(batch(index + warmup_rounds), batch_size=batch_size, **options)
                latencies_ms.append((time.perf_counter() - start) * 1000)
        finally:
            elapsed = time.perf_counter() - start_all
            stopped.set()
            sampler.join(timeout=1)
        rss_end, cpu_end = _resources(process)
        rows.append({
            "bucket": bucket, "batch_size": batch_size, "rounds": measurement_rounds,
            "distinct_input_texts": len(texts),
            "docs_per_second": batch_size * measurement_rounds / elapsed,
            "latency_ms": _percentiles(latencies_ms), "latency_mean_ms": statistics.fmean(latencies_ms),
            "process_tree_cpu_seconds": cpu_end - cpu_start,
            "rss_bytes_before": rss_start, "rss_bytes_after": rss_end,
            "sampled_peak_rss_bytes": max(peak[0], rss_end), "rss_sampling_interval_ms": 20,
        })
    return rows


def run_speed(adapter: Any, document_texts: list[str], query_texts: list[str] | None = None, *,
              batch_sizes: tuple[int, ...] = (1, 8, 32, 64), warmup_rounds: int = 3,
              measurement_rounds: int = 20, cold_load_seconds: float | None = None,
              instruction: str | None = None) -> dict[str, Any]:
    if not document_texts or measurement_rounds < 1 or any(size < 1 for size in batch_sizes):
        raise ValueError("speed measurements require texts, positive rounds, and positive batches")
    query_texts = query_texts or document_texts[:8]
    buckets = {}
    token_counts = []
    if hasattr(adapter, "tokenizer"):
        prepared = adapter.prepare_documents(document_texts)
        token_counts = [len(ids) for ids in adapter.tokenizer(prepared, truncation=False)["input_ids"]]
        for text, count in zip(document_texts, token_counts, strict=True):
            bucket = next(label for limit, label in ((64, "1-64"), (128, "65-128"), (256, "129-256"),
                         (512, "257-512"), (10**12, "513+")) if count <= limit)
            buckets.setdefault(bucket, []).append(text)
    else:
        buckets = {"token_count_not_measured": document_texts}
    return {
        "status": "measured", "variant_id": adapter.variant["variant_id"],
        "dimension": adapter.dimension, "max_tokens": adapter.max_tokens,
        "runtime": getattr(adapter, "runtime", {}), "measurement_rounds": measurement_rounds,
        "instruction": instruction, "query_batch_size": 1,
        "cold_load_seconds": cold_load_seconds,
        "cold_load_semantics": "fresh Python process/adapter; filesystem cache is not flushed",
        "memory_semantics": "sum of parent and child RSS; sampled peak at 20 ms, shared pages may be counted twice",
        "token_counts_before_truncation": token_counts,
        "document_batches": [row for bucket, texts in buckets.items() for row in _measure(
            adapter, texts, kind="document", batch_sizes=batch_sizes, bucket=bucket,
            warmup_rounds=warmup_rounds, measurement_rounds=measurement_rounds)],
        "query_latency": _measure(adapter, query_texts, kind="query", batch_sizes=(1,), bucket="mixed_queries",
            warmup_rounds=warmup_rounds, measurement_rounds=max(measurement_rounds, 50), instruction=instruction),
    }

