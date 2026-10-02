# Methodology

## Retrieval objective

Retrieve a high-recall shortlist for specific company product, capability, customer and exclusion criteria. Embeddings feed downstream reasoning; cosine scores alone do not establish that a company satisfies an exclusivity condition.

All embeddings become contiguous float32, are checked for finite values and nonzero norms, and are L2 normalized. Exact dot-product search aggregates at company ID before ranking. Ties use stable company IDs.

## Distinct evidence tracks

**Controlled:** 2,400 synthetic records, 120 paraphrased criteria and exhaustive labels derived from explicit attributes. Positives and customer-channel near misses share product language. Filler companies diversify the corpus. Template repetition and a small number of underlying profiles can make this dataset easy; it is a controlled diagnostic.

**Public companies:** 130 records, 38 queries, 215 pooled judgments and 55 annotated hard negatives. Descriptions are authored factual summaries of official sources, not licensed commercial descriptions. Large businesses and brands are included; middle-market status is unverified. Only 4.35% of query/company pairs are judged.

Direct-match recall requires relevance >= 2 and is averaged over 35 strict-eligible queries. Candidate recall includes relevance >= 1 across all 38. Three exclusion probes lack verified strict positives and are excluded from strict denominators. Missing judgments remain unknown. Precision and MAP are not emitted for this track. nDCG/MRR use conventional pooled-qrel gains, with unjudged gains of zero; incomplete labels can bias these scores.

**SciFact:** the pinned BEIR test corpus with 5,183 abstracts and 300 test queries, preserving official qrels and licenses. Standard retrieval results support comparability, not company suitability.

## Input budgets and representations

- Description-only preserves the sparse business description.
- Description + keywords appends the supplied keyword list.
- Enriched one-vector adds products/services, customers/end markets and capabilities.
- Fields-max embeds semantic fields separately and uses the maximum score.
- Fields-weighted uses normalized fixed weights 0.35/0.30/0.20/0.15.

Enriched fields may contain facts deliberately omitted from the short description. Their gains cannot be attributed to a model alone. Each result states its input budget.

Model prompts, pooling, output projection and dimensions come from pinned official sources. BGE/GTE/Arctic use CLS, E5/Nomic/Voyage use masked means, Qwen uses the last nonpadding token, and Voyage includes its learned embedding projection. Qwen company tracks use one fixed task instruction; generic retrieval uses its official default. No tuning uses held-out labels.

## Artifact correctness

Payload checksums precede custom-code execution on validation and benchmark commands. Each artifact is isolated in a process. Native dimensions, finite/unit vectors, repeat determinism and padding consistency are checked. Same-batch repeats have strict numerical tolerances; different-padding batches require cosine >= 0.999 for native/unquantized rows. Quantized cross-batch cosine is reported diagnostically because dynamic activation scales can change with batch composition. The BGE, GTE, Arctic and Nomic INT8 graph files contain the DynamicQuantizeLinear operator. See [the ONNX Runtime documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Quality queries always use batch 1. Documents use a fixed ingestion batch (default 8); changing it requires a new matching receipt. Receipts bind dataset, source/catalog, hardware, role, threads, effective instruction and batching. Speed experiments deliberately vary document batches and retain the sampled reference-validation scope. Qwen ONNX receives position IDs derived from the attention mask and zero-length KV caches sized from its pinned configuration; only its last-hidden-state output is used.

Converted FP32 embeddings require paired query/document cosine p5 >= 0.99 and minimum >= 0.98; quantized candidates require p5 >= 0.90 and minimum >= 0.80. The latter gate detects gross conversion errors. Retrieval deltas are still necessary.

A quality delta gate permits at most a one-point loss in Recall@50, Recall@100 and nDCG@20. It does not alone warrant the term effectively lossless; a measured compute/storage benefit and representative data are also required. Each run's reference-validation scope states the text sample and lengths.

Custom CPU loading leaves parameters empty while constructing nonpersistent position/rotary buffers on CPU, then assigns the checkpoint tensors. Nomic's absent optional BERT pooler is removed; its official embedding uses the last hidden state's masked mean.

## CPU measurements

The fair comparison fixes threads and uses a fresh process per row. Available tokenizer-length buckets are recorded; absent buckets are not synthesized into performance claims. Document batches are 1/8/32/64. Full runs use three warmups, 20 document repetitions and at least 50 single-query repetitions, rotating texts through the available sample.

Report p50/p90/p95/p99, throughput, runtime/provider/thread configuration, parent-plus-child CPU time, and resident memory. Peak RSS is sampled every 20 ms and may miss transient peaks; summed processes may double-count shared pages. GGUF uses the pinned official CPU server and sequential document requests.

Load time is fresh-process adapter construction, including runtime imports, but the operating-system file cache is not flushed. Label short development performance runs as diagnostics. Measure target VDI performance separately.

## Context and unresolved evidence

Context views append neutral padding after relevant facts. They measure truncation/compute behavior, not difficult late-document evidence discovery. Supported maximum context sizes are model specifications until separately validated at those lengths.

The public pilot is single-curator, incompletely judged and subject to source selection bias. It does not validate transfer to PitchBook or prove production-scale recall. Select a deployment after complete comparable VDI runs and representative company labels; preserve skips, failures and uncertainty.

