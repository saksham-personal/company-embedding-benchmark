# Company Embedding Benchmark Architecture

## Scope and evidence boundary

This repository is a reproducible benchmark kit for a CPU-only VDI that can
reach GitHub but cannot reach Hugging Face. It separates three kinds of
evidence so that results cannot be overstated:

1. **Controlled company screening** is deterministic, synthetic, and
   exhaustively labelled. It measures precise constraint retrieval, hard
   negatives, representation strategies, and long-document behaviour.
2. **Source-backed company screening** contains public company facts with a
   citation for every asserted field. Its relevance labels are pooled and
   incomplete; unjudged companies remain unknown rather than being counted as
   negatives. It must never be described as PitchBook data.
3. **Public retrieval** imports a pinned, established retrieval dataset and its
   qrels without changing their judgement semantics.

Results produced on the personal computer are smoke or development results.
The final model recommendation is generated only from a completed VDI run.

## Package boundaries

```text
configs/
  models.json       # root-owned official model facts and revisions
  artifacts.json    # root-owned GitHub Release packages and checksums
  variants.json     # executable benchmark rows referencing the two files above
  benchmark.json    # fixed seeds, thread budgets, metric cutoffs, timing policy
data/
  curated/          # root-owned source-backed company facts and citations
  synthetic/        # deterministic generated fixtures and gold judgements
  public/           # imported public retrieval corpus/query/qrels
  processed/        # standard schema assembled by dataset builders
src/embedding_bench/
  adapters/         # official query/document conventions and runtime backends
  datasets/         # validation, deterministic generation, import, rendering
  evaluation/       # exact float32 cosine search and judgement-aware metrics
  experiments/      # quality, speed, quantization, fields, context, dimensions
  reporting/        # provenance-preserving JSON/CSV/Markdown/HTML reports
  cli.py             # stable entry points used by scripts and VDI agent
scripts/             # thin wrappers for common VDI commands
tests/               # metrics, prompting, exact search, datasets, provenance
models/              # ignored reconstructed artifacts
results/             # measured outputs, each with run provenance
reports/             # generated reports and methodology
```

The benchmark core never downloads from Hugging Face. The downloader resolves
only entries in `configs/artifacts.json`, downloads GitHub Release assets,
verifies the package SHA-256, extracts into a staging directory, verifies every
payload file, and atomically installs the model directory.

## Configuration contracts

All configuration files carry `schema_version: 1`.

### `configs/models.json` (root-owned)

Each `models[]` entry supplies immutable model facts:

```json
{
  "model_id": "e5-base-v2",
  "family": "e5",
  "upstream": {"repo_id": "intfloat/e5-base-v2", "revision": "<sha>"},
  "license": {"spdx": "MIT", "redistribution_reviewed": true},
  "architecture": {
    "native_dimensions": [768], "max_tokens": 512,
    "pooling": "masked_mean", "l2_normalize": true
  },
  "text_convention": {
    "query": {"prefix": "query: ", "instruction_template": null},
    "document": {"prefix": "passage: ", "instruction_template": null}
  },
  "tokenizer": {"repo_id": "intfloat/e5-base-v2", "revision": "<sha>"},
  "requires_remote_code": false,
  "sources": [{"kind": "model_card", "url": "...", "accessed": "YYYY-MM-DD"}]
}
```

Model instructions are data, but adapters implement and test their rendering.
No default prefix or pooling rule is inferred from the family name.

### `configs/artifacts.json` (root-owned)

Each `artifacts[]` entry identifies one redistributable package:

```json
{
  "artifact_id": "e5-base-v2-onnx-int8-avx512-vnni",
  "model_id": "e5-base-v2",
  "runtime": "onnxruntime",
  "precision": "int8",
  "source": {"kind": "upstream-or-local-export", "revision": "<sha>"},
  "github": {"repository": "owner/repo", "release": "models-v1", "asset": "...tar.zst"},
  "archive_sha256": "<hex>",
  "files": [{"path": "model.onnx", "size": 0, "sha256": "<hex>"}],
  "size_bytes": 0,
  "cpu_features_required": ["avx512f", "avx512_vnni"],
  "redistribution": {"allowed": true, "basis": "license", "notes": "..."}
}
```

Unavailable, gated, non-redistributable, or invalid exports are explicit
records with `status` and `reason`; the downloader skips them visibly.

### `configs/variants.json` (benchmark-owned)

Each `variants[]` row is the actual comparison unit:

```json
{
  "variant_id": "e5-base-v2__onnx-int8__768d",
  "model_id": "e5-base-v2",
  "artifact_id": "e5-base-v2-onnx-int8-avx512-vnni",
  "backend": "onnxruntime",
  "dimensions": 768,
  "max_tokens": 512,
  "weight_quantization": "int8",
  "output_dtype": "float32",
  "pooling_override": null,
  "trust_remote_code": false,
  "enabled": true
}
```

The runner rejects an override that conflicts with immutable model facts unless
the model record explicitly lists it as supported (for example Matryoshka
dimensions). Weight quantization and output-vector quantization are distinct
fields; the primary benchmark requires normalized float32 output vectors.

## Standard dataset schema

Processed datasets use JSONL plus TSV qrels. IDs are stable ASCII strings.
Every dataset directory includes `dataset.json` with version, license, source
URLs/revisions, creation command, seed, file hashes, judgement completeness,
and corpus/query counts.

### `companies.jsonl`

```json
{
  "company_id": "syn-co-000001",
  "track": "controlled",
  "name": "Northline Timber Exchange",
  "description": "...",
  "keywords": ["..."],
  "fields": {
    "core_business": "...", "products_services": "...",
    "customers_end_markets": "...", "capabilities": "..."
  },
  "attributes": {
    "sector": "...", "subsector": "...", "offering": ["..."],
    "capability": ["..."], "customer_type": ["..."],
    "end_market": ["..."], "business_model": ["..."],
    "geography": ["..."], "negative_attributes": ["..."]
  },
  "source_refs": ["src-..."],
  "generation": {"generator": "controlled-v1", "seed": 20261002}
}
```

Source-backed records set `track: source_backed`, omit `generation`, and may
leave unknown attributes null. Synthetic and source-backed records never share
an evaluation pool.

### `queries.jsonl`

```json
{
  "query_id": "syn-q-0001",
  "text": "...",
  "category": "capability_customer_exclusion",
  "constraints": {
    "must": [{"field": "customer_type", "op": "contains_any", "values": ["wood manufacturer"]}],
    "must_not": [{"field": "customer_type", "op": "contains_any", "values": ["contractor"]}]
  },
  "source_refs": [],
  "judgement_scope": "exhaustive"
}
```

Constraints generate synthetic gold labels and document why a company is a
positive or hard negative. They are not exposed to embedding models.

### `qrels.tsv`

Columns are `query_id`, `company_id`, `relevance`, `judgement_source`, and
`rationale_id`. Relevance uses `2` for a direct match, `1` for a useful partial
match, and `0` only for an explicitly judged non-match. Missing pairs are
unknown on pooled datasets. Controlled data is exhaustive and therefore every
pair is derivable, though the file may store positives and annotated hard
negatives compactly.

### `hard_negatives.jsonl`

```json
{
  "query_id": "syn-q-0001", "company_id": "syn-co-000041",
  "shared_facets": ["lumber distribution"],
  "violated_constraints": ["sells primarily to contractors"],
  "rationale_id": "rat-..."
}
```

`sources.jsonl` contains citation records (`source_ref`, URL, title, publisher,
accessed date, licence/terms note, and a short factual excerpt or fact summary).

## Evaluation semantics

All primary quality runs convert adapter output to contiguous float32, reject
NaNs and zero norms, L2-normalize, and compute exact matrix dot products. Ties
are stable by company ID. Approximate indexes are outside the primary run.

Metrics are Recall@10/25/50/100, HitRate@10/25/50/100, nDCG@10/20/50,
MRR@10, MAP@10, and Precision@10/20/50. Precision and MAP are emitted only
for exhaustive or officially complete qrels. On pooled source-backed data the
report labels recall as `known-positive recall`, keeps unjudged items unknown,
and emphasizes hard-negative violation rate: the fraction of annotated hard
negatives appearing in the top K, reported both macro per query and micro.

Quality rows include dataset ID/hash, variant ID, representation, dimension,
max tokens, query/document convention digest, runtime versions, machine ID,
Git commit, seed, and command. This prevents incomparable rows from being
merged silently.

## Experiments

1. **Quality:** exact retrieval for each valid variant and dataset track.
2. **Representation:** one compact document vector versus field vectors using
   both max-field and fixed, declared weights. Aggregation happens at company
   ID before ranking.
3. **Context:** deterministic short, medium, long, and very-long renderings;
   compare whole-record truncation with field/chunk representation.
4. **Dimensions:** use only official supported truncation dimensions and
   renormalize after truncation.
5. **Quantization:** paired reference/candidate texts, per-row embedding cosine
   distribution, retrieval deltas, and speed deltas. A candidate cannot enter
   the main tables until shape, finite values, norm, convention, determinism,
   and reference-similarity gates pass.
6. **Performance:** cold load in a fresh process, warmed steady-state batches
   1/8/32/64, realistic token buckets, query p50/p90/p95/p99, docs/sec, peak
   RSS, and optional tokenize/infer/postprocess split. A fixed fair thread
   budget is reported separately from per-variant best local settings.

Hardware inspection records CPU, flags, virtualization, memory, storage,
Python/runtime versions, execution providers, and all relevant thread settings.

## Execution and failure policy

Every command supports `--smoke`, `--dataset`, `--variant`, `--output-dir`, and
an explicit `--work-dir` suitable for `E:\\Codex\\company-embedding-benchmark`.
Network access is not required after GitHub Release assets and public dataset
packages are downloaded and verified. Missing CPU features, gated artefacts,
licence restrictions, checksum failures, and failed validation produce
structured `SKIPPED` or `FAILED` records; there are no silent substitutions.

The report generator recommends a model only when comparable VDI quality and
performance rows exist. Development-machine rows are marked non-decisional.

## Validation gates

- schema and cross-reference validation for every config and dataset file;
- deterministic controlled-data hashes for a fixed seed;
- exhaustive synthetic qrels agree with constraints;
- prompt rendering and pooling tests for each enabled family;
- exact-search and metric tests with hand-calculated fixtures and ties;
- incomplete-qrels tests proving unjudged items are not negatives;
- archive traversal protection and SHA-256 enforcement;
- reference/candidate quantization validation before benchmarking;
- provenance completeness before a row can enter generated reports.

## Main risks

- Model licences can prohibit redistribution even when weights are accessible;
  such artifacts remain unavailable with documented setup instructions.
- Official model behaviour may depend on custom code. Pin and package reviewed
  code, default `trust_remote_code` to false, and enable it only per manifest.
- Public company labels are incomplete and can bias conventional precision.
  Keep them separate and report judgement coverage.
- Template leakage can make controlled data too easy. Use withheld paraphrase
  families, lexical-overlap diagnostics, and explicit hard negatives.
- Personal-PC speed and quality cannot stand in for the VDI. Preserve hardware
  provenance and defer the final deployment choice until a VDI run exists.
