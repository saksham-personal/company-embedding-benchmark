# VDI Agent Runbook

## Task and acceptance criteria

Use this repository to download the pinned artifacts from GitHub, verify them, run CPU retrieval and performance measurements on the actual VDI, and produce traceable reports. Follow the user's public-sources-now/PitchBook-later choice.

Success means:

- Each requested configuration has either measured results or an explicit failed/skipped reason.
- Converted artifacts pass same-family reference conformance before being benchmarked.
- Company tracks and SciFact remain separate evaluation pools.
- Sparse description and description + keyword results are distinguishable from enriched fields.
- Performance records actual machine, threads, batching, runtime and memory.
- A deployment decision is supported by comparable VDI evidence; a tiny public pilot does not establish production recall.

## 1. Establish paths and environment

Clone the repository and inspect disk/RAM/CPU. Use an absolute work directory with at least 25 GB free. Windows x64 is the packaged environment target. The user's expected Xeon/8-vCPU/64-GB VDI must be confirmed by measurements.

~~~powershell
git clone https://github.com/saksham-personal/company-embedding-benchmark.git
cd company-embedding-benchmark
$repo = (Get-Location).Path
$work = Join-Path $env:USERPROFILE 'EmbeddingBench'
.\scripts\setup_offline.ps1 -WorkDir $work
$py = Join-Path $work '.venv\Scripts\python.exe'
$models = Join-Path $work 'models'
$llama = Join-Path $work 'runtime\llama-b11146\llama-server.exe'
$env:PYTHONPATH = Join-Path $repo 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
~~~

The setup script downloads only the GitHub-hosted environment ZIP, verifies its manifest, creates a Python 3.12.10 venv and installs the pinned CPU wheels without package indexes. Use `-ArchivePath` for a local bundle. If scripting policy prevents execution, unpack and verify the bundle using `reports/OFFLINE_ENVIRONMENT.json`, then run:

~~~powershell
$bundle = Join-Path $work 'offline-environment'
& (Join-Path $bundle 'python\python.exe') -m venv (Join-Path $work '.venv')
& $py -m pip install --no-index --find-links (Join-Path $bundle 'wheels') --require-hashes -r (Join-Path $bundle 'requirements.lock')
~~~

If package indexes are available, `uv sync --frozen --extra test --extra transformers --extra onnx` is the alternative. Keep the configured CPU index. Do not upgrade packages or replace artifacts mid-comparison.

## 2. Download and inspect

~~~powershell
& $py scripts/download_models.py --model-root $models --work-dir $work
& $py scripts/setup_llama_runtime.py --work-dir $work
& $py scripts/inspect_machine.py --output (Join-Path $work 'results\raw\hardware.json')
& $py -m pytest
~~~

Confirm logical cores, RAM, CPU/OS feature flags, CPUExecutionProvider and CPU PyTorch. Fix the fair thread budget, normally 8 on the described VDI. Use a lower batch size after a memory failure, and retain the failure and retry settings. Do not infer AVX-512/VNNI availability from the processor name alone.

EmbeddingGemma is gated and absent. E5 INT8 is unavailable when required CPU/OS features are missing. Preserve those reasons. Required custom code is pinned inside packages; trust is enabled only for those declared families.

## 3. Build and validate all three data tracks

~~~powershell
$controlled = Join-Path $work 'data\controlled-v1'
$scifact = Join-Path $work 'data\scifact'
& $py scripts/build_dataset.py --output $controlled
& $py scripts/download_public_dataset.py --dataset-id beir-scifact --work-dir $work --output $scifact
& $py -m embedding_bench.cli validate-dataset $controlled
& $py -m embedding_bench.cli validate-dataset data/curated
& $py data/curated/validate_curated.py
& $py -m embedding_bench.cli validate-dataset $scifact
~~~

Expected counts: controlled 2,400/120; curated 130/38; SciFact test 5,183/300. The public pilot has 215 pooled judgments, 55 annotated hard negatives and 4.35% judgment coverage. Source-backed strict recall uses 35 eligible queries with relevance-2 positives; three exclusion probes contribute only candidate recall.

## 4. Validate artifacts

~~~powershell
$validationDir = Join-Path $work 'validation'
& $py scripts/validate_artifacts.py --model-root $models --dataset data/curated --output-dir $validationDir --threads 8 --batch-size 8 --llama-server $llama --machine-role vdi
$validation = Join-Path $validationDir 'artifact-validation.json'
~~~

The script hashes all payloads before executing custom code and exports up to 100 descriptions and 100 queries in separate processes. It checks finite/unit vectors, supported dimensions, deterministic same-batch repeats, and cross-padding cosine >= 0.999 for native/unquantized artifacts. For quantized artifacts, cross-batch cosine is a diagnostic: dynamic activation scales depend on the batch. Repeat determinism and reference cosine gates remain mandatory. It then compares queries and documents to the matching PyTorch reference. All quality queries use batch 1; document ingestion uses the declared batch size.

- FP32 conversion: cosine p5 >= 0.99 and minimum >= 0.98.
- Quantized conversion: p5 >= 0.90 and minimum >= 0.80.

A gross-error gate does not establish equal retrieval quality. A failed reference blocks its derivative rows. Preserve failures; investigate prompts, pooling, projection, padding and normalization before interpreting a derivative. Re-run validation after code/catalog changes. Cached exports include catalog, dataset, source code, hardware, machine role, threads, document/query batch policy and effective instructions. Each benchmark rehashes the payload before loading it.

Receipts cannot be reused across datasets, hardware, machine roles, threads, instructions or quality document batch sizes. Create a separate receipt for each track. Omit `--validation-file` below to let the runner validate that track automatically. A failed validation subprocess stops the automatic run; inspect its fresh receipt before explicitly reusing it to run passed rows and record failed/skipped configurations.

The gate samples the supplied dataset's text lengths. Validate on SciFact or dedicated context data too if long-document behavior matters. Do not generalize a short-description gate to an untested 8K/32K maximum.

## 5. Smoke, then full company runs

~~~powershell
& $py scripts/run_all.py --model-root $models --dataset data/curated --threads 8 --machine-role vdi --llama-server $llama --validation-file $validation --smoke --output-dir (Join-Path $work 'results\curated-smoke')

& $py scripts/run_all.py --model-root $models --dataset data/curated --threads 8 --machine-role vdi --llama-server $llama --validation-file $validation --output-dir (Join-Path $work 'results\curated')

& $py scripts/run_all.py --model-root $models --dataset $controlled --threads 8 --machine-role vdi --llama-server $llama --output-dir (Join-Path $work 'results\controlled')
$controlledValidation = Join-Path $work 'results\controlled\validation\artifact-validation.json'
~~~

Keep smoke and full performance output directories separate. Quality uses the entire dataset even in smoke mode. By default the full company runner tests:

1. description only;
2. description + keywords;
3. enriched one-vector;
4. separate semantic fields with max aggregation;
5. separate fields with fixed normalized weights 0.35/0.30/0.20/0.15.

`--variant` and `--representation` are repeatable selectors; `--quality-only` omits speed. Every row runs in a fresh process. Avoid concurrent inference or model downloads during final performance measurements.

For large matrices, prioritize BGE, E5, GTE, Arctic, Qwen and Voyage as the user requested. Explicitly record any reduced scope rather than implying completion.

## 6. Standard retrieval

~~~powershell
& $py scripts/run_all.py --model-root $models --dataset $scifact --threads 8 --machine-role vdi --llama-server $llama --quality-only --output-dir (Join-Path $work 'results\scifact')
~~~

Use all 5,183 documents and the official 300 test queries. SciFact is supporting evidence, not a substitute for company screening. Report nDCG@10 alongside recall metrics. Preserve official licenses: claims/qrels CC BY 4.0 and abstracts ODC-By 1.0.

## 7. Quantization, dimensions and CPU tuning

~~~powershell
& $py scripts/benchmark_quantization.py --model-root $models --dataset $controlled --reference bge-base-en-v1.5__pytorch-fp32__768d --candidate bge-base-en-v1.5__onnx-int8__768d --representation description_keywords --threads 8 --machine-role vdi --validation-file $controlledValidation --output (Join-Path $work 'results\comparisons\bge-int8.json')
~~~

Repeat paired comparisons for each valid candidate. A one-percentage-point loss budget on Recall@50, Recall@100 and nDCG@20 is a **quality delta gate**. Call a candidate effectively lossless only with an observed CPU/storage benefit and representative query coverage. The paired script measures reference/candidate in separate processes to avoid keeping both large models in RAM.

Supported dimension rows already exist: Arctic 768/256; Qwen 1024/768; Voyage 1024/512. Lower dimensions are output truncation + normalization, not weight quantization. Nomic's reduced dimensions need full-vector layer normalization first; the configured Nomic rows use native 768D.

For shortlisted variants, measure `benchmark_speed.py` separately at 1, 2, 4 and 8 threads. Store a distinct output filename for each. Preserve the fixed eight-thread comparison and label per-model best settings separately. The speed runner reports available token buckets, batch sizes 1/8/32/64, 20 measurement rounds, 50 single-query repetitions, sampled parent/server RSS and fresh-process load time. GGUF batch measurements group serial requests rather than batched inference.

Optional context stress:

~~~powershell
& $py scripts/build_context_datasets.py --output-root (Join-Path $work 'data\context')
& $py scripts/benchmark_context.py --model-root $models --variant gte-base-en-v1.5__pytorch-fp32__768d --dataset (Join-Path $work 'data\context\short') --dataset (Join-Path $work 'data\context\long') --threads 8 --batch-size 2 --machine-role vdi
~~~

Context views append neutral padding after relevant facts. They test costs/truncation and are not a validated test of finding evidence buried late in a document.

## 8. Report and choose

~~~powershell
& $py scripts/generate_report.py --results (Join-Path $work 'results') --reports reports
~~~

Check raw JSON before choosing a model. Compare matching dataset hashes, representation, hardware identity, thread budget, source/catalog hashes and runtime/provider. The report never names a production winner from one VDI row.

Prioritize Recall@50/100 and category-level failures. Inspect annotated hard negatives and downstream candidate workload. If recall differences are small, prefer the faster/smaller/simpler valid configuration. Choose Top-K from the measured recall curve and reranker capacity. Do not select solely from the synthetic template corpus or the 130-company public pilot.

Commit only authorized source/configuration/public data, reports and small non-sensitive result files. Keep weights, vector caches and any licensed PitchBook data outside Git.

## 9. PitchBook transfer test

When the user supplies an authorized export, keep it private. Preserve the original sparse descriptions and keywords as independent representations. Build explicit criterion labels without using model scores; keep unverified exclusivity unknown. Include near misses differing in product, capability, customer channel and exclusion. Use multiple reviewers, record label coverage and verify middle-market eligibility. Run the same artifact and metric pipeline with `track: private_company`, keeping results outside this public repository.

