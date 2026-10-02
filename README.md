# Company Embedding Benchmark

A CPU benchmark kit for **company screening** on a Windows VDI where Hugging Face is blocked. Models, tokenizers, custom model code, SciFact and the llama.cpp runtime are hosted on [GitHub Releases](https://github.com/saksham-personal/company-embedding-benchmark/releases/tag/assets-v1). Large binaries stay outside Git.

The primary question is which embedding configuration retrieves relevant companies at Top-25/50/100 before a reranker applies detailed criteria. Download the assets while connected to GitHub, or transfer the checked-out repository and verified assets to a disconnected machine. Inference uses local files and offline Hugging Face settings.

## What is included

| Family | Published artifacts | Executable dimensions |
|---|---|---|
| BGE base en v1.5 | PyTorch FP32, ONNX FP32, ONNX INT8 | 768 |
| E5 base v2 | PyTorch FP32, ONNX FP32, AVX-512/VNNI INT8 | 768 |
| GTE base en v1.5 | PyTorch FP32, ONNX FP32, ONNX INT8 | 768 |
| Arctic embed m v2 | PyTorch FP32, ONNX FP32, ONNX INT8 | 768, 256 |
| Qwen3 embedding 0.6B | PyTorch reference, community ONNX INT8 | 1024, 768 |
| Voyage 4 nano | PyTorch reference, community ONNX INT8 | 1024, 512 |
| Nomic embed text v1.5 | PyTorch reference, ONNX INT8, official Q4_K_M GGUF | 768 |
| EmbeddingGemma 300M | **Skipped: upstream access is gated** | No runnable artifact |

There are **19 model packages and 25 configured rows** across seven accessible families. E5 INT8 is skipped if the CPU and OS do not expose the required AVX-512/VNNI features. A configured row must pass local reference validation before the complete runner benchmarks it; available files do not imply a valid conversion.

The model ZIPs total about **9.8 GB**. Reserve at least **25 GB** for archives, extracted models, Python and result caches. Use a drive with space; a VDI need not use the personal computer's E: path.

## Dataset tracks

| Track | Size | What it establishes |
|---|---|---|
| Controlled company screening | 2,400 synthetic companies / 120 queries | Reproducible constraint and hard-negative tests with exhaustive gold labels |
| Public company pilot | 130 company/brand records / 38 queries | Variable short descriptions, keywords and cited company facts; incomplete pooled labels |
| BEIR SciFact test | 5,183 documents / 300 queries | A standard retrieval comparison using official test qrels |

The public pilot is **not PitchBook data or a verified middle-market sample**. It includes large enterprises and operating brands. Its 35 strict queries and three exploratory exclusion probes are reported separately. At 130 companies, Recall@100 is weak evidence for model selection. See [the public dataset card](data/curated/README.md).

The public pilot defaults to sparse **description + keywords**. Description-only and separately enriched field representations are explicit alternatives; richer fields may restore information omitted from a short description. Absence of evidence for an excluded customer channel is unknown.

## Setup on the VDI

Clone this repository, then choose an accessible work directory on a drive with space.

~~~powershell
git clone https://github.com/saksham-personal/company-embedding-benchmark.git
cd company-embedding-benchmark
$work = Join-Path $env:USERPROFILE 'EmbeddingBench'
~~~

### GitHub-only dependency setup

The release also contains a portable **Windows x64 CPython 3.12.10 + CPU wheel bundle**. This path does not require PyPI or the PyTorch package index.

~~~powershell
.\scripts\setup_offline.ps1 -WorkDir $work
$py = Join-Path $work '.venv\Scripts\python.exe'
$models = Join-Path $work 'models'
~~~

The script verifies the archive and payload hashes and installs the locked wheels with `--no-index --require-hashes`. To install without any network, provide a previously downloaded bundle using `-ArchivePath <ZIP>`. It sets `PYTHONPATH` for the current PowerShell session. In a new shell, set `$env:PYTHONPATH = Join-Path (Get-Location) 'src'` again. Retain the bundled Python and third-party licenses.

PowerShell must allow a local script to run under your organization's policy. If needed, use the equivalent unpack/venv/pip commands in [the agent guide](reports/VDI_AGENT_GUIDE.md).

### Alternative: package-index access is available

With `uv` installed:

~~~powershell
uv python install 3.12.10
uv sync --frozen --extra test --extra transformers --extra onnx
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$models = Join-Path $work 'models'
~~~

`uv.lock` pins the environment and explicitly selects CPU PyTorch. The portable wheel bundle supports Windows x64/Python 3.12; use the uv route for other supported environments. The included GGUF server package is Windows x64.

## Download models and data

~~~powershell
& $py scripts/download_models.py --model-root $models --work-dir $work
& $py scripts/setup_llama_runtime.py --work-dir $work
$llama = Join-Path $work 'runtime\llama-b11146\llama-server.exe'

& $py scripts/build_dataset.py --output (Join-Path $work 'data\controlled-v1')
& $py scripts/download_public_dataset.py --dataset-id beir-scifact --work-dir $work --output (Join-Path $work 'data\scifact')
& $py -m embedding_bench.cli validate-dataset data/curated
& $py -m embedding_bench.cli validate-dataset (Join-Path $work 'data\controlled-v1')
& $py -m embedding_bench.cli validate-dataset (Join-Path $work 'data\scifact')
~~~

The downloader verifies archive and individual file SHA-256 hashes and rejects unsafe ZIP paths. Existing verified models are reused. Select a family with `--model e5-base-v2` or an artifact with `--artifact <ARTIFACT_ID>`. Public assets require no GitHub login. Private releases require `GH_TOKEN` or `GITHUB_TOKEN` in the environment; do not store tokens in files.

## Validate, then benchmark

Inspect the actual VDI before choosing a fair thread budget:

~~~powershell
& $py scripts/inspect_machine.py --output (Join-Path $work 'results\raw\hardware.json')
& $py scripts/validate_artifacts.py --model-root $models --dataset data/curated --output-dir (Join-Path $work 'validation') --threads 8 --llama-server $llama --machine-role vdi
~~~

Validation runs in isolated processes, checks payload hashes, repeated embeddings, padding consistency, dimensions and reference cosine distributions for queries and documents. FP32 conversions require tighter similarity than quantized artifacts. This is an artifact correctness gate; retrieval deltas must still be measured.

Run an initial company-pilot benchmark:

~~~powershell
& $py scripts/run_all.py --model-root $models --dataset data/curated --machine-role vdi --threads 8 --llama-server $llama --validation-file (Join-Path $work 'validation\artifact-validation.json') --smoke --output-dir (Join-Path $work 'results\curated')
~~~

`--smoke` shortens performance repetitions; **quality still uses the entire supplied dataset**. For the full run, remove `--smoke`. It tests description-only, description + keywords, enriched one-vector, fields-max and fixed fields-weighted representations. Run the controlled and SciFact tracks separately using their dataset paths. SciFact uses its title + abstract representation.

One row or a paired comparison:

~~~powershell
& $py scripts/benchmark_quality.py --model-root $models --dataset data/curated --variant bge-base-en-v1.5__onnx-int8__768d --representation description_keywords --threads 8 --machine-role vdi --validation-file (Join-Path $work 'validation\artifact-validation.json')

& $py scripts/benchmark_quantization.py --model-root $models --dataset data/curated --reference bge-base-en-v1.5__pytorch-fp32__768d --candidate bge-base-en-v1.5__onnx-int8__768d --threads 8 --machine-role vdi --validation-file (Join-Path $work 'validation\artifact-validation.json')
~~~

Qwen uses its official instruction template with a fixed company-screening instruction for company tracks, and the official generic retrieval instruction for SciFact. `--instruction` on the quality command explicitly overrides the Qwen instruction. Other families retain their official prefixes and pooling.

## Results and limits

~~~powershell
& $py scripts/generate_report.py --results (Join-Path $work 'results') --reports reports
~~~

Raw JSON records source/configuration/dataset hashes, actual runtime settings, hardware identity, Git state and commands. CSV and Markdown/HTML reports preserve measured values. Rows lacking a passed validation receipt are exploratory. Compare rows only when dataset, representation, machine, threads and source/configuration identities match.

The report defers a production recommendation until complete VDI measurements and representative company judgements exist. Personal-PC results in this repository are development evidence. Quantization is not called effectively lossless from cosine similarity alone.

- [Executed local validation](reports/LOCAL_VALIDATION.md)
- [VDI agent runbook](reports/VDI_AGENT_GUIDE.md)
- [Methodology](reports/METHODOLOGY.md)
- [Architecture](ARCHITECTURE.md)
- [Third-party sources and licenses](THIRD_PARTY.md)

## Add licensed PitchBook data later

Keep exports and results outside the public repository, or use the ignored `data/private/` directory. Convert an authorized export into the dataset schema in [ARCHITECTURE.md](ARCHITECTURE.md), preserve sparse descriptions and original keywords, and label product/customer/exclusion criteria independently of model rankings. Unknown facts must remain unknown. The current public pilot does not validate transfer to actual PitchBook descriptions.

## Maintainer preparation

`scripts/prepare_assets.py --work-dir <DIR> --download-upstream` is a maintainer-only Hugging Face packaging tool. Upstream revisions, weight hashes, custom-code revisions and CPU compatibility modifications are recorded in the manifests and each package's `PROVENANCE.json`. The VDI does not run this tool. Model licenses travel with every package. See the agent guide for reproducibility checks and acceptance criteria.

