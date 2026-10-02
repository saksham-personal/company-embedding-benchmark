# Local Validation and Delivery Record

## Delivered scope

The GitHub kit contains 19 model packages across seven accessible families and 25 configured artifact/dimension rows, a portable Windows CPU Python environment, three distinct dataset tracks, reference gates, benchmark scripts and a VDI agent guide. EmbeddingGemma is unavailable because upstream access is gated.

The user requested: **“Finish the kit; run larger models on the VDI.”** Arctic, Qwen, Voyage and Nomic full conformance/retrieval and all full SciFact retrieval remain deferred. Their released files and the full standard dataset are available. Target compatibility and conversion validity must be checked using the supplied gates on the VDI.

## Completed development checks

- CPython 3.12.10 / CPU PyTorch 2.9.1 / Transformers 5.3.0 / ONNX Runtime 1.24.4 pinned and installed.
- **37 regression tests passed**; source compilation passed. The two memory regressions failed before the fix and passed afterward. Two sandbox test attempts failed because Windows denied access to temporary fixture directories; the unrestricted retry passed all tests.
- Portable environment downloaded from GitHub into a fresh work directory, archive/payload verified, and all **53 locked CPU wheels** installed with `--no-index --require-hashes`; offline BGE INT8 encoding passed.
- All **23 model/data/runtime/environment assets** matched GitHub's server-computed SHA-256 digests. The kit ZIP is checked against the external `KIT_SHA256.txt`. The ZIP snapshot's [ASSET_VERIFICATION.json](ASSET_VERIFICATION.json) covers those 23 input assets; the [final online verification record](https://github.com/saksham-personal/company-embedding-benchmark/blob/embedding-benchmark/reports/ASSET_VERIFICATION.json) is updated after packaging to include the kit and checksum, for 25 total. An archive cannot contain its own final checksum.
- A fresh Git checkout preserved exact source bytes and curated dataset hashes before the memory fix. Final kit extraction is separately checked before delivery.
- Controlled dataset validated: **2,400 companies / 120 queries**.
- Curated public pilot validated: **130 records / 38 queries / 215 pooled judgments**, 35 strict-eligible queries and 4.35% coverage.
- SciFact downloaded through GitHub with Hugging Face offline; official test split imported and validated: **5,183 documents / 300 queries**.
- Nomic GGUF fresh GitHub download/basic offline encoding and Qwen ONNX 2-query/2-document 1024D smoke encoding passed. These probes do not replace full current reference conformance for deferred families.

## Executed retrieval and performance

Retained measurements use the development Ryzen 5 4600H Windows laptop and **4 CPU threads**. Quality uses document batch **8** and independent query batch **1**.

| Track | Completed measurements |
|---|---|
| Public company pilot | 16 quality rows: eight passed BGE/E5/GTE variants × description-only and description + keywords; all 130 records and 38 queries |
| Controlled | BGE FP32/INT8 paired quality on all 2,400 records / 120 queries |
| SciFact | Dataset import and sampled BGE reference/INT8 conformance only; both full retrieval attempts failed; no score is claimed |
| CPU diagnostics | BGE FP32/INT8 after the memory fix; three document repetitions per available token bucket/batch and 50 single-query measurements |

The public-pilot conformance receipt contains **eight passes and one hardware skip**, repeated successfully after the memory fix. E5 INT8 requires AVX-512F/VNNI absent on this laptop. Separate pre-fix BGE-pair receipts passed on controlled and SciFact text samples. Receipts describe sampled conformance; they do not validate maximum 8K/32K lengths.

### Controlled quantization result

BGE INT8 passed embedding conformance but **failed the one-percentage-point retrieval loss budget**: Recall@50 changed by -1.15 percentage points, Recall@100 by -0.94, and nDCG@20 by -1.10. The script leaves `effectively_lossless` unset. This synthetic-dataset result is not a deployment recommendation.

See [FINAL_BENCHMARK.md](FINAL_BENCHMARK.md), the CSVs and [raw development results](../results/development/) for values and commands. Short personal-PC diagnostics do not establish stable VDI performance.

## Memory correction and evidence versions

CLS pooling returned a view of each full token output. Accumulating those views retained all prior token buffers instead of only pooled embedding vectors, increasing memory with document count and padded sequence length. Commit `4bfd911` copies each pooled chunk in both PyTorch and ONNX adapters. Regression tests inspect chunks before concatenation for compact owned storage and exact vector equality. GPT-6 Luna independently verified the defect and reviewed the correction.

Historical quality measurements keep their original commit `20db533` and source digest `acdcf68972da7b262db3b664ca2088f36958b5930379b289a5fe11c81548ebd1`. Post-fix sampled conformance and CPU diagnostics use commit `4bfd911` and digest `cf1aa346ee12583a6587ab8489896972427497d70743c61d4693ac9bc134bbb2`. Delivery documentation was edited during collection, so provenance reports a dirty Git worktree. The copy changes storage ownership without changing vector values; old receipts still **cannot authorize execution under the current code**. Revalidate on the target VDI. Do not treat the old quality/new speed combination as a fresh paired quantization experiment.

## Failed and deferred work

Arctic reference attempts reported Windows paging-file exhaustion (`1455`). Qwen reference children exited `0xC0000005` without a trace; low commit headroom was observed but the crash cause is unresolved. Shell/Python startup also failed during peak pressure. Both full BGE SciFact retrieval runs failed with allocation errors before the CLS copy correction. The correction has been verified with regression tests and short actual model runs; **full SciFact has not been retried** following the user's deferral.

[DEVELOPMENT_SCOPE.json](DEVELOPMENT_SCOPE.json) retains the larger-reference failure records and scope. [The original execution summary](../results/development/execution-summary.json) remains failed, while [the separate bounded post-fix check](../results/development/postfix/execution-summary.json) completed. Failed/deferred runs are never reported as measured quality.

Complete larger-model conformance, full SciFact retrieval, the comparison matrix, long-context experiments and stable performance on the actual VDI. Add authorized PitchBook descriptions and independently reviewed labels privately later. The public pilot contains large enterprises/brands and incomplete judgments; it is not a verified middle-market sample. Its high known-positive recall cannot establish production recall.

## Engineering and review record

Sol supplied architecture and the primary harness; root integrated packaging, data, fixes, validation and publication. Terra reviewed strict recall, Qwen instructions, report selection, receipt integrity/scope/cache and runner behavior; findings were corrected with regressions. The user then requested **GPT-6 Luna agents and direct execution without AI Flow**. Luna agents reviewed the baseline source/evidence and memory fix. Their review does not assert VDI compatibility.

No DeepSeek work was executed: the attempted provider was unavailable. Changes cover `src/embedding_bench`, `scripts`, `configs`, locked dependencies, `tests`, public data, reports and the VDI runbook. Large binaries are release assets. Runtime decisions preserve CPU-only execution, official prefixes/pooling, explicit query batching, source-bound receipts and separate dataset tracks.

Executed checks include frozen dependency installation; `pytest -q -p no:cacheprovider --basetemp <owned-directory>`; `compileall`; fresh `setup_offline.ps1`; `validate_curated.py`; `validate_artifacts.py` per track; `run_all.py`; `benchmark_quantization.py`; standalone `speed`; `generate_report.py`; GitHub upload/server-digest verification; fresh checkout and source archive checks. Exact inference commands, source/data/catalog hashes, hardware and runtime settings are in raw provenance.

Final review status: **completed baseline evidence and the memory correction reviewed by GPT-6 Luna; larger-model/full-SciFact/VDI work deferred**. No production winner is named.
