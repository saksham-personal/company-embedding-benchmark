# Third-party artifacts and attribution

All model packages retain an upstream model card, license and `PROVENANCE.json`.
`configs/models.json`, `configs/artifacts.json` and `configs/preparation.json`
pin model revisions, file hashes, code dependencies and packaging changes.

| Model | Official source | License |
|---|---|---|
| BGE | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | MIT |
| E5 | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | MIT |
| GTE | [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) | Apache 2.0 |
| Arctic | [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) | Apache 2.0 |
| Qwen | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache 2.0 |
| Voyage | [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | Apache 2.0 |
| Nomic | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Apache 2.0 |
| EmbeddingGemma | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | Gemma terms; gated and not redistributed here |

Some ONNX derivatives are community exports from Xenova/onnx-community.
Their exact source repositories and revisions are recorded per artifact;
their existence is not a claim of official conversion validation. Conformance
must be established by this kit's reference comparisons.

Required GTE, Arctic, Nomic and Voyage custom Python code retains its license
headers and is pinned in each package. Local packaging rewrites remote code
references to bundled files. Arctic CPU configuration disables unpadding and
memory-efficient attention settings; its original configuration is retained.

The Nomic Q4_K_M GGUF comes from the [official Nomic GGUF repository](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF).
The Windows CPU server is the [official llama.cpp b11146 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11146).

[BEIR SciFact](https://github.com/beir-cellar/beir) uses the official test corpus
and qrels. [SciFact's license](https://github.com/allenai/scifact/blob/master/LICENSE.md)
specifies CC BY 4.0 for claims/evidence and ODC-By 1.0 for abstracts. The release
also hosts the source license file. Preserve the original corpus and qrels;
cite the BEIR and SciFact projects when publishing benchmark results.

The public company summaries, queries and annotations have a separate CC BY
4.0 license and attribution in `data/curated/dataset.json`. Company web prose
and third-party trademarks are excluded from that grant.

The portable environment repacks the tested uv-installed CPython 3.12.10 from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone).
The original build tag was not captured; the exact archive and every payload
file are pinned in `reports/OFFLINE_ENVIRONMENT.json`. CPython's `LICENSE.txt`
is retained. Third-party package licenses remain inside their wheels. This
environment contains CPU PyTorch, Transformers, ONNX Runtime and their locked
dependencies; these do not inherit the benchmark's MIT license.
