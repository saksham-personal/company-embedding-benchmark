# Local Validation and Delivery Record

Validation is being completed on the personal computer. These measurements
cannot stand in for the target VDI or a licensed PitchBook transfer test.

## Completed checks

- Pinned CPython 3.12.10 / CPU PyTorch 2.9.1 / Transformers 5.3.0 / ONNX Runtime 1.24.4 environment installed.
- Latest regression suite: **24 passed**.
- Controlled dataset: 2,400 companies / 120 queries; schema and hash validation passed.
- Public company pilot: 130 records / 38 queries / 215 qrels; curation validation passed.
- SciFact downloaded through GitHub with Hugging Face offline, imported and verified: 5,183 documents / 300 test queries.
- Nomic GGUF downloaded from the new release into a fresh model directory; archive/payload verification and offline encoding passed.
- Arctic reference loading corrected to initialize CPU buffers while assigning empty parameters from the checkpoint; offline smoke check passed.
- Qwen official template now honors the instruction override; repeat determinism and batch-padding checks are distinct.
- Report regression prevents an isolated VDI row or failed result from establishing a model winner.

Full artifact conformance, the fresh portable environment installation and
development retrieval measurements are in progress. This file will be updated
with the final measured scope and failures before delivery.

## Engineering review status

Sol supplied the architecture and primary harness. Terra independently found
the strict-recall call mismatch, ignored Qwen instruction, and incomplete-VDI
leader logic. Those findings were corrected and regression checks pass.

Sol and Terra then reached the account's usage limit. The root agent completed
integration and validation directly. **A final independent Terra approval has
not been issued.** The attempted DeepSeek explorer was unavailable through this
account; no DeepSeek provider work or external provider execution is claimed.

Detailed executed commands and measured values remain in raw result JSON.
