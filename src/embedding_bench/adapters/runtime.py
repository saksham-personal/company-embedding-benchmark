from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import AdapterError, BaseAdapter, UnsupportedHardware


def _masked_mean(hidden: Any, mask: Any) -> Any:
    import torch
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)


def _last_token(hidden: Any, mask: Any) -> Any:
    import torch
    positions = torch.arange(mask.shape[1], device=mask.device)
    lengths = (mask * positions).max(dim=1).values
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, lengths]


def pool_torch(output: Any, mask: Any, method: str) -> Any:
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    if method in {"mean", "masked_mean"}:
        return _masked_mean(hidden, mask)
    if method == "cls":
        return hidden[:, 0]
    if method in {"last", "last_token", "last_nonpadding"}:
        return _last_token(hidden, mask)
    raise AdapterError(f"unsupported pooling method: {method}")


def pool_numpy(hidden: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    if hidden.ndim == 2:
        return hidden
    if hidden.ndim != 3:
        raise AdapterError(f"unsupported ONNX output shape: {hidden.shape}")
    if method in {"mean", "masked_mean"}:
        expanded = mask[..., None].astype(hidden.dtype)
        return (hidden * expanded).sum(1) / np.maximum(expanded.sum(1), 1e-9)
    if method == "cls":
        return hidden[:, 0]
    if method in {"last", "last_token", "last_nonpadding"}:
        lengths = (mask * np.arange(mask.shape[1])).max(1).astype(np.int64)
        return hidden[np.arange(hidden.shape[0]), lengths]
    raise AdapterError(f"unsupported pooling method: {method}")


def onnx_feed(tokens: Any, inputs: Sequence[Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Supply a full-sequence embedding export, including an empty decoder cache."""
    feed = {}
    mask = np.asarray(tokens["attention_mask"], dtype=np.int64)
    dtypes = {"tensor(int64)": np.int64, "tensor(float)": np.float32,
              "tensor(float16)": np.float16, "tensor(bool)": np.bool_}
    for item in inputs:
        if item.type not in dtypes:
            raise AdapterError(f"unsupported ONNX input type: {item.name}: {item.type}")
        dtype = dtypes[item.type]
        if item.name in tokens:
            feed[item.name] = np.asarray(tokens[item.name], dtype=dtype)
        elif item.name == "position_ids":
            feed[item.name] = np.maximum(np.cumsum(mask, axis=1) - 1, 0).astype(dtype)
        elif item.name.startswith("past_key_values.") and item.name.endswith((".key", ".value")):
            if config.get("model_type") != "qwen3" or len(item.shape) != 4:
                raise AdapterError(f"unsupported decoder cache input: {item.name}")
            heads, width = config["num_key_value_heads"], config["head_dim"]
            if item.shape[1] != heads or item.shape[3] != width:
                raise AdapterError(f"decoder cache shape does not match pinned config: {item.name}")
            feed[item.name] = np.empty((mask.shape[0], heads, 0, width), dtype=dtype)
        else:
            raise AdapterError(f"required ONNX input is unsupported: {item.name}")
    return feed


class TransformersAdapter(BaseAdapter):
    def __init__(self, model: dict[str, Any], variant: dict[str, Any], model_dir: Path,
                 artifact: dict[str, Any], threads: int | None = None) -> None:
        super().__init__(model, variant)
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise AdapterError("install the transformers optional dependencies") from exc
        if threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        trust = bool(variant.get("trust_remote_code", model.get("requires_remote_code", False)))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=trust
        )
        if trust and artifact.get("cpu_safe_init", True):
            # Keep parameters empty while constructing every nonpersistent
            # position/rotary buffer on CPU. Transformers' generic meta-device
            # constructor does not reliably restore buffers in pinned custom
            # architectures. Assigning checkpoint tensors also avoids a second
            # full copy of large embedding tables on Windows.
            from accelerate import init_empty_weights
            from safetensors.torch import load_file
            from transformers.initialization import no_init_weights
            config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            with init_empty_weights(include_buffers=False), no_init_weights():
                self.backend_model = AutoModel.from_config(config, trust_remote_code=True)
            weight_name = artifact.get("entrypoint") or "model.safetensors"
            missing, unexpected = self.backend_model.load_state_dict(
                load_file(str(model_dir / weight_name), device="cpu", backend="pread"), strict=False, assign=True
            )
            if unexpected:
                raise AdapterError(f"unexpected custom-code checkpoint keys: {unexpected[:10]}")
            persistent_missing = [
                name for name in missing
                if not name.endswith(("position_ids", "cos_cached", "sin_cached"))
                and not (model["model_id"] == "nomic-embed-text-v1.5" and name.startswith("pooler."))
            ]
            if persistent_missing:
                raise AdapterError(f"missing custom-code checkpoint keys: {persistent_missing[:10]}")
            if model["model_id"] == "nomic-embed-text-v1.5" and any(name.startswith("pooler.") for name in missing):
                # The embedding checkpoint omits BERT's optional pooler; its
                # official masked-mean embedding uses last_hidden_state only.
                self.backend_model.pooler = None
            if any(buffer.device.type == "meta" for buffer in self.backend_model.buffers()):
                raise AdapterError("a nonpersistent model buffer was left uninitialized")
            self.backend_model.eval()
        else:
            self.backend_model = AutoModel.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=trust
            ).eval()
        self.pooling = variant.get("pooling_override") or model["architecture"]["pooling"]
        self.runtime = {"backend": "transformers", "torch": torch.__version__, "device": "cpu",
                        "threads": torch.get_num_threads(), "inter_op_threads": torch.get_num_interop_threads(),
                        "parameter_dtype": str(next(self.backend_model.parameters()).dtype)}
        self.torch = torch

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=self.max_tokens, return_tensors="pt"
            )
            with self.torch.inference_mode():
                output = self.backend_model(**tokens)
                embeddings = pool_torch(output, tokens["attention_mask"], self.pooling)
            chunks.append(embeddings.detach().cpu().float().numpy())
        return np.concatenate(chunks, axis=0)


class OnnxAdapter(BaseAdapter):
    def __init__(self, model: dict[str, Any], variant: dict[str, Any], model_dir: Path,
                 artifact: dict[str, Any], threads: int | None = None) -> None:
        super().__init__(model, variant)
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise AdapterError("install the onnx optional dependencies") from exc
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_rel = variant.get("model_file") or artifact.get("entrypoint") or "model.onnx"
        self.session = ort.InferenceSession(
            str(model_dir / onnx_rel), sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        trust = bool(variant.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=trust
        )
        self.inputs = self.session.get_inputs()
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.output_name = variant.get("output_name") or artifact.get("output_name")
        self.pooling = variant.get("pooling_override") or model["architecture"]["pooling"]
        self.runtime = {"backend": "onnxruntime", "version": ort.__version__,
                        "providers": self.session.get_providers(), "threads": options.intra_op_num_threads,
                        "inter_op_threads": options.inter_op_num_threads}

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=self.max_tokens, return_tensors="np"
            )
            feed = onnx_feed(tokens, self.inputs, self.config)
            selected_output = self.output_name or self.output_names[0]
            outputs = self.session.run([selected_output], feed)
            if self.output_name:
                if self.output_name not in self.output_names:
                    raise AdapterError(f"ONNX output {self.output_name!r} is unavailable: {self.output_names}")
            output = outputs[0]
            chunks.append(pool_numpy(output, tokens["attention_mask"], self.pooling))
        return np.concatenate(chunks, axis=0)


class LlamaCppAdapter(BaseAdapter):
    def __init__(self, model: dict[str, Any], variant: dict[str, Any], model_dir: Path,
                 artifact: dict[str, Any], threads: int | None = None) -> None:
        super().__init__(model, variant)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise AdapterError("install the gguf optional dependencies") from exc
        gguf_rel = variant.get("model_file") or artifact.get("entrypoint")
        if not gguf_rel:
            raise AdapterError("GGUF artifact needs entrypoint or variant model_file")
        self.backend_model = Llama(
            model_path=str(model_dir / gguf_rel), embedding=True,
            n_ctx=self.max_tokens, n_threads=threads
        )

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        rows = []
        for text in texts:
            result = self.backend_model.create_embedding(text)
            rows.append(result["data"][0]["embedding"])
        return np.asarray(rows, dtype=np.float32)


class LlamaServerAdapter(BaseAdapter):
    """Pinned official llama.cpp server adapter for GGUF embeddings."""

    def __init__(self, model: dict[str, Any], variant: dict[str, Any], model_dir: Path,
                 artifact: dict[str, Any], server_path: Path, threads: int | None = None) -> None:
        super().__init__(model, variant)
        gguf_rel = variant.get("model_file") or artifact.get("entrypoint")
        if not gguf_rel:
            raise AdapterError("GGUF artifact needs entrypoint or variant model_file")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            str(server_path), "-m", str(model_dir / gguf_rel), "-ngl", "0",
            "--embedding", "--pooling", "mean", "-c", str(self.max_tokens),
            "-t", str(threads or 1), "-tb", str(threads or 1), "--host", "127.0.0.1", "--port", str(port),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.server_log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self.process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=self.server_log,
            text=True, creationflags=creationflags
        )
        self.runtime = {"backend": "llama.cpp-server", "executable": str(server_path),
                        "threads": threads or 1, "gpu_layers": 0,
                        "context_tokens": self.max_tokens, "request_policy": "serial_single_document"}
        self.base_url = f"http://127.0.0.1:{port}"
        atexit.register(self.close)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.server_log.seek(0)
                detail = self.server_log.read()[-2000:]
                self.close()
                raise AdapterError(f"llama-server exited during startup: {detail}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        else:
            self.close()
            raise AdapterError("llama-server did not become healthy within 60 seconds")

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log = getattr(self, "server_log", None)
        if log and not log.closed:
            log.close()

    def _request_one(self, text: str) -> list[float]:
        payload = json.dumps({"content": text}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/embedding", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.load(response)
        embedding = None
        if isinstance(value, list) and value and "embedding" in value[0]:
            embedding = value[0]["embedding"]
        elif isinstance(value, dict) and value.get("data"):
            embedding = value["data"][0]["embedding"]
        elif isinstance(value, dict) and "embedding" in value:
            embedding = value["embedding"]
        if embedding is None:
            raise AdapterError(f"unexpected llama-server embedding response shape: {type(value).__name__}")
        vector = np.asarray(embedding, dtype=np.float32).squeeze()
        if vector.ndim != 1:
            raise AdapterError(f"unexpected llama-server vector shape: {vector.shape}")
        return vector.tolist()

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return np.asarray([self._request_one(text) for text in texts], dtype=np.float32)


def load_adapter(catalog: Any, variant_id: str, model_root: str | Path,
                 threads: int | None = None) -> BaseAdapter:
    try:
        variant = catalog.variants[variant_id]
        model = catalog.models[variant["model_id"]]
        artifact = catalog.artifacts[variant["artifact_id"]]
    except KeyError as exc:
        raise AdapterError(f"unknown variant or catalog reference: {exc}") from exc
    status = artifact.get("status", "ready")
    if status not in {"ready", "available"}:
        raise AdapterError(f"artifact {variant['artifact_id']} is {status}: {artifact.get('reason', '')}")
    required = {str(value).lower() for value in artifact.get("cpu_features_required", [])}
    if required:
        from embedding_bench.hardware import cpu_feature_flags
        missing = required - cpu_feature_flags()
        if missing:
            raise UnsupportedHardware(
                f"artifact {variant['artifact_id']} requires unavailable CPU/OS features: {sorted(missing)}"
            )
    model_dir = Path(model_root) / variant["artifact_id"]
    if not model_dir.is_dir():
        raise AdapterError(f"model directory does not exist: {model_dir}")
    backend = variant["backend"]
    if backend in {"transformers", "sentence_transformers"}:
        return TransformersAdapter(model, variant, model_dir, artifact, threads)
    if backend == "onnxruntime":
        return OnnxAdapter(model, variant, model_dir, artifact, threads)
    if backend in {"llama_cpp", "llama.cpp", "gguf"}:
        server = variant.get("llama_server") or os.environ.get("LLAMA_SERVER")
        if server:
            server_path = Path(server)
            if not server_path.is_file():
                raise AdapterError(f"llama-server executable does not exist: {server_path}")
            return LlamaServerAdapter(model, variant, model_dir, artifact, server_path, threads)
        return LlamaCppAdapter(model, variant, model_dir, artifact, threads)
    raise AdapterError(f"unsupported backend: {backend}")

