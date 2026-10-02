from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import numpy as np


class AdapterError(RuntimeError):
    pass


class UnsupportedHardware(AdapterError):
    pass


def prepare_text(text: str, convention: dict[str, Any], instruction: str | None = None) -> str:
    """Render a manifest-pinned query or document convention.

    Supported fields are deliberately small and explicit. The model catalog
    owns the exact strings; the adapter never guesses from the model name.
    """
    if not isinstance(text, str) or not text.strip():
        raise AdapterError("text must be a non-empty string")
    prefix = convention.get("prefix") or ""
    suffix = convention.get("suffix") or ""
    template = convention.get("instruction_template")
    if template:
        if instruction is None:
            instruction = convention.get("default_instruction")
        if not instruction:
            raise AdapterError("this model requires a query instruction")
        try:
            return template.format(instruction=instruction, text=text)
        except KeyError as exc:
            raise AdapterError(f"unsupported instruction template field: {exc}") from exc
    return f"{prefix}{text}{suffix}"


def normalize_float32(vectors: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise AdapterError(f"expected 2-D embeddings, got {array.shape}")
    if not np.isfinite(array).all():
        raise AdapterError("embedding contains NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise AdapterError("embedding contains a zero vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class BaseAdapter(ABC):
    def __init__(self, model: dict[str, Any], variant: dict[str, Any]) -> None:
        self.model = model
        self.variant = variant
        self.dimension = int(variant["dimensions"])
        self.max_tokens = int(variant.get("max_tokens") or model["architecture"]["max_tokens"])

    def prepare_queries(self, texts: Iterable[str], instruction: str | None = None) -> list[str]:
        convention = self.model["text_convention"]["query"]
        return [prepare_text(text, convention, instruction) for text in texts]

    def prepare_documents(self, texts: Iterable[str]) -> list[str]:
        convention = self.model["text_convention"]["document"]
        return [prepare_text(text, convention) for text in texts]

    def encode_queries(self, texts: Sequence[str], *, instruction: str | None = None,
                       batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode(self.prepare_queries(texts, instruction), batch_size=batch_size))

    def encode_documents(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        return self._finish(self._encode(self.prepare_documents(texts), batch_size=batch_size))

    def _finish(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise AdapterError(f"backend returned shape {vectors.shape}")
        if vectors.shape[1] < self.dimension:
            raise AdapterError(f"backend returned {vectors.shape[1]}D, requested {self.dimension}D")
        processing = self.model["architecture"].get("reduced_dimension_processing", "truncate_then_l2")
        native_dimensions = self.model["architecture"].get("native_dimensions", [vectors.shape[1]])
        is_reduced = self.dimension < max(native_dimensions)
        if is_reduced and processing == "full_layer_norm_then_truncate_l2":
            mean = vectors.mean(axis=1, keepdims=True)
            variance = ((vectors - mean) ** 2).mean(axis=1, keepdims=True)
            vectors = (vectors - mean) / np.sqrt(variance + 1e-5)
        elif is_reduced and processing != "truncate_then_l2":
            raise AdapterError(f"unsupported reduced-dimension processing: {processing}")
        vectors = vectors[:, : self.dimension]
        return normalize_float32(vectors)

    @abstractmethod
    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        raise NotImplementedError

