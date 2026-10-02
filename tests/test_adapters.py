import numpy as np
import pytest

from embedding_bench.adapters.base import AdapterError, BaseAdapter, normalize_float32, prepare_text
from embedding_bench.adapters.runtime import pool_numpy
from embedding_bench.hardware import cpu_feature_flags


def test_prepare_prefix_and_instruction_are_manifest_driven():
    assert prepare_text("abc", {"prefix": "query: "}) == "query: abc"
    convention = {"instruction_template": "Instruct: {instruction}\nQuery:{text}",
                  "default_instruction": "retrieve evidence"}
    assert prepare_text("abc", convention) == "Instruct: retrieve evidence\nQuery:abc"
    assert prepare_text("abc", convention, "find companies") == "Instruct: find companies\nQuery:abc"


def test_required_instruction_fails_closed():
    with pytest.raises(AdapterError):
        prepare_text("abc", {"instruction_template": "{instruction}: {text}"})


def test_normalization_rejects_zero_and_nan():
    value = normalize_float32(np.array([[3.0, 4.0]], dtype=np.float64))
    assert value.dtype == np.float32
    assert np.allclose(np.linalg.norm(value, axis=1), 1)
    with pytest.raises(AdapterError):
        normalize_float32(np.zeros((1, 2)))
    with pytest.raises(AdapterError):
        normalize_float32(np.array([[np.nan, 1.0]]))


def test_numpy_pooling_respects_attention_mask():
    hidden = np.array([[[1, 0], [3, 2], [100, 100]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert np.allclose(pool_numpy(hidden, mask, "masked_mean"), [[2, 1]])
    assert np.allclose(pool_numpy(hidden, mask, "cls"), [[1, 0]])
    assert np.allclose(pool_numpy(hidden, mask, "last_nonpadding"), [[3, 2]])


class FakeAdapter(BaseAdapter):
    def _encode(self, texts, *, batch_size):
        return np.tile(np.array([[1., 2., 9.]], dtype=np.float32), (len(texts), 1))


def test_nomic_reduced_dimension_layer_norm_happens_before_truncation():
    model = {"architecture": {"native_dimensions": [3], "max_tokens": 10,
                               "reduced_dimension_processing": "full_layer_norm_then_truncate_l2"},
             "text_convention": {"query": {}, "document": {}}}
    adapter = FakeAdapter(model, {"dimensions": 2})
    actual = adapter.encode_documents(["x"])[0]
    full = np.array([1., 2., 9.], dtype=np.float32)
    normalized = (full - full.mean()) / np.sqrt(((full - full.mean()) ** 2).mean() + 1e-5)
    expected = normalized[:2] / np.linalg.norm(normalized[:2])
    assert np.allclose(actual, expected)


def test_cpu_features_are_normalized_strings():
    assert all(value == value.lower() for value in cpu_feature_flags())


def test_qwen_catalog_uses_default_and_explicit_instruction():
    from pathlib import Path
    from embedding_bench.config import Catalog
    convention = Catalog.load(Path(__file__).resolve().parents[1] / "configs").models[
        "qwen3-embedding-0.6b"]["text_convention"]["query"]
    assert prepare_text("steel companies", convention) == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: steel companies")
    assert prepare_text("steel companies", convention, "Find matching companies") == (
        "Instruct: Find matching companies\nQuery: steel companies")


def test_last_nonpadding_handles_left_padding():
    hidden = np.array([[[99, 99], [1, 2], [3, 4]]], dtype=np.float32)
    assert np.allclose(pool_numpy(hidden, np.array([[0, 1, 1]]), "last_nonpadding"), [[3, 4]])


def test_qwen_onnx_feed_positions_and_empty_cache_follow_config():
    from types import SimpleNamespace
    from embedding_bench.adapters.runtime import onnx_feed
    inputs = [SimpleNamespace(name=name, type=dtype, shape=shape) for name, dtype, shape in (
        ("input_ids", "tensor(int64)", ["batch", "sequence"]),
        ("attention_mask", "tensor(int64)", ["batch", "sequence"]),
        ("position_ids", "tensor(int64)", ["batch", "sequence"]),
        ("past_key_values.0.key", "tensor(float)", ["batch", 8, "past", 128]),
        ("past_key_values.0.value", "tensor(float)", ["batch", 8, "past", 128]))]
    tokens = {"input_ids": np.array([[0, 1, 2], [3, 4, 0]]),
              "attention_mask": np.array([[0, 1, 1], [1, 1, 0]])}
    feed = onnx_feed(tokens, inputs, {"model_type": "qwen3", "num_key_value_heads": 8, "head_dim": 128})
    assert feed["position_ids"].tolist() == [[0, 0, 1], [0, 1, 1]]
    assert feed["past_key_values.0.key"].shape == (2, 8, 0, 128)
    assert feed["past_key_values.0.value"].dtype == np.float32
    inputs.append(SimpleNamespace(name="unsupported_required", type="tensor(int64)", shape=[]))
    with pytest.raises(AdapterError, match="required ONNX input"):
        onnx_feed(tokens, inputs, {"model_type": "qwen3", "num_key_value_heads": 8, "head_dim": 128})

