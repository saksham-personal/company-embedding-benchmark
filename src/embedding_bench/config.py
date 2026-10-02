from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ConfigError(f"{path} must be a schema_version 1 JSON object")
    return value


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Catalog:
    models: dict[str, dict[str, Any]]
    artifacts: dict[str, dict[str, Any]]
    variants: dict[str, dict[str, Any]]
    digest: str

    @classmethod
    def load(cls, config_dir: str | Path) -> "Catalog":
        config_dir = Path(config_dir)
        model_doc = load_json(config_dir / "models.json")
        artifact_doc = load_json(config_dir / "artifacts.json")
        variant_doc = load_json(config_dir / "variants.json")
        models = _index_unique(model_doc.get("models", []), "model_id", "models")
        artifacts = _index_unique(artifact_doc.get("artifacts", []), "artifact_id", "artifacts")
        variants = _index_unique(variant_doc.get("variants", []), "variant_id", "variants")
        for artifact_id, artifact in artifacts.items():
            if artifact.get("model_id") not in models:
                raise ConfigError(f"artifact {artifact_id} references unknown model")
        for variant_id, variant in variants.items():
            model_id = variant.get("model_id")
            artifact_id = variant.get("artifact_id")
            if model_id not in models:
                raise ConfigError(f"variant {variant_id} references unknown model")
            if artifact_id not in artifacts:
                raise ConfigError(f"variant {variant_id} references unknown artifact")
            if artifacts[artifact_id].get("model_id") != model_id:
                raise ConfigError(f"variant {variant_id} crosses model/artifact families")
            if variant.get("output_dtype", "float32") != "float32":
                raise ConfigError(f"variant {variant_id}: primary output_dtype must be float32")
            supported = models[model_id].get("architecture", {}).get("supported_dimensions")
            dimensions = variant.get("dimensions")
            if supported and dimensions not in supported:
                raise ConfigError(f"variant {variant_id}: unsupported dimension {dimensions}")
        digest = canonical_digest({"models": models, "artifacts": artifacts, "variants": variants})
        return cls(models, artifacts, variants, digest)


def _index_unique(rows: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ConfigError(f"{label} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise ConfigError(f"every {label} entry needs string {key}")
        if row[key] in result:
            raise ConfigError(f"duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result

