from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .profiles import PROFILES, QUERY_FRAMES


OPENERS = [
    "The business", "This privately held company", "The organization", "The enterprise"
]
VERBS = ["provides", "develops", "manufactures", "specializes in"]
FILLER_OFFERINGS = [
    "packaging equipment", "facility maintenance", "workflow software", "precision components",
    "specialty coatings", "logistics services", "quality consulting", "network monitoring"
]
FILLER_CUSTOMERS = [
    "regional manufacturers", "commercial property operators", "healthcare providers",
    "consumer brands", "public agencies", "industrial distributors"
]
FILLER_CAPABILITIES = [
    "custom engineering", "field installation", "regulatory reporting", "predictive analytics",
    "contract production", "supplier coordination"
]


def _record(company_id: str, profile: dict[str, str], customer: str, *, kind: str,
            rng: random.Random) -> dict[str, Any]:
    offering = profile["offering"]
    capability = profile["capability"]
    document_offering = profile.get("doc_offering", offering)
    document_capability = profile.get("doc_capability", capability)
    core = f"{rng.choice(OPENERS)} {document_offering}."
    market = f"Its work is designed for {customer}."
    capabilities = f"Its specialists {document_capability}."
    description = f"{core} {market}" if kind != "positive" or rng.random() < .5 else f"{market} {core}"
    return {
        "company_id": company_id,
        "track": "controlled",
        "name": f"{profile['slug'].replace('-', ' ').title()} {company_id[-4:]}",
        "description": description,
        "keywords": [],
        "fields": {
            "core_business": core,
            "products_services": f"The main service {document_offering}.",
            "customers_end_markets": market,
            "capabilities": capabilities,
        },
        "attributes": {
            "sector": profile["sector"], "subsector": profile["offering"],
            "offering": [offering], "capability": [capability],
            "customer_type": [customer], "end_market": [customer],
            "business_model": ["business-to-business"], "geography": ["United States"],
            "negative_attributes": [profile["excluded_customer"]] if kind == "hard_negative" else [],
        },
        "source_refs": [],
        "generation": {"generator": "controlled-v1", "kind": kind},
    }


def _filler(company_id: str, rng: random.Random) -> dict[str, Any]:
    offering = rng.choice(FILLER_OFFERINGS)
    customer = rng.choice(FILLER_CUSTOMERS)
    capability = rng.choice(FILLER_CAPABILITIES)
    profile = {
        "slug": offering.replace(" ", "-"), "sector": "diversified",
        "offering": offering, "capability": capability,
        "customer": customer, "excluded_customer": "none"
    }
    return _record(company_id, profile, customer, kind="filler", rng=rng)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build_controlled_dataset(output_dir: str | Path, *, company_count: int = 2400,
                             queries_per_profile: int = 10,
                             seed: int = 20261002,
                             description_words: int | None = None) -> Path:
    if company_count < len(PROFILES) * 16:
        raise ValueError(f"company_count must be at least {len(PROFILES) * 16}")
    if not 1 <= queries_per_profile <= len(QUERY_FRAMES):
        raise ValueError(f"queries_per_profile must be 1..{len(QUERY_FRAMES)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    companies: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: list[tuple[str, str, int, str, str]] = []
    hard: list[dict[str, Any]] = []

    counter = 1
    for profile in PROFILES:
        positives: list[str] = []
        hard_ids: list[str] = []
        for _ in range(8):
            company_id = f"syn-co-{counter:06d}"
            companies.append(_record(company_id, profile, profile["customer"], kind="positive", rng=rng))
            positives.append(company_id)
            counter += 1
        for _ in range(8):
            company_id = f"syn-co-{counter:06d}"
            companies.append(_record(company_id, profile, profile["excluded_customer"], kind="hard_negative", rng=rng))
            hard_ids.append(company_id)
            counter += 1
        for frame_index, frame in enumerate(QUERY_FRAMES[:queries_per_profile], 1):
            query_id = f"syn-q-{profile['slug']}-{frame_index:02d}"
            queries.append({
                "query_id": query_id,
                "text": frame.format(**profile),
                "category": "capability_customer_exclusion",
                "constraints": {
                    "must": [
                        {"field": "offering", "op": "contains", "value": profile["offering"]},
                        {"field": "capability", "op": "contains", "value": profile["capability"]},
                        {"field": "customer_type", "op": "contains", "value": profile["customer"]},
                    ],
                    "must_not": [
                        {"field": "customer_type", "op": "contains", "value": profile["excluded_customer"]}
                    ],
                },
                "source_refs": [], "judgement_scope": "exhaustive",
            })
            for company_id in positives:
                qrels.append((query_id, company_id, 2, "controlled_constraints", f"{query_id}:{company_id}"))
            for company_id in hard_ids:
                rationale_id = f"{query_id}:{company_id}"
                qrels.append((query_id, company_id, 0, "controlled_constraints", rationale_id))
                hard.append({
                    "query_id": query_id, "company_id": company_id,
                    "shared_facets": [profile["offering"], profile["capability"]],
                    "violated_constraints": [f"customer is {profile['excluded_customer']}"],
                    "rationale_id": rationale_id,
                })

    while len(companies) < company_count:
        company_id = f"syn-co-{counter:06d}"
        companies.append(_filler(company_id, rng))
        counter += 1

    if description_words:
        neutral = ("The company describes its operating history, regional service process, account "
                   "management approach, staffing model, supplier relationships, quality procedures, "
                   "and routine administrative capabilities. ")
        for company in companies:
            original = company["description"]
            padding = neutral
            while len((original + padding).split()) < description_words:
                padding += neutral
            padding = " ".join(padding.split()[: max(0, description_words - len(original.split()))])
            numeric_id = int(company["company_id"].rsplit("-", 1)[-1])
            company["description"] = f"{padding} {original}" if numeric_id % 2 else f"{original} {padding}"

    rng.shuffle(companies)
    _write_jsonl(output / "companies.jsonl", companies)
    _write_jsonl(output / "queries.jsonl", queries)
    _write_jsonl(output / "hard_negatives.jsonl", hard)
    (output / "sources.jsonl").write_text("", encoding="utf-8")
    lines = ["query_id\tcompany_id\trelevance\tjudgement_source\trationale_id\n"]
    lines.extend("\t".join(map(str, row)) + "\n" for row in qrels)
    (output / "qrels.tsv").write_text("".join(lines), encoding="utf-8")
    files = ["companies.jsonl", "queries.jsonl", "qrels.tsv", "hard_negatives.jsonl", "sources.jsonl"]
    manifest = {
        "schema_version": 1,
        "dataset_id": "controlled-company-screening-v1" + (f"-{description_words}w" if description_words else ""),
        "track": "controlled", "license": "CC0-1.0", "seed": seed,
        "judgement_completeness": "exhaustive", "company_count": len(companies),
        "query_count": len(queries), "generator": "controlled-v1",
        "description_target_words": description_words,
        "files": {name: _sha256(output / name) for name in files},
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

