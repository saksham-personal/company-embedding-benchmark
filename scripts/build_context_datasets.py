from __future__ import annotations

import argparse
from pathlib import Path

from embedding_bench.datasets import build_controlled_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic context-length views")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--companies", type=int, default=2400)
    parser.add_argument("--queries-per-profile", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    for label, words in (("short", 150), ("medium", 480), ("long", 1800), ("very-long", 4200)):
        path = args.output_root / label
        build_controlled_dataset(path, company_count=args.companies,
                                 queries_per_profile=args.queries_per_profile,
                                 seed=args.seed, description_words=words)
        print(path)
    return 0


raise SystemExit(main())

