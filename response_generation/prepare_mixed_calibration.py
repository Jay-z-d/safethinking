#!/usr/bin/env python3
"""Build a balanced vanilla benign/harmful refusal calibration set."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benign-limit", type=int, default=500)
    parser.add_argument("--harmful-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--benign-field", default="benign_vanilla")
    parser.add_argument("--harmful-field", default="harmful_vanilla")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collect_unique(
    path: Path,
    field: str,
    label: str,
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = normalize_text(str(row.get(field) or ""))
            if not query or query in seen:
                continue
            seen.add(query)
            records.append(
                {
                    "query": query,
                    "gold_label": label,
                    "data_type": f"vanilla_{label}",
                    "source": "wildjailbreak_pairs_vanilla_fields",
                    "source_pair_id": row.get("pair_id"),
                    "source_line": line_number,
                    "source_field": field,
                }
            )
    return records


def sample(records: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[: min(limit, len(shuffled))]


def main() -> None:
    args = parse_args()
    if args.benign_limit < 1 or args.harmful_limit < 1:
        raise ValueError("--benign-limit and --harmful-limit must be positive")

    rng = random.Random(args.seed)
    benign = sample(
        collect_unique(args.input_pairs, args.benign_field, "benign"),
        args.benign_limit,
        rng,
    )
    harmful = sample(
        collect_unique(args.input_pairs, args.harmful_field, "harmful"),
        args.harmful_limit,
        rng,
    )

    mixed = []
    for index, row in enumerate(benign, start=1):
        mixed.append(
            {
                "id": f"mixed_vanilla_benign_{index:06d}",
                "pair_id": f"mixed_vanilla_benign_{index:06d}",
                **row,
            }
        )
    for index, row in enumerate(harmful, start=1):
        mixed.append(
            {
                "id": f"mixed_vanilla_harmful_{index:06d}",
                "pair_id": f"mixed_vanilla_harmful_{index:06d}",
                **row,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "benign": len(benign),
                "harmful": len(harmful),
                "total": len(mixed),
                "seed": args.seed,
                "benign_field": args.benign_field,
                "harmful_field": args.harmful_field,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
