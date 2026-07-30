#!/usr/bin/env python3
"""Extract WildJailbreak vanilla harmful prompts for refusal-pattern induction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-type", default="vanilla_harmful")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input_tsv.open(newline="", encoding="utf-8") as source, args.output.open(
        "w",
        encoding="utf-8",
    ) as target:
        reader = csv.DictReader(source, delimiter="\t")
        if "vanilla" not in (reader.fieldnames or []):
            raise ValueError(f"{args.input_tsv} has no 'vanilla' column")
        for row_number, row in enumerate(reader, start=2):
            if row.get("data_type") != args.data_type:
                continue
            prompt = row.get("vanilla")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            count += 1
            target.write(
                json.dumps(
                    {
                        "id": f"{args.data_type}_{count:06d}",
                        "source_line": row_number,
                        "query": prompt,
                        "gold_label": "harmful",
                        "source": "wildjailbreak_train",
                        "data_type": args.data_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.limit is not None and count >= args.limit:
                break
    print(f"Wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
