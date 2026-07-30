#!/usr/bin/env python3
"""Merge newly computed refusal fields into existing guard-scored rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REFUSAL_FIELDS = (
    "refusal_scoring_context_type",
    "refusal_logprob_sum",
    "refusal_pattern_aggregation",
    "refusal_pattern_count",
    "refusal_pattern_scores",
)
OLD_REFUSAL_FIELDS = ("refusal_score", "refusal_calibration", "refusal_topk_logprob")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-input", type=Path, required=True)
    parser.add_argument("--refusal-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    pair_id = str(row.get("pair_id") or "")
    run_id = int(row.get("run_id", row.get("seed", 0)))
    side = str(row.get("side") or row.get("gold_label") or "")
    if not pair_id or not side:
        raise ValueError(f"Cannot construct merge key from row: {row}")
    return pair_id, run_id, side


def main() -> None:
    args = parse_args()
    refusal_rows = {row_key(row): row for row in read_jsonl(args.refusal_input)}
    if not refusal_rows:
        raise ValueError("Refusal input contains no rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    matched = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for guard_row in read_jsonl(args.guard_input):
            key = row_key(guard_row)
            refusal_row = refusal_rows.get(key)
            if refusal_row is None:
                raise ValueError(f"No refusal row for guard row {key}")
            merged = dict(guard_row)
            for field in OLD_REFUSAL_FIELDS + REFUSAL_FIELDS:
                merged.pop(field, None)
            for field in REFUSAL_FIELDS:
                if field in refusal_row:
                    merged[field] = refusal_row[field]
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1
            matched += 1

    if matched != len(refusal_rows):
        raise ValueError(f"Refusal rows not matched: {len(refusal_rows) - matched}")
    print(json.dumps({"output": str(args.output), "rows": written, "matched": matched}, indent=2))


if __name__ == "__main__":
    main()
