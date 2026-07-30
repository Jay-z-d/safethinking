"""Input loading utilities for paired and flat query JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_records(path: Path, query_field: str) -> Iterable[dict[str, Any]]:
    """Yield normalized query records from WildJailbreak pairs or flat JSONL."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "benign_prompt" in row and "harmful_prompt" in row:
                pair_id = row.get("pair_id")
                if not pair_id:
                    raise ValueError(f"Missing pair_id on line {line_number}")
                for source_field, gold_label in (
                    ("benign_prompt", "benign"),
                    ("harmful_prompt", "harmful"),
                ):
                    query = row.get(source_field)
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError(f"Missing {source_field} on line {line_number}")
                    yield {
                        "record_id": f"{pair_id}:{gold_label}",
                        "pair_id": str(pair_id),
                        "source_line": line_number,
                        "source_field": source_field,
                        "query": query,
                        "gold_label": gold_label,
                    }
                continue

            query = row.get(query_field)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Missing {query_field!r} on line {line_number}; expected "
                    "WildJailbreak pair fields or a flat query field"
                )
            record_id = row.get("id") or row.get("record_id") or row.get("pair_id")
            if record_id is None:
                record_id = str(line_number)
            yield {
                "record_id": str(record_id),
                "pair_id": str(row.get("pair_id", record_id)),
                "source_line": line_number,
                "source_field": query_field,
                "query": query,
                "gold_label": row.get("gold_label") or row.get("label"),
            }
