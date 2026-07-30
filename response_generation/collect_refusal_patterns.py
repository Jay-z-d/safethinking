#!/usr/bin/env python3
"""Collect method-specific refusal patterns from generated refusal-induction rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REFUSAL_CUE_RE = re.compile(
    r"^\s*(i\s*(?:am|'m|’m)\s*sorry|sorry|i\s*(?:can(?:not|'t|’t)|won(?:'t|’t)|am\s+unable|"
    r"cannot|can't|won't|unable)|for\s+safety\s+reasons|as\s+an\s+ai|"
    r"抱歉|对不起|我不能|我无法|无法协助|不能帮助)",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", help="Override method name in output.")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=24)
    parser.add_argument("--max-patterns", type=int, default=50)
    parser.add_argument(
        "--require-refusal-cue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only prefixes that look like refusals.",
    )
    return parser.parse_args()


def normalized_text(row: dict[str, Any]) -> str:
    text = row.get("final_response") or row.get("response") or row.get("raw_response")
    return str(text or "").strip()


def words(text: str) -> list[str]:
    return [part for part in re.split(r"\s+", text.strip()) if part]


def extract_pattern(text: str, min_words: int, max_words: int) -> str | None:
    if not text:
        return None
    first_sentence = SENTENCE_END_RE.split(text, maxsplit=1)[0].strip()
    parts = words(first_sentence)
    if len(parts) < min_words:
        parts = words(text)
    if len(parts) < min_words:
        return None
    return " ".join(parts[:max_words]).strip()


def main() -> None:
    args = parse_args()
    counter: Counter[str] = Counter()
    method_names: Counter[str] = Counter()
    rows = 0
    kept = 0

    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            method_names[str(row.get("method_name") or row.get("method"))] += 1
            text = normalized_text(row)
            if args.require_refusal_cue and not REFUSAL_CUE_RE.search(text):
                continue
            pattern = extract_pattern(text, args.min_words, args.max_words)
            if pattern is None:
                continue
            counter[pattern] += 1
            kept += 1

    method_name = args.method_name
    if method_name is None and method_names:
        method_name = method_names.most_common(1)[0][0]
    result = {
        "method_name": method_name,
        "source": str(args.input),
        "rows": rows,
        "kept_refusal_like_rows": kept,
        "require_refusal_cue": args.require_refusal_cue,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_patterns": args.max_patterns,
        "refusal_patterns": [
            {"text": text, "count": count}
            for text, count in counter.most_common(args.max_patterns)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
