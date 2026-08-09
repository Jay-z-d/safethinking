#!/usr/bin/env python3
"""Analyze CoT statistics (presence, length, think tags, stages, refusal overlap)."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from response_generation.collect_refusal_patterns import REFUSAL_CUE_RE


HEADLINE_STAGES = (
    "intention_analysis",
    "sage_safety_analysis",
    "goal_prioritization_internal_thoughts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "boundary_generation.jsonl files; glob patterns such as "
            "dir/*.boundary_generation.jsonl are expanded."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help=(
            "Optional Hugging Face tokenizer for exact token counts; "
            "defaults to a heuristic (~4 characters per token)."
        ),
    )
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = glob.glob(pattern.replace("\\", "/"))
        else:
            matches = [pattern]
        if not matches:
            raise ValueError(f"No input files match: {pattern}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise ValueError(f"Input file does not exist: {path}")
            if path not in seen:
                seen.add(path)
                files.append(path)
    if not files:
        raise ValueError("No input files found")
    return files


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError("No rows found in any input file")
    return rows


class _HeuristicTokenCounter:
    """Token estimate using the common ~4 characters per token heuristic."""

    name = "heuristic_chars_div_4"

    def count(self, text: str) -> int:
        return len(text) // 4


def load_token_counter(path: Path | None) -> tuple[str, Callable[[str], int]]:
    if path is None:
        heuristic = _HeuristicTokenCounter()
        return heuristic.name, heuristic.count
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    return f"transformers:{path}", lambda text: len(tokenizer.encode(text))


def method_name_of(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("method_name") or "unknown")


def length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def analyze_rows(
    rows: list[dict[str, Any]],
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    total = len(rows)
    cot_texts: list[str] = []
    model_think_tag_rows = 0
    think_text_rows = 0
    stage_trace_counts: dict[str, int] = {}
    stage_row_counts: dict[str, int] = {}
    refusal_like_cot_rows = 0

    for row in rows:
        cot = row.get("cot")
        if cot is not None:
            text = str(cot)
            cot_texts.append(text)
            final_response = str(row.get("final_response") or row.get("response") or "")
            if REFUSAL_CUE_RE.search(final_response):
                refusal_like_cot_rows += 1

        traces = row.get("cot_traces")
        if not isinstance(traces, list):
            continue
        stages_seen: set[str] = set()
        row_has_think_tag = False
        row_has_think_text = False
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            stage = str(trace.get("stage") or "unknown")
            stage_trace_counts[stage] = stage_trace_counts.get(stage, 0) + 1
            stages_seen.add(stage)
            if trace.get("trace_type") == "model_think_tag":
                row_has_think_tag = True
            if str(trace.get("think_text") or "").strip():
                row_has_think_text = True
        for stage in stages_seen:
            stage_row_counts[stage] = stage_row_counts.get(stage, 0) + 1
        if row_has_think_tag:
            model_think_tag_rows += 1
        if row_has_think_text:
            think_text_rows += 1

    return {
        "rows": total,
        "cot_presence": {
            "present": len(cot_texts),
            "absent": total - len(cot_texts),
            "rate": len(cot_texts) / total,
        },
        "cot_length_chars": length_stats([len(text) for text in cot_texts]),
        "cot_length_tokens": length_stats([count_tokens(text) for text in cot_texts]),
        "think_tag": {
            "model_think_tag_rows": model_think_tag_rows,
            "model_think_tag_rate": model_think_tag_rows / total,
            "think_text_rows": think_text_rows,
            "think_text_rate": think_text_rows / total,
        },
        "stages": {
            stage: {
                "trace_count": stage_trace_counts.get(stage, 0),
                "row_count": stage_row_counts.get(stage, 0),
                "row_rate": stage_row_counts.get(stage, 0) / total,
            }
            for stage in HEADLINE_STAGES
        },
        "other_stages": {
            stage: {
                "trace_count": stage_trace_counts.get(stage, 0),
                "row_count": stage_row_counts.get(stage, 0),
                "row_rate": stage_row_counts.get(stage, 0) / total,
            }
            for stage in sorted(set(stage_trace_counts) - set(HEADLINE_STAGES))
        },
        "refusal_cue_overlap": {
            "cot_rows": len(cot_texts),
            "refusal_like_final_response": refusal_like_cot_rows,
            "overlap_rate": refusal_like_cot_rows / len(cot_texts) if cot_texts else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    rows = read_rows(files)
    token_source, count_tokens = load_token_counter(args.tokenizer)

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(method_name_of(row), []).append(row)

    result = {
        "input_files": [str(path) for path in files],
        "metadata": {
            "rows": len(rows),
            "methods": sorted(by_method),
            "token_counts": token_source,
            "refusal_cue_regex": REFUSAL_CUE_RE.pattern,
        },
        "methods": {
            name: analyze_rows(method_rows, count_tokens)
            for name, method_rows in sorted(by_method.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
