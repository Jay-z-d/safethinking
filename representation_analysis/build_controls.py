#!/usr/bin/env python3
"""Attach fold-local, length-matched IA representation controls to generations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


IA_METHOD = "safe_llm_intention_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--length-tolerance-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-token-tolerance", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_fold_manifest(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for row in read_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        group_id = str(row.get("harmful_source_index") or "")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold row for pair_id={pair_id!r}") from exc
        if not pair_id or not group_id or fold < 0:
            raise ValueError(f"Invalid fold row for pair_id={pair_id!r}")
        if pair_id in result:
            raise ValueError(f"Duplicate pair_id in fold manifest: {pair_id}")
        result[pair_id] = (fold, group_id)
    return result


def _stored_final_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt_messages = row.get("prompt_messages")
    if isinstance(prompt_messages, dict):
        final = prompt_messages.get("final")
    else:
        trace = row.get("method_trace")
        intermediate = trace.get("intermediate_messages") if isinstance(trace, dict) else None
        final = intermediate.get("final") if isinstance(intermediate, dict) else None
    if not isinstance(final, list) or len(final) < 3:
        raise ValueError("IA row lacks stored stage-2 final messages")
    normalized: list[dict[str, str]] = []
    for message in final:
        if not isinstance(message, dict):
            raise ValueError("IA final messages contain a non-object entry")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("IA final messages require string role/content")
        normalized.append({"role": role, "content": content})
    if [message["role"] for message in normalized[:3]] != ["user", "assistant", "user"]:
        raise ValueError("IA final messages must begin user/assistant/user")
    if not normalized[1]["content"].strip():
        raise ValueError("IA row has an empty stage-1 analysis")
    return normalized


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    pair_id = str(row.get("pair_id") or "")
    side = str(row.get("side") or row.get("gold_label") or "")
    try:
        run_id = int(row.get("run_id", row.get("seed")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid run_id for pair={pair_id!r}, side={side!r}") from exc
    if not pair_id or side not in {"benign", "harmful"}:
        raise ValueError(f"Invalid pair/side for control row: {pair_id!r}/{side!r}")
    return pair_id, side, run_id


def validate_complete_runs(rows: list[dict[str, Any]]) -> None:
    by_pair_run: dict[tuple[str, int], set[str]] = defaultdict(set)
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate IA generation row: {key}")
        seen.add(key)
        by_pair_run[(key[0], key[2])].add(key[1])
    incomplete = [key for key, sides in by_pair_run.items() if sides != {"benign", "harmful"}]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} pair/run groups lack benign and harmful rows; "
            f"examples: {incomplete[:3]}"
        )


def attach_controls(
    rows: list[dict[str, Any]],
    folds: dict[str, tuple[int, str]],
    tokenizer: Any,
    tolerance_ratio: float,
    minimum_tolerance: int,
) -> list[dict[str, Any]]:
    if tolerance_ratio < 0 or minimum_tolerance < 0:
        raise ValueError("Token-length tolerances must be non-negative")
    validate_complete_runs(rows)
    prepared: list[dict[str, Any]] = []
    buckets: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("method") != IA_METHOD:
            raise ValueError(f"Control construction only accepts method={IA_METHOD}")
        pair_id, side, run_id = _row_key(row)
        if pair_id not in folds:
            raise ValueError(f"Pair {pair_id!r} is missing from fold manifest")
        fold, group_id = folds[pair_id]
        messages = _stored_final_messages(row)
        analysis = messages[1]["content"]
        token_count = len(tokenizer.encode(analysis, add_special_tokens=False))
        if token_count < 1:
            raise ValueError(f"Tokenizer produced no analysis tokens for {_row_key(row)}")
        copy = dict(row)
        copy["formal_fold"] = fold
        copy["source_group"] = group_id
        copy["harmful_source_index"] = row.get("harmful_source_index", group_id)
        copy["_control_work"] = {
            "analysis": analysis,
            "analysis_token_count": token_count,
            "messages": messages,
        }
        index = len(prepared)
        prepared.append(copy)
        buckets[(fold, run_id, side)].append(index)

    # For each recipient label, alternate donor labels. Therefore donor label is
    # balanced within both benign and harmful recipients and cannot predict it.
    for recipient_bucket in sorted(buckets):
        fold, run_id, recipient_side = recipient_bucket
        recipient_indices = sorted(
            buckets[recipient_bucket],
            key=lambda index: _row_key(prepared[index]),
        )
        for rank, recipient_index in enumerate(recipient_indices):
            recipient = prepared[recipient_index]
            recipient_pair, recipient_side_value, _ = _row_key(recipient)
            target_side = ("benign", "harmful")[rank % 2]
            candidates = [
                candidate_index
                for candidate_index in buckets[(fold, run_id, target_side)]
                if _row_key(prepared[candidate_index])[0] != recipient_pair
                and str(prepared[candidate_index]["source_group"])
                != str(recipient["source_group"])
            ]
            if not candidates:
                raise ValueError(
                    f"No fold-local {target_side} donor for pair={recipient_pair}, "
                    f"fold={fold}, run={run_id}"
                )
            recipient_tokens = int(recipient["_control_work"]["analysis_token_count"])
            donor_index = min(
                candidates,
                key=lambda index: (
                    abs(
                        int(prepared[index]["_control_work"]["analysis_token_count"])
                        - recipient_tokens
                    ),
                    _row_key(prepared[index]),
                ),
            )
            donor = prepared[donor_index]
            donor_pair, donor_side, _ = _row_key(donor)
            donor_group = str(donor["source_group"])
            donor_tokens = int(donor["_control_work"]["analysis_token_count"])
            difference = abs(donor_tokens - recipient_tokens)
            tolerance = max(minimum_tolerance, math.ceil(recipient_tokens * tolerance_ratio))
            if difference > tolerance:
                raise ValueError(
                    "No sufficiently length-matched shuffled analysis for "
                    f"{recipient_pair}/{recipient_side_value}/run{run_id}: "
                    f"recipient={recipient_tokens}, nearest={donor_tokens}, tolerance={tolerance}"
                )
            recipient["representation_controls"] = {
                "true": {
                    "analysis_token_count": recipient_tokens,
                },
                "shuffled": {
                    "analysis": donor["_control_work"]["analysis"],
                    "analysis_token_count": donor_tokens,
                    "donor_pair_id": donor_pair,
                    "donor_side": donor_side,
                    "donor_run_id": run_id,
                    "donor_source_group": donor_group,
                    "token_difference": difference,
                },
                "empty": {
                    "analysis": "",
                    "analysis_token_count": 0,
                },
            }

    for row in prepared:
        row.pop("_control_work", None)
    return prepared


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Output file already exists: {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer),
        trust_remote_code=args.trust_remote_code,
    )
    rows = read_jsonl(args.input)
    controlled = attach_controls(
        rows,
        load_fold_manifest(args.fold_manifest),
        tokenizer,
        args.length_tolerance_ratio,
        args.minimum_token_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in controlled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(controlled),
                "pairs": len({str(row["pair_id"]) for row in controlled}),
                "runs": sorted({int(row["run_id"]) for row in controlled}),
                "folds": sorted({int(row["formal_fold"]) for row in controlled}),
                "controls": ["true", "shuffled", "empty"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
