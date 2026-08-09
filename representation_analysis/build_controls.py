#!/usr/bin/env python3
"""Attach fold-local, length-matched IA representation controls to generations."""

from __future__ import annotations

import argparse
import json
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


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[Any]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode_without_cleanup(tokenizer: Any, token_ids: list[Any]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # Small test tokenizers need not implement the complete Hugging Face API.
        return tokenizer.decode(token_ids)


def _compose_exact_length_control(
    prepared: list[dict[str, Any]],
    candidate_indices: list[int],
    target_token_count: int,
    tokenizer: Any,
) -> tuple[str, list[int], int]:
    """Compose unrelated donor analyses and truncate to an exact token count."""
    ordered = sorted(
        candidate_indices,
        key=lambda index: (
            abs(
                int(prepared[index]["_control_work"]["analysis_token_count"])
                - target_token_count
            ),
            _row_key(prepared[index]),
        ),
    )
    if not ordered:
        raise ValueError("Cannot compose a shuffled control without donors")

    pieces: list[str] = []
    used_indices: list[int] = []
    composed_ids: list[Any] = []
    cursor = 0
    # Cycling is only needed when a fold contains many very short refusals.
    # Each non-empty donor must increase the composed token count, so the bound
    # is conservative and protects custom tokenizers from an infinite loop.
    maximum_segments = target_token_count + len(ordered)
    while len(composed_ids) < target_token_count:
        donor_index = ordered[cursor % len(ordered)]
        pieces.append(str(prepared[donor_index]["_control_work"]["analysis"]))
        used_indices.append(donor_index)
        composed_ids = _encode_without_special_tokens(tokenizer, "\n\n".join(pieces))
        cursor += 1
        if cursor > maximum_segments:
            raise ValueError(
                "Unable to construct an exact-length shuffled control from "
                f"{len(ordered)} non-empty donors"
            )

    untruncated_count = len(composed_ids)
    analysis = _decode_without_cleanup(tokenizer, composed_ids[:target_token_count])
    realized_count = len(_encode_without_special_tokens(tokenizer, analysis))
    if realized_count != target_token_count:
        raise ValueError(
            "Tokenizer decode/encode did not preserve the requested shuffled-control "
            f"length: requested={target_token_count}, realized={realized_count}"
        )
    return analysis, used_indices, untruncated_count


def attach_controls(
    rows: list[dict[str, Any]],
    folds: dict[str, tuple[int, str]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
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
        token_count = len(_encode_without_special_tokens(tokenizer, analysis))
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
            recipient_pair, _, _ = _row_key(recipient)
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
            shuffled_analysis, used_indices, untruncated_count = (
                _compose_exact_length_control(
                    prepared,
                    candidates,
                    recipient_tokens,
                    tokenizer,
                )
            )
            donor = prepared[used_indices[0]]
            donor_pair, donor_side, _ = _row_key(donor)
            donor_group = str(donor["source_group"])
            unique_used_indices = list(dict.fromkeys(used_indices))
            donor_pair_ids = [
                _row_key(prepared[index])[0] for index in unique_used_indices
            ]
            donor_source_groups = [
                str(prepared[index]["source_group"]) for index in unique_used_indices
            ]
            recipient["representation_controls"] = {
                "true": {
                    "analysis_token_count": recipient_tokens,
                },
                "shuffled": {
                    "analysis": shuffled_analysis,
                    "analysis_token_count": recipient_tokens,
                    "donor_pair_id": donor_pair,
                    "donor_pair_ids": donor_pair_ids,
                    "donor_side": donor_side,
                    "donor_run_id": run_id,
                    "donor_source_group": donor_group,
                    "donor_source_groups": donor_source_groups,
                    "segments": len(used_indices),
                    "untruncated_token_count": untruncated_count,
                    "construction": "fold_local_target_label_compose_then_truncate",
                    "token_difference": 0,
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
