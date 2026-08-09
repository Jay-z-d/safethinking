#!/usr/bin/env python3
"""Prepare, validate, split, and merge immutable formal-experiment shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FORMAL_CHECKPOINTS = (
    "h_analysis_boundary_empty",
    "h_analysis_boundary_shuffled",
    "h_analysis_boundary_true",
    "h_guided",
    "h_preanswer_empty",
    "h_preanswer_shuffled",
    "h_preanswer_true",
    "h_query",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield line_number, row


def prepare_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(directory: Path) -> dict[str, Any]:
    if not (directory / "_SUCCESS").exists():
        raise ValueError(f"Shard directory lacks _SUCCESS: {directory}")
    path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(f"Shard directory lacks manifest.json: {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_rows(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for _line_number, row in read_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        group = str(row.get("harmful_source_index") or row.get("source_group") or "")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold row for pair={pair_id!r}") from exc
        if not pair_id or not group or fold < 0 or pair_id in result:
            raise ValueError(f"Invalid or duplicate fold row for pair={pair_id!r}")
        result[pair_id] = (fold, group)
    return result


def prepare_pair_shards(
    input_path: Path,
    fold_manifest: Path,
    output_dir: Path,
    pairs_per_shard: int,
) -> dict[str, Any]:
    if pairs_per_shard < 1:
        raise ValueError("--pairs-per-shard must be positive")
    folds = load_fold_rows(fold_manifest)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in read_jsonl(input_path):
        pair_id = str(row.get("pair_id") or "")
        group = str(row.get("harmful_source_index") or "")
        if not pair_id or pair_id in seen:
            raise ValueError(f"Missing or duplicate pair_id at {input_path}:{line_number}")
        if pair_id not in folds or group != folds[pair_id][1]:
            raise ValueError(f"Fold/source mismatch for pair={pair_id!r}")
        for field in ("benign_prompt", "harmful_prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Pair {pair_id!r} lacks non-empty {field}")
        seen.add(pair_id)
        rows.append(row)
    if seen != set(folds):
        missing = sorted(set(folds) - seen)
        extra = sorted(seen - set(folds))
        raise ValueError(f"Pair/fold sets differ; missing={missing[:3]}, extra={extra[:3]}")
    if not rows:
        raise ValueError(f"No pairs found in {input_path}")

    prepare_empty_dir(output_dir)
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(rows), pairs_per_shard)):
        shard_rows = rows[start : start + pairs_per_shard]
        name = f"pairs-{shard_index:05d}.jsonl"
        path = output_dir / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in shard_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        shards.append(
            {
                "index": shard_index,
                "file": name,
                "pairs": len(shard_rows),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "created_at": utc_now(),
        "kind": "formal_pair_shards",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "fold_manifest": str(fold_manifest),
        "fold_manifest_sha256": sha256_file(fold_manifest),
        "pair_count": len(rows),
        "pairs_per_shard": pairs_per_shard,
        "shard_count": len(shards),
        "shards": shards,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def pair_shard_expectations(
    pair_shards_dir: Path,
) -> tuple[dict[str, Any], dict[str, tuple[int, str]], dict[int, set[str]]]:
    manifest = load_manifest(pair_shards_dir)
    indices = [int(shard["index"]) for shard in manifest["shards"]]
    if indices != list(range(int(manifest["shard_count"]))):
        raise ValueError(f"Pair-shard indices must be contiguous from zero: {indices}")
    pair_info: dict[str, tuple[int, str]] = {}
    shard_pairs: dict[int, set[str]] = {}
    for shard in manifest["shards"]:
        index = int(shard["index"])
        path = pair_shards_dir / str(shard["file"])
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Pair shard checksum mismatch: {path}")
        pairs: set[str] = set()
        for _line_number, row in read_jsonl(path):
            pair_id = str(row.get("pair_id") or "")
            group = str(row.get("harmful_source_index") or "")
            if not pair_id or not group or pair_id in pair_info:
                raise ValueError(f"Invalid or duplicate pair in {path}: {pair_id!r}")
            pair_info[pair_id] = (index, group)
            pairs.add(pair_id)
        if len(pairs) != int(shard["pairs"]):
            raise ValueError(f"Pair count mismatch in {path}")
        shard_pairs[index] = pairs
    if len(pair_info) != int(manifest["pair_count"]):
        raise ValueError("Pair-shard manifest total does not match shard contents")
    return manifest, pair_info, shard_pairs


def normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(value)))


def merge_generation_shards(
    pair_shards_dir: Path,
    generation_dir: Path,
    output: Path,
    seeds: list[int],
    method: str,
    model: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be non-empty and unique")
    pair_manifest, pair_info, shard_pairs = pair_shard_expectations(pair_shards_dir)
    expected_model = normalized_path(model)
    seen: set[tuple[str, str, int]] = set()
    shard_summaries: list[dict[str, Any]] = []
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Temporary merge output already exists: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            for shard in pair_manifest["shards"]:
                index = int(shard["index"])
                path = generation_dir / f"generation-{index:05d}.jsonl"
                if not path.exists():
                    raise ValueError(f"Missing generation shard: {path}")
                local_seen: set[tuple[str, str, int]] = set()
                for line_number, row in read_jsonl(path):
                    pair_id = str(row.get("pair_id") or "")
                    side = str(row.get("side") or row.get("gold_label") or "")
                    try:
                        run_id = int(row.get("run_id", row.get("seed")))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid seed at {path}:{line_number}") from exc
                    key = (pair_id, side, run_id)
                    if (
                        pair_id not in shard_pairs[index]
                        or side not in {"benign", "harmful"}
                        or run_id not in seeds
                        or key in seen
                    ):
                        raise ValueError(f"Unexpected/duplicate generation key {key} in {path}")
                    if str(row.get("method")) != method:
                        raise ValueError(f"Method mismatch for {key}: {row.get('method')!r}")
                    if normalized_path(str(row.get("model_path") or "")) != expected_model:
                        raise ValueError(f"Model mismatch for {key}: {row.get('model_path')!r}")
                    if str(row.get("source_group") or "") != pair_info[pair_id][1]:
                        raise ValueError(f"Source-group mismatch for {key}")
                    seen.add(key)
                    local_seen.add(key)
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                expected_local = {
                    (pair_id, side, seed)
                    for pair_id in shard_pairs[index]
                    for side in ("benign", "harmful")
                    for seed in seeds
                }
                if local_seen != expected_local:
                    missing = sorted(expected_local - local_seen)
                    raise ValueError(
                        f"Generation shard {index} is incomplete; missing examples={missing[:3]}"
                    )
                shard_summaries.append(
                    {
                        "index": index,
                        "file": str(path),
                        "rows": len(local_seen),
                        "sha256": sha256_file(path),
                    }
                )
        expected_all = len(pair_info) * 2 * len(seeds)
        if len(seen) != expected_all:
            raise ValueError(f"Merged rows {len(seen)} != expected {expected_all}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "created_at": utc_now(),
        "kind": "formal_generation_merge",
        "pair_shards": str(pair_shards_dir),
        "generation_dir": str(generation_dir),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "pairs": len(pair_info),
        "rows": len(seen),
        "seeds": seeds,
        "method": method,
        "model": str(model),
        "shards": shard_summaries,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    output.with_name(output.name + "._SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def split_controlled_generation(
    input_path: Path,
    pair_shards_dir: Path,
    output_dir: Path,
    seeds: list[int],
) -> dict[str, Any]:
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be non-empty and unique")
    _pair_manifest, pair_info, shard_pairs = pair_shard_expectations(pair_shards_dir)
    prepare_empty_dir(output_dir)
    handles: dict[int, Any] = {}
    paths: dict[int, Path] = {}
    seen: set[tuple[str, str, int]] = set()
    counts: Counter[int] = Counter()
    try:
        for index in sorted(shard_pairs):
            path = output_dir / f"controlled-{index:05d}.jsonl"
            paths[index] = path
            handles[index] = path.open("w", encoding="utf-8", newline="\n")
        for line_number, row in read_jsonl(input_path):
            pair_id = str(row.get("pair_id") or "")
            side = str(row.get("side") or row.get("gold_label") or "")
            try:
                run_id = int(row.get("run_id", row.get("seed")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid run at {input_path}:{line_number}") from exc
            key = (pair_id, side, run_id)
            if (
                pair_id not in pair_info
                or side not in {"benign", "harmful"}
                or run_id not in seeds
                or key in seen
            ):
                raise ValueError(f"Unexpected/duplicate controlled key: {key}")
            if not isinstance(row.get("representation_controls"), dict):
                raise ValueError(f"Controlled row lacks controls: {key}")
            if str(row.get("source_group") or "") != pair_info[pair_id][1]:
                raise ValueError(f"Controlled source-group mismatch: {key}")
            index = pair_info[pair_id][0]
            handles[index].write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(key)
            counts[index] += 1
    finally:
        for handle in handles.values():
            handle.close()

    expected_all = len(pair_info) * 2 * len(seeds)
    if len(seen) != expected_all:
        raise ValueError(f"Controlled rows {len(seen)} != expected {expected_all}")
    shards: list[dict[str, Any]] = []
    for index, pairs in sorted(shard_pairs.items()):
        expected = len(pairs) * 2 * len(seeds)
        if counts[index] != expected:
            raise ValueError(f"Controlled shard {index}: {counts[index]} != {expected}")
        shards.append(
            {
                "index": index,
                "file": paths[index].name,
                "pairs": len(pairs),
                "rows": counts[index],
                "sha256": sha256_file(paths[index]),
            }
        )
    manifest = {
        "created_at": utc_now(),
        "kind": "formal_controlled_shards",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "pair_shards": str(pair_shards_dir),
        "pairs": len(pair_info),
        "rows": len(seen),
        "seeds": seeds,
        "shard_count": len(shards),
        "shards": shards,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def merge_representation_shards(
    controlled_shards_dir: Path,
    representation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    controlled_manifest = load_manifest(controlled_shards_dir)
    prepare_empty_dir(output_dir)
    combined_metadata = output_dir / "metadata.jsonl"
    common: dict[str, Any] | None = None
    checkpoint_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, int, str]] = set()
    output_shard_index = 0
    link_modes: Counter[str] = Counter()
    sources: list[dict[str, Any]] = []

    with combined_metadata.open("w", encoding="utf-8", newline="\n") as target_metadata:
        for controlled_shard in controlled_manifest["shards"]:
            index = int(controlled_shard["index"])
            source_dir = representation_root / f"repr-{index:05d}"
            source_manifest = load_manifest(source_dir)
            expected_rows = int(controlled_shard["rows"])
            expected_checkpoints = expected_rows * len(FORMAL_CHECKPOINTS)
            if int(source_manifest["checkpoints"]) != expected_checkpoints:
                raise ValueError(
                    f"Representation shard {index} checkpoints "
                    f"{source_manifest['checkpoints']} != {expected_checkpoints}"
                )
            for checkpoint in FORMAL_CHECKPOINTS:
                if int(source_manifest["checkpoint_counts"].get(checkpoint, 0)) != expected_rows:
                    raise ValueError(f"Representation shard {index} lacks {checkpoint}")
            compatibility = {
                key: source_manifest[key]
                for key in (
                    "model",
                    "tokenizer",
                    "model_dtype",
                    "output_dtype",
                    "pooling",
                    "requested_layers",
                    "resolved_hidden_state_indices",
                )
            }
            if common is None:
                common = compatibility
            elif common != compatibility:
                raise ValueError(f"Representation settings differ in shard {index}")

            shard_map: dict[str, str] = {}
            source_embedding_paths = sorted(source_dir.glob("embeddings-*.safetensors"))
            if len(source_embedding_paths) != int(source_manifest["shards"]):
                raise ValueError(f"Embedding-file count mismatch in {source_dir}")
            for source_path in source_embedding_paths:
                target_name = f"embeddings-{output_shard_index:05d}.safetensors"
                link_modes[link_or_copy(source_path, output_dir / target_name)] += 1
                shard_map[source_path.name] = target_name
                output_shard_index += 1

            metadata_count = 0
            for line_number, row in read_jsonl(
                source_dir / str(source_manifest.get("metadata", "metadata.jsonl"))
            ):
                old_shard = str(row.get("shard") or "")
                if old_shard not in shard_map:
                    raise ValueError(
                        f"Unknown embedding shard at {source_dir}:{line_number}: {old_shard}"
                    )
                key = (
                    str(row.get("pair_id") or ""),
                    str(row.get("side") or ""),
                    int(row.get("run_id", -1)),
                    str(row.get("checkpoint") or ""),
                )
                if key in seen:
                    raise ValueError(f"Duplicate representation metadata key: {key}")
                seen.add(key)
                metadata_count += 1
                checkpoint_counts[key[3]] += 1
                method_counts[str(row.get("method") or "unknown")] += 1
                row["shard"] = shard_map[old_shard]
                row["source_representation_shard"] = index
                target_metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
            if metadata_count != expected_checkpoints:
                raise ValueError(
                    f"Representation metadata shard {index}: "
                    f"{metadata_count} != {expected_checkpoints}"
                )
            sources.append(
                {
                    "index": index,
                    "directory": str(source_dir),
                    "checkpoints": metadata_count,
                    "embedding_shards": len(source_embedding_paths),
                }
            )

    if common is None:
        raise ValueError("No representation shards were merged")
    expected_total = int(controlled_manifest["rows"]) * len(FORMAL_CHECKPOINTS)
    if len(seen) != expected_total:
        raise ValueError(f"Merged representations {len(seen)} != {expected_total}")
    manifest = {
        "created_at": utc_now(),
        "input": controlled_manifest["input"],
        **common,
        "source_row_limit": None,
        "checkpoints": len(seen),
        "checkpoint_counts": dict(sorted(checkpoint_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "shards": output_shard_index,
        "metadata": combined_metadata.name,
        "merge_link_modes": dict(sorted(link_modes.items())),
        "source_representation_directories": sources,
        "notes": "Merged immutable formal representation shards.",
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def merge_probe_chunks(
    input_dir: Path,
    prefix: str,
    chunks: int,
    output: Path,
) -> dict[str, Any]:
    if chunks < 1:
        raise ValueError("--chunks must be positive")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    from representation_analysis.probe_separability import (
        bootstrap_intervals_from_distributions,
    )

    combined = {
        state: {metric: [] for metric in (
            "roc_auc",
            "balanced_accuracy",
            "signed_margin",
            "benign_signed_margin",
            "harmful_signed_margin",
        )}
        for state in ("before", "after", "delta")
    }
    ignored = {
        "created_at",
        "bootstrap_source_refit_95_ci",
        "bootstrap_samples",
        "bootstrap_start_index",
        "bootstrap_draws",
        "metrics",
        "fold_metrics",
        "pooled_oof_diagnostic",
    }
    base: dict[str, Any] | None = None
    base_configuration: dict[str, Any] | None = None
    expected_start = 0
    sources: list[dict[str, Any]] = []
    for chunk_index in range(chunks):
        path = input_dir / f"{prefix}.chunk-{chunk_index:05d}.json"
        if not path.exists():
            raise ValueError(f"Missing probe chunk: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        configuration = {key: value for key, value in row.items() if key not in ignored}
        if base is None:
            base = row
            base_configuration = configuration
        elif configuration != base_configuration:
            raise ValueError(f"Probe configuration differs in {path}")
        if base is not None:
            for outcome_key in ("metrics", "fold_metrics", "pooled_oof_diagnostic"):
                if not nested_numeric_close(base.get(outcome_key), row.get(outcome_key)):
                    raise ValueError(f"Probe point result {outcome_key} differs in {path}")
        start = int(row.get("bootstrap_start_index", -1))
        samples = int(row.get("bootstrap_samples", 0))
        if start != expected_start or samples < 1:
            raise ValueError(
                f"Probe chunk coverage is not contiguous at {path}: "
                f"start={start}, expected={expected_start}, samples={samples}"
            )
        draws = row.get("bootstrap_draws")
        if not isinstance(draws, dict):
            raise ValueError(f"Probe chunk lacks stored bootstrap draws: {path}")
        for state, metrics in combined.items():
            for metric, values in metrics.items():
                incoming = draws.get(state, {}).get(metric)
                if not isinstance(incoming, list) or len(incoming) != samples:
                    raise ValueError(f"Invalid draws for {state}/{metric} in {path}")
                values.extend(float(value) for value in incoming)
        expected_start += samples
        sources.append(
            {
                "index": chunk_index,
                "file": str(path),
                "start": start,
                "samples": samples,
                "sha256": sha256_file(path),
            }
        )

    if base is None:
        raise ValueError("No probe chunks were merged")
    result = dict(base)
    result["created_at"] = utc_now()
    result["bootstrap_source_refit_95_ci"] = bootstrap_intervals_from_distributions(
        combined
    )
    result["bootstrap_samples"] = expected_start
    result["bootstrap_start_index"] = 0
    result["bootstrap_draws"] = combined
    result["bootstrap_sharded_equivalent"] = True
    result["bootstrap_chunks"] = sources
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def nested_numeric_close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            nested_numeric_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            nested_numeric_close(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid comma-separated seeds: {value!r}") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-pairs")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--fold-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--pairs-per-shard", type=int, default=250)

    merge_generation = subparsers.add_parser("merge-generations")
    merge_generation.add_argument("--pair-shards-dir", type=Path, required=True)
    merge_generation.add_argument("--generation-dir", type=Path, required=True)
    merge_generation.add_argument("--output", type=Path, required=True)
    merge_generation.add_argument("--seeds", default="42,43,44")
    merge_generation.add_argument("--method", required=True)
    merge_generation.add_argument("--model", type=Path, required=True)

    split_controlled = subparsers.add_parser("split-controlled")
    split_controlled.add_argument("--input", type=Path, required=True)
    split_controlled.add_argument("--pair-shards-dir", type=Path, required=True)
    split_controlled.add_argument("--output-dir", type=Path, required=True)
    split_controlled.add_argument("--seeds", default="42,43,44")

    merge_representations = subparsers.add_parser("merge-representations")
    merge_representations.add_argument(
        "--controlled-shards-dir", type=Path, required=True
    )
    merge_representations.add_argument("--representation-root", type=Path, required=True)
    merge_representations.add_argument("--output-dir", type=Path, required=True)

    merge_probes = subparsers.add_parser("merge-probes")
    merge_probes.add_argument("--input-dir", type=Path, required=True)
    merge_probes.add_argument("--prefix", required=True)
    merge_probes.add_argument("--chunks", type=int, required=True)
    merge_probes.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-pairs":
        result = prepare_pair_shards(
            args.input,
            args.fold_manifest,
            args.output_dir,
            args.pairs_per_shard,
        )
    elif args.command == "merge-generations":
        result = merge_generation_shards(
            args.pair_shards_dir,
            args.generation_dir,
            args.output,
            parse_seeds(args.seeds),
            args.method,
            args.model,
        )
    elif args.command == "split-controlled":
        result = split_controlled_generation(
            args.input,
            args.pair_shards_dir,
            args.output_dir,
            parse_seeds(args.seeds),
        )
    elif args.command == "merge-representations":
        result = merge_representation_shards(
            args.controlled_shards_dir,
            args.representation_root,
            args.output_dir,
        )
    elif args.command == "merge-probes":
        result = merge_probe_chunks(
            args.input_dir,
            args.prefix,
            args.chunks,
            args.output,
        )
    else:  # pragma: no cover - argparse enforces subcommands.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
