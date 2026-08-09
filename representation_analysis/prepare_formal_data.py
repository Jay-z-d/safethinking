#!/usr/bin/env python3
"""Audit boundary pairs and freeze formal/pilot evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "pair_id",
    "benign_source_index",
    "harmful_source_index",
    "benign_prompt",
    "harmful_prompt",
    "embedding_score",
    "reranker_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-pairs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from exc
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing fields {missing} at {path}:{line_number}")
            pair_id = str(row["pair_id"])
            if not pair_id:
                raise ValueError(f"Empty pair_id at {path}:{line_number}")
            if pair_id in seen_pairs:
                raise ValueError(f"Duplicate pair_id {pair_id!r} at {path}:{line_number}")
            for field in ("benign_prompt", "harmful_prompt"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"Empty {field} at {path}:{line_number}")
            for field in ("benign_source_index", "harmful_source_index"):
                if row[field] is None or str(row[field]) == "":
                    raise ValueError(f"Empty {field} at {path}:{line_number}")
            for field in ("embedding_score", "reranker_score"):
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {field} at {path}:{line_number}") from exc
                if not value == value:
                    raise ValueError(f"Non-finite {field} at {path}:{line_number}")
            seen_pairs.add(pair_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No pairs found in {path}")
    return rows


def _canonical_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -float(row["reranker_score"]),
        -float(row["embedding_score"]),
        str(row["pair_id"]),
    )


def canonical_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_harmful: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_harmful[str(row["harmful_source_index"])].append(row)
    return [
        min(group_rows, key=_canonical_key)
        for _, group_rows in sorted(by_harmful.items())
    ]


def assign_group_folds(
    rows: list[dict[str, Any]],
    folds: int,
    seed: int,
) -> tuple[dict[str, int], list[int]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["harmful_source_index"])].append(str(row["pair_id"]))
    if folds < 2 or folds > len(groups):
        raise ValueError(f"--folds must be between 2 and {len(groups)}")
    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id in groups}
    ordered = sorted(
        groups,
        key=lambda group_id: (-len(groups[group_id]), tie_break[group_id], group_id),
    )
    fold_loads = [0] * folds
    pair_folds: dict[str, int] = {}
    for group_id in ordered:
        fold = min(range(folds), key=lambda index: (fold_loads[index], index))
        for pair_id in groups[group_id]:
            pair_folds[pair_id] = fold
        fold_loads[fold] += len(groups[group_id])
    return pair_folds, fold_loads


def _quantile_bins(values: list[float], bins: int = 4) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0] * len(values)
    for rank, index in enumerate(order):
        result[index] = min(bins - 1, rank * bins // len(values))
    return result


def select_pilot(
    canonical: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 1 or count > len(canonical):
        raise ValueError(f"--pilot-pairs must be between 1 and {len(canonical)}")
    reranker_bins = _quantile_bins([float(row["reranker_score"]) for row in canonical])
    length_bins = _quantile_bins(
        [len(row["benign_prompt"]) + len(row["harmful_prompt"]) for row in canonical]
    )
    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row, reranker_bin, length_bin in zip(
        canonical,
        reranker_bins,
        length_bins,
        strict=True,
    ):
        strata[(reranker_bin, length_bin)].append(row)
    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    stratum_keys = sorted(strata)
    while len(selected) < count:
        made_progress = False
        for key in stratum_keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
                made_progress = True
        if not made_progress:
            raise RuntimeError("Pilot stratification exhausted before reaching target size")
    return sorted(selected, key=lambda row: str(row["pair_id"]))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_formal_data(
    input_path: Path,
    output_dir: Path,
    pilot_pair_count: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_pairs(input_path)
    canonical = canonical_pairs(rows)
    pair_folds, fold_loads = assign_group_folds(rows, folds, seed)
    pilot = select_pilot(canonical, pilot_pair_count, seed)
    pilot_pair_folds, pilot_fold_loads = assign_group_folds(pilot, folds, seed)
    pilot_ids = {str(row["pair_id"]) for row in pilot}
    canonical_ids = {str(row["pair_id"]) for row in canonical}
    harmful_counts = Counter(str(row["harmful_source_index"]) for row in rows)

    pilot_path = output_dir / "pilot_pairs.jsonl"
    canonical_path = output_dir / "canonical_pairs.jsonl"
    folds_path = output_dir / "formal_folds.jsonl"
    pilot_folds_path = output_dir / "pilot_folds.jsonl"
    write_jsonl(pilot_path, pilot)
    write_jsonl(canonical_path, sorted(canonical, key=lambda row: str(row["pair_id"])))
    write_jsonl(
        folds_path,
        (
            {
                "pair_id": str(row["pair_id"]),
                "harmful_source_index": str(row["harmful_source_index"]),
                "fold": pair_folds[str(row["pair_id"])],
                "harmful_source_reuse": harmful_counts[str(row["harmful_source_index"])],
                "source_weight": 1.0 / harmful_counts[str(row["harmful_source_index"])],
                "is_pilot": str(row["pair_id"]) in pilot_ids,
                "is_canonical": str(row["pair_id"]) in canonical_ids,
            }
            for row in sorted(rows, key=lambda item: str(item["pair_id"]))
        ),
    )
    write_jsonl(
        pilot_folds_path,
        (
            {
                "pair_id": str(row["pair_id"]),
                "harmful_source_index": str(row["harmful_source_index"]),
                "fold": pilot_pair_folds[str(row["pair_id"])],
                "harmful_source_reuse": 1,
                "source_weight": 1.0,
                "is_pilot": True,
                "is_canonical": True,
            }
            for row in sorted(pilot, key=lambda item: str(item["pair_id"]))
        ),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": seed,
        "folds": folds,
        "pair_count": len(rows),
        "unique_benign_sources": len({str(row["benign_source_index"]) for row in rows}),
        "unique_harmful_sources": len(harmful_counts),
        "reused_harmful_sources": sum(count > 1 for count in harmful_counts.values()),
        "max_harmful_source_reuse": max(harmful_counts.values()),
        "fold_pair_counts": fold_loads,
        "pilot_fold_pair_counts": pilot_fold_loads,
        "pilot_pair_count": len(pilot),
        "pilot_unique_harmful_sources": len(
            {str(row["harmful_source_index"]) for row in pilot}
        ),
        "canonical_pair_count": len(canonical),
        "selection_rule": (
            "one pair per harmful_source_index by highest reranker_score, then "
            "highest embedding_score, then lexicographically smallest pair_id"
        ),
        "files": {
            "pilot_pairs": pilot_path.name,
            "canonical_pairs": canonical_path.name,
            "formal_folds": folds_path.name,
            "pilot_folds": pilot_folds_path.name,
        },
    }
    for key, filename in manifest["files"].items():
        manifest.setdefault("sha256", {})[key] = sha256_file(output_dir / filename)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_formal_data(
        args.input,
        args.output_dir,
        args.pilot_pairs,
        args.folds,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
