#!/usr/bin/env python3
"""Compare held-out benign/harmful linear separability at two checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LABELS = {"benign": -1, "harmful": 1}
METRIC_NAMES = (
    "roc_auc",
    "balanced_accuracy",
    "signed_margin",
    "benign_signed_margin",
    "harmful_signed_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--before", default="h_query")
    parser.add_argument("--after", default="h_reasoned")
    parser.add_argument(
        "--layer-column",
        type=int,
        default=-1,
        help="Column in the extracted layer list; -1 selects the last stored layer.",
    )
    parser.add_argument("--probe", choices=("logistic", "svm"), default="logistic")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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
    return rows


def load_checkpoint_vectors(
    input_dir: Path,
    method: str,
    checkpoints: set[str],
    layer_column: int,
) -> tuple[dict[tuple[str, str, int], dict[str, np.ndarray]], int, dict[str, Any]]:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    layer_indices = list(manifest["resolved_hidden_state_indices"])
    resolved_column = layer_column if layer_column >= 0 else len(layer_indices) + layer_column
    if resolved_column < 0 or resolved_column >= len(layer_indices):
        raise ValueError(
            f"--layer-column {layer_column} is invalid for stored layers {layer_indices}"
        )

    metadata_rows = [
        row
        for row in read_jsonl(input_dir / str(manifest.get("metadata", "metadata.jsonl")))
        if row.get("method") == method and row.get("checkpoint") in checkpoints
    ]
    if not metadata_rows:
        raise ValueError(f"No rows found for method={method!r}, checkpoints={sorted(checkpoints)}")

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        by_shard[str(row["shard"])].append(row)

    records: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for shard_name, shard_rows in by_shard.items():
        embeddings = load_file(input_dir / shard_name)["embeddings"]
        for row in shard_rows:
            key = (str(row["pair_id"]), str(row["side"]), int(row["run_id"]))
            checkpoint = str(row["checkpoint"])
            target = records.setdefault(key, {})
            if checkpoint in target:
                raise ValueError(f"Duplicate activation for key={key}, checkpoint={checkpoint}")
            target[checkpoint] = np.asarray(
                embeddings[int(row["row_index"]), resolved_column],
                dtype=np.float32,
            )
    return records, int(layer_indices[resolved_column]), manifest


def paired_matrices(
    records: dict[tuple[str, str, int], dict[str, np.ndarray]],
    before: str,
    after: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str, int]]]:
    incomplete = [key for key, states in records.items() if before not in states or after not in states]
    if incomplete:
        examples = ", ".join(map(str, incomplete[:3]))
        raise ValueError(
            f"{len(incomplete)} samples lack a paired {before}/{after} state; examples: {examples}"
        )
    keys = sorted(records)
    before_matrix = np.stack([records[key][before] for key in keys])
    after_matrix = np.stack([records[key][after] for key in keys])
    labels = np.asarray([LABELS.get(key[1], 0) for key in keys], dtype=np.int8)
    groups = np.asarray([key[0] for key in keys])
    if np.any(labels == 0):
        raise ValueError("Activations contain a side other than benign/harmful")
    for pair_id in np.unique(groups):
        pair_labels = set(labels[groups == pair_id].tolist())
        if pair_labels != {-1, 1}:
            raise ValueError(f"Pair {pair_id} does not contain both benign and harmful samples")
    return before_matrix, after_matrix, labels, groups, keys


def make_probe(kind: str, c_value: float, seed: int) -> Any:
    if c_value <= 0:
        raise ValueError("--c must be positive")
    if kind == "logistic":
        return LogisticRegression(C=c_value, max_iter=5000, solver="liblinear", random_state=seed)
    return LinearSVC(C=c_value, dual="auto", max_iter=10000, random_state=seed)


def normalized_decision(probe: Any, matrix: np.ndarray) -> np.ndarray:
    coefficient_norm = float(np.linalg.norm(probe.coef_))
    if not np.isfinite(coefficient_norm) or coefficient_norm <= 0:
        raise ValueError("Linear probe has a zero or non-finite coefficient norm")
    return np.asarray(probe.decision_function(matrix), dtype=np.float64) / coefficient_norm


def metric_values(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    signed = labels * decisions
    return {
        "roc_auc": float(roc_auc_score(labels == 1, decisions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, np.where(decisions >= 0, 1, -1))),
        "signed_margin": float(np.mean(signed)),
        "benign_signed_margin": float(np.mean(signed[labels == -1])),
        "harmful_signed_margin": float(np.mean(signed[labels == 1])),
    }


def cross_validated_decisions(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    probe_kind: str,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if folds < 2 or folds > len(unique_groups):
        raise ValueError(f"--folds must be between 2 and the number of pairs ({len(unique_groups)})")
    shuffled_groups = unique_groups.copy()
    np.random.default_rng(seed).shuffle(shuffled_groups)
    held_out_group_folds = np.array_split(shuffled_groups, folds)
    before_decisions = np.full(len(labels), np.nan, dtype=np.float64)
    after_decisions = np.full(len(labels), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int16)

    for fold_id, held_out_groups in enumerate(held_out_group_folds):
        test_mask = np.isin(groups, held_out_groups)
        test = np.flatnonzero(test_mask)
        train = np.flatnonzero(~test_mask)
        # One scaler fitted on both training checkpoints keeps the coordinate
        # scale shared while using no held-out examples.
        scaler = StandardScaler().fit(np.concatenate([before[train], after[train]], axis=0))
        before_train = scaler.transform(before[train])
        after_train = scaler.transform(after[train])
        before_test = scaler.transform(before[test])
        after_test = scaler.transform(after[test])

        before_probe = make_probe(probe_kind, c_value, seed + fold_id)
        after_probe = make_probe(probe_kind, c_value, seed + fold_id)
        before_probe.fit(before_train, labels[train])
        after_probe.fit(after_train, labels[train])
        before_decisions[test] = normalized_decision(before_probe, before_test)
        after_decisions[test] = normalized_decision(after_probe, after_test)
        fold_ids[test] = fold_id

    if np.isnan(before_decisions).any() or np.isnan(after_decisions).any() or np.any(fold_ids < 0):
        raise RuntimeError("Cross-validation did not produce exactly one prediction per sample")
    return before_decisions, after_decisions, fold_ids


def bootstrap_intervals(
    labels: np.ndarray,
    groups: np.ndarray,
    before_decisions: np.ndarray,
    after_decisions: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    if samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    pair_ids = np.unique(groups)
    group_indices = {pair_id: np.flatnonzero(groups == pair_id) for pair_id in pair_ids}
    distributions = {
        state: {metric: [] for metric in METRIC_NAMES}
        for state in ("before", "after", "delta")
    }
    for _ in range(samples):
        sampled_pairs = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        indices = np.concatenate([group_indices[pair_id] for pair_id in sampled_pairs])
        before_metrics = metric_values(labels[indices], before_decisions[indices])
        after_metrics = metric_values(labels[indices], after_decisions[indices])
        for metric in METRIC_NAMES:
            distributions["before"][metric].append(before_metrics[metric])
            distributions["after"][metric].append(after_metrics[metric])
            distributions["delta"][metric].append(after_metrics[metric] - before_metrics[metric])
    return {
        state: {
            metric: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for metric, values in metrics.items()
        }
        for state, metrics in distributions.items()
    }


def main() -> None:
    args = parse_args()
    records, hidden_state_index, manifest = load_checkpoint_vectors(
        args.input_dir,
        args.method,
        {args.before, args.after},
        args.layer_column,
    )
    before, after, labels, groups, keys = paired_matrices(records, args.before, args.after)
    before_decisions, after_decisions, fold_ids = cross_validated_decisions(
        before,
        after,
        labels,
        groups,
        args.folds,
        args.probe,
        args.c,
        args.seed,
    )
    before_metrics = metric_values(labels, before_decisions)
    after_metrics = metric_values(labels, after_decisions)
    delta_metrics = {
        metric: after_metrics[metric] - before_metrics[metric] for metric in METRIC_NAMES
    }
    intervals = bootstrap_intervals(
        labels,
        groups,
        before_decisions,
        after_decisions,
        args.bootstrap_samples,
        args.seed,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "source_input": manifest.get("input"),
        "method": args.method,
        "before": args.before,
        "after": args.after,
        "hidden_state_index": hidden_state_index,
        "layer_column": args.layer_column,
        "pooling": manifest.get("pooling"),
        "probe": args.probe,
        "c": args.c,
        "folds": args.folds,
        "pair_grouped_cross_validation": True,
        "shared_train_only_scaler": True,
        "samples": len(labels),
        "pairs": len(np.unique(groups)),
        "class_counts": {
            "benign": int(np.sum(labels == -1)),
            "harmful": int(np.sum(labels == 1)),
        },
        "metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "delta_after_minus_before": delta_metrics,
        },
        "bootstrap_pair_95_ci": intervals,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions_output = args.predictions_output or args.output.with_suffix(".predictions.jsonl")
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf-8") as handle:
        for index, key in enumerate(keys):
            pair_id, side, run_id = key
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "side": side,
                        "run_id": run_id,
                        "label": int(labels[index]),
                        "fold": int(fold_ids[index]),
                        "before_normalized_decision": float(before_decisions[index]),
                        "after_normalized_decision": float(after_decisions[index]),
                        "before_signed_margin": float(labels[index] * before_decisions[index]),
                        "after_signed_margin": float(labels[index] * after_decisions[index]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({**result, "predictions_output": str(predictions_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
