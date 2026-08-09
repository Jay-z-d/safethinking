#!/usr/bin/env python3
"""Compare held-out benign/harmful separability with source-grouped probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
SampleKey = tuple[str, str, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--before", default="h_query")
    parser.add_argument("--after", default="h_analysis_boundary_true")
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--pair-filter", type=Path)
    parser.add_argument(
        "--run-id",
        type=int,
        help="Optional single generation seed/run to analyze independently.",
    )
    parser.add_argument(
        "--group-field",
        default="source_group",
        help="Metadata field defining leakage/cluster groups.",
    )
    parser.add_argument(
        "--weighting",
        choices=("pair", "source_balanced"),
        default="pair",
    )
    parser.add_argument(
        "--layer-column",
        type=int,
        default=-1,
        help="Column in extracted layers; -1 selects the last stored layer.",
    )
    parser.add_argument("--probe", choices=("logistic", "svm"), default="logistic")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--bootstrap-start-index",
        type=int,
        default=0,
        help="Zero-based first bootstrap draw; enables deterministic job sharding.",
    )
    parser.add_argument(
        "--store-bootstrap-draws",
        action="store_true",
        help="Store per-draw metrics so independently computed chunks can be merged.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not write redundant OOF predictions for bootstrap chunks.",
    )
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


def read_pair_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    pair_ids = {str(row.get("pair_id") or "") for row in read_jsonl(path)}
    pair_ids.discard("")
    if not pair_ids:
        raise ValueError(f"Pair filter contains no pair_id values: {path}")
    return pair_ids


def read_fold_manifest(path: Path | None) -> dict[str, tuple[int, str]] | None:
    if path is None:
        return None
    result: dict[str, tuple[int, str]] = {}
    for row in read_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        group_id = str(row.get("harmful_source_index") or row.get("source_group") or "")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold manifest row for pair={pair_id!r}") from exc
        if not pair_id or not group_id or fold < 0:
            raise ValueError(f"Invalid fold manifest row for pair={pair_id!r}")
        if pair_id in result:
            raise ValueError(f"Duplicate pair in fold manifest: {pair_id}")
        result[pair_id] = (fold, group_id)
    return result


def load_checkpoint_vectors(
    input_dir: Path,
    method: str,
    checkpoints: set[str],
    layer_column: int,
    group_field: str = "source_group",
    pair_filter: set[str] | None = None,
    run_id: int | None = None,
) -> tuple[dict[SampleKey, dict[str, np.ndarray]], int, dict[str, Any]]:
    success_path = input_dir / "_SUCCESS"
    if not success_path.exists():
        raise ValueError(f"Representation directory lacks completion marker: {success_path}")
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
        if row.get("method") == method
        and row.get("checkpoint") in checkpoints
        and (pair_filter is None or str(row.get("pair_id")) in pair_filter)
        and (run_id is None or int(row.get("run_id", -1)) == run_id)
    ]
    if not metadata_rows:
        raise ValueError(f"No rows found for method={method!r}, checkpoints={sorted(checkpoints)}")

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        by_shard[str(row["shard"])].append(row)

    records: dict[SampleKey, dict[str, np.ndarray]] = {}
    for shard_name, shard_rows in by_shard.items():
        embeddings = load_file(input_dir / shard_name)["embeddings"]
        for row in shard_rows:
            pair_id = str(row["pair_id"])
            group_id = str(row.get(group_field) or row.get("harmful_source_index") or pair_id)
            key: SampleKey = (
                pair_id,
                str(row["side"]),
                int(row["run_id"]),
                group_id,
            )
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
    records: dict[SampleKey, dict[str, np.ndarray]],
    before: str,
    after: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[SampleKey]]:
    expected_states = {before, after}
    invalid_states = [key for key, states in records.items() if set(states) != expected_states]
    if invalid_states:
        examples = ", ".join(map(str, invalid_states[:3]))
        raise ValueError(
            f"{len(invalid_states)} samples do not have exactly {sorted(expected_states)}; "
            f"examples: {examples}"
        )
    by_pair_run: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for pair_id, side, run_id, group_id in records:
        if side not in LABELS:
            raise ValueError(f"Activation side must be benign/harmful, got {side!r}")
        pair_run = (pair_id, run_id)
        if side in by_pair_run[pair_run]:
            raise ValueError(f"Duplicate side for pair/run={pair_run}, side={side}")
        by_pair_run[pair_run][side] = group_id
    incomplete = [
        pair_run
        for pair_run, sides in by_pair_run.items()
        if set(sides) != {"benign", "harmful"} or len(set(sides.values())) != 1
    ]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} pair/run groups lack matched benign/harmful states or "
            f"a shared source group; examples: {incomplete[:3]}"
        )

    keys = sorted(records)
    before_matrix = np.stack([records[key][before] for key in keys])
    after_matrix = np.stack([records[key][after] for key in keys])
    labels = np.asarray([LABELS[key[1]] for key in keys], dtype=np.int8)
    groups = np.asarray([key[3] for key in keys])
    return before_matrix, after_matrix, labels, groups, keys


def make_probe(kind: str, c_value: float, seed: int) -> Any:
    if c_value <= 0:
        raise ValueError("--c must be positive")
    if kind == "logistic":
        return LogisticRegression(
            C=c_value,
            max_iter=5000,
            solver="liblinear",
            random_state=seed,
        )
    return LinearSVC(C=c_value, dual="auto", max_iter=10000, random_state=seed)


def sample_weights(groups: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "pair":
        return np.ones(len(groups), dtype=np.float64)
    if weighting != "source_balanced":
        raise ValueError(f"Unknown weighting: {weighting}")
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return weights / np.mean(weights)


def raw_space_decision(probe: Any, scaler: StandardScaler, matrix: np.ndarray) -> np.ndarray:
    raw_coefficient = np.asarray(probe.coef_, dtype=np.float64).reshape(-1) / scaler.scale_
    coefficient_norm = float(np.linalg.norm(raw_coefficient))
    if not np.isfinite(coefficient_norm) or coefficient_norm <= 0:
        raise ValueError("Linear probe has a zero or non-finite raw-space coefficient norm")
    transformed = scaler.transform(matrix)
    return np.asarray(probe.decision_function(transformed), dtype=np.float64) / coefficient_norm


def balanced_group_fold_ids(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    counts = Counter(groups.tolist())
    if folds < 2 or folds > len(counts):
        raise ValueError(f"--folds must be between 2 and source groups ({len(counts)})")
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in counts}
    ordered = sorted(counts, key=lambda group: (-counts[group], tie_break[group], str(group)))
    loads = [0] * folds
    group_folds: dict[Any, int] = {}
    for group in ordered:
        fold = min(range(folds), key=lambda index: (loads[index], index))
        group_folds[group] = fold
        loads[fold] += counts[group]
    return np.asarray([group_folds[group] for group in groups], dtype=np.int16)


def manifest_fold_ids(
    keys: list[SampleKey],
    fold_manifest: dict[str, tuple[int, str]],
) -> np.ndarray:
    result: list[int] = []
    group_folds: dict[str, int] = {}
    for pair_id, _side, _run_id, group_id in keys:
        if pair_id not in fold_manifest:
            raise ValueError(f"Pair {pair_id!r} is absent from --fold-manifest")
        fold, manifest_group = fold_manifest[pair_id]
        if group_id != manifest_group:
            raise ValueError(
                f"Source group mismatch for pair {pair_id}: activations={group_id}, "
                f"manifest={manifest_group}"
            )
        previous = group_folds.setdefault(group_id, fold)
        if previous != fold:
            raise ValueError(f"Source group {group_id!r} spans multiple manifest folds")
        result.append(fold)
    return np.asarray(result, dtype=np.int16)


def cross_validated_decisions(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    probe_kind: str,
    c_value: float,
    seed: int,
    *,
    fixed_fold_ids: np.ndarray | None = None,
    weighting: str = "pair",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if before.shape != after.shape or before.shape[0] != len(labels) or len(labels) != len(groups):
        raise ValueError("Before/after matrices, labels, and groups must have matching rows")
    fold_ids = (
        balanced_group_fold_ids(groups, folds, seed)
        if fixed_fold_ids is None
        else np.asarray(fixed_fold_ids, dtype=np.int16)
    )
    if len(fold_ids) != len(labels):
        raise ValueError("Fixed fold IDs have the wrong length")
    unique_folds = sorted(np.unique(fold_ids).tolist())
    if len(unique_folds) != folds or unique_folds != list(range(folds)):
        raise ValueError(f"Fold IDs must contain every fold 0..{folds - 1}, got {unique_folds}")
    for group in np.unique(groups):
        if len(np.unique(fold_ids[groups == group])) != 1:
            raise ValueError(f"Source group {group!r} spans multiple folds")

    weights = sample_weights(groups, weighting)
    before_decisions = np.full(len(labels), np.nan, dtype=np.float64)
    after_decisions = np.full(len(labels), np.nan, dtype=np.float64)
    for fold_id in range(folds):
        test = np.flatnonzero(fold_ids == fold_id)
        train = np.flatnonzero(fold_ids != fold_id)
        if set(labels[train]) != {-1, 1} or set(labels[test]) != {-1, 1}:
            raise ValueError(f"Fold {fold_id} does not contain both classes in train/test")
        # Freeze the coordinate transform on before-state training examples.
        # Changing the after distribution can no longer change the before baseline.
        scaler = StandardScaler().fit(before[train], sample_weight=weights[train])
        before_train = scaler.transform(before[train])
        after_train = scaler.transform(after[train])
        before_probe = make_probe(probe_kind, c_value, seed + fold_id)
        after_probe = make_probe(probe_kind, c_value, seed + fold_id)
        before_probe.fit(before_train, labels[train], sample_weight=weights[train])
        after_probe.fit(after_train, labels[train], sample_weight=weights[train])
        before_decisions[test] = raw_space_decision(before_probe, scaler, before[test])
        after_decisions[test] = raw_space_decision(after_probe, scaler, after[test])

    if np.isnan(before_decisions).any() or np.isnan(after_decisions).any():
        raise RuntimeError("Cross-validation did not predict every sample")
    return before_decisions, after_decisions, fold_ids


def metric_values(
    labels: np.ndarray,
    decisions: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    signed = labels * decisions

    def weighted_mean(mask: np.ndarray) -> float:
        return float(np.average(signed[mask], weights=weights[mask]))

    return {
        "roc_auc": float(roc_auc_score(labels == 1, decisions, sample_weight=weights)),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                np.where(decisions >= 0, 1, -1),
                sample_weight=weights,
            )
        ),
        "signed_margin": weighted_mean(np.ones(len(labels), dtype=bool)),
        "benign_signed_margin": weighted_mean(labels == -1),
        "harmful_signed_margin": weighted_mean(labels == 1),
    }


def fold_metric_values(
    labels: np.ndarray,
    decisions: np.ndarray,
    fold_ids: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_fold: list[dict[str, float]] = []
    for fold_id in sorted(np.unique(fold_ids).tolist()):
        mask = fold_ids == fold_id
        values = metric_values(labels[mask], decisions[mask], weights[mask])
        per_fold.append({"fold": int(fold_id), "samples": int(np.sum(mask)), **values})
    aggregate = {
        metric: float(np.mean([row[metric] for row in per_fold]))
        for metric in METRIC_NAMES
    }
    return aggregate, per_fold


def bootstrap_cluster_rows(
    groups: np.ndarray,
    sampled_sources: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand source-cluster draws without leaking duplicate rows across folds."""
    source_indices = {
        source: np.flatnonzero(groups == source) for source in np.unique(groups)
    }
    indices_parts: list[np.ndarray] = []
    draw_groups: list[str] = []
    original_groups: list[str] = []
    for draw_index, source in enumerate(sampled_sources):
        if source not in source_indices:
            raise ValueError(f"Bootstrap source {source!r} is absent from groups")
        indices = source_indices[source]
        indices_parts.append(indices)
        draw_groups.extend([f"{draw_index}:{source}"] * len(indices))
        original_groups.extend([str(source)] * len(indices))
    if not indices_parts:
        raise ValueError("Bootstrap requires at least one sampled source")

    indices = np.concatenate(indices_parts)
    synthetic_groups = np.asarray(draw_groups)
    # Repeated draws keep distinct weight-cluster IDs, but every copy of the
    # same original source receives one fold. This preserves multiplicity
    # without putting identical rows in both probe training and test sets.
    fixed_fold_ids = balanced_group_fold_ids(
        np.asarray(original_groups), folds, seed
    )
    return indices, synthetic_groups, fixed_fold_ids


def bootstrap_refit_distributions(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    probe_kind: str,
    c_value: float,
    weighting: str,
    samples: int,
    seed: int,
    start_index: int = 0,
) -> dict[str, dict[str, list[float]]]:
    if samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if start_index < 0:
        raise ValueError("--bootstrap-start-index must be non-negative")
    rng = np.random.default_rng(seed)
    source_ids = np.unique(groups)
    distributions = {
        state: {metric: [] for metric in METRIC_NAMES}
        for state in ("before", "after", "delta")
    }
    # Consume preceding draws so chunk [start, start+n) is bit-for-bit
    # equivalent to the same slice of one monolithic seeded run.
    for _ in range(start_index):
        rng.choice(source_ids, size=len(source_ids), replace=True)
    for local_index in range(samples):
        bootstrap_index = start_index + local_index
        sampled_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
        indices, bootstrap_groups, fixed_fold_ids = bootstrap_cluster_rows(
            groups,
            sampled_sources,
            folds,
            seed + bootstrap_index + 1,
        )
        bootstrap_before, bootstrap_after, bootstrap_folds = cross_validated_decisions(
            before[indices],
            after[indices],
            labels[indices],
            bootstrap_groups,
            folds,
            probe_kind,
            c_value,
            seed + bootstrap_index + 1,
            fixed_fold_ids=fixed_fold_ids,
            weighting=weighting,
        )
        weights = sample_weights(bootstrap_groups, weighting)
        before_metrics, _ = fold_metric_values(
            labels[indices], bootstrap_before, bootstrap_folds, weights
        )
        after_metrics, _ = fold_metric_values(
            labels[indices], bootstrap_after, bootstrap_folds, weights
        )
        for metric in METRIC_NAMES:
            distributions["before"][metric].append(before_metrics[metric])
            distributions["after"][metric].append(after_metrics[metric])
            distributions["delta"][metric].append(
                after_metrics[metric] - before_metrics[metric]
            )
    return distributions


def bootstrap_intervals_from_distributions(
    distributions: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    expected_states = {"before", "after", "delta"}
    if set(distributions) != expected_states:
        raise ValueError(f"Bootstrap distributions require states {sorted(expected_states)}")
    lengths = {
        len(values)
        for metrics in distributions.values()
        for values in metrics.values()
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("Bootstrap distributions have inconsistent/empty draw counts")
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


def bootstrap_refit_intervals(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    probe_kind: str,
    c_value: float,
    weighting: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    distributions = bootstrap_refit_distributions(
        before,
        after,
        labels,
        groups,
        folds,
        probe_kind,
        c_value,
        weighting,
        samples,
        seed,
    )
    return bootstrap_intervals_from_distributions(distributions)


def main() -> None:
    args = parse_args()
    pair_filter = read_pair_filter(args.pair_filter)
    records, hidden_state_index, manifest = load_checkpoint_vectors(
        args.input_dir,
        args.method,
        {args.before, args.after},
        args.layer_column,
        args.group_field,
        pair_filter,
        args.run_id,
    )
    before, after, labels, groups, keys = paired_matrices(records, args.before, args.after)
    fold_manifest = read_fold_manifest(args.fold_manifest)
    fixed_fold_ids = manifest_fold_ids(keys, fold_manifest) if fold_manifest else None
    before_decisions, after_decisions, fold_ids = cross_validated_decisions(
        before,
        after,
        labels,
        groups,
        args.folds,
        args.probe,
        args.c,
        args.seed,
        fixed_fold_ids=fixed_fold_ids,
        weighting=args.weighting,
    )
    weights = sample_weights(groups, args.weighting)
    before_metrics, before_folds = fold_metric_values(
        labels, before_decisions, fold_ids, weights
    )
    after_metrics, after_folds = fold_metric_values(
        labels, after_decisions, fold_ids, weights
    )
    delta_metrics = {
        metric: after_metrics[metric] - before_metrics[metric] for metric in METRIC_NAMES
    }
    bootstrap_draws = bootstrap_refit_distributions(
        before,
        after,
        labels,
        groups,
        args.folds,
        args.probe,
        args.c,
        args.weighting,
        args.bootstrap_samples,
        args.seed,
        args.bootstrap_start_index,
    )
    intervals = bootstrap_intervals_from_distributions(bootstrap_draws)
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
        "group_field": args.group_field,
        "grouped_cross_validation": True,
        "fold_manifest": str(args.fold_manifest) if args.fold_manifest else None,
        "pair_filter": str(args.pair_filter) if args.pair_filter else None,
        "run_id": args.run_id,
        "weighting": args.weighting,
        "scaler_fit": "before_train_only",
        "margin_space": "raw_hidden_state",
        "metric_aggregation": "unweighted_mean_over_outer_folds",
        "samples": len(labels),
        "pairs": len({key[0] for key in keys}),
        "source_groups": len(np.unique(groups)),
        "class_counts": {
            "benign": int(np.sum(labels == -1)),
            "harmful": int(np.sum(labels == 1)),
        },
        "metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "delta_after_minus_before": delta_metrics,
        },
        "fold_metrics": {
            "before": before_folds,
            "after": after_folds,
        },
        "pooled_oof_diagnostic": {
            "before": metric_values(labels, before_decisions, weights),
            "after": metric_values(labels, after_decisions, weights),
        },
        "bootstrap_source_refit_95_ci": intervals,
        "bootstrap_duplicate_source_policy": "same_original_source_same_fold",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_start_index": args.bootstrap_start_index,
        "seed": args.seed,
    }
    if args.store_bootstrap_draws:
        result["bootstrap_draws"] = bootstrap_draws
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions_output: Path | None = None
    if not args.skip_predictions:
        predictions_output = args.predictions_output or args.output.with_suffix(
            ".predictions.jsonl"
        )
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", encoding="utf-8", newline="\n") as handle:
            for index, key in enumerate(keys):
                pair_id, side, run_id, group_id = key
                handle.write(
                    json.dumps(
                        {
                            "pair_id": pair_id,
                            "side": side,
                            "run_id": run_id,
                            "source_group": group_id,
                            "label": int(labels[index]),
                            "fold": int(fold_ids[index]),
                            "sample_weight": float(weights[index]),
                            "before_raw_space_decision": float(before_decisions[index]),
                            "after_raw_space_decision": float(after_decisions[index]),
                            "before_signed_margin": float(
                                labels[index] * before_decisions[index]
                            ),
                            "after_signed_margin": float(
                                labels[index] * after_decisions[index]
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(
        json.dumps(
            {
                **result,
                "predictions_output": (
                    str(predictions_output) if predictions_output is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
