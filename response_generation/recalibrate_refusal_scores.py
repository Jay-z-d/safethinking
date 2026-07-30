#!/usr/bin/env python3
"""Recalibrate refusal scores from a scored calibration distribution."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--calibration-name", default="mixed_vanilla_benign_harmful")
    parser.add_argument("--raw-field", default="refusal_logprob_sum")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--store-previous", action="store_true")
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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute percentile of empty list")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def calibration_stats(path: Path, epsilon: float, name: str, raw_field: str) -> dict[str, Any]:
    values = []
    labels: dict[str, int] = {}
    for row in read_jsonl(path):
        try:
            values.append(float(row[raw_field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid {raw_field} in {path}") from exc
        label = str(row.get("gold_label") or row.get("side") or "unknown")
        labels[label] = labels.get(label, 0) + 1
    if not values:
        raise ValueError(f"{path} contains no calibration rows")
    q25 = percentile(values, 0.25)
    q75 = percentile(values, 0.75)
    return {
        "name": name,
        "source": str(path),
        "n": len(values),
        "label_counts": labels,
        "median": statistics.median(values),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "gamma": q75 - q25 + epsilon,
        "epsilon": epsilon,
        "raw_field": raw_field,
        "score_formula": f"sigmoid(({raw_field} - median) / gamma)",
    }


def recalibrate_row(
    row: dict[str, Any],
    stats: dict[str, Any],
    store_previous: bool,
    raw_field: str,
) -> dict[str, Any]:
    try:
        raw = float(row[raw_field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Input row missing/invalid {raw_field}") from exc
    updated = dict(row)
    if store_previous:
        updated["previous_refusal_score"] = row.get("refusal_score")
        updated["previous_refusal_calibration"] = row.get("refusal_calibration")
    updated["refusal_calibration"] = stats
    updated["refusal_score"] = sigmoid((raw - float(stats["median"])) / float(stats["gamma"]))
    return updated


def main() -> None:
    args = parse_args()
    stats = calibration_stats(args.calibration, args.epsilon, args.calibration_name, args.raw_field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.input):
            handle.write(
                json.dumps(
                    recalibrate_row(row, stats, args.store_previous, args.raw_field),
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(
            json.dumps({**stats, "recalibrated_rows": rows}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({**stats, "output": str(args.output), "recalibrated_rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
