#!/usr/bin/env python3
"""Score refusal tendency with per-template normalized logprobs (mean of top-5 template averages)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from response_generation.registry import get_method, method_names
from response_generation.runtime import GenerationContext, chat_template_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", choices=method_names(), required=True)
    parser.add_argument(
        "--calibration-input",
        type=Path,
        help="Generated refusal-induction rows used to estimate median/IQR.",
    )
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument(
        "--methods-root",
        type=Path,
        default=Path("methods"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Deprecated compatibility option; refusal is now aggregated as mean of top-5 template avg logprobs.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Write raw top-5-mean refusal logprob values without applying calibration.",
    )
    parser.add_argument("--scoring-batch-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enable-thinking", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_patterns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    patterns = payload.get("refusal_patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"No refusal_patterns found in {path}")
    result = []
    for item in patterns:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text") or "")
        else:
            text = ""
        text = text.strip()
        if text:
            result.append(text)
    if not result:
        raise ValueError(f"No non-empty refusal patterns found in {path}")
    return result


def logprob_value(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError("Missing prompt logprob entry")
    value = None
    if isinstance(entry, dict):
        value = entry.get(token_id)
        if value is None:
            value = entry.get(str(token_id))
    if value is None:
        raise ValueError(
            f"Actual token id {token_id} not found in prompt_logprobs entry; "
            "try increasing --prompt-logprobs"
        )
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def candidate_token_ids(tokenizer: Any, context_prompt: str, full_prompt: str) -> list[int]:
    context_ids = tokenizer.encode(context_prompt)
    full_ids = tokenizer.encode(full_prompt)
    candidate_ids = full_ids[len(context_ids) :]
    if not candidate_ids:
        raise ValueError("Candidate produced no new tokens after context")
    return candidate_ids


def render_refusal_context(ctx: GenerationContext, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    method = get_method(ctx.method_name)
    context = method.refusal_context(ctx, row)
    messages = context["messages"]
    add_generation_prompt = bool(context.get("add_generation_prompt", True))
    return ctx.render_messages(messages, add_generation_prompt=add_generation_prompt), context


def score_pattern_batch(
    ctx: GenerationContext,
    llm: Any,
    requests: list[dict[str, Any]],
    prompt_logprobs: int,
) -> None:
    from vllm import SamplingParams

    prompts = [request["full_prompt"] for request in requests]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=prompt_logprobs,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    for request, output in zip(requests, outputs, strict=True):
        prompt_logprobs = output.prompt_logprobs
        start = request["candidate_start"]
        candidate_ids = request["candidate_ids"]
        logprobs = []
        for offset, token_id in enumerate(candidate_ids):
            logprobs.append(logprob_value(prompt_logprobs[start + offset], token_id))
        request["total_logprob"] = math.fsum(logprobs)
        request["token_count"] = len(candidate_ids)


def score_rows(
    ctx: GenerationContext,
    llm: Any,
    rows: list[dict[str, Any]],
    patterns: list[str],
    scoring_batch_size: int,
    prompt_logprobs: int,
) -> list[dict[str, Any]]:
    scored_rows = []
    progress = tqdm(total=len(rows), desc="Scoring refusal", unit="row")
    for row_batch in batched(rows, max(1, scoring_batch_size // max(1, len(patterns)))):
        requests = []
        contexts = []
        for row in row_batch:
            context_prompt, context_info = render_refusal_context(ctx, row)
            contexts.append(context_info)
            context_ids = ctx.tokenizer.encode(context_prompt)
            for pattern in patterns:
                full_prompt = context_prompt + pattern
                requests.append(
                    {
                        "row": row,
                        "pattern": pattern,
                        "context_prompt": context_prompt,
                        "full_prompt": full_prompt,
                        "candidate_start": len(context_ids),
                        "candidate_ids": candidate_token_ids(
                            ctx.tokenizer,
                            context_prompt,
                            full_prompt,
                        ),
                    }
                )
        for request_batch in batched(requests, scoring_batch_size):
            score_pattern_batch(ctx, llm, request_batch, prompt_logprobs)

        by_row_id: dict[int, list[dict[str, Any]]] = {}
        for request in requests:
            by_row_id.setdefault(id(request["row"]), []).append(request)
        for row, context_info in zip(row_batch, contexts, strict=True):
            row_requests = by_row_id[id(row)]
            pattern_scores = [
                {
                    "pattern": item["pattern"],
                    "total_logprob": item["total_logprob"],
                    "token_count": item["token_count"],
                    "avg_logprob": item["total_logprob"] / item["token_count"],
                }
                for item in row_requests
            ]
            pattern_scores.sort(key=lambda item: item["avg_logprob"], reverse=True)
            top5 = pattern_scores[:5]
            refusal_raw = math.fsum(item["avg_logprob"] for item in top5) / len(top5)
            scored = {
                **row,
                "refusal_scoring_context_type": context_info.get("context_type"),
                "refusal_logprob_sum": refusal_raw,
                "refusal_pattern_aggregation": "mean_top5_template_avg_logprob",
                "refusal_pattern_count": len(pattern_scores),
                "refusal_pattern_scores": pattern_scores,
                "refusal_top5_patterns": top5,
            }
            scored_rows.append(scored)
            progress.update(1)
    progress.close()
    return scored_rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
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


def calibration_stats(rows: list[dict[str, Any]], epsilon: float) -> dict[str, float | str]:
    values = [float(row["refusal_logprob_sum"]) for row in rows]
    if not values:
        raise ValueError("Cannot calibrate refusal score from zero rows")
    q25 = percentile(values, 0.25)
    q75 = percentile(values, 0.75)
    return {
        "median": statistics.median(values),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "gamma": q75 - q25 + epsilon,
        "n": len(values),
        "raw_field": "refusal_logprob_sum",
        "score_formula": "sigmoid((refusal_logprob_sum - median) / gamma)",
    }


def apply_refusal_score(rows: list[dict[str, Any]], stats: dict[str, float]) -> list[dict[str, Any]]:
    median = float(stats["median"])
    gamma = float(stats["gamma"])
    return [
        {
            **row,
            "refusal_calibration": stats,
            "refusal_score": sigmoid((float(row["refusal_logprob_sum"]) - median) / gamma),
        }
        for row in rows
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    patterns = load_patterns(args.patterns)
    method = get_method(args.method)

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    ctx = GenerationContext(
        args=args,
        tokenizer=tokenizer,
        llm=None,
        method_name=method.name,
        method_description=method.description,
        method_kind=method.kind,
        template_kwargs=chat_template_kwargs(args.enable_thinking),
    )
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )

    if args.calibration_input:
        calibration_rows = read_jsonl(args.calibration_input, limit=args.limit)
        scored_calibration = score_rows(
            ctx,
            llm,
            calibration_rows,
            patterns,
            args.scoring_batch_size,
            args.prompt_logprobs,
        )
        stats = calibration_stats(scored_calibration, args.epsilon)
        if not args.raw_only:
            scored_calibration = apply_refusal_score(scored_calibration, stats)
        if args.calibration_output:
            write_jsonl(args.calibration_output, scored_calibration)
    else:
        stats = {
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 1.0,
            "gamma": 1.0,
            "n": 0,
            "warning": "no calibration input; scores use sigmoid(raw refusal_logprob_sum)",
            "raw_field": "refusal_logprob_sum",
        }

    rows = read_jsonl(args.input, limit=args.limit)
    scored = score_rows(
        ctx,
        llm,
        rows,
        patterns,
        args.scoring_batch_size,
        args.prompt_logprobs,
    )
    if not args.raw_only:
        scored = apply_refusal_score(scored, stats)
    write_jsonl(args.output, scored)
    print(f"Wrote {len(scored)} rows to {args.output}")


if __name__ == "__main__":
    main()
