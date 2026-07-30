#!/usr/bin/env python3
"""Generate responses and method-induced traces with vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from response_generation.data import iter_records
from response_generation.registry import get_method, method_names, methods
from response_generation.runtime import GenerationContext, chat_template_kwargs


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--method",
        choices=method_names(),
        help="Registered response-generation method.",
    )
    parser.add_argument(
        "--method-name",
        help=(
            "Experiment-facing method name, e.g. qwen3_base or llama31_ia. "
            "Defaults to --method."
        ),
    )
    parser.add_argument(
        "--methods-root",
        type=Path,
        default=Path("methods"),
        help="Directory containing external method implementations.",
    )
    parser.add_argument(
        "--query-field",
        default="query",
        help=(
            "Field to read when input is a flat JSONL query file. "
            "WildJailbreak pair files are detected automatically."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--enable-thinking",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "Optional chat-template enable_thinking value. Use false for Qwen3 "
            "no-thinking generation; auto leaves the tokenizer default unchanged."
        ),
    )
    parser.add_argument(
        "--store-prompts",
        action="store_true",
        help="Store chat messages used by each stage. This increases output size.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, method registration, and prompt budgets without loading vLLM.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print registered methods and exit.",
    )
    return parser.parse_args()


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def completed_keys(path: Path, method: str) -> set[tuple[str, str, int, str]]:
    completed: set[tuple[str, str, int, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("method") != method:
                    continue
                completed.add(
                    (
                        str(row["record_id"]),
                        str(row["source_field"]),
                        int(row["seed"]),
                        str(row["method"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid existing output at {path}:{line_number}"
                ) from exc
    return completed


def record_key(record: dict[str, Any], seed: int, method: str) -> tuple[str, str, int, str]:
    return (
        str(record["record_id"]),
        str(record["source_field"]),
        int(seed),
        method,
    )


def print_registered_methods() -> None:
    for name, spec in methods().items():
        print(f"{name}\t{spec.kind}\t{spec.description}")


def main() -> None:
    args = parse_args()
    if args.list_methods:
        print_registered_methods()
        return
    for arg_name in ("input", "output", "model", "method"):
        if getattr(args, arg_name) is None:
            raise ValueError(f"--{arg_name.replace('_', '-')} is required")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.stage1_max_new_tokens < 1:
        raise ValueError("--stage1-max-new-tokens must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    method = get_method(args.method)
    records = list(iter_records(args.input, args.query_field))
    if args.limit is not None:
        records = records[: args.limit]

    from transformers import AutoTokenizer

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

    previews = method.preview(ctx, records)
    print(
        f"Loaded {len(records)} queries; method={method.name}; "
        f"seeds={args.seeds}",
        flush=True,
    )
    for preview in previews:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_keys(args.output, method.name)
    from vllm import LLM

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
    ctx.llm = llm

    total_pending = sum(
        record_key(record, seed, method.name) not in completed
        for seed in args.seeds
        for record in records
    )
    progress = tqdm(total=total_pending, desc=f"Generating {method.name}", unit="sample")

    with args.output.open("a", encoding="utf-8", buffering=1) as output_handle:
        for seed in args.seeds:
            pending = [
                record
                for record in records
                if record_key(record, seed, method.name) not in completed
            ]
            for batch in batched(pending, args.batch_size):
                rows = method.generate_batch(ctx, batch, seed)
                for row in rows:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed.add(
                        (
                            str(row["record_id"]),
                            str(row["source_field"]),
                            int(row["seed"]),
                            str(row["method"]),
                        )
                    )
                    progress.update(1)
    progress.close()
    print(f"Wrote {len(completed)} completed samples to {args.output}", flush=True)


if __name__ == "__main__":
    main()
