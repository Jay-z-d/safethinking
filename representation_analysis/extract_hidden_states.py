#!/usr/bin/env python3
"""Extract selected hidden-state checkpoints from generation JSONL rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from representation_analysis.checkpoints import CheckpointText, build_checkpoint_texts


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--layers",
        default="-1",
        help=(
            "Comma-separated indices into Hugging Face hidden_states; 0 is the "
            "embedding output and -1 is the final transformer layer."
        ),
    )
    parser.add_argument("--pooling", choices=("last_token", "mean"), default="last_token")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit source JSONL rows.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", *DTYPES), default="auto")
    parser.add_argument("--output-dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_layer_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --layers value: {value!r}") from exc
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--layers must contain one or more unique integer indices")
    return indices


def resolve_layer_indices(requested: list[int], hidden_state_count: int) -> list[int]:
    resolved = [index if index >= 0 else hidden_state_count + index for index in requested]
    if any(index < 0 or index >= hidden_state_count for index in resolved):
        raise ValueError(
            f"Layer indices {requested} are invalid for {hidden_state_count} hidden states"
        )
    return resolved


def read_rows(path: Path, limit: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        rows = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield line_number, row
            rows += 1
            if limit is not None and rows >= limit:
                return


def batched(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def checkpoint_jobs(
    tokenizer: Any,
    rows: Iterable[tuple[int, dict[str, Any]]],
) -> Iterable[tuple[dict[str, Any], CheckpointText]]:
    for line_number, row in rows:
        for checkpoint in build_checkpoint_texts(tokenizer, row):
            metadata = {
                "input_line": line_number,
                "record_id": str(row.get("record_id") or ""),
                "pair_id": str(row.get("pair_id") or ""),
                "side": str(row.get("side") or row.get("gold_label") or ""),
                "run_id": int(row.get("run_id", row.get("seed", 0))),
                "seed": int(row.get("seed", row.get("run_id", 0))),
                "method": str(row.get("method") or "unknown"),
                "method_name": str(row.get("method_name") or row.get("method") or "unknown"),
                "model_path": str(row.get("model_path") or ""),
                "checkpoint": checkpoint.name,
                "checkpoint_source": checkpoint.source,
            }
            if not metadata["pair_id"] or metadata["side"] not in {"benign", "harmful"}:
                raise ValueError(
                    f"Row {line_number} lacks a valid pair_id or benign/harmful side"
                )
            yield metadata, checkpoint


def pool_hidden_state(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "last_token":
        positions = attention_mask.sum(dim=1) - 1
        return hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            positions,
        ]
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("--batch-size and --shard-size must be positive")
    prepare_output_dir(args.output_dir)
    requested_layers = parse_layer_indices(args.layers)

    from transformers import AutoModel, AutoTokenizer

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype != "auto":
        model_kwargs["torch_dtype"] = DTYPES[args.dtype]
    # The base model avoids allocating vocabulary logits that are unnecessary
    # for representation extraction and can otherwise dominate GPU memory.
    model = AutoModel.from_pretrained(str(args.model), **model_kwargs)
    model.to(args.device)
    model.eval()

    metadata_path = args.output_dir / "metadata.jsonl"
    shard_embeddings: list[torch.Tensor] = []
    shard_metadata: list[dict[str, Any]] = []
    shard_index = 0
    total_checkpoints = 0
    resolved_layers: list[int] | None = None
    checkpoint_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    output_dtype = DTYPES[args.output_dtype]

    def flush_shard(metadata_handle: Any) -> None:
        nonlocal shard_index, total_checkpoints
        if not shard_embeddings:
            return
        shard_name = f"embeddings-{shard_index:05d}.safetensors"
        tensor = torch.stack(shard_embeddings)
        save_file({"embeddings": tensor.contiguous()}, args.output_dir / shard_name)
        for row_index, metadata in enumerate(shard_metadata):
            output_row = {
                **metadata,
                "shard": shard_name,
                "row_index": row_index,
                "layer_indices": resolved_layers,
                "pooling": args.pooling,
                "embedding_dtype": args.output_dtype,
                "embedding_shape": list(tensor.shape[1:]),
            }
            metadata_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        total_checkpoints += len(shard_embeddings)
        shard_embeddings.clear()
        shard_metadata.clear()
        shard_index += 1

    rows = read_rows(args.input, args.limit)
    jobs = checkpoint_jobs(tokenizer, rows)
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        progress = tqdm(desc="Extracting hidden states", unit="checkpoint")
        for batch in batched(jobs, args.batch_size):
            metadata_rows = [item[0] for item in batch]
            texts = [item[1].text for item in batch]
            tokenized = tokenizer(
                texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            token_counts = tokenized["attention_mask"].sum(dim=1).tolist()
            if args.max_length is not None and max(token_counts) > args.max_length:
                offending = max(token_counts)
                raise ValueError(
                    f"Checkpoint length {offending} exceeds --max-length {args.max_length}; "
                    "refusing to truncate hidden-state inputs"
                )
            model_inputs = {name: value.to(args.device) for name, value in tokenized.items()}
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states")
            current_layers = resolve_layer_indices(requested_layers, len(hidden_states))
            if resolved_layers is None:
                resolved_layers = current_layers
            elif resolved_layers != current_layers:
                raise RuntimeError("Model returned an inconsistent hidden-state count")
            pooled = torch.stack(
                [
                    pool_hidden_state(hidden_states[index], model_inputs["attention_mask"], args.pooling)
                    for index in resolved_layers
                ],
                dim=1,
            ).to(dtype=output_dtype, device="cpu")

            for offset, metadata in enumerate(metadata_rows):
                metadata["token_count"] = int(token_counts[offset])
                shard_embeddings.append(pooled[offset])
                shard_metadata.append(metadata)
                checkpoint_counts[metadata["checkpoint"]] += 1
                method_counts[metadata["method"]] += 1
                progress.update(1)
                if len(shard_embeddings) >= args.shard_size:
                    flush_shard(metadata_handle)
        flush_shard(metadata_handle)
        progress.close()

    if total_checkpoints == 0 or resolved_layers is None:
        raise ValueError(f"No checkpoints extracted from {args.input}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "model": str(args.model),
        "tokenizer": str(tokenizer_path),
        "device": args.device,
        "model_dtype": args.dtype,
        "output_dtype": args.output_dtype,
        "pooling": args.pooling,
        "requested_layers": requested_layers,
        "resolved_hidden_state_indices": resolved_layers,
        "source_row_limit": args.limit,
        "checkpoints": total_checkpoints,
        "checkpoint_counts": dict(sorted(checkpoint_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "shards": shard_index,
        "metadata": metadata_path.name,
        "notes": (
            "h_query is the original-query prompt end; h_guided is the stored method "
            "prompt end; h_reasoned is IA stage-2 or a reconstructed saved generation prefix."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
