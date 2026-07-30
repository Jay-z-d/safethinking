#!/usr/bin/env python3
"""Build semantically matched adversarial benign/harmful WildJailbreak pairs.

The pipeline has three resumable stages:
1. Encode adversarial_harmful and adversarial_benign prompts.
2. Retrieve exact top-k harmful neighbors for every benign prompt.
3. Rerank candidates and keep high-scoring opposite-label pairs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("wildjailbreak_pairs")

DEFAULT_EMBED_INSTRUCTION = (
    "Given an adversarial prompt, retrieve prompts that share the same underlying "
    "topic, target entity, scenario, and requested capability. Ignore differences "
    "in jailbreak framing and whether the intent is benign or harmful."
)
DEFAULT_RERANK_INSTRUCTION = (
    "Judge whether the Query and Document share the same underlying topic, target "
    "entity, scenario, and requested capability. Ignore differences in jailbreak "
    "framing and whether the intent is benign or harmful."
)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    source_index: int
    vanilla: str
    adversarial: str
    data_type: str

    def text(self, field: str) -> str:
        value = getattr(self, field).strip()
        if value:
            return value
        return self.vanilla.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the WildJailbreak training TSV.",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        required=True,
        help="Path to a local Qwen3 embedding model.",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        required=True,
        help="Path to a local Qwen3 reranker model.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("wildjailbreak_pair_work"))
    parser.add_argument("--output", type=Path, default=Path("wildjailbreak_pairs.jsonl"))
    parser.add_argument("--text-field", choices=("adversarial", "vanilla"), default="adversarial")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--embed-max-length", type=int, default=2048)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--rerank-max-length", type=int, default=2048)
    parser.add_argument("--rerank-query-chunk", type=int, default=100)
    parser.add_argument("--reranker-threshold", type=float, default=0.5)
    parser.add_argument("--min-embedding-score", type=float, default=-1.0)
    parser.add_argument(
        "--max-pairs-per-benign",
        type=int,
        default=1,
        help="Keep this many reranked pairs per benign prompt; 0 keeps all above threshold.",
    )
    parser.add_argument(
        "--unique-harmful",
        action="store_true",
        help="Apply greedy one-to-one matching so neither side is reused.",
    )
    parser.add_argument("--max-benign", type=int, default=0, help="0 means all records.")
    parser.add_argument("--max-harmful", type=int, default=0, help="0 means all records.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--embedding-instruction", default=DEFAULT_EMBED_INSTRUCTION)
    parser.add_argument("--rerank-instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()

    positive_args = (
        "top_k",
        "embed_batch_size",
        "embed_max_length",
        "retrieval_batch_size",
        "rerank_batch_size",
        "rerank_max_length",
        "rerank_query_chunk",
    )
    for name in positive_args:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_pairs_per_benign < 0:
        parser.error("--max-pairs-per-benign cannot be negative")
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_records(
    dataset: Path,
    text_field: str,
    max_benign: int,
    max_harmful: int,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    set_csv_field_limit()
    benign: list[PromptRecord] = []
    harmful: list[PromptRecord] = []
    required = {"vanilla", "adversarial", "data_type"}

    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

        for source_index, row in enumerate(reader):
            data_type = row["data_type"].strip()
            record = PromptRecord(
                source_index=source_index,
                vanilla=row["vanilla"],
                adversarial=row["adversarial"],
                data_type=data_type,
            )
            if not record.text(text_field):
                continue
            if data_type == "adversarial_benign" and (not max_benign or len(benign) < max_benign):
                benign.append(record)
            elif data_type == "adversarial_harmful" and (
                not max_harmful or len(harmful) < max_harmful
            ):
                harmful.append(record)
            if (
                max_benign
                and max_harmful
                and len(benign) >= max_benign
                and len(harmful) >= max_harmful
            ):
                break

    if not benign or not harmful:
        raise ValueError(
            f"Need both adversarial_benign and adversarial_harmful records; "
            f"found {len(benign)} and {len(harmful)}"
        )
    LOGGER.info("Loaded %d benign and %d harmful prompts", len(benign), len(harmful))
    return benign, harmful


def records_fingerprint(records: Sequence[PromptRecord], text_field: str) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.source_index).encode())
        digest.update(b"\0")
        digest.update(record.data_type.encode())
        digest.update(b"\0")
        digest.update(record.text(text_field).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, lengths]


def embedding_cache_valid(path: Path, expected: dict) -> bool:
    meta = read_json(path.with_suffix(".meta.json"))
    if not path.exists() or meta is None:
        return False
    return all(meta.get(key) == value for key, value in expected.items())


def encode_records(
    model: torch.nn.Module,
    tokenizer,
    records: Sequence[PromptRecord],
    text_field: str,
    output_path: Path,
    cache_descriptor: dict,
    batch_size: int,
    max_length: int,
    device: torch.device,
    is_query: bool,
    instruction: str,
) -> Path:
    if embedding_cache_valid(output_path, cache_descriptor):
        LOGGER.info("Using cached embeddings: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    hidden_size = int(model.config.hidden_size)
    mmap = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), hidden_size),
    )

    LOGGER.info("Encoding %d %s prompts", len(records), "query" if is_query else "document")
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            texts = [record.text(text_field) for record in batch_records]
            if is_query:
                texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs, use_cache=False, return_dict=True)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            end = start + len(batch_records)
            mmap[start:end] = embeddings.cpu().numpy().astype(np.float16)
            if start == 0 or end % (batch_size * 100) == 0 or end == len(records):
                LOGGER.info("Encoded %d/%d", end, len(records))

    mmap.flush()
    del mmap
    os.replace(temp_path, output_path)
    cache_descriptor = {**cache_descriptor, "dimension": hidden_size, "dtype": "float16"}
    atomic_json_dump(cache_descriptor, output_path.with_suffix(".meta.json"))
    return output_path


def load_embedding_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrieval_cache_valid(path: Path, expected: dict) -> bool:
    meta = read_json(path.with_suffix(".meta.json"))
    return path.exists() and meta is not None and all(
        meta.get(key) == value for key, value in expected.items()
    )


def exact_topk(
    query_embeddings_path: Path,
    corpus_embeddings_path: Path,
    output_path: Path,
    cache_descriptor: dict,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    if retrieval_cache_valid(output_path, cache_descriptor):
        LOGGER.info("Using cached retrieval candidates: %s", output_path)
        return output_path

    queries = np.load(query_embeddings_path, mmap_mode="r")
    corpus = np.load(corpus_embeddings_path, mmap_mode="r")
    effective_k = min(top_k, len(corpus))
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    corpus_array = np.asarray(corpus, dtype=np.float16 if device.type == "cuda" else np.float32)
    corpus_tensor = torch.from_numpy(np.array(corpus_array, copy=True)).to(
        device=device, dtype=compute_dtype
    )

    all_indices = np.empty((len(queries), effective_k), dtype=np.int32)
    all_scores = np.empty((len(queries), effective_k), dtype=np.float32)
    LOGGER.info("Computing exact top-%d for %d queries", effective_k, len(queries))
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            query_array = np.array(queries[start:end], copy=True)
            query_tensor = torch.from_numpy(query_array).to(device=device, dtype=compute_dtype)
            scores = query_tensor @ corpus_tensor.T
            values, indices = torch.topk(scores, k=effective_k, dim=1, largest=True, sorted=True)
            all_indices[start:end] = indices.cpu().numpy().astype(np.int32)
            all_scores[start:end] = values.float().cpu().numpy()
            if start == 0 or end % (batch_size * 20) == 0 or end == len(queries):
                LOGGER.info("Retrieved %d/%d", end, len(queries))

    del corpus_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez(handle, indices=all_indices, scores=all_scores)
    os.replace(temp_path, output_path)
    atomic_json_dump({**cache_descriptor, "effective_top_k": effective_k}, output_path.with_suffix(".meta.json"))
    return output_path


class QwenReranker:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        attn_implementation: str,
        max_length: int,
        instruction: str,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.backbone = getattr(self.model, self.model.base_model_prefix)
        self.false_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.true_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct provided. Note that the answer can "
            "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        if len(self.prefix_tokens) + len(self.suffix_tokens) >= max_length:
            raise ValueError("Reranker max length is too small for the prompt template")

    def _format_pair(self, query: str, document: str) -> list[int]:
        text = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        available = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        middle = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=available,
        )
        return self.prefix_tokens + middle + self.suffix_tokens

    @torch.inference_mode()
    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        result: list[float] = []
        for start in range(0, len(pairs), batch_size):
            ids = [self._format_pair(query, document) for query, document in pairs[start : start + batch_size]]
            features = [
                {"input_ids": item, "attention_mask": [1] * len(item)}
                for item in ids
            ]
            inputs = self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.device)
            outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
            last_hidden = outputs.last_hidden_state[:, -1, :]
            logits = self.model.lm_head(last_hidden)
            binary_logits = torch.stack(
                [logits[:, self.false_token_id], logits[:, self.true_token_id]], dim=1
            )
            result.extend(torch.softmax(binary_logits.float(), dim=1)[:, 1].cpu().tolist())
        return result


def rerank_signature(args: argparse.Namespace, benign_fp: str, harmful_fp: str) -> str:
    payload = {
        "benign_fingerprint": benign_fp,
        "harmful_fingerprint": harmful_fp,
        "reranker_model": str(args.reranker_model.resolve()),
        "rerank_instruction": args.rerank_instruction,
        "rerank_max_length": args.rerank_max_length,
        "reranker_threshold": args.reranker_threshold,
        "min_embedding_score": args.min_embedding_score,
        "max_pairs_per_benign": args.max_pairs_per_benign,
        "top_k": args.top_k,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def iter_ranges(length: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def write_jsonl_atomic(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def rerank_candidates(
    args: argparse.Namespace,
    benign: Sequence[PromptRecord],
    harmful: Sequence[PromptRecord],
    candidates_path: Path,
    benign_fp: str,
    harmful_fp: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Path:
    signature = rerank_signature(args, benign_fp, harmful_fp)
    parts_dir = args.work_dir / "rerank_parts" / signature
    ranges = list(iter_ranges(len(benign), args.rerank_query_chunk))
    missing = [
        (start, end)
        for start, end in ranges
        if not (parts_dir / f"part_{start:08d}_{end:08d}.jsonl").exists()
    ]
    candidates = np.load(candidates_path, mmap_mode="r")
    indices = candidates["indices"]
    embedding_scores = candidates["scores"]

    reranker = None
    if missing:
        LOGGER.info("Loading reranker; %d/%d output parts remain", len(missing), len(ranges))
        reranker = QwenReranker(
            args.reranker_model,
            device,
            dtype,
            args.attn_implementation,
            args.rerank_max_length,
            args.rerank_instruction,
        )

    for part_number, (start, end) in enumerate(missing, start=1):
        flat_meta: list[tuple[int, int, int, float]] = []
        flat_pairs: list[tuple[str, str]] = []
        for benign_index in range(start, end):
            query = benign[benign_index].text(args.text_field)
            for embedding_rank, (harmful_index, embedding_score) in enumerate(
                zip(indices[benign_index], embedding_scores[benign_index]), start=1
            ):
                score = float(embedding_score)
                if score < args.min_embedding_score:
                    continue
                harmful_index = int(harmful_index)
                flat_meta.append((benign_index, harmful_index, embedding_rank, score))
                flat_pairs.append((query, harmful[harmful_index].text(args.text_field)))

        reranker_scores = reranker.score(flat_pairs, args.rerank_batch_size) if flat_pairs else []
        grouped: dict[int, list[tuple[float, tuple[int, int, int, float]]]] = {}
        for meta, reranker_score in zip(flat_meta, reranker_scores):
            if reranker_score >= args.reranker_threshold:
                grouped.setdefault(meta[0], []).append((reranker_score, meta))

        output_records: list[dict] = []
        for benign_index in range(start, end):
            ranked = sorted(grouped.get(benign_index, []), key=lambda item: item[0], reverse=True)
            if args.max_pairs_per_benign:
                ranked = ranked[: args.max_pairs_per_benign]
            benign_record = benign[benign_index]
            for reranker_rank, (reranker_score, meta) in enumerate(ranked, start=1):
                _, harmful_index, embedding_rank, embedding_score = meta
                harmful_record = harmful[harmful_index]
                if benign_record.data_type == harmful_record.data_type:
                    raise AssertionError("Pair labels must be opposite")
                output_records.append(
                    {
                        "pair_id": f"wjb_{benign_record.source_index}_{harmful_record.source_index}",
                        "benign_source_index": benign_record.source_index,
                        "harmful_source_index": harmful_record.source_index,
                        "benign_label": benign_record.data_type,
                        "harmful_label": harmful_record.data_type,
                        "benign_prompt": benign_record.text(args.text_field),
                        "harmful_prompt": harmful_record.text(args.text_field),
                        "benign_vanilla": benign_record.vanilla,
                        "harmful_vanilla": harmful_record.vanilla,
                        "embedding_score": embedding_score,
                        "embedding_rank": embedding_rank,
                        "reranker_score": float(reranker_score),
                        "reranker_rank": reranker_rank,
                    }
                )

        part_path = parts_dir / f"part_{start:08d}_{end:08d}.jsonl"
        write_jsonl_atomic(output_records, part_path)
        LOGGER.info(
            "Reranked part %d/%d (%d-%d): kept %d pairs",
            part_number,
            len(missing),
            start,
            end,
            len(output_records),
        )

    if reranker is not None:
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return merge_parts(parts_dir, args.output, args.unique_harmful)


def merge_parts(parts_dir: Path, output: Path, unique_harmful: bool) -> Path:
    parts = sorted(parts_dir.glob("part_*.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if not unique_harmful:
        with temp.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target)
    else:
        records: list[dict] = []
        for part in parts:
            with part.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        records.sort(key=lambda item: item["reranker_score"], reverse=True)
        used_benign: set[int] = set()
        used_harmful: set[int] = set()
        selected: list[dict] = []
        for record in records:
            benign_id = record["benign_source_index"]
            harmful_id = record["harmful_source_index"]
            if benign_id in used_benign or harmful_id in used_harmful:
                continue
            used_benign.add(benign_id)
            used_harmful.add(harmful_id)
            selected.append(record)
        selected.sort(key=lambda item: item["benign_source_index"])
        with temp.open("w", encoding="utf-8") as target:
            for record in selected:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, output)
    LOGGER.info("Wrote final pairs to %s", output)
    return output


def validate_model_files(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing model index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        expected = sorted(set(json.load(handle)["weight_map"].values()))
    for filename in expected:
        path = model_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if not list(tensors.keys()):
                raise ValueError(f"No tensors found in {path}")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_model_files(args.embedding_model)
    validate_model_files(args.reranker_model)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_cache:
        for path in (
            args.work_dir / "harmful_embeddings.npy",
            args.work_dir / "harmful_embeddings.meta.json",
            args.work_dir / "benign_embeddings.npy",
            args.work_dir / "benign_embeddings.meta.json",
            args.work_dir / "retrieval_topk.npz",
            args.work_dir / "retrieval_topk.meta.json",
        ):
            path.unlink(missing_ok=True)

    benign, harmful = read_records(
        args.dataset, args.text_field, args.max_benign, args.max_harmful
    )
    benign_fp = records_fingerprint(benign, args.text_field)
    harmful_fp = records_fingerprint(harmful, args.text_field)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    LOGGER.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_embedding_model(
        args.embedding_model, device, dtype, args.attn_implementation
    )
    common_descriptor = {
        "dataset": str(args.dataset.resolve()),
        "embedding_model": str(args.embedding_model.resolve()),
        "text_field": args.text_field,
        "max_length": args.embed_max_length,
    }
    harmful_path = encode_records(
        model,
        tokenizer,
        harmful,
        args.text_field,
        args.work_dir / "harmful_embeddings.npy",
        {
            **common_descriptor,
            "role": "document",
            "count": len(harmful),
            "fingerprint": harmful_fp,
            "instruction": None,
        },
        args.embed_batch_size,
        args.embed_max_length,
        device,
        False,
        args.embedding_instruction,
    )
    benign_path = encode_records(
        model,
        tokenizer,
        benign,
        args.text_field,
        args.work_dir / "benign_embeddings.npy",
        {
            **common_descriptor,
            "role": "query",
            "count": len(benign),
            "fingerprint": benign_fp,
            "instruction": args.embedding_instruction,
        },
        args.embed_batch_size,
        args.embed_max_length,
        device,
        True,
        args.embedding_instruction,
    )
    release_model(model)
    del model, tokenizer

    candidates_path = exact_topk(
        benign_path,
        harmful_path,
        args.work_dir / "retrieval_topk.npz",
        {
            "benign_fingerprint": benign_fp,
            "harmful_fingerprint": harmful_fp,
            "top_k": args.top_k,
            "benign_count": len(benign),
            "harmful_count": len(harmful),
        },
        args.top_k,
        args.retrieval_batch_size,
        device,
    )
    rerank_candidates(
        args,
        benign,
        harmful,
        candidates_path,
        benign_fp,
        harmful_fp,
        device,
        dtype,
    )


if __name__ == "__main__":
    main()
