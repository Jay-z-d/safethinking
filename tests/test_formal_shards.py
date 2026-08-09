from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import save_file

from representation_analysis.formal_shards import (
    FORMAL_CHECKPOINTS,
    merge_generation_shards,
    merge_probe_chunks,
    merge_representation_shards,
    prepare_pair_shards,
    split_controlled_generation,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def pair_row(index: int) -> dict:
    return {
        "pair_id": f"p{index}",
        "benign_source_index": index,
        "harmful_source_index": 100 + index,
        "benign_prompt": f"benign {index}",
        "harmful_prompt": f"harmful {index}",
    }


def generation_row(pair: dict, side: str, seed: int, model: Path) -> dict:
    pair_id = pair["pair_id"]
    return {
        "record_id": f"{pair_id}:{side}",
        "pair_id": pair_id,
        "side": side,
        "gold_label": side,
        "run_id": seed,
        "seed": seed,
        "method": "safe_llm_intention_analysis",
        "model_path": str(model),
        "source_group": str(pair["harmful_source_index"]),
        "representation_controls": {
            "true": {"analysis_token_count": 2},
            "shuffled": {"analysis": "other text", "analysis_token_count": 2},
            "empty": {"analysis": "", "analysis_token_count": 0},
        },
    }


class FormalShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.pairs = [pair_row(index) for index in range(5)]
        self.pair_input = self.root / "pairs.jsonl"
        self.fold_manifest = self.root / "folds.jsonl"
        write_jsonl(self.pair_input, self.pairs)
        write_jsonl(
            self.fold_manifest,
            [
                {
                    "pair_id": row["pair_id"],
                    "harmful_source_index": str(row["harmful_source_index"]),
                    "fold": index % 2,
                }
                for index, row in enumerate(self.pairs)
            ],
        )
        self.pair_shards = self.root / "pair-shards"
        self.pair_manifest = prepare_pair_shards(
            self.pair_input,
            self.fold_manifest,
            self.pair_shards,
            pairs_per_shard=2,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_generation_shards(self) -> Path:
        output = self.root / "generation-shards"
        output.mkdir()
        for shard in self.pair_manifest["shards"]:
            shard_pairs = [
                json.loads(line)
                for line in (self.pair_shards / shard["file"]).read_text().splitlines()
            ]
            rows = [
                generation_row(pair, side, seed, self.model)
                for pair in shard_pairs
                for side in ("benign", "harmful")
                for seed in (42, 43)
            ]
            write_jsonl(output / f"generation-{shard['index']:05d}.jsonl", rows)
        return output

    def test_prepare_merge_and_split_are_complete(self):
        self.assertEqual(self.pair_manifest["pair_count"], 5)
        self.assertEqual(self.pair_manifest["shard_count"], 3)
        generation_dir = self.make_generation_shards()
        merged = self.root / "full.generation.jsonl"
        manifest = merge_generation_shards(
            self.pair_shards,
            generation_dir,
            merged,
            [42, 43],
            "safe_llm_intention_analysis",
            self.model,
        )
        self.assertEqual(manifest["rows"], 20)
        self.assertTrue(merged.with_name(merged.name + "._SUCCESS").exists())

        controlled_shards = self.root / "controlled-shards"
        controlled_manifest = split_controlled_generation(
            merged,
            self.pair_shards,
            controlled_shards,
            [42, 43],
        )
        self.assertEqual(controlled_manifest["rows"], 20)
        self.assertEqual(
            [shard["rows"] for shard in controlled_manifest["shards"]],
            [8, 8, 4],
        )

    def test_merge_representations_rewrites_shard_names(self):
        generation_dir = self.make_generation_shards()
        merged_generation = self.root / "full.generation.jsonl"
        merge_generation_shards(
            self.pair_shards,
            generation_dir,
            merged_generation,
            [42, 43],
            "safe_llm_intention_analysis",
            self.model,
        )
        controlled_shards = self.root / "controlled-shards"
        controlled_manifest = split_controlled_generation(
            merged_generation,
            self.pair_shards,
            controlled_shards,
            [42, 43],
        )

        representation_root = self.root / "representation-shards"
        for shard in controlled_manifest["shards"]:
            index = shard["index"]
            source_dir = representation_root / f"repr-{index:05d}"
            source_dir.mkdir(parents=True)
            controlled_rows = [
                json.loads(line)
                for line in (controlled_shards / shard["file"]).read_text().splitlines()
            ]
            metadata = []
            for row in controlled_rows:
                for checkpoint in FORMAL_CHECKPOINTS:
                    metadata.append(
                        {
                            "pair_id": row["pair_id"],
                            "side": row["side"],
                            "run_id": row["run_id"],
                            "source_group": row["source_group"],
                            "method": row["method"],
                            "checkpoint": checkpoint,
                            "shard": "embeddings-00000.safetensors",
                            "row_index": len(metadata),
                        }
                    )
            save_file(
                {"embeddings": np.zeros((len(metadata), 2, 3), dtype=np.float16)},
                source_dir / "embeddings-00000.safetensors",
            )
            write_jsonl(source_dir / "metadata.jsonl", metadata)
            manifest = {
                "input": str(controlled_shards / shard["file"]),
                "model": str(self.model),
                "tokenizer": str(self.model),
                "model_dtype": "bfloat16",
                "output_dtype": "float16",
                "pooling": "last_token",
                "requested_layers": [0, 32],
                "resolved_hidden_state_indices": [0, 32],
                "checkpoints": len(metadata),
                "checkpoint_counts": {
                    checkpoint: len(controlled_rows) for checkpoint in FORMAL_CHECKPOINTS
                },
                "method_counts": {
                    "safe_llm_intention_analysis": len(metadata)
                },
                "shards": 1,
                "metadata": "metadata.jsonl",
            }
            (source_dir / "manifest.json").write_text(json.dumps(manifest))
            (source_dir / "_SUCCESS").write_text("complete\n")

        combined = self.root / "representations"
        manifest = merge_representation_shards(
            controlled_shards,
            representation_root,
            combined,
        )
        self.assertEqual(manifest["checkpoints"], 20 * len(FORMAL_CHECKPOINTS))
        self.assertEqual(manifest["shards"], 3)
        self.assertTrue((combined / "_SUCCESS").exists())
        combined_metadata = [
            json.loads(line)
            for line in (combined / "metadata.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(combined_metadata), manifest["checkpoints"])
        self.assertEqual(
            {row["shard"] for row in combined_metadata},
            {f"embeddings-{index:05d}.safetensors" for index in range(3)},
        )

    def test_merge_rejects_incomplete_generation_shard(self):
        generation_dir = self.make_generation_shards()
        first = generation_dir / "generation-00000.jsonl"
        lines = first.read_text().splitlines()
        first.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_generation_shards(
                self.pair_shards,
                generation_dir,
                self.root / "invalid.jsonl",
                [42, 43],
                "safe_llm_intention_analysis",
                self.model,
            )

    def test_merge_probe_chunks_requires_contiguous_equivalent_draws(self):
        chunks = self.root / "probe-chunks"
        chunks.mkdir()
        metrics = (
            "roc_auc",
            "balanced_accuracy",
            "signed_margin",
            "benign_signed_margin",
            "harmful_signed_margin",
        )
        for chunk_index, start in enumerate((0, 2)):
            draws = {
                state: {
                    metric: [float(start), float(start + 1)] for metric in metrics
                }
                for state in ("before", "after", "delta")
            }
            row = {
                "created_at": f"time-{chunk_index}",
                "before": "h_guided",
                "after": "h_analysis_boundary_true",
                "metrics": {"fixed": 1.0},
                "bootstrap_source_refit_95_ci": {},
                "bootstrap_duplicate_source_policy": "same_original_source_same_fold",
                "bootstrap_samples": 2,
                "bootstrap_start_index": start,
                "bootstrap_draws": draws,
                "seed": 42,
            }
            (chunks / f"guided.chunk-{chunk_index:05d}.json").write_text(
                json.dumps(row), encoding="utf-8"
            )
        output = self.root / "guided.full.json"
        result = merge_probe_chunks(chunks, "guided", 2, output)
        self.assertEqual(result["bootstrap_samples"], 4)
        self.assertTrue(result["bootstrap_sharded_equivalent"])
        self.assertEqual(len(result["bootstrap_draws"]["delta"]["roc_auc"]), 4)
        self.assertEqual(len(result["bootstrap_chunks"]), 2)


if __name__ == "__main__":
    unittest.main()
