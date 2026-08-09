from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from representation_analysis.build_controls import attach_controls, validate_complete_runs
from representation_analysis.prepare_formal_data import prepare_formal_data
from response_generation.data import iter_records


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def decode(self, token_ids, **kwargs):
        return " ".join(token_ids)


def pair_row(pair_id, benign_index, harmful_index, reranker, embedding=0.6):
    return {
        "pair_id": pair_id,
        "benign_source_index": benign_index,
        "harmful_source_index": harmful_index,
        "benign_prompt": f"benign prompt {pair_id}",
        "harmful_prompt": f"harmful prompt {pair_id}",
        "embedding_score": embedding,
        "reranker_score": reranker,
    }


def generation_row(pair_id, side, run_id, fold, group, analysis=None):
    analysis = analysis or f"analysis content for {pair_id} {side}"
    return {
        "record_id": f"{pair_id}:{side}",
        "pair_id": pair_id,
        "side": side,
        "gold_label": side,
        "run_id": run_id,
        "seed": run_id,
        "method": "safe_llm_intention_analysis",
        "harmful_source_index": group,
        "source_group": group,
        "prompt_messages": {
            "stage1": [{"role": "user", "content": f"analyze {pair_id} {side}"}],
            "final": [
                {"role": "user", "content": f"analyze {pair_id} {side}"},
                {"role": "assistant", "content": analysis},
                {"role": "user", "content": "answer safely"},
            ],
        },
        "formal_fold": fold,
    }


class FormalDataTests(unittest.TestCase):
    def test_freezes_grouped_folds_pilot_and_canonical_selection(self):
        rows = [
            pair_row("p0", 0, 10, 0.7),
            pair_row("p1", 1, 10, 0.9),
            pair_row("p2", 2, 11, 0.8),
            pair_row("p3", 3, 12, 0.6),
            pair_row("p4", 4, 13, 0.5),
            pair_row("p5", 5, 14, 0.4),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pairs.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            output_dir = root / "formal"
            manifest = prepare_formal_data(input_path, output_dir, 3, 2, 42)
            self.assertEqual(manifest["pair_count"], 6)
            self.assertEqual(manifest["unique_harmful_sources"], 5)
            self.assertEqual(manifest["pilot_unique_harmful_sources"], 3)
            self.assertEqual(manifest["canonical_pair_count"], 5)
            self.assertTrue((output_dir / "_SUCCESS").exists())
            canonical = [
                json.loads(line)
                for line in (output_dir / "canonical_pairs.jsonl").read_text().splitlines()
            ]
            self.assertIn("p1", {row["pair_id"] for row in canonical})
            self.assertNotIn("p0", {row["pair_id"] for row in canonical})
            folds = [
                json.loads(line)
                for line in (output_dir / "formal_folds.jsonl").read_text().splitlines()
            ]
            repeated = [row for row in folds if row["harmful_source_index"] == "10"]
            self.assertEqual(len({row["fold"] for row in repeated}), 1)
            pilot_folds = [
                json.loads(line)
                for line in (output_dir / "pilot_folds.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(pilot_folds), 3)
            self.assertEqual({row["fold"] for row in pilot_folds}, {0, 1})
            self.assertEqual(sum(manifest["pilot_fold_pair_counts"]), 3)

    def test_generation_records_preserve_harmful_source_group(self):
        row = pair_row("p0", 0, 99, 0.8)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = list(iter_records(path, "query"))
        self.assertEqual(len(records), 2)
        self.assertEqual({record["source_group"] for record in records}, {"99"})
        self.assertEqual({record["harmful_source_index"] for record in records}, {99})


class ControlTests(unittest.TestCase):
    def test_controls_are_fold_local_balanced_and_length_matched(self):
        rows = []
        folds = {}
        for index in range(4):
            pair_id = f"p{index}"
            group = f"g{index}"
            folds[pair_id] = (0, group)
            rows.extend(
                [
                    generation_row(pair_id, "benign", 42, 0, group),
                    generation_row(pair_id, "harmful", 42, 0, group),
                ]
            )
        controlled = attach_controls(rows, folds, WordTokenizer())
        donor_sides = {"benign": [], "harmful": []}
        for row in controlled:
            control = row["representation_controls"]["shuffled"]
            self.assertNotEqual(control["donor_pair_id"], row["pair_id"])
            self.assertNotEqual(control["donor_source_group"], row["source_group"])
            self.assertEqual(control["token_difference"], 0)
            self.assertEqual(
                control["analysis_token_count"],
                row["representation_controls"]["true"]["analysis_token_count"],
            )
            self.assertEqual(row["formal_fold"], 0)
            donor_sides[row["side"]].append(control["donor_side"])
        self.assertEqual(donor_sides["benign"].count("benign"), 2)
        self.assertEqual(donor_sides["benign"].count("harmful"), 2)
        self.assertEqual(donor_sides["harmful"].count("benign"), 2)
        self.assertEqual(donor_sides["harmful"].count("harmful"), 2)

    def test_composes_short_donors_to_exact_recipient_length(self):
        rows = []
        folds = {}
        for index in range(4):
            pair_id = f"p{index}"
            group = f"g{index}"
            folds[pair_id] = (0, group)
            benign_analysis = (
                "one two three four five six seven eight nine ten eleven twelve"
                if index == 0
                else f"short benign {index}"
            )
            rows.extend(
                [
                    generation_row(
                        pair_id, "benign", 42, 0, group, benign_analysis
                    ),
                    generation_row(
                        pair_id, "harmful", 42, 0, group, f"brief harmful {index}"
                    ),
                ]
            )

        controlled = attach_controls(rows, folds, WordTokenizer())
        recipient = next(
            row
            for row in controlled
            if row["pair_id"] == "p0" and row["side"] == "benign"
        )
        shuffled = recipient["representation_controls"]["shuffled"]
        self.assertEqual(shuffled["analysis_token_count"], 12)
        self.assertEqual(shuffled["token_difference"], 0)
        self.assertGreater(shuffled["segments"], 1)
        self.assertNotIn("p0", shuffled["donor_pair_ids"])
        self.assertNotIn("g0", shuffled["donor_source_groups"])

    def test_rejects_incomplete_pair_seed(self):
        rows = [generation_row("p0", "benign", 42, 0, "g0")]
        with self.assertRaisesRegex(ValueError, "lack benign and harmful"):
            validate_complete_runs(rows)


if __name__ == "__main__":
    unittest.main()
