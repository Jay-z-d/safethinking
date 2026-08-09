from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import save_file

from representation_analysis.checkpoints import build_checkpoint_texts
from representation_analysis.probe_separability import (
    balanced_group_fold_ids,
    bootstrap_cluster_rows,
    bootstrap_refit_intervals,
    cross_validated_decisions,
    load_checkpoint_vectors,
    metric_values,
    paired_matrices,
    sample_weights,
)
from response_generation.case_study import behavioral_boundary_margin

try:
    import torch

    from representation_analysis.extract_hidden_states import (
        checkpoint_jobs,
        pool_hidden_state,
        resolve_layer_indices,
    )
except ModuleNotFoundError:
    torch = None


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        **kwargs,
    ):
        assert tokenize is False
        body = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        boundary = "GEN" if add_generation_prompt else "EOT"
        return f"{body}|{boundary}|thinking={kwargs.get('enable_thinking', 'auto')}"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


class CheckpointTests(unittest.TestCase):
    def test_ia_separates_analysis_boundary_from_preanswer(self):
        row = {
            "query": "question",
            "method": "safe_llm_intention_analysis",
            "enable_thinking": False,
            "method_trace": {
                "intermediate_messages": {
                    "stage1": [{"role": "user", "content": "analyze question"}],
                    "final": [
                        {"role": "user", "content": "analyze question"},
                        {"role": "assistant", "content": "benign intent"},
                        {"role": "user", "content": "answer safely"},
                    ],
                }
            },
        }
        checkpoints = build_checkpoint_texts(DummyTokenizer(), row)
        self.assertEqual(
            [checkpoint.name for checkpoint in checkpoints],
            [
                "h_query",
                "h_guided",
                "h_analysis_boundary_true",
                "h_preanswer_true",
            ],
        )
        analysis = checkpoints[-2]
        preanswer = checkpoints[-1]
        self.assertIn("assistant:benign intent", analysis.text)
        self.assertIn("|EOT|", analysis.text)
        self.assertNotIn("answer safely", analysis.text)
        self.assertIn("answer safely", preanswer.text)
        self.assertIn("|GEN|", preanswer.text)

    def test_ia_emits_true_shuffled_and_empty_controls(self):
        row = {
            "query": "question",
            "method": "safe_llm_intention_analysis",
            "method_trace": {
                "intermediate_messages": {
                    "stage1": [{"role": "user", "content": "analyze question"}],
                    "final": [
                        {"role": "user", "content": "analyze question"},
                        {"role": "assistant", "content": "true analysis"},
                        {"role": "user", "content": "answer safely"},
                    ],
                }
            },
            "representation_controls": {
                "true": {"analysis_token_count": 2},
                "shuffled": {
                    "analysis": "donor analysis",
                    "donor_pair_id": "p2",
                    "analysis_token_count": 2,
                },
                "empty": {"analysis": "", "analysis_token_count": 0},
            },
        }
        checkpoints = build_checkpoint_texts(DummyTokenizer(), row)
        self.assertEqual(
            [checkpoint.name for checkpoint in checkpoints],
            [
                "h_query",
                "h_guided",
                "h_analysis_boundary_true",
                "h_preanswer_true",
                "h_analysis_boundary_shuffled",
                "h_preanswer_shuffled",
                "h_analysis_boundary_empty",
                "h_preanswer_empty",
            ],
        )
        shuffled = next(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.name == "h_analysis_boundary_shuffled"
        )
        self.assertIn("donor analysis", shuffled.text)
        self.assertEqual(shuffled.metadata["donor_pair_id"], "p2")

    def test_wrapper_uses_prefix_before_final_answer(self):
        row = {
            "query": "question",
            "method": "goal_prioritization",
            "raw_generation": "[Internal thoughts] inspect risk [Final response] safe answer",
            "final_response": "safe answer",
            "method_trace": {
                "intermediate_messages": {
                    "final": [{"role": "user", "content": "wrapped question"}],
                }
            },
        }
        checkpoints = build_checkpoint_texts(DummyTokenizer(), row)
        self.assertEqual(
            [checkpoint.name for checkpoint in checkpoints],
            ["h_query", "h_guided", "h_reasoned"],
        )
        self.assertTrue(checkpoints[-1].text.endswith("[Final response] "))


@unittest.skipUnless(torch is not None, "PyTorch is not installed in the local test environment")
class ExtractionMathTests(unittest.TestCase):
    def test_layer_resolution(self):
        self.assertEqual(resolve_layer_indices([0, -1, -2], 5), [0, 4, 3])
        with self.assertRaises(ValueError):
            resolve_layer_indices([-6], 5)

    def test_last_and_mean_pooling_ignore_padding(self):
        hidden = torch.tensor(
            [
                [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],
                [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        np.testing.assert_allclose(
            pool_hidden_state(hidden, mask, "last_token").numpy(),
            [[3.0, 3.0], [6.0, 6.0]],
        )
        np.testing.assert_allclose(
            pool_hidden_state(hidden, mask, "mean").numpy(),
            [[2.0, 2.0], [4.0, 4.0]],
        )

    def test_extractor_rejects_generation_from_a_different_model(self):
        rows = [
            (
                1,
                {
                    "query": "question",
                    "pair_id": "p0",
                    "side": "benign",
                    "run_id": 42,
                    "method": "direct",
                    "model_path": "/models/wrong",
                },
            )
        ]
        with self.assertRaisesRegex(ValueError, "model_path mismatch"):
            next(checkpoint_jobs(DummyTokenizer(), rows, Path("/models/right")))


class ProbeTests(unittest.TestCase):
    def test_loads_sharded_checkpoint_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = np.asarray(
                [
                    [[-1.0, 0.0]],
                    [[-2.0, 0.0]],
                    [[1.0, 0.0]],
                    [[2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            save_file({"embeddings": embeddings}, root / "embeddings-00000.safetensors")
            metadata = [
                ("benign", "h_query", 0),
                ("benign", "h_reasoned", 1),
                ("harmful", "h_query", 2),
                ("harmful", "h_reasoned", 3),
            ]
            with (root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                import json

                for side, checkpoint, row_index in metadata:
                    handle.write(
                        json.dumps(
                            {
                                "pair_id": "p0",
                                "side": side,
                                "run_id": 42,
                                "source_group": "g0",
                                "method": "ia",
                                "checkpoint": checkpoint,
                                "shard": "embeddings-00000.safetensors",
                                "row_index": row_index,
                            }
                        )
                        + "\n"
                    )
            (root / "manifest.json").write_text(
                '{"resolved_hidden_state_indices":[32],"metadata":"metadata.jsonl"}\n',
                encoding="utf-8",
            )
            (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            records, layer, _ = load_checkpoint_vectors(
                root,
                "ia",
                {"h_query", "h_reasoned"},
                -1,
            )
            before, after, labels, groups, keys = paired_matrices(
                records,
                "h_query",
                "h_reasoned",
            )
            self.assertEqual(layer, 32)
            self.assertEqual(before.shape, (2, 2))
            self.assertEqual(after.shape, (2, 2))
            self.assertEqual(set(labels.tolist()), {-1, 1})
            self.assertEqual(set(groups.tolist()), {"g0"})
            self.assertEqual(len(keys), 2)

    def test_after_state_is_more_separable_on_held_out_groups(self):
        rng = np.random.default_rng(7)
        pair_count = 30
        labels = np.tile(np.asarray([-1, 1], dtype=np.int8), pair_count)
        groups = np.repeat(np.asarray([f"g{i:02d}" for i in range(pair_count)]), 2)
        before = rng.normal(size=(pair_count * 2, 6)).astype(np.float32)
        after = rng.normal(scale=0.2, size=(pair_count * 2, 6)).astype(np.float32)
        after[:, 0] += labels * 3.0
        before_scores, after_scores, folds = cross_validated_decisions(
            before,
            after,
            labels,
            groups,
            folds=5,
            probe_kind="logistic",
            c_value=1.0,
            seed=42,
        )
        before_metrics = metric_values(labels, before_scores)
        after_metrics = metric_values(labels, after_scores)
        self.assertTrue(np.all(folds >= 0))
        self.assertGreater(after_metrics["roc_auc"], 0.99)
        self.assertGreater(
            after_metrics["signed_margin"],
            before_metrics["signed_margin"],
        )

    def test_before_baseline_is_invariant_to_after_distribution(self):
        rng = np.random.default_rng(9)
        pair_count = 20
        labels = np.tile(np.asarray([-1, 1], dtype=np.int8), pair_count)
        groups = np.repeat(np.asarray([f"g{i:02d}" for i in range(pair_count)]), 2)
        before = rng.normal(size=(pair_count * 2, 5)).astype(np.float32)
        before[:, 0] += labels * 0.5
        after = rng.normal(size=before.shape).astype(np.float32)
        baseline_a, _, folds_a = cross_validated_decisions(
            before, after, labels, groups, 5, "logistic", 1.0, 42
        )
        baseline_b, _, folds_b = cross_validated_decisions(
            before, after * 1000.0 + 500.0, labels, groups, 5, "logistic", 1.0, 42
        )
        np.testing.assert_array_equal(folds_a, folds_b)
        np.testing.assert_allclose(baseline_a, baseline_b, rtol=0, atol=1e-10)

    def test_grouped_folds_never_split_a_reused_source(self):
        groups = np.asarray(["large"] * 8 + [f"g{i}" for i in range(8)])
        fold_ids = balanced_group_fold_ids(groups, folds=3, seed=42)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(fold_ids[groups == group])), 1)

    def test_rejects_missing_side_for_one_run(self):
        vector = np.asarray([0.0, 1.0], dtype=np.float32)
        records = {
            ("p0", "benign", 42, "g0"): {
                "h_query": vector,
                "h_analysis_boundary_true": vector,
            },
            ("p0", "harmful", 42, "g0"): {
                "h_query": vector,
                "h_analysis_boundary_true": vector,
            },
            ("p0", "benign", 43, "g0"): {
                "h_query": vector,
                "h_analysis_boundary_true": vector,
            },
        }
        with self.assertRaisesRegex(ValueError, "pair/run groups"):
            paired_matrices(records, "h_query", "h_analysis_boundary_true")

    def test_source_balanced_weights_give_each_source_equal_mass(self):
        groups = np.asarray(["large", "large", "large", "small"])
        weights = sample_weights(groups, "source_balanced")
        self.assertAlmostEqual(float(np.sum(weights[groups == "large"])), 2.0)
        self.assertAlmostEqual(float(np.sum(weights[groups == "small"])), 2.0)

    def test_source_cluster_bootstrap_refits_and_returns_intervals(self):
        rng = np.random.default_rng(11)
        group_count = 12
        labels = np.tile(np.asarray([-1, 1], dtype=np.int8), group_count)
        groups = np.repeat(np.asarray([f"g{i:02d}" for i in range(group_count)]), 2)
        before = rng.normal(size=(group_count * 2, 4)).astype(np.float32)
        after = rng.normal(scale=0.3, size=before.shape).astype(np.float32)
        after[:, 0] += labels * 2.0
        intervals = bootstrap_refit_intervals(
            before,
            after,
            labels,
            groups,
            folds=3,
            probe_kind="logistic",
            c_value=1.0,
            weighting="pair",
            samples=3,
            seed=42,
        )
        self.assertEqual(set(intervals), {"before", "after", "delta"})
        self.assertEqual(len(intervals["delta"]["roc_auc"]), 2)

    def test_bootstrap_duplicate_source_never_crosses_folds(self):
        groups = np.repeat(np.asarray([f"g{i}" for i in range(6)]), 2)
        sampled_sources = np.asarray(
            ["g0", "g0", "g1", "g2", "g3", "g4", "g5"]
        )
        indices, draw_groups, fold_ids = bootstrap_cluster_rows(
            groups,
            sampled_sources,
            folds=3,
            seed=42,
        )
        original_sources = groups[indices]

        # Duplicate draws remain distinct weighting clusters...
        self.assertNotEqual(draw_groups[0], draw_groups[2])
        # ...but identical original-source rows can never cross CV folds.
        for source in np.unique(original_sources):
            self.assertEqual(
                len(np.unique(fold_ids[original_sources == source])),
                1,
            )


class CaseStudyMetricTests(unittest.TestCase):
    def test_uses_directed_core_boundary_margin(self):
        benign = [{"harmful_score": 0.1, "refusal_score": 0.2}]
        harmful = [{"harmful_score": 0.3, "refusal_score": 0.8}]
        expected = (1.0 - 0.2) * (1.0 - 0.1) - (1.0 - 0.8) * 0.3
        self.assertAlmostEqual(behavioral_boundary_margin(benign, harmful), expected)


if __name__ == "__main__":
    unittest.main()
