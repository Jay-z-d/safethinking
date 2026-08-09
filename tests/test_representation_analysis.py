from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import save_file

from representation_analysis.checkpoints import build_checkpoint_texts
from representation_analysis.probe_separability import (
    cross_validated_decisions,
    load_checkpoint_vectors,
    metric_values,
    paired_matrices,
)
from response_generation.case_study import behavioral_boundary_margin

try:
    import torch

    from representation_analysis.extract_hidden_states import (
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
        assert add_generation_prompt is True
        body = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return f"{body}|GEN|thinking={kwargs.get('enable_thinking', 'auto')}"


class CheckpointTests(unittest.TestCase):
    def test_ia_has_query_guided_and_reasoned_states(self):
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
            ["h_query", "h_guided", "h_reasoned"],
        )
        self.assertIn("assistant:benign intent", checkpoints[-1].text)

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
            self.assertEqual(set(groups.tolist()), {"p0"})
            self.assertEqual(len(keys), 2)

    def test_after_state_is_more_separable_on_held_out_pairs(self):
        rng = np.random.default_rng(7)
        pair_count = 30
        labels = np.tile(np.asarray([-1, 1], dtype=np.int8), pair_count)
        groups = np.repeat(np.asarray([f"p{i:02d}" for i in range(pair_count)]), 2)
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


class CaseStudyMetricTests(unittest.TestCase):
    def test_uses_directed_core_boundary_margin(self):
        benign = [{"harmful_score": 0.1, "refusal_score": 0.2}]
        harmful = [{"harmful_score": 0.3, "refusal_score": 0.8}]
        expected = (1.0 - 0.2) * (1.0 - 0.1) - (1.0 - 0.8) * 0.3
        self.assertAlmostEqual(behavioral_boundary_margin(benign, harmful), expected)


if __name__ == "__main__":
    unittest.main()
