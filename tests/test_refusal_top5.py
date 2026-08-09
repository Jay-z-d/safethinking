from __future__ import annotations

import unittest
from unittest.mock import patch

import response_generation.score_refusal_vllm as scoring


class DummyTokenizer:
    def encode(self, text):
        return list(range(len(text)))


class DummyContext:
    tokenizer = DummyTokenizer()


class RefusalTop5Tests(unittest.TestCase):
    def test_uses_mean_of_five_highest_per_token_scores(self):
        patterns = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
        averages = {pattern: -float(index) for index, pattern in enumerate(patterns, 1)}

        def fake_score(ctx, llm, requests, prompt_logprobs):
            for request in requests:
                request["token_count"] = len(request["candidate_ids"])
                request["total_logprob"] = (
                    averages[request["pattern"]] * request["token_count"]
                )

        with (
            patch.object(
                scoring,
                "render_refusal_context",
                return_value=("CTX|", {"context_type": "test"}),
            ),
            patch.object(scoring, "score_pattern_batch", side_effect=fake_score),
        ):
            rows = scoring.score_rows(
                DummyContext(),
                None,
                [{"id": 1}],
                patterns,
                scoring_batch_size=99,
                prompt_logprobs=1,
            )

        self.assertEqual(rows[0]["refusal_logprob_sum"], -3.0)
        self.assertEqual(
            [item["pattern"] for item in rows[0]["refusal_top5_patterns"]],
            patterns[:5],
        )
        self.assertEqual(
            rows[0]["refusal_pattern_aggregation"],
            "mean_top5_template_avg_logprob",
        )


if __name__ == "__main__":
    unittest.main()

