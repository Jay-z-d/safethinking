"""Shared runtime helpers for method adapters."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def chat_template_kwargs(enable_thinking: str) -> dict[str, bool]:
    if enable_thinking == "auto":
        return {}
    return {"enable_thinking": enable_thinking == "true"}


def enable_thinking_value(enable_thinking: str) -> bool | None:
    if enable_thinking == "auto":
        return None
    return enable_thinking == "true"


def split_think_tags(text: str) -> tuple[str | None, str]:
    matches = list(THINK_RE.finditer(text))
    if not matches:
        return None, text.strip()
    think_text = "\n\n".join(
        match.group(1).strip() for match in matches if match.group(1).strip()
    )
    visible_text = THINK_RE.sub("", text).strip()
    return think_text or None, visible_text


@dataclass
class GenerationContext:
    args: argparse.Namespace
    tokenizer: Any
    llm: Any
    method_name: str
    method_description: str
    method_kind: str
    template_kwargs: dict[str, bool]

    def render(self, messages: list[dict[str, str]]) -> str:
        return self.render_messages(messages, add_generation_prompt=True)

    def render_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )

    def ensure_prompt_budget(
        self,
        prompts: list[str],
        max_new_tokens: int,
        stage_name: str,
    ) -> int:
        max_prompt_tokens = max(
            (len(self.tokenizer.encode(prompt)) for prompt in prompts),
            default=0,
        )
        if max_prompt_tokens + max_new_tokens > self.args.max_model_len:
            raise ValueError(
                f"{stage_name} prompt plus output budget exceeds --max-model-len: "
                f"{max_prompt_tokens} + {max_new_tokens} > {self.args.max_model_len}"
            )
        return max_prompt_tokens

    def preview_messages(
        self,
        stage_name: str,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int,
    ) -> dict[str, int | str]:
        prompts = [self.render(messages) for messages in messages_batch]
        max_prompt_tokens = self.ensure_prompt_budget(
            prompts,
            max_new_tokens=max_new_tokens,
            stage_name=stage_name,
        )
        return {
            "stage": stage_name,
            "queries": len(messages_batch),
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
        }

    def generate_texts(
        self,
        stage_name: str,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        if self.llm is None:
            raise RuntimeError("Cannot generate without an initialized vLLM instance")
        from vllm import SamplingParams

        prompts = [self.render(messages) for messages in messages_batch]
        self.ensure_prompt_budget(
            prompts,
            max_new_tokens=max_new_tokens,
            stage_name=stage_name,
        )
        sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text.strip() for output in outputs]

    def base_result(self, record: dict[str, Any], seed: int) -> dict[str, Any]:
        side = record.get("side") or record.get("gold_label")
        method_name = getattr(self.args, "method_name", None) or self.method_name
        return {
            "record_id": record["record_id"],
            "pair_id": record["pair_id"],
            "side": side,
            "run_id": seed,
            "source_line": record["source_line"],
            "source_field": record["source_field"],
            "prompt": record["query"],
            "query": record["query"],
            "gold_label": record.get("gold_label"),
            "seed": seed,
            "model_path": str(self.args.model),
            "enable_thinking": enable_thinking_value(self.args.enable_thinking),
            "method_name": method_name,
            "method": self.method_name,
            "method_kind": self.method_kind,
            "method_description": self.method_description,
            "generation_mode": "response_generation",
            "generation_config": {
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "stage1_max_new_tokens": self.args.stage1_max_new_tokens,
                "max_new_tokens": self.args.max_new_tokens,
            },
        }

    def append_response_fields(
        self,
        row: dict[str, Any],
        raw_response: str,
        cot_traces: list[dict[str, Any]],
        method_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response_think, response_text = split_think_tags(raw_response)
        if response_think:
            cot_traces.append(
                {
                    "stage": "final_response_think",
                    "trace_type": "model_think_tag",
                    "text": response_think,
                }
            )
        cot_parts = [
            str(trace.get("text") or trace.get("think_text") or "").strip()
            for trace in cot_traces
            if str(trace.get("text") or trace.get("think_text") or "").strip()
        ]
        row["cot_traces"] = cot_traces
        row["cot"] = "\n\n".join(cot_parts) if cot_parts else None
        row["raw_response"] = raw_response
        row["response"] = response_text
        row["raw_generation"] = raw_response
        row["final_response"] = response_text
        row["method_trace"] = method_trace or {}
        row["finish_reason"] = None
        return row
