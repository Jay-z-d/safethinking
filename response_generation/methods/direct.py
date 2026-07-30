"""Direct response generation with the base model."""

from __future__ import annotations

from typing import Any

from response_generation.registry import MethodSpec, register_method
from response_generation.runtime import GenerationContext


METHOD_NAME = "direct"
DESCRIPTION = "Base model direct response to the original user query."


def _messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": record["query"]}]


def preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    return [
        ctx.preview_messages(
            "direct.final",
            [_messages(record) for record in records],
            max_new_tokens=ctx.args.max_new_tokens,
        )
    ]


def generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    messages_batch = [_messages(record) for record in records]
    raw_responses = ctx.generate_texts(
        "direct.final",
        messages_batch,
        max_new_tokens=ctx.args.max_new_tokens,
        seed=seed,
    )

    rows = []
    for record, messages, raw_response in zip(records, messages_batch, raw_responses, strict=True):
        row = ctx.base_result(record, seed)
        if ctx.args.store_prompts:
            row["prompt_messages"] = {"final": messages}
        rows.append(ctx.append_response_fields(row, raw_response, []))
    return rows


def refusal_context(ctx: GenerationContext, row: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "user", "content": row["query"]}]
    return {
        "messages": messages,
        "context_type": "prompt_only",
        "add_generation_prompt": True,
    }


register_method(
    MethodSpec(
        name=METHOD_NAME,
        kind="single_stage",
        description=DESCRIPTION,
        preview=preview,
        generate_batch=generate_batch,
        refusal_context=refusal_context,
    )
)
