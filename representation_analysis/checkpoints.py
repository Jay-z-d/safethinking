"""Build reproducible text checkpoints from saved response-generation rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IA_METHOD = "safe_llm_intention_analysis"


@dataclass(frozen=True)
class CheckpointText:
    """One model input whose final token represents a research checkpoint."""

    name: str
    text: str
    source: str


def _stored_messages(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    prompt_messages = row.get("prompt_messages")
    if isinstance(prompt_messages, dict):
        return prompt_messages
    method_trace = row.get("method_trace")
    if isinstance(method_trace, dict):
        messages = method_trace.get("intermediate_messages")
        if isinstance(messages, dict):
            return messages
    return {}


def _valid_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        result.append({"role": role, "content": content})
    return result


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool | None,
) -> str:
    kwargs: dict[str, bool] = {}
    if isinstance(enable_thinking, bool):
        kwargs["enable_thinking"] = enable_thinking
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    )


def generated_reasoning_prefix(row: dict[str, Any]) -> str | None:
    """Return the saved raw prefix that precedes the visible final answer.

    This covers labelled internal-analysis output and native ``<think>`` output.
    It is unavailable when the saved raw output contains only the final answer.
    """

    raw = row.get("raw_generation") or row.get("raw_response")
    final = row.get("final_response") or row.get("response")
    if not isinstance(raw, str) or not isinstance(final, str) or not final:
        return None
    boundary = raw.rfind(final)
    if boundary <= 0:
        return None
    prefix = raw[:boundary]
    return prefix if prefix.strip() else None


def build_checkpoint_texts(tokenizer: Any, row: dict[str, Any]) -> list[CheckpointText]:
    """Construct query, guided-prompt, and post-reasoning checkpoints.

    ``h_query`` is method-independent. ``h_guided`` captures a method wrapper
    before generated analysis. ``h_reasoned`` captures an IA stage-2 prompt or
    a reconstructable generated reasoning prefix immediately before the final
    answer. Identical checkpoints are omitted.
    """

    query = row.get("query") or row.get("prompt")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Generation row is missing a non-empty query/prompt")
    enable_thinking = row.get("enable_thinking")
    if not isinstance(enable_thinking, bool):
        enable_thinking = None

    query_text = render_messages(
        tokenizer,
        [{"role": "user", "content": query}],
        enable_thinking,
    )
    checkpoints = [CheckpointText("h_query", query_text, "original_query")]
    seen_texts = {query_text}

    messages = _stored_messages(row)
    method = str(row.get("method") or "")
    guided_key = "stage1" if method == IA_METHOD else "final"
    guided_messages = _valid_messages(messages.get(guided_key))
    guided_text: str | None = None
    if guided_messages:
        guided_text = render_messages(tokenizer, guided_messages, enable_thinking)
        if guided_text not in seen_texts:
            checkpoints.append(
                CheckpointText(
                    "h_guided",
                    guided_text,
                    f"stored_messages.{guided_key}",
                )
            )
            seen_texts.add(guided_text)

    if method == IA_METHOD:
        reasoned_messages = _valid_messages(messages.get("final"))
        if reasoned_messages:
            reasoned_text = render_messages(tokenizer, reasoned_messages, enable_thinking)
            if reasoned_text not in seen_texts:
                checkpoints.append(
                    CheckpointText(
                        "h_reasoned",
                        reasoned_text,
                        "stored_messages.final_after_stage1_analysis",
                    )
                )
                seen_texts.add(reasoned_text)
    else:
        prefix = generated_reasoning_prefix(row)
        if prefix:
            base_text = guided_text or query_text
            reasoned_text = base_text + prefix
            if reasoned_text not in seen_texts:
                checkpoints.append(
                    CheckpointText(
                        "h_reasoned",
                        reasoned_text,
                        "reconstructed_saved_generation_prefix",
                    )
                )

    return checkpoints

