"""Build reproducible text checkpoints from saved response-generation rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


IA_METHOD = "safe_llm_intention_analysis"


@dataclass(frozen=True)
class CheckpointText:
    """One model input whose final token represents a research checkpoint."""

    name: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    *,
    add_generation_prompt: bool = True,
) -> str:
    kwargs: dict[str, bool] = {}
    if isinstance(enable_thinking, bool):
        kwargs["enable_thinking"] = enable_thinking
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
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
    """Construct exact prompt, analysis-boundary, and pre-answer checkpoints.

    ``h_query`` is method-independent. ``h_guided`` captures a method wrapper
    before generated analysis. IA rows add fixed-boundary true/shuffled/empty
    analysis states and pre-answer states. Non-IA methods retain the legacy
    reconstructable ``h_reasoned`` checkpoint when one is available.
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
            if len(reasoned_messages) < 3 or [
                message["role"] for message in reasoned_messages[:3]
            ] != ["user", "assistant", "user"]:
                raise ValueError("IA final messages must begin user/assistant/user")
            controls = row.get("representation_controls")
            true_control: dict[str, Any] = {
                "analysis": reasoned_messages[1]["content"]
            }
            if isinstance(controls, dict) and isinstance(controls.get("true"), dict):
                true_control.update(controls["true"])
                true_control["analysis"] = reasoned_messages[1]["content"]
            control_map: dict[str, dict[str, Any]] = {"true": true_control}
            if isinstance(controls, dict):
                for control_name in ("shuffled", "empty"):
                    value = controls.get(control_name)
                    if not isinstance(value, dict) or not isinstance(value.get("analysis"), str):
                        raise ValueError(f"IA row has an invalid {control_name} control")
                    control_map[control_name] = value
            for control_name, control in control_map.items():
                controlled_messages = [dict(message) for message in reasoned_messages]
                controlled_messages[1]["content"] = str(control["analysis"])
                analysis_messages = controlled_messages[:2]
                analysis_text = render_messages(
                    tokenizer,
                    analysis_messages,
                    enable_thinking,
                    add_generation_prompt=False,
                )
                control_metadata = {
                    key: value
                    for key, value in control.items()
                    if key != "analysis"
                }
                checkpoints.append(
                    CheckpointText(
                        f"h_analysis_boundary_{control_name}",
                        analysis_text,
                        f"stored_messages.final.analysis_boundary.{control_name}",
                        {"control": control_name, **control_metadata},
                    )
                )
                preanswer_text = render_messages(
                    tokenizer,
                    controlled_messages,
                    enable_thinking,
                )
                checkpoints.append(
                    CheckpointText(
                        f"h_preanswer_{control_name}",
                        preanswer_text,
                        f"stored_messages.final.preanswer.{control_name}",
                        {"control": control_name, **control_metadata},
                    )
                )
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
