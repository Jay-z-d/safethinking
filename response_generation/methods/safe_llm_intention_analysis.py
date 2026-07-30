"""SafeLLM with Intention Analysis adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from response_generation.registry import MethodSpec, register_method
from response_generation.runtime import GenerationContext, split_think_tags


METHOD_NAME = "safe_llm_intention_analysis"
DESCRIPTION = (
    "SafeLLM Intention Analysis: first elicit essential intention, then answer "
    "the original query under normal safety restrictions."
)


def _method_dir(ctx: GenerationContext) -> Path:
    return Path(ctx.args.methods_root) / "SafeLLM_with_IntentionAnalysis"


def _demo_path(ctx: GenerationContext) -> Path:
    return _method_dir(ctx) / "demo" / "IA_demo.py"


def _string_constant(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _load_prompts(ctx: GenerationContext) -> tuple[str, str, str]:
    path = _demo_path(ctx)
    if not path.exists():
        raise RuntimeError(
            "Intention Analysis requires the official repository at "
            f"{_method_dir(ctx)}; missing prompt source: {path}"
        )
    ia_prompt = _string_constant(path, "IA_PROMPT")
    ct_prompt = _string_constant(path, "CT_PROMPT")
    if not ia_prompt or not ct_prompt:
        raise RuntimeError(
            f"Could not load IA_PROMPT and CT_PROMPT from official source: {path}"
        )
    return ia_prompt, ct_prompt, str(path)


def _stage1_user_prompt(ctx: GenerationContext, query: str) -> str:
    ia_prompt, _, _ = _load_prompts(ctx)
    return f"{ia_prompt}'''\n{query}\n'''"


def _stage1_messages(ctx: GenerationContext, record: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": _stage1_user_prompt(ctx, record["query"])}]


def _stage2_messages(
    stage1_user_prompt: str,
    stage1_response: str,
    ct_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": stage1_user_prompt},
        {"role": "assistant", "content": stage1_response},
        {"role": "user", "content": ct_prompt},
    ]


def preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    _, ct_prompt, source = _load_prompts(ctx)
    stage1_messages = [_stage1_messages(ctx, record) for record in records]
    stage1_preview = ctx.preview_messages(
        "safe_llm_intention_analysis.stage1",
        stage1_messages,
        max_new_tokens=ctx.args.stage1_max_new_tokens,
    )
    stage1_preview["method_prompt_source"] = source

    placeholder = "The essential intention of the query is to be determined."
    stage2_preview = ctx.preview_messages(
        "safe_llm_intention_analysis.stage2_placeholder",
        [
            _stage2_messages(messages[0]["content"], placeholder, ct_prompt)
            for messages in stage1_messages
        ],
        max_new_tokens=ctx.args.max_new_tokens,
    )
    stage2_preview["method_prompt_source"] = source
    return [stage1_preview, stage2_preview]


def generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    _, ct_prompt, source = _load_prompts(ctx)
    first_messages = [_stage1_messages(ctx, record) for record in records]
    stage1_raw_outputs = ctx.generate_texts(
        "safe_llm_intention_analysis.stage1",
        first_messages,
        max_new_tokens=ctx.args.stage1_max_new_tokens,
        seed=seed,
    )

    traces = []
    stage1_visible_outputs = []
    for raw_output in stage1_raw_outputs:
        think_text, visible_text = split_think_tags(raw_output)
        stage1_visible_outputs.append(visible_text)
        trace = {
            "stage": "intention_analysis",
            "trace_type": "method_induced_analysis",
            "method_prompt_source": source,
            "raw_output": raw_output,
            "text": visible_text,
        }
        if think_text:
            trace["think_text"] = think_text
        traces.append(trace)

    second_messages = [
        _stage2_messages(
            first_message[0]["content"],
            stage1_visible if stage1_visible else stage1_raw,
            ct_prompt,
        )
        for first_message, stage1_visible, stage1_raw in zip(
            first_messages,
            stage1_visible_outputs,
            stage1_raw_outputs,
            strict=True,
        )
    ]
    raw_responses = ctx.generate_texts(
        "safe_llm_intention_analysis.stage2",
        second_messages,
        max_new_tokens=ctx.args.max_new_tokens,
        seed=seed,
    )

    rows = []
    for record, first_message, second_message, trace, raw_response in zip(
        records,
        first_messages,
        second_messages,
        traces,
        raw_responses,
        strict=True,
    ):
        row = ctx.base_result(record, seed)
        row["method_source_path"] = source
        method_trace = {
            "analysis": trace.get("text"),
            "intermediate_messages": {
                "stage1": first_message,
                "final": second_message,
            },
            "selected_agent": None,
            "extra_info": {
                "method_prompt_source": source,
                "stage1_raw_output": trace.get("raw_output"),
            },
        }
        if ctx.args.store_prompts:
            row["prompt_messages"] = {
                "stage1": first_message,
                "final": second_message,
            }
        rows.append(
            ctx.append_response_fields(
                row,
                raw_response,
                [trace],
                method_trace=method_trace,
            )
        )
    return rows


def refusal_context(ctx: GenerationContext, row: dict[str, Any]) -> dict[str, Any]:
    _, ct_prompt, source = _load_prompts(ctx)
    trace = row.get("method_trace") or {}
    analysis = trace.get("analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        traces = row.get("cot_traces") or []
        if traces:
            analysis = str(traces[0].get("text") or traces[0].get("raw_output") or "")
    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError(
            "IA refusal scoring requires the stage-1 intention analysis in "
            "method_trace.analysis or cot_traces"
        )

    stage1_user_prompt = _stage1_user_prompt(ctx, row["query"])
    return {
        "messages": _stage2_messages(stage1_user_prompt, analysis, ct_prompt),
        "context_type": "method_trace_context",
        "add_generation_prompt": True,
        "method_prompt_source": source,
    }


register_method(
    MethodSpec(
        name=METHOD_NAME,
        kind="two_stage",
        description=DESCRIPTION,
        preview=preview,
        generate_batch=generate_batch,
        refusal_context=refusal_context,
    )
)
