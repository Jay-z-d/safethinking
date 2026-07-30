"""SAGE prompt-wrapper adapter."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

from response_generation.registry import MethodSpec, register_method
from response_generation.runtime import GenerationContext


METHOD_NAME = "sage"
DESCRIPTION = (
    "SAGE inference prompt: perform semantic and task-structure safety checks "
    "before responding."
)

ANALYSIS_RE = re.compile(
    r"(?P<analysis>(?:semantic\s+analysis|task\s+structure\s+analysis|safety\s+analysis).*?)"
    r"(?=(?:final\s+response|response\s+protocol|answer)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_LABEL_RE = re.compile(
    r"(?:final\s+response|answer)\s*:\s*(?P<final>.*)\Z",
    re.IGNORECASE | re.DOTALL,
)


def _sage_repo(ctx: GenerationContext) -> Path:
    return Path(ctx.args.methods_root) / "SAGE"


def _external_make_sage_prompt(
    ctx: GenerationContext,
) -> tuple[Callable[[str], str] | None, str]:
    path = _sage_repo(ctx) / "defense_prompts.py"
    if not path.exists():
        return None, f"missing:{path}"
    spec = importlib.util.spec_from_file_location("sage_external_defense_prompts", path)
    if spec is None or spec.loader is None:
        return None, f"unloadable:{path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None, f"import_failed:{path}"
    make_prompt = getattr(module, "make_sage_prompt", None)
    if not callable(make_prompt):
        return None, f"missing_make_sage_prompt:{path}"
    return make_prompt, str(path)


def _sage_prompt(ctx: GenerationContext, query: str) -> tuple[str, str]:
    make_prompt, source = _external_make_sage_prompt(ctx)
    if make_prompt is None:
        raise RuntimeError(
            f"SAGE requires the official repository at {_sage_repo(ctx)}; "
            f"adapter status: {source}"
        )
    try:
        return str(make_prompt(query)), source
    except Exception as exc:
        raise RuntimeError(
            f"Official SAGE make_sage_prompt() failed: {source}"
        ) from exc


def _messages(
    ctx: GenerationContext,
    record_or_row: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    query = str(record_or_row.get("query") or record_or_row.get("prompt") or "")
    prompt, source = _sage_prompt(ctx, query)
    return [{"role": "user", "content": prompt}], source


def _parse_optional_analysis(raw_response: str) -> tuple[str | None, str]:
    analysis_match = ANALYSIS_RE.search(raw_response)
    final_match = FINAL_LABEL_RE.search(raw_response)
    analysis = analysis_match.group("analysis").strip() if analysis_match else None
    final = final_match.group("final").strip() if final_match else raw_response.strip()
    return analysis, final


def preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    messages_batch = [_messages(ctx, record)[0] for record in records]
    preview_info = ctx.preview_messages(
        "sage.final",
        messages_batch,
        max_new_tokens=ctx.args.max_new_tokens,
    )
    return [preview_info]


def generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    message_source_pairs = [_messages(ctx, record) for record in records]
    messages_batch = [item[0] for item in message_source_pairs]
    sources = [item[1] for item in message_source_pairs]
    raw_responses = ctx.generate_texts(
        "sage.final",
        messages_batch,
        max_new_tokens=ctx.args.max_new_tokens,
        seed=seed,
    )

    rows = []
    for record, messages, source, raw_response in zip(
        records,
        messages_batch,
        sources,
        raw_responses,
        strict=True,
    ):
        analysis, final_response = _parse_optional_analysis(raw_response)
        traces = []
        if analysis:
            traces.append(
                {
                    "stage": "sage_safety_analysis",
                    "trace_type": "method_induced_analysis",
                    "method_prompt_source": source,
                    "text": analysis,
                }
            )
        row = ctx.base_result(record, seed)
        row["method_source_path"] = source
        method_trace = {
            "analysis": analysis,
            "wrapper_prompt_source": source,
            "intermediate_messages": {"final": messages},
        }
        if ctx.args.store_prompts:
            row["prompt_messages"] = {"final": messages}
        row = ctx.append_response_fields(
            row,
            final_response,
            traces,
            method_trace=method_trace,
        )
        row["raw_response"] = raw_response
        row["raw_generation"] = raw_response
        rows.append(row)
    return rows


def refusal_context(ctx: GenerationContext, row: dict[str, Any]) -> dict[str, Any]:
    messages, source = _messages(ctx, row)
    return {
        "messages": messages,
        "context_type": "sage_prompt",
        "add_generation_prompt": True,
        "method_prompt_source": source,
    }


register_method(
    MethodSpec(
        name=METHOD_NAME,
        kind="single_stage_prompt_wrapper",
        description=DESCRIPTION,
        preview=preview,
        generate_batch=generate_batch,
        refusal_context=refusal_context,
    )
)
