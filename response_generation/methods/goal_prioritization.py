"""Goal Prioritization prompt-wrapper adapters."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

from response_generation.registry import MethodSpec, register_method
from response_generation.runtime import GenerationContext


PRIORITY_METHOD_NAME = "goal_prioritization"
PRIORITY_LLAMA_METHOD_NAME = "goal_prioritization_llama"
DESCRIPTION = (
    "Inference-only Goal Prioritization: wrap the user query with a prompt "
    "that prioritizes safety goals over helpfulness goals."
)
LLAMA_DESCRIPTION = (
    "Inference-only Goal Prioritization using the revised priority_llama "
    "wrapper from the official implementation."
)

INTERNAL_RE = re.compile(
    r"\[internal\s+thoughts?\](.*?)(?=\[final\s+response\]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_RE = re.compile(r"\[final\s+response\](.*)\Z", re.IGNORECASE | re.DOTALL)


def _goal_priority_repo(ctx: GenerationContext) -> Path:
    return Path(ctx.args.methods_root) / "JailbreakDefense_GoalPriority"


def _external_add_defense_from_ast(
    path: Path,
) -> tuple[Callable[[str, str], str] | None, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return None, f"import_failed:{path}"
    add_defense_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "add_defense":
            add_defense_node = node
            break
    if add_defense_node is None:
        return None, f"missing_add_defense:{path}"
    namespace: dict[str, Any] = {}
    try:
        module = ast.Module(body=[add_defense_node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, filename=str(path), mode="exec"), namespace)
    except Exception:
        return None, f"ast_extract_failed:{path}"
    add_defense = namespace.get("add_defense")
    if not callable(add_defense):
        return None, f"missing_add_defense:{path}"
    return add_defense, f"{path}:add_defense_ast"


def _external_add_defense(
    ctx: GenerationContext,
) -> tuple[Callable[[str, str], str] | None, str]:
    path = _goal_priority_repo(ctx) / "utils" / "utils.py"
    if not path.exists():
        return None, f"missing:{path}"
    spec = importlib.util.spec_from_file_location("goal_priority_external_utils", path)
    if spec is None or spec.loader is None:
        return None, f"unloadable:{path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return _external_add_defense_from_ast(path)
    add_defense = getattr(module, "add_defense", None)
    if not callable(add_defense):
        return None, f"missing_add_defense:{path}"
    return add_defense, str(path)


def _wrap_prompt(ctx: GenerationContext, query: str, defense_type: str) -> tuple[str, str]:
    add_defense, source = _external_add_defense(ctx)
    if add_defense is None:
        raise RuntimeError(
            "Goal Prioritization requires the official repository at "
            f"{_goal_priority_repo(ctx)}; adapter status: {source}"
        )
    try:
        return str(add_defense(query, defense_type=defense_type)), source
    except Exception as exc:
        raise RuntimeError(
            f"Official Goal Prioritization add_defense() failed: {source}"
        ) from exc


def _messages(
    ctx: GenerationContext,
    record_or_row: dict[str, Any],
    defense_type: str,
) -> tuple[list[dict[str, str]], str]:
    query = str(record_or_row.get("query") or record_or_row.get("prompt") or "")
    prompt, source = _wrap_prompt(ctx, query, defense_type)
    return [{"role": "user", "content": prompt}], source


def _parse_output(raw_response: str) -> tuple[str | None, str]:
    internal_match = INTERNAL_RE.search(raw_response)
    final_match = FINAL_RE.search(raw_response)
    internal = internal_match.group(1).strip() if internal_match else None
    final = final_match.group(1).strip() if final_match else raw_response.strip()
    return internal, final


def _preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    defense_type: str,
    stage_name: str,
) -> list[dict[str, int | str]]:
    messages_batch = [_messages(ctx, record, defense_type)[0] for record in records]
    preview = ctx.preview_messages(
        stage_name,
        messages_batch,
        max_new_tokens=ctx.args.max_new_tokens,
    )
    preview["defense_type"] = defense_type
    return [preview]


def _generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
    defense_type: str,
) -> list[dict[str, Any]]:
    message_source_pairs = [_messages(ctx, record, defense_type) for record in records]
    messages_batch = [item[0] for item in message_source_pairs]
    sources = [item[1] for item in message_source_pairs]
    raw_responses = ctx.generate_texts(
        f"goal_prioritization.{defense_type}.final",
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
        internal, final_response = _parse_output(raw_response)
        traces = []
        if internal:
            traces.append(
                {
                    "stage": "goal_prioritization_internal_thoughts",
                    "trace_type": "method_induced_analysis",
                    "method_prompt_source": source,
                    "text": internal,
                }
            )
        row = ctx.base_result(record, seed)
        row["method_source_path"] = source
        method_trace = {
            "analysis": internal,
            "defense_type": defense_type,
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


def _refusal_context(
    ctx: GenerationContext,
    row: dict[str, Any],
    defense_type: str,
) -> dict[str, Any]:
    messages, source = _messages(ctx, row, defense_type)
    return {
        "messages": messages,
        "context_type": f"goal_prioritization_{defense_type}_prompt",
        "add_generation_prompt": True,
        "method_prompt_source": source,
        "defense_type": defense_type,
    }


def priority_preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    return _preview(ctx, records, "priority", "goal_prioritization.final")


def priority_generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    return _generate_batch(ctx, records, seed, "priority")


def priority_refusal_context(
    ctx: GenerationContext,
    row: dict[str, Any],
) -> dict[str, Any]:
    return _refusal_context(ctx, row, "priority")


def priority_llama_preview(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    return _preview(
        ctx,
        records,
        "priority_llama",
        "goal_prioritization_llama.final",
    )


def priority_llama_generate_batch(
    ctx: GenerationContext,
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    return _generate_batch(ctx, records, seed, "priority_llama")


def priority_llama_refusal_context(
    ctx: GenerationContext,
    row: dict[str, Any],
) -> dict[str, Any]:
    return _refusal_context(ctx, row, "priority_llama")


register_method(
    MethodSpec(
        name=PRIORITY_METHOD_NAME,
        kind="single_stage_prompt_wrapper",
        description=DESCRIPTION,
        preview=priority_preview,
        generate_batch=priority_generate_batch,
        refusal_context=priority_refusal_context,
    )
)

register_method(
    MethodSpec(
        name=PRIORITY_LLAMA_METHOD_NAME,
        kind="single_stage_prompt_wrapper",
        description=LLAMA_DESCRIPTION,
        preview=priority_llama_preview,
        generate_batch=priority_llama_generate_batch,
        refusal_context=priority_llama_refusal_context,
    )
)
