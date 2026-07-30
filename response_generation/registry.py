"""Method adapter registry for response-generation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from response_generation.runtime import GenerationContext


PreviewFn = Callable[
    [GenerationContext, list[dict[str, Any]]],
    list[dict[str, int | str]],
]
GenerateBatchFn = Callable[
    [GenerationContext, list[dict[str, Any]], int],
    list[dict[str, Any]],
]
RefusalContextFn = Callable[
    [GenerationContext, dict[str, Any]],
    dict[str, Any],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    description: str
    preview: PreviewFn
    generate_batch: GenerateBatchFn
    refusal_context: RefusalContextFn


_REGISTRY: dict[str, MethodSpec] = {}


def register_method(spec: MethodSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Duplicate method registration: {spec.name}")
    _REGISTRY[spec.name] = spec


def get_method(name: str) -> MethodSpec:
    import_builtin_methods()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown method {name!r}; available methods: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def method_names() -> list[str]:
    import_builtin_methods()
    return sorted(_REGISTRY)


def methods() -> dict[str, MethodSpec]:
    import_builtin_methods()
    return dict(sorted(_REGISTRY.items()))


_BUILTINS_IMPORTED = False


def import_builtin_methods() -> None:
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    _BUILTINS_IMPORTED = True
    from response_generation.methods import direct  # noqa: F401
    from response_generation.methods import goal_prioritization  # noqa: F401
    from response_generation.methods import sage  # noqa: F401
    from response_generation.methods import safe_llm_intention_analysis  # noqa: F401
