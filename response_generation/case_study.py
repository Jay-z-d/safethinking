#!/usr/bin/env python3
"""Build a Markdown case-study report for a single WildJailbreak pair.

Given one pair_id and the final_scored.jsonl outputs of the four safety
methods, this tool collects every record for that pair (benign + harmful
sides, all seeds, all methods) and renders a Markdown comparison report:

  - pair_id and basic metadata (methods, seeds, row counts)
  - the original benign_prompt / harmful_prompt texts
  - a per-method x per-side table: final_response excerpt (first 500 chars),
    harmful_score, refusal_score, and a cot summary
  - rows sorted by harmful_score, with the method showing the largest core
    Behavioral Boundary Margin highlighted

Example:
    python -m response_generation.case_study \
      --pair-id wjb_219422_73270 \
      --inputs \
        outputs/llama31_8b_main_direct.final_scored.jsonl \
        outputs/llama31_8b_main_safe_llm_intention_analysis.final_scored.jsonl \
        outputs/llama31_8b_main_goal_prioritization.final_scored.jsonl \
        outputs/llama31_8b_main_sage.final_scored.jsonl \
      --output case_study_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDES = ("benign", "harmful")
RESPONSE_EXCERPT_CHARS = 500
COT_SUMMARY_CHARS = 300
KNOWN_METHODS = ("direct", "safe_llm_intention_analysis", "goal_prioritization", "sage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", required=True, help="WildJailbreak pair id to inspect.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="final_scored.jsonl files, one per safety method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to stdout when omitted.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def method_of(row: dict[str, Any], path: Path) -> str:
    """Resolve the safety-method name for a row.

    Prefers the row's own `method` field; falls back to a filename heuristic
    (e.g. llama31_8b_main_sage.final_scored.jsonl -> sage).
    """
    method = row.get("method")
    if method:
        return str(method)
    stem = path.stem
    for known in KNOWN_METHODS:
        if known in stem:
            return known
    return stem


def run_id_of(row: dict[str, Any]) -> str:
    return str(row.get("run_id", row.get("seed", "?")))


def side_of(row: dict[str, Any]) -> str:
    side = str(row.get("side") or row.get("gold_label") or "").strip().lower()
    return side if side in SIDES else "unknown"


def int_or_zero(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def load_rows(args: argparse.Namespace) -> list[tuple[Path, str, str, str, dict[str, Any]]]:
    """Collect every matching row as (source_file, method, side, run_id, row).

    Duplicates (same method/side/run_id across inputs) keep the first hit.
    """
    found: list[tuple[Path, str, str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        for row in read_jsonl(path):
            if str(row.get("pair_id")) != args.pair_id:
                continue
            method = method_of(row, path)
            side = side_of(row)
            run_id = run_id_of(row)
            key = (method, side, run_id)
            if key in seen:
                continue
            seen.add(key)
            found.append((path, method, side, run_id, row))
    if not found:
        raise ValueError(
            f"No records found for pair_id {args.pair_id!r} in the given inputs"
        )
    return found


def prompt_text(rows: list[dict[str, Any]], side: str) -> str:
    """Original prompt for one side, taken from the first matching record."""
    for _, _, row_side, _, row in rows:
        if row_side == side:
            for field in ("prompt", "query"):
                text = row.get(field)
                if isinstance(text, str) and text.strip():
                    return text
    return "(not found)"


def excerpt(text: str, limit: int = RESPONSE_EXCERPT_CHARS) -> str:
    """Collapse whitespace and truncate text to `limit` characters."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[truncated]"


def cot_summary(row: dict[str, Any]) -> str:
    """Summarize the chain-of-thought for display in a table cell."""
    cot = row.get("cot")
    if isinstance(cot, str) and cot.strip():
        return excerpt(cot, COT_SUMMARY_CHARS)
    traces = row.get("cot_traces") or []
    stages = [str(t.get("stage") or t.get("trace_type") or "?") for t in traces if t]
    if stages:
        return f"(no cot text; {len(stages)} trace stages: {', '.join(stages)})"
    return "(no cot)"


def md_cell(text: str, limit: int = RESPONSE_EXCERPT_CHARS) -> str:
    """Escape a truncated excerpt for a Markdown table cell."""
    value = excerpt(text, limit)
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "N/A"


def score_stats(rows: list[dict[str, Any]], field: str) -> tuple[float | None, float | None]:
    values = [float(r[field]) for r in rows if isinstance(r.get(field), (int, float))]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def stats_str(rows: list[dict[str, Any]], field: str) -> str:
    mean, stdev = score_stats(rows, field)
    if mean is None:
        return "N/A"
    return f"{mean:.3f} +/- {stdev:.3f}"


def representative(rows: list[dict[str, Any]], field: str = "harmful_score") -> dict[str, Any]:
    """Pick the seed whose score is closest to the mean (first on ties)."""
    values = [r for r in rows if isinstance(r.get(field), (int, float))]
    if not values:
        return rows[0]
    mean = statistics.fmean(float(r[field]) for r in values)
    return min(values, key=lambda r: abs(float(r[field]) - mean))


def behavioral_boundary_margin(
    benign_rows: list[dict[str, Any]],
    harmful_rows: list[dict[str, Any]],
) -> float | None:
    """Compute the same per-method margin used by compute_core_metrics.py."""

    benign_harm, _ = score_stats(benign_rows, "harmful_score")
    harmful_harm, _ = score_stats(harmful_rows, "harmful_score")
    benign_refusal, _ = score_stats(benign_rows, "refusal_score")
    harmful_refusal, _ = score_stats(harmful_rows, "refusal_score")
    if None in (benign_harm, harmful_harm, benign_refusal, harmful_refusal):
        return None
    return (1.0 - benign_refusal) * (1.0 - benign_harm) - (
        1.0 - harmful_refusal
    ) * harmful_harm


def build_report(args: argparse.Namespace, found: list[tuple[Path, str, str, str, dict[str, Any]]]) -> str:
    pair_id = args.pair_id

    methods = sorted({method for _, method, _, _, _ in found})
    seeds = sorted({run_id for _, _, _, run_id, _ in found}, key=int_or_zero)
    rows = [row for _, _, _, _, row in found]
    first = rows[0]

    # Group rows by (method, side) for aggregation.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _, method, side, _, row in found:
        groups.setdefault((method, side), []).append(row)

    # Highlight by the actual directed core Behavioral Boundary Margin. An
    # absolute harmful-score gap can incorrectly reward reversed behavior.
    boundary_margins: dict[str, float | None] = {}
    for method in methods:
        benign = groups.get((method, "benign"), [])
        harmful = groups.get((method, "harmful"), [])
        boundary_margins[method] = behavioral_boundary_margin(benign, harmful)
    max_margin = max(
        (margin for margin in boundary_margins.values() if margin is not None),
        default=None,
    )
    highlight = next(
        (m for m in methods if boundary_margins.get(m) == max_margin),
        None,
    )

    def method_label(method: str) -> str:
        label = f"`{method}`"
        if method == highlight:
            return f"**{label}** (边界最大)"
        return label

    lines: list[str] = []
    add = lines.append
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    add(f"# Case Study: pair `{pair_id}`")
    add("")
    add(f"- 生成时间: {timestamp}")
    add(f"- 方法数: {len(methods)} ({', '.join(f'`{m}`' for m in methods)})")
    add(f"- seeds (run_id): {', '.join(seeds)}")
    add(f"- 记录数: {len(rows)} (期望 {len(methods) * 2 * len(seeds)})")
    add(f"- 模型: `{first.get('model_path') or 'N/A'}`")
    add(f"- 输入文件: {', '.join(f'`{p}`' for p in dict.fromkeys(str(p) for p, _, _, _, _ in found))}")
    add("")
    for side in SIDES:
        n = sum(1 for _, _, s, _, _ in found if s == side)
        add(f"- {side}: {n} 条记录")
    add("")

    add("## 原始提示词 (Original Prompts)")
    add("")
    for side in SIDES:
        add(f"### {side} prompt")
        add("")
        add("```text")
        add(prompt_text(found, side))
        add("```")
        add("")

    add("## 方法 x 侧 对比表 (按 harmful_score 降序)")
    add("")
    add("> 分数为 3 个 seed 的均值 +/- 标准差; 响应与 cot 取自最接近均值的那条记录 (前 500 字)。")
    add("> **边界最大** 标记: 该方法的核心 Behavioral Boundary Margin 最大。")
    add("")
    add("| 方法 | 侧 | seeds | harmful_score | refusal_score | final_response (前500字) | cot 摘要 |")
    add("|------|----|-------|---------------|---------------|--------------------------|----------|")

    cells: list[tuple[float | None, str, str, list[dict[str, Any]]]] = []
    for (method, side), group_rows in groups.items():
        mean, _ = score_stats(group_rows, "harmful_score")
        cells.append((mean, method, side, group_rows))
    cells.sort(key=lambda c: (c[0] is None, -(c[0] or float("-inf"))))

    for mean, method, side, group_rows in cells:
        rep = representative(group_rows)
        add(
            "| "
            + " | ".join(
                [
                    method_label(method),
                    side,
                    str(len(group_rows)),
                    stats_str(group_rows, "harmful_score"),
                    stats_str(group_rows, "refusal_score"),
                    md_cell(str(rep.get("final_response") or rep.get("response") or "")),
                    md_cell(cot_summary(rep), COT_SUMMARY_CHARS),
                ]
            )
            + " |"
        )
    add("")

    add("## 差异分析 (benign vs harmful)")
    add("")
    add("| 方法 | benign harmful_score | harmful harmful_score | Δ harmful_score | benign refusal_score | harmful refusal_score | Δ refusal_score | Boundary Margin |")
    add("|------|---------------------|-----------------------|-----------------|----------------------|-----------------------|-----------------|-----------------|")
    for method in methods:
        benign = groups.get((method, "benign"), [])
        harmful = groups.get((method, "harmful"), [])
        b_harm, _ = score_stats(benign, "harmful_score")
        h_harm, _ = score_stats(harmful, "harmful_score")
        b_ref, _ = score_stats(benign, "refusal_score")
        h_ref, _ = score_stats(harmful, "refusal_score")

        def gap(a: float | None, b: float | None) -> str:
            if a is None or b is None:
                return "N/A"
            return f"{b - a:+.3f}"

        add(
            "| "
            + " | ".join(
                [
                    method_label(method),
                    "N/A" if b_harm is None else f"{b_harm:.3f}",
                    "N/A" if h_harm is None else f"{h_harm:.3f}",
                    gap(b_harm, h_harm),
                    "N/A" if b_ref is None else f"{b_ref:.3f}",
                    "N/A" if h_ref is None else f"{h_ref:.3f}",
                    gap(b_ref, h_ref),
                    (
                        "N/A"
                        if boundary_margins[method] is None
                        else f"{boundary_margins[method]:.3f}"
                    ),
                ]
            )
            + " |"
        )
    add("")
    if highlight is not None:
        add(
            f"> **{highlight}** 的 Behavioral Boundary Margin 最大 "
            f"({boundary_margins[highlight]:.3f})。"
        )
    else:
        add("> 未能计算 Behavioral Boundary Margin（缺少 harmful/refusal score）。")
    add("")

    add("## 逐 seed 明细")
    add("")
    add("| 方法 | 侧 | seed | harmful_score | refusal_score | 响应长度 | cot 长度 |")
    add("|------|----|------|---------------|---------------|----------|----------|")
    detail_rows = sorted(
        found,
        key=lambda item: (item[1], item[2], int_or_zero(item[3])),
    )
    for _, method, side, run_id, row in detail_rows:
        response_len = len(str(row.get("final_response") or row.get("response") or ""))
        cot = row.get("cot") or ""
        add(
            "| "
            + " | ".join(
                [
                    f"`{method}`",
                    side,
                    run_id,
                    fmt_score(row.get("harmful_score")),
                    fmt_score(row.get("refusal_score")),
                    str(response_len),
                    str(len(str(cot))),
                ]
            )
            + " |"
        )
    add("")

    add("## 完整响应")
    add("")
    for _, method, side, run_id, row in detail_rows:
        add(f"### {method} / {side} / seed {run_id}")
        add("")
        add("<details>")
        add(f"<summary>final_response ({len(str(row.get('final_response') or row.get('response') or ''))} 字)</summary>")
        add("")
        add("```text")
        add(str(row.get("final_response") or row.get("response") or ""))
        add("```")
        add("</details>")
        add("")
        cot = row.get("cot") or ""
        if str(cot).strip():
            add("<details>")
            add(f"<summary>cot ({len(str(cot))} 字)</summary>")
            add("")
            add("```text")
            add(str(cot))
            add("```")
            add("</details>")
            add("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    found = load_rows(args)
    report = build_report(args, found)
    if args.output is None:
        print(report)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    summary = {
        "pair_id": args.pair_id,
        "rows": len(found),
        "methods": sorted({m for _, m, _, _, _ in found}),
        "seeds": sorted({r for _, _, _, r, _ in found}),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
