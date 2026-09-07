"""Report persistence for CV experiments."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev

ScalarMap = Mapping[str, float]

type JsonPayload = (
    Mapping[str, "JsonPayload | float | int | str"] | Sequence["JsonPayload | float | int | str"]
)


def write_json(path: Path, payload: JsonPayload) -> None:
    """Write a JSON payload with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def summarize_folds(folds: Sequence[ScalarMap]) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation for scalar fold metrics."""
    metric_names = sorted({name for fold in folds for name in fold})
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in folds if metric_name in fold]
        summary[metric_name] = {
            "mean": fmean(values),
            "std": pstdev(values),
        }
    return summary


def write_markdown_summary(path: Path, summary: Mapping[str, Mapping[str, float]]) -> None:
    """Write a compact human-readable metrics summary."""
    lines = ["# Evaluation Summary", "", "| Metric | Mean | Std |", "|---|---:|---:|"]
    for metric_name, stats in sorted(summary.items()):
        lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
