"""Produce a transparent audit of dataset properties used to compile hidden checks."""

from __future__ import annotations

from pathlib import Path

from automl_eval.core.session import RuntimeSession
from automl_eval.domain.task_registry import TaskRegistry


def collect_audit(
    tasks_dir: str | Path = "automl_eval/tasks",
) -> list[dict[str, object]]:
    registry = TaskRegistry()
    registry.load_directory(tasks_dir)
    rows: list[dict[str, object]] = []
    for task in registry:
        session = RuntimeSession(task)
        session.initialize()
        insights = session.data_insights
        assert insights is not None and session.initial_profile is not None
        rows.append(
            {
                "task_id": task.task_id,
                "problem_type": task.task_type.value,
                "iid_tabular": task.metadata.get("evaluation_mode")
                == "iid_tabular_holdout"
                and not task.metadata.get("time_series_task"),
                "missing": insights.has_missing,
                "categorical": bool(insights.categorical_columns),
                "duplicates": insights.has_duplicates,
                "high_correlation": insights.has_high_correlation,
                "outliers": insights.has_outliers,
                "skew": insights.has_high_skew,
                "scale_disparity": insights.scale_range_ratio > 10,
                "class_imbalance_ratio": insights.class_imbalance_ratio,
                "identifier_like_columns": session.initial_profile.identifier_like_columns,
            }
        )
    return rows


def to_markdown(rows: list[dict[str, object]]) -> str:
    fields = [
        "task_id",
        "problem_type",
        "iid_tabular",
        "missing",
        "categorical",
        "duplicates",
        "high_correlation",
        "outliers",
        "skew",
        "scale_disparity",
        "identifier_like_columns",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(to_markdown(collect_audit()))
