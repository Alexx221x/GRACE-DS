"""Task configuration for stage-aware tabular AutoML episodes."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

_VALIDATE_WARNED_TASK_IDS: set[str] = set()
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class MetricName(str, Enum):
    ROC_AUC = "roc_auc"
    ACCURACY = "accuracy"
    F1 = "f1"
    LOG_LOSS = "log_loss"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


@dataclass
class PlanChecklistItem:
    """Legacy task author hint used internally when compiling plan expectations."""

    id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0
    required: bool = False

    def check(self, plan_text: str) -> bool:
        text_lower = plan_text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)


@dataclass(frozen=True)
class StageLimit:
    """Soft/hard local governance for one workflow stage."""

    max_steps: int
    max_seconds: float
    max_consecutive_steps: int


def default_stage_limits() -> dict[str, StageLimit]:
    return {
        "PLAN": StageLimit(max_steps=3, max_seconds=45.0, max_consecutive_steps=2),
        "EDA": StageLimit(max_steps=5, max_seconds=90.0, max_consecutive_steps=3),
        "FEATURE_ENGINEERING": StageLimit(
            max_steps=6, max_seconds=120.0, max_consecutive_steps=3
        ),
        "MODEL": StageLimit(max_steps=6, max_seconds=180.0, max_consecutive_steps=3),
        "VALIDATE": StageLimit(max_steps=5, max_seconds=60.0, max_consecutive_steps=2),
        "CODE": StageLimit(max_steps=6, max_seconds=150.0, max_consecutive_steps=3),
        "CODE_FIX": StageLimit(max_steps=5, max_seconds=120.0, max_consecutive_steps=3),
        "FINAL_SUBMIT": StageLimit(
            max_steps=1, max_seconds=60.0, max_consecutive_steps=1
        ),
    }


@dataclass
class SplitStrategy:
    """Evaluator-owned train/validation/test split policy."""

    method: str = "random"
    train_size: float = 0.7
    valid_size: float = 0.15
    test_size: float = 0.15
    stratify: bool | None = None
    group_column: str | None = None
    sort_by: list[str] = field(default_factory=list)
    split_column: str | None = None
    train_values: list[Any] = field(default_factory=lambda: ["train"])
    valid_values: list[Any] = field(
        default_factory=lambda: ["valid", "validation", "dev"]
    )
    test_values: list[Any] = field(default_factory=lambda: ["test"])
    drop_split_columns: bool = True
    rationale: str | None = None

    def validate(self) -> None:
        self.method = self.method.lower()
        allowed = {"random", "ordered", "group", "predefined"}
        if self.method not in allowed:
            raise ValueError(
                f"Unsupported split_strategy.method '{self.method}'. Expected one of {sorted(allowed)}."
            )
        if self.method == "group" and not self.group_column:
            raise ValueError(
                "split_strategy.group_column is required for method='group'."
            )
        if self.method == "predefined" and not self.split_column:
            raise ValueError(
                "split_strategy.split_column is required for method='predefined'."
            )
        if self.method != "predefined":
            sizes = (self.train_size, self.valid_size, self.test_size)
            if any(size <= 0 or size >= 1 for size in sizes):
                raise ValueError(
                    "split_strategy train/valid/test sizes must be between 0 and 1."
                )
            if abs(sum(sizes) - 1.0) > 1e-9:
                raise ValueError(
                    "split_strategy train_size + valid_size + test_size must equal 1.0."
                )

    def split_only_columns(self) -> list[str]:
        columns: list[str] = []
        if self.drop_split_columns:
            if self.group_column:
                columns.append(self.group_column)
            if self.split_column:
                columns.append(self.split_column)
        return list(dict.fromkeys(columns))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "train_size": self.train_size,
            "valid_size": self.valid_size,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "group_column": self.group_column,
            "sort_by": self.sort_by,
            "split_column": self.split_column,
            "train_values": self.train_values,
            "valid_values": self.valid_values,
            "test_values": self.test_values,
            "drop_split_columns": self.drop_split_columns,
            "rationale": self.rationale,
        }

    def is_default(self) -> bool:
        return self.to_dict() == SplitStrategy().to_dict()

    def summary(self) -> str:
        parts = [
            f"method={self.method}",
            f"train/valid/test={self.train_size:.2f}/{self.valid_size:.2f}/{self.test_size:.2f}",
        ]
        parts.append(f"stratify={'auto' if self.stratify is None else self.stratify}")
        if self.group_column:
            parts.append(f"group_column={self.group_column}")
        if self.sort_by:
            parts.append("sort_by=" + ",".join(self.sort_by))
        if self.split_column:
            parts.append(f"split_column={self.split_column}")
        if self.rationale:
            parts.append(f"rationale={self.rationale}")
        return "; ".join(parts)


@dataclass
class Task:
    """Full description of one regression or classification benchmark task."""

    task_id: str
    dataset_path: str
    target_column: str
    task_type: TaskType
    metric: MetricName
    description: str
    plan_checklist: list[PlanChecklistItem] = field(default_factory=list)
    feature_columns: list[str] | None = None
    drop_columns: list[str] = field(default_factory=list)
    time_budget_seconds: float = 300.0
    max_steps: int = 15
    oracle_score: float | None = None
    baseline_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    split_strategy: SplitStrategy = field(default_factory=SplitStrategy)
    stage_limits: dict[str, StageLimit] = field(default_factory=default_stage_limits)

    @classmethod
    def from_json(cls, path: str | Path) -> "Task":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        checklist = [
            PlanChecklistItem(**item) for item in data.pop("plan_checklist", [])
        ]
        raw_limits = data.pop("stage_limits", None)
        limits = default_stage_limits()
        if raw_limits:
            for name, values in raw_limits.items():
                limits[name.upper()] = StageLimit(**values)
        raw_split = data.pop("split_strategy", None)
        split_strategy = SplitStrategy(**raw_split) if raw_split else SplitStrategy()
        split_strategy.validate()
        data["task_type"] = TaskType(data["task_type"])
        data["metric"] = MetricName(data["metric"])
        task = cls(
            **data,
            plan_checklist=checklist,
            split_strategy=split_strategy,
            stage_limits=limits,
        )
        task.validate()
        return task

    def validate(self) -> None:
        """Non-fatal sanity checks on the task definition."""
        b, o = self.baseline_score, self.oracle_score
        if (
            b is not None
            and o is not None
            and b >= o
            and self.task_id not in _VALIDATE_WARNED_TASK_IDS
        ):
            _VALIDATE_WARNED_TASK_IDS.add(self.task_id)
            warnings.warn(
                f"Task {self.task_id!r}: baseline_score ({b}) >= oracle_score ({o}). "
                f"GRACE uses a higher-is-better convention (metric {self.metric.value!r}); "
                "for error metrics the baseline/oracle must be stored NEGATED so that "
                "oracle > baseline. As written, normalize_score would reward worse models. "
                "Fix the task JSON's baseline_score/oracle_score signs.",
                stacklevel=2,
            )

    def to_json(self, path: str | Path) -> None:
        data = {
            "task_id": self.task_id,
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "task_type": self.task_type.value,
            "metric": self.metric.value,
            "description": self.description,
            "plan_checklist": [
                {
                    "id": item.id,
                    "description": item.description,
                    "keywords": item.keywords,
                    "weight": item.weight,
                    "required": item.required,
                }
                for item in self.plan_checklist
            ],
            "feature_columns": self.feature_columns,
            "time_budget_seconds": self.time_budget_seconds,
            "max_steps": self.max_steps,
            "oracle_score": self.oracle_score,
            "baseline_score": self.baseline_score,
            "metadata": self.metadata,
            "stage_limits": {
                name: {
                    "max_steps": limit.max_steps,
                    "max_seconds": limit.max_seconds,
                    "max_consecutive_steps": limit.max_consecutive_steps,
                }
                for name, limit in self.stage_limits.items()
            },
        }
        if self.drop_columns:
            data["drop_columns"] = self.drop_columns
        if not self.split_strategy.is_default():
            data["split_strategy"] = self.split_strategy.to_dict()
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def stage_limit(self, stage_name: str) -> StageLimit:
        return self.stage_limits.get(stage_name.upper(), default_stage_limits()["CODE"])

    def observation_text(self) -> str:
        lines = [
            f"Task ID: {self.task_id}",
            f"Task type: {self.task_type.value}",
            f"Target column: {self.target_column}",
            f"Metric: {self.metric.value}",
            f"Split strategy: {self.split_strategy.summary()}",
            f"Total time budget: {self.time_budget_seconds}s",
            f"Maximum actions: {self.max_steps}",
            "",
            f"Description: {self.description}",
        ]
        if self.metadata.get("column_descriptions"):
            lines.extend(["", "Column descriptions:"])
            for col, desc in self.metadata["column_descriptions"].items():
                lines.append(f"  - {col}: {desc}")
        if self.drop_columns:
            lines.extend(
                ["", "Dropped source columns: " + ", ".join(self.drop_columns)]
            )
        return "\n".join(lines)
