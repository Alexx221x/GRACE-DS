"""Mutable episode state with evaluator-private labels and protected raw snapshots."""

from __future__ import annotations

import copy
import hashlib
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from automl_eval.evaluation.data_insights import DataInsights, analyze_dataset
from automl_eval.domain.task import SplitStrategy, Task, TaskType

if TYPE_CHECKING:
    from automl_eval.evaluation.reward import RewardBreakdown


class ActionType(str, Enum):
    PLAN = "PLAN"
    EDA = "EDA"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL = "MODEL"
    VALIDATE = "VALIDATE"
    CODE = "CODE"
    CODE_FIX = "CODE_FIX"
    FINAL_SUBMIT = "FINAL_SUBMIT"


@dataclass
class StepRecord:
    step_idx: int
    action_type: ActionType
    action_text: str
    state_before: str
    state_after: str
    reward: float
    execution_success: bool
    error_message: str | None = None
    metric_value: float | None = None
    code_body: str = ""
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    reward_breakdown: "RewardBreakdown | None" = None


@dataclass
class DatasetStateProfile:
    missing_cells: int
    duplicate_rows: int
    categorical_columns: int
    high_cardinality_columns: int
    identifier_like_columns: int
    class_imbalance_ratio: float | None


@dataclass
class StageUsage:
    steps: int = 0
    elapsed_seconds: float = 0.0
    violations: int = 0


class RuntimeSession:
    """State for one evaluator-owned split and an agent-visible workspace."""

    def __init__(self, task: Task, seed: int = 42, subsample_factor: int = 1) -> None:
        self.task = task
        self.seed = seed
        self.subsample_factor = int(subsample_factor) if subsample_factor else 1
        self.train_df: pd.DataFrame | None = None  # internal train + target
        self.valid_df: pd.DataFrame | None = None  # internal validation + target
        self.visible_valid_df: pd.DataFrame | None = (
            None  # sandbox-facing features only
        )
        self.private_dev_df: pd.DataFrame | None = None
        self.hidden_test_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None  # internal compatibility alias
        self._protected_train_snapshot: pd.DataFrame | None = None
        self._protected_valid_snapshot: pd.DataFrame | None = None
        self._train_original_hash: str | None = None
        self._valid_original_hash: str | None = None
        self.steps: list[StepRecord] = []
        self.current_step = 0
        self.plan_text: str | None = None
        self.applied_transforms: list[dict[str, Any]] = []
        self.trained_models: list[dict[str, Any]] = []
        self.current_metric: float | None = None
        self.best_metric: float | None = None
        self.current_validated_candidate_version: int | None = None
        self.best_candidate_version: int | None = None
        self.best_metric_step: int | None = None
        self.hidden_test_metric: float | None = None
        self.predictions: np.ndarray | None = None
        self.final_submitted = False
        self.finalization_reason: str | None = None
        self.final_submit_blocked = False
        self.final_submit_block_reason: str | None = None
        self.current_submission_kind: str | None = None
        self.current_submission_replayable = False
        self.candidate_raw_input_compatible: bool | None = None
        self.candidate_probe_kind: str | None = None
        self.candidate_probe_error: str | None = None
        self.candidate_probe_step: int | None = None
        self.validation_error: str | None = None
        self.validation_requests = 0
        self.test_evaluation_count = 0
        self.candidate_version = 0
        self.sandbox_namespace: dict[str, Any] = {}
        self.repair_attempts = 0
        self.repair_execution_successes = 0
        self.repair_successes = 0  # evaluator-verified fixes, not merely executing code
        self.pending_repair_verified = False
        self.consecutive_failures = 0
        self.data_insights: DataInsights | None = None
        self.metric_history: list[tuple[int, float]] = []
        self.cycle_count = 0
        self._last_phase = "init"
        self.done = False
        self.wall_start_time = 0.0
        self.execution_time_seconds = 0.0
        self.initial_profile: DatasetStateProfile | None = None
        self.current_profile: DatasetStateProfile | None = None
        self.previous_profile: DatasetStateProfile | None = None
        self.stage_usage: dict[str, StageUsage] = defaultdict(StageUsage)
        self.last_stage: str | None = None
        self.consecutive_stage_steps = 0
        self.stage_budget_message: str | None = None
        self.hidden_checklist: Any = None
        self.split_indices: dict[str, list[int]] = {}
        self.split_manifest: dict[str, Any] = {}
        self._full_dataset_row_count: int | None = None
        self._subsample_applied: bool = False

    def initialize(self) -> None:
        df = pd.read_csv(self.task.dataset_path)
        self._full_dataset_row_count = int(len(df))
        df = self._subsample_frame(df)
        feature_cols = self._feature_columns(df)
        train_idx, valid_idx, test_idx = self._split_indices(df)
        self.split_indices = {
            "train": [int(index) for index in train_idx],
            "validation": [int(index) for index in valid_idx],
            "hidden_test": [int(index) for index in test_idx],
        }
        self.split_manifest = self._build_split_manifest(
            df, train_idx, valid_idx, test_idx
        )
        self.train_df = self._frame_for_indices(df, train_idx, feature_cols)
        self.valid_df = self._frame_for_indices(df, valid_idx, feature_cols)
        self.visible_valid_df = self.valid_df.drop(
            columns=[self.task.target_column]
        ).copy(deep=True)
        self.private_dev_df = self.valid_df.copy(deep=True)
        self.hidden_test_df = self._frame_for_indices(df, test_idx, feature_cols)
        self.test_df = self.hidden_test_df

        self._protected_train_snapshot = self.train_df.copy(deep=True)
        self._protected_valid_snapshot = self.visible_valid_df.copy(deep=True)
        self._train_original_hash = self._hash_df(self._protected_train_snapshot)
        self._valid_original_hash = self._hash_df(self._protected_valid_snapshot)
        self.sandbox_namespace = {
            "train_df": self._protected_train_snapshot.copy(deep=True),
            "valid_df": self._protected_valid_snapshot.copy(deep=True),
            "train_df_original": self._protected_train_snapshot.copy(deep=True),
            "valid_df_original": self._protected_valid_snapshot.copy(deep=True),
            "target_column": self.task.target_column,
            "pd": pd,
            "np": np,
        }
        self.data_insights = analyze_dataset(self.train_df, self.task.target_column)
        self.initial_profile = self._build_profile(self.train_df)
        self.current_profile = self.initial_profile
        self.wall_start_time = time.time()

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        if self.task.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.task.target_column}' is not present in {self.task.dataset_path}."
            )
        missing_drop_columns = [
            column for column in self.task.drop_columns if column not in df.columns
        ]
        if missing_drop_columns:
            raise ValueError(
                f"Configured drop_columns are missing from dataset: {missing_drop_columns}"
            )
        configured = self.task.feature_columns or [
            c for c in df.columns if c != self.task.target_column
        ]
        drop_columns = set(self.task.drop_columns) | set(
            self.task.split_strategy.split_only_columns()
        )
        return [
            column
            for column in configured
            if column != self.task.target_column and column not in drop_columns
        ]

    def _split_indices(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split = self.task.split_strategy
        split.validate()
        if split.method == "random":
            return self._random_split_indices(df, split)
        if split.method == "ordered":
            return self._ordered_split_indices(df, split)
        if split.method == "group":
            return self._group_split_indices(df, split)
        if split.method == "predefined":
            return self._predefined_split_indices(df, split)
        raise ValueError(f"Unsupported split strategy: {split.method}")

    def _subsample_seed(self) -> int:
        """Stable 32-bit seed derived from BOTH the split seed and the factor."""
        return (int(self.seed) * 1_000_003 + int(self.subsample_factor) * 9_176_117) % (
            2**32
        )

    def _min_rows_for_safe_split(self, df: pd.DataFrame) -> int:
        """Lower bound on kept rows so the configured split cannot degenerate."""
        floor = 3
        if self.task.task_type != TaskType.REGRESSION:
            try:
                n_classes = int(df[self.task.target_column].nunique(dropna=False))
            except Exception:
                n_classes = 2
            floor = max(floor, 4 * max(n_classes, 2))
        return max(10, floor)

    def _subsample_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce the dataset to ~1/factor of its rows, deterministically."""
        factor = self.subsample_factor
        if factor is None or factor <= 1:
            return df
        n_total = len(df)
        floor = self._min_rows_for_safe_split(df)
        if n_total <= floor:
            warnings.warn(
                f"Task {self.task.task_id!r}: dataset has {n_total} rows; "
                f"subsample_factor={factor} skipped (already at/below the safe-split "
                f"floor of {floor}).",
                stacklevel=2,
            )
            return df
        target_n = max(floor, int(round(n_total / factor)))
        if target_n >= n_total:
            return df
        rng = np.random.RandomState(self._subsample_seed())
        split = self.task.split_strategy
        method = split.method
        if method == "ordered":
            keep = self._subsample_ordered(df, split, target_n)
        elif method == "group":
            keep = self._subsample_groups(df, split, target_n, rng)
        elif method == "predefined":
            keep = self._subsample_stratified(df, df[split.split_column], target_n, rng)
        else:  # random / stratified
            strat = None
            if (
                self._stratify_values(
                    df[self.task.target_column], np.arange(n_total), split
                )
                is not None
            ):
                strat = df[self.task.target_column]
            keep = self._subsample_stratified(df, strat, target_n, rng)
        keep = np.unique(np.asarray(keep, dtype=int))
        reduced = df.iloc[np.sort(keep)].reset_index(drop=True)
        self._subsample_applied = True
        return reduced

    @staticmethod
    def _subsample_ordered(
        df: pd.DataFrame, split: SplitStrategy, target_n: int
    ) -> np.ndarray:
        if split.sort_by and all(c in df.columns for c in split.sort_by):
            order = df.sort_values(split.sort_by, kind="mergesort").index.to_numpy()
        else:
            order = df.index.to_numpy()
        pos = np.unique(np.linspace(0, len(order) - 1, target_n).round().astype(int))
        return order[pos]

    @staticmethod
    def _subsample_groups(
        df: pd.DataFrame,
        split: SplitStrategy,
        target_n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        groups = (
            df[split.group_column]
            .astype(object)
            .where(df[split.group_column].notna(), "__NA__")
        )
        unique_groups = np.sort(pd.unique(groups).astype(str))
        frac = target_n / len(df)
        n_keep_groups = max(3, int(round(len(unique_groups) * frac)))
        n_keep_groups = min(n_keep_groups, len(unique_groups))
        chosen = set(rng.permutation(unique_groups)[:n_keep_groups].tolist())
        mask = groups.astype(str).isin(chosen).to_numpy()
        return np.flatnonzero(mask)

    @staticmethod
    def _subsample_stratified(
        df: pd.DataFrame,
        strat: pd.Series | None,
        target_n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n = len(df)
        positions = np.arange(n)
        if strat is None:
            return rng.choice(positions, size=min(target_n, n), replace=False)
        frac = target_n / n
        keys = strat.astype(object).where(strat.notna(), "__NA__").to_numpy()
        kept: list[np.ndarray] = []
        for value in sorted({str(k) for k in keys}):
            grp = positions[np.asarray([str(k) == value for k in keys])]
            if grp.size == 0:
                continue
            k = max(1, int(round(grp.size * frac)))
            kept.append(rng.choice(grp, size=min(k, grp.size), replace=False))
        return np.concatenate(kept) if kept else positions[: min(target_n, n)]

    def _frame_for_indices(
        self, df: pd.DataFrame, indices: np.ndarray, feature_cols: list[str]
    ) -> pd.DataFrame:
        columns = feature_cols + [self.task.target_column]
        return df.iloc[indices][columns].reset_index(drop=True)

    @staticmethod
    def _hash_indices(indices: np.ndarray) -> str:
        serialised = ",".join(
            str(int(index)) for index in np.asarray(indices).tolist()
        ).encode("utf-8")
        return hashlib.sha256(serialised).hexdigest()

    def _build_split_manifest(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> dict[str, Any]:
        """Return label-free evaluator metadata proving deterministic split reuse."""
        hashes = {
            "train_indices_sha256": self._hash_indices(train_idx),
            "validation_indices_sha256": self._hash_indices(valid_idx),
            "hidden_test_indices_sha256": self._hash_indices(test_idx),
        }
        split_id_source = "|".join(
            [
                self.task.task_id,
                str(self.seed),
                f"sub{self.subsample_factor}",
                *hashes.values(),
            ]
        ).encode("utf-8")
        configured_split = self.task.split_strategy.to_dict()
        applied_stratification = bool(
            self.task.split_strategy.method == "random"
            and self._stratify_values(
                df[self.task.target_column],
                np.arange(len(df)),
                self.task.split_strategy,
            )
            is not None
        )
        return {
            "task_id": self.task.task_id,
            "split_seed": self.seed,
            "split_strategy": configured_split,
            "split_strategy_resolved": {
                **configured_split,
                "stratification_applied": applied_stratification,
                "stratification_values_exported": False,
            },
            "dataset_row_count": int(len(df)),
            "full_dataset_row_count": int(
                self._full_dataset_row_count
                if self._full_dataset_row_count is not None
                else len(df)
            ),
            "subsample_factor": int(self.subsample_factor),
            "subsample_applied": bool(self._subsample_applied),
            "train_row_count": int(len(train_idx)),
            "validation_row_count": int(len(valid_idx)),
            "hidden_test_row_count": int(len(test_idx)),
            **hashes,
            "split_id": hashlib.sha256(split_id_source).hexdigest(),
            "contains_hidden_test_labels": False,
        }

    def evaluator_split_manifest(
        self, *, include_indices: bool = False
    ) -> dict[str, Any]:
        """Return experiment-runner metadata without exposing target values."""
        manifest = copy.deepcopy(self.split_manifest)
        if include_indices:
            manifest["indices"] = copy.deepcopy(self.split_indices)
        return manifest

    def _random_split_indices(
        self, df: pd.DataFrame, split: SplitStrategy
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(df))
        y = df[self.task.target_column]
        temp_size = split.valid_size + split.test_size
        stratify = self._stratify_values(y, indices, split)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=temp_size,
            random_state=self.seed,
            stratify=stratify,
        )
        temp_stratify = self._stratify_values(y, temp_idx, split)
        relative_test_size = split.test_size / temp_size
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_size,
            random_state=self.seed,
            stratify=temp_stratify,
        )
        return np.asarray(train_idx), np.asarray(valid_idx), np.asarray(test_idx)

    def _ordered_split_indices(
        self, df: pd.DataFrame, split: SplitStrategy
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split.sort_by:
            missing = [column for column in split.sort_by if column not in df.columns]
            if missing:
                raise ValueError(
                    f"split_strategy.sort_by columns are missing from dataset: {missing}"
                )
            ordered = df.sort_values(split.sort_by, kind="mergesort").index.to_numpy()
        else:
            ordered = df.index.to_numpy()
        n_train, n_valid, _ = self._split_counts(len(ordered), split)
        train_idx = ordered[:n_train]
        valid_idx = ordered[n_train : n_train + n_valid]
        test_idx = ordered[n_train + n_valid :]
        return train_idx, valid_idx, test_idx

    def _group_split_indices(
        self, df: pd.DataFrame, split: SplitStrategy
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert split.group_column is not None
        if split.group_column not in df.columns:
            raise ValueError(
                f"split_strategy.group_column '{split.group_column}' is not present in dataset."
            )
        groups = df[split.group_column]
        if groups.nunique(dropna=False) < 3:
            raise ValueError("Group split requires at least three distinct groups.")
        temp_size = split.valid_size + split.test_size
        first = GroupShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=self.seed
        )
        train_idx, temp_idx = next(first.split(df, groups=groups))
        relative_test_size = split.test_size / temp_size
        second = GroupShuffleSplit(
            n_splits=1, test_size=relative_test_size, random_state=self.seed
        )
        temp_df = df.iloc[temp_idx]
        temp_groups = groups.iloc[temp_idx]
        valid_pos, test_pos = next(second.split(temp_df, groups=temp_groups))
        return (
            np.asarray(train_idx),
            np.asarray(temp_idx[valid_pos]),
            np.asarray(temp_idx[test_pos]),
        )

    def _predefined_split_indices(
        self, df: pd.DataFrame, split: SplitStrategy
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert split.split_column is not None
        if split.split_column not in df.columns:
            raise ValueError(
                f"split_strategy.split_column '{split.split_column}' is not present in dataset."
            )
        values = df[split.split_column]
        train_mask = values.isin(split.train_values)
        valid_mask = values.isin(split.valid_values)
        test_mask = values.isin(split.test_values)
        assigned = (
            train_mask.astype(int) + valid_mask.astype(int) + test_mask.astype(int)
        )
        if not (assigned == 1).all():
            raise ValueError(
                "Predefined split requires every row to match exactly one train/valid/test value."
            )
        train_idx = np.flatnonzero(train_mask.to_numpy())
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if min(len(train_idx), len(valid_idx), len(test_idx)) == 0:
            raise ValueError(
                "Predefined split produced an empty train, validation, or test split."
            )
        return train_idx, valid_idx, test_idx

    def _stratify_values(
        self, y: pd.Series, indices: np.ndarray, split: SplitStrategy
    ) -> pd.Series | None:
        should_stratify = split.stratify
        if should_stratify is None:
            should_stratify = self.task.task_type != TaskType.REGRESSION
        if not should_stratify or self.task.task_type == TaskType.REGRESSION:
            return None
        selected = y.iloc[indices]
        counts = selected.value_counts(dropna=False)
        if len(counts) < 2 or counts.min() < 2:
            return None
        return selected

    @staticmethod
    def _split_counts(n_rows: int, split: SplitStrategy) -> tuple[int, int, int]:
        if n_rows < 3:
            raise ValueError(
                "At least three rows are required for train/validation/test splitting."
            )
        n_train = int(n_rows * split.train_size)
        n_valid = int(n_rows * split.valid_size)
        n_train = max(1, min(n_train, n_rows - 2))
        n_valid = max(1, min(n_valid, n_rows - n_train - 1))
        n_test = n_rows - n_train - n_valid
        return n_train, n_valid, n_test

    def elapsed_seconds(self) -> float:
        """Budgeted sandbox/environment execution time; excludes router wait time."""
        return self.execution_time_seconds

    def wall_clock_seconds(self) -> float:
        return max(0.0, time.time() - self.wall_start_time)

    def is_over_budget(self) -> bool:
        return self.elapsed_seconds() > self.task.time_budget_seconds

    def is_over_steps(self) -> bool:
        return self.current_step >= self.task.max_steps

    def record_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        self.current_step += 1
        self.execution_time_seconds += max(0.0, record.duration_seconds)
        stage = record.action_type.value
        usage = self.stage_usage[stage]
        usage.steps += 1
        usage.elapsed_seconds += max(0.0, record.duration_seconds)
        if self.last_stage == stage:
            self.consecutive_stage_steps += 1
        else:
            self.last_stage = stage
            self.consecutive_stage_steps = 1
        if record.action_type == ActionType.CODE_FIX:
            self.repair_attempts += 1
            self.repair_execution_successes += int(record.execution_success)
            self.repair_successes += int(
                record.execution_success and self.pending_repair_verified
            )
            self.pending_repair_verified = False
        self.consecutive_failures = (
            0 if record.execution_success else self.consecutive_failures + 1
        )
        if record.action_type in (
            ActionType.EDA,
            ActionType.FEATURE_ENGINEERING,
            ActionType.CODE_FIX,
        ):
            self._last_phase = "data"
        elif record.action_type == ActionType.CODE:
            if ".fit" in (record.code_body or record.action_text):
                if self._last_phase == "data":
                    self.cycle_count += 1
                self._last_phase = "model"
            else:
                self._last_phase = "data"
        elif record.action_type in (ActionType.MODEL, ActionType.VALIDATE):
            if record.action_type == ActionType.MODEL and self._last_phase == "data":
                self.cycle_count += 1
            self._last_phase = "model"
        if record.metric_value is not None:
            self.metric_history.append((record.step_idx, record.metric_value))

    def stage_budget_status(self, stage: str) -> tuple[str | None, bool, float]:
        if stage == ActionType.FINAL_SUBMIT.value:
            return None, False, 0.0
        usage = self.stage_usage[stage]
        limit = self.task.stage_limit(stage)
        over = (
            usage.steps > limit.max_steps
            or usage.elapsed_seconds > limit.max_seconds
            or (
                self.last_stage == stage
                and self.consecutive_stage_steps > limit.max_consecutive_steps
            )
        )
        if not over:
            return None, False, 0.0
        usage.violations += 1
        message = f"Stage budget exceeded for {stage}; move to a different stage or submit the current solution."
        if usage.violations == 1:
            return message, False, 0.05
        return (
            message + " The latest replayable solution will now be finalised.",
            True,
            0.15,
        )

    def check_data_intact(self) -> bool:
        train_original = self.sandbox_namespace.get("train_df_original")
        valid_original = self.sandbox_namespace.get("valid_df_original")
        if not isinstance(train_original, pd.DataFrame) or not isinstance(
            valid_original, pd.DataFrame
        ):
            return False
        return (
            self._hash_df(train_original) == self._train_original_hash
            and self._hash_df(valid_original) == self._valid_original_hash
        )

    def capture_workspace(self) -> dict[str, Any]:
        """Best-effort transactional snapshot used to undo failing or forbidden actions."""
        snapshot: dict[str, Any] = {}
        for key, value in self.sandbox_namespace.items():
            if key == "__builtins__":
                continue
            try:
                snapshot[key] = copy.deepcopy(value)
            except Exception:
                snapshot[key] = value
        return snapshot

    def restore_workspace(self, snapshot: dict[str, Any]) -> None:
        self.sandbox_namespace.clear()
        self.sandbox_namespace.update(snapshot)
        self.restore_protected_snapshots()

    def restore_protected_snapshots(self) -> None:
        assert (
            self._protected_train_snapshot is not None
            and self._protected_valid_snapshot is not None
        )
        self.sandbox_namespace["train_df_original"] = (
            self._protected_train_snapshot.copy(deep=True)
        )
        self.sandbox_namespace["valid_df_original"] = (
            self._protected_valid_snapshot.copy(deep=True)
        )

    def invalidate_candidate_validation(self) -> None:
        self.current_metric = None
        self.current_validated_candidate_version = None
        self.current_submission_kind = None
        self.current_submission_replayable = False
        self.validation_error = None
        self.candidate_raw_input_compatible = None
        self.candidate_probe_kind = None
        self.candidate_probe_error = None
        self.candidate_probe_step = None
        self.candidate_version += 1

    def state_summary(self, show_validation_metrics: bool = True) -> str:
        lines = [
            f"Step: {self.current_step} / {self.task.max_steps}",
            f"Sandbox execution time: {self.elapsed_seconds():.1f}s / {self.task.time_budget_seconds}s",
            f"Wall-clock elapsed (audit only): {self.wall_clock_seconds():.1f}s",
            f"Plan: {'submitted' if self.plan_text else 'not yet submitted'}",
            f"Transforms recorded: {len(self.applied_transforms)}",
            f"Replayable model candidates recorded: {len(self.trained_models)}",
        ]
        if show_validation_metrics:
            lines.append(
                f"Current validation {self.task.metric.value}: "
                + (
                    "N/A"
                    if self.current_metric is None
                    else f"{self.current_metric:.4f}"
                )
            )
            lines.append(
                f"Best validation {self.task.metric.value}: "
                + ("N/A" if self.best_metric is None else f"{self.best_metric:.4f}")
            )
        lines.append(
            f"Candidate replayability verified by VALIDATE: {'yes' if self.current_submission_replayable else 'no'}"
        )
        raw_probe = (
            "not checked"
            if self.candidate_raw_input_compatible is None
            else ("passed" if self.candidate_raw_input_compatible else "failed")
        )
        lines.append(f"Candidate raw-input smoke check (no metric): {raw_probe}")
        active_usage = [
            f"{name}={usage.steps}"
            for name, usage in sorted(self.stage_usage.items())
            if usage.steps
        ]
        if active_usage:
            lines.append("Stage actions used: " + ", ".join(active_usage))
        if self.repair_attempts:
            lines.append(
                f"Self-repair: {self.repair_successes}/{self.repair_attempts} fixes evaluator-verified "
                f"({self.repair_execution_successes}/{self.repair_attempts} code actions executed)"
            )
        return "\n".join(lines)

    def refresh_current_profile(self) -> None:
        self.previous_profile = self.current_profile
        candidate = self.sandbox_namespace.get("train_df")
        if isinstance(candidate, pd.DataFrame):
            self.current_profile = self._build_profile(candidate)

    def profile_delta_messages(self) -> list[str]:
        if self.previous_profile is None or self.current_profile is None:
            return []
        messages: list[str] = []
        if self.current_profile.missing_cells < self.previous_profile.missing_cells:
            messages.append(
                "Missing-value burden improved in the working training frame."
            )
        elif self.current_profile.missing_cells > self.previous_profile.missing_cells:
            messages.append(
                "Missing-value burden increased in the working training frame."
            )
        if self.current_profile.duplicate_rows < self.previous_profile.duplicate_rows:
            messages.append(
                "Duplicate-feature rows decreased in the working training frame."
            )
        elif self.current_profile.duplicate_rows > self.previous_profile.duplicate_rows:
            messages.append(
                "Duplicate-feature rows increased in the working training frame."
            )
        return messages

    def _build_profile(self, df: pd.DataFrame) -> DatasetStateProfile:
        feature_df = df.drop(columns=[self.task.target_column], errors="ignore")
        high_cardinality = sum(
            feature_df[column].nunique(dropna=True)
            > max(20, int(0.2 * len(feature_df)))
            for column in feature_df.columns
        )
        identifier_like = sum(
            feature_df[column].nunique(dropna=True) >= int(0.95 * len(feature_df))
            for column in feature_df.columns
        )
        imbalance = None
        if (
            self.task.task_type != TaskType.REGRESSION
            and self.task.target_column in df.columns
        ):
            proportions = df[self.task.target_column].value_counts(
                normalize=True, dropna=False
            )
            if len(proportions) > 1:
                imbalance = float(
                    proportions.iloc[0] / max(proportions.iloc[-1], 1e-12)
                )
        return DatasetStateProfile(
            missing_cells=int(feature_df.isna().sum().sum()),
            duplicate_rows=int(feature_df.duplicated().sum()),
            categorical_columns=int(
                len(feature_df.select_dtypes(include=["object", "category"]).columns)
            ),
            high_cardinality_columns=int(high_cardinality),
            identifier_like_columns=int(identifier_like),
            class_imbalance_ratio=imbalance,
        )

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        schema = repr(
            (list(df.columns), [str(dtype) for dtype in df.dtypes], df.shape)
        ).encode("utf-8")
        return hashlib.sha256(schema + values).hexdigest()
