""""""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_FEATURE_DF_NAMES = ["X_train", "X", "features", "df_train", "train_processed"]


class FeaturePipelineValidator(BaseValidator):
    """Checks the actual state of data in the sandbox after agent transformations."""

    name = "feature_pipeline"

    def __init__(
        self,
        nan_penalty: float = 0.12,
        unencoded_penalty: float = 0.10,
        no_scaling_penalty: float = 0.05,
        no_fe_penalty: float = 0.05,
        scale_threshold: float = 100.0,
    ) -> None:
        self.nan_penalty = nan_penalty
        self.unencoded_penalty = unencoded_penalty
        self.no_scaling_penalty = no_scaling_penalty
        self.no_fe_penalty = no_fe_penalty
        self.scale_threshold = scale_threshold

    def validate(self, session: RuntimeSession) -> ValidationResult:
        ns = session.sandbox_namespace
        insights = session.data_insights

        working_df = self._find_working_df(ns)
        if working_df is None or insights is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No transformed data found in sandbox yet.",
            )

        critical_issues: list[str] = []
        soft_issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0

        # ── 1. NaN check (critical) ─────────────────────────────────
        total_nan = int(working_df.isnull().sum().sum())
        if total_nan > 0:
            nan_cols = [c for c in working_df.columns if working_df[c].isnull().any()]
            critical_issues.append(
                f"Working data still has {total_nan} NaN values "
                f"in columns: {', '.join(nan_cols[:5])}"
            )
            penalty += self.nan_penalty
        else:
            if insights.has_missing:
                bonuses.append("all missing values handled")

        # ── 2. Unencoded categoricals (critical) ────────────────────
        object_cols = working_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if object_cols:
            critical_issues.append(
                f"Unencoded categorical columns remain: {', '.join(object_cols[:5])} — "
                f"model cannot handle raw strings"
            )
            penalty += self.unencoded_penalty
        else:
            if insights.categorical_columns:
                bonuses.append("all categoricals encoded")

        # ── 3. Scaling (soft — adds penalty but not a hard fail) ────
        num_cols = working_df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            stds = working_df[num_cols].std().replace(0, np.nan).dropna()
            if len(stds) >= 2:
                ratio = float(stds.max() / stds.min())
                if ratio > self.scale_threshold:
                    soft_issues.append(
                        f"Feature scales differ by {ratio:.0f}x "
                        f"(max_std/min_std) — consider scaling"
                    )
                    penalty += self.no_scaling_penalty

        # ── 4. Feature engineering (soft) ────────────────────────────
        original_cols = set(insights.numeric_columns + insights.categorical_columns)
        current_cols = set(working_df.columns.tolist())
        new_cols = current_cols - original_cols
        dropped_cols = original_cols - current_cols
        if new_cols:
            bonuses.append(f"{len(new_cols)} new feature(s) created")
        if dropped_cols:
            bonuses.append(f"{len(dropped_cols)} column(s) dropped")
        if not new_cols and not dropped_cols and len(original_cols) > 3:
            soft_issues.append(
                "No feature engineering detected (same columns as original)"
            )
            penalty += self.no_fe_penalty

        # ── 5. Skewness improvement ──────────────────────────────────
        if insights.has_high_skew:
            still_skewed = 0
            for si in insights.skewed_columns:
                if si.column in working_df.columns:
                    current_skew = abs(float(working_df[si.column].dropna().skew()))
                    if current_skew > 2.0:
                        still_skewed += 1
            if still_skewed == 0:
                bonuses.append("skewed features normalized")
            elif still_skewed < len(insights.skewed_columns):
                bonuses.append(
                    f"some skewed features addressed ({still_skewed} remain)"
                )

        # ── 6. Datetime handling ──────────────────────────────────────
        if insights.datetime_like_columns:
            still_string = [
                c
                for c in insights.datetime_like_columns
                if c in working_df.columns and working_df[c].dtype == "object"
            ]
            if still_string:
                critical_issues.append(
                    f"Datetime-like columns still as strings: {', '.join(still_string)}"
                )

        score = max(0.0, 1.0 - penalty)
        passed = len(critical_issues) == 0

        all_issues = critical_issues + soft_issues
        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if all_issues:
            parts.append("Issues: " + "; ".join(all_issues))
        details = ". ".join(parts) if parts else "Feature pipeline looks reasonable."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _find_working_df(self, ns: dict) -> pd.DataFrame | None:
        """Finds the working DataFrame with features in the sandbox namespace."""
        for name in _FEATURE_DF_NAMES:
            obj = ns.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                return obj

        train = ns.get("train_df")
        if isinstance(train, pd.DataFrame):
            return train

        return None
