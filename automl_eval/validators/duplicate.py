""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_DEDUP_CODE = re.compile(
    r"(drop_duplicates|duplicated\(\)|\.unique\(\)|дубл|duplicate)",
    re.IGNORECASE,
)
_DEDUP_PLAN = re.compile(
    r"(duplicate|дубл|дубликат|dedup|повтор|одинаков)",
    re.IGNORECASE,
)


class DuplicateValidator(BaseValidator):
    """Checks handling of explicit (and near) duplicates."""

    name = "duplicates"

    def __init__(
        self,
        no_handling_penalty: float = 0.08,
        near_dup_col_threshold: int = 2,
    ) -> None:
        self.no_handling_penalty = no_handling_penalty
        self.near_dup_col_threshold = near_dup_col_threshold

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        insights = session.data_insights
        if insights is None or not insights.has_duplicates:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No duplicates in original data.",
            )

        all_code = ""
        plan_text = session.plan_text or ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (
                ActionType.EDA,
                ActionType.CODE,
                ActionType.FEATURE_ENGINEERING,
                ActionType.CODE_FIX,
            ):
                all_code += "\n" + text

        issues: list[str] = []
        penalty = 0.0

        mentioned = bool(_DEDUP_CODE.search(all_code) or _DEDUP_PLAN.search(plan_text))

        ns = session.sandbox_namespace
        current_train = ns.get("train_df")
        remaining_dups = 0
        if isinstance(current_train, pd.DataFrame):
            target = session.task.target_column
            feat_cols = [c for c in current_train.columns if c != target]
            if feat_cols:
                remaining_dups = int(current_train[feat_cols].duplicated().sum())

        if not mentioned and remaining_dups > 0:
            issues.append(
                f"Dataset had {insights.duplicate_count} duplicate rows; "
                f"{remaining_dups} still remain — not addressed in code or plan"
            )
            penalty += self.no_handling_penalty
        elif mentioned and remaining_dups > 0:
            issues.append(
                f"Duplicates mentioned but {remaining_dups} still remain "
                f"(original: {insights.duplicate_count})"
            )
            penalty += self.no_handling_penalty * 0.5

        # Near-duplicates: informational note, not a blocking issue
        notes: list[str] = []
        near_dup_count = self._count_near_duplicates(session)
        if near_dup_count > 0:
            notes.append(
                f"Note: {near_dup_count} near-duplicate row(s) "
                f"(differ in <={self.near_dup_col_threshold} columns)"
            )

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        if passed:
            if insights.duplicate_count > 0:
                details = f"Duplicates handled: {insights.duplicate_count} original → {remaining_dups} remaining."
            else:
                details = "No duplicates in dataset."
        else:
            details = "; ".join(issues)

        if notes:
            details += " | " + "; ".join(notes)

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _count_near_duplicates(self, session: RuntimeSession) -> int:
        """Count near-duplicates: rows differing in <= N columns."""
        ns = session.sandbox_namespace
        df = ns.get("train_df")
        if not isinstance(df, pd.DataFrame) or len(df) > 500:
            return 0

        target = session.task.target_column
        feat_cols = [c for c in df.columns if c != target]
        if len(feat_cols) < 3:
            return 0

        try:
            numeric_df = df[feat_cols].select_dtypes(include="number")
            if len(numeric_df.columns) < 3 or len(numeric_df) < 5:
                return 0

            count = 0
            sample = numeric_df.head(200)
            vals = sample.values
            for i in range(len(vals)):
                for j in range(i + 1, min(i + 50, len(vals))):
                    diffs = (vals[i] != vals[j]).sum()
                    if 0 < diffs <= self.near_dup_col_threshold:
                        count += 1
                        if count >= 10:
                            return count
            return count
        except Exception:
            return 0
