""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_FILL_PATTERNS = re.compile(
    r"(\.fillna\(|\.fill\(|SimpleImputer|KNNImputer|IterativeImputer|MissingIndicator|"
    r"\.interpolate\(|\.bfill\(|\.ffill\(|imputer|impute)",
    re.IGNORECASE,
)
_DROPNA_PATTERN = re.compile(r"\.dropna\s*\(", re.IGNORECASE)
_DROP_COL_PATTERN = re.compile(r"\.drop\s*\(", re.IGNORECASE)
_MISSING_PLAN = re.compile(
    r"(missing|пропус|null|nan|fillna|imput|dropna|недостающ)",
    re.IGNORECASE,
)


class MissingValuesValidator(BaseValidator):
    """Checks missing value handling based on the environment's pre-analysis."""

    name = "missing_values"

    def __init__(
        self,
        no_handling_penalty: float = 0.15,
        excessive_dropna_penalty: float = 0.10,
        bad_strategy_penalty: float = 0.05,
        max_dropna_loss_frac: float = 0.15,
    ) -> None:
        self.no_handling_penalty = no_handling_penalty
        self.excessive_dropna_penalty = excessive_dropna_penalty
        self.bad_strategy_penalty = bad_strategy_penalty
        self.max_dropna_loss_frac = max_dropna_loss_frac

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        insights = session.data_insights
        if insights is None or not insights.has_missing:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No missing values in dataset.",
            )

        all_code = ""
        plan_text = session.plan_text or ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (
                ActionType.CODE,
                ActionType.FEATURE_ENGINEERING,
                ActionType.MODEL,
                ActionType.CODE_FIX,
            ):
                all_code += "\n" + text

        issues: list[str] = []
        penalty = 0.0

        # 1) Is missing value handling mentioned at all?
        mentions_missing = bool(
            _FILL_PATTERNS.search(all_code)
            or _DROPNA_PATTERN.search(all_code)
            or _MISSING_PLAN.search(plan_text)
        )
        if not mentions_missing:
            missing_cols = [m.column for m in insights.missing_columns]
            issues.append(
                f"No missing-value handling found — "
                f"columns with nulls: {', '.join(missing_cols)}"
            )
            penalty += self.no_handling_penalty

        # 2) Check each column with missing values
        handled_cols: list[str] = []
        unhandled_cols: list[str] = []

        for cm in insights.missing_columns:
            col_esc = re.escape(cm.column)
            col_mentioned = bool(re.search(r"['\"]" + col_esc + r"['\"]", all_code))
            col_filled = col_mentioned and bool(_FILL_PATTERNS.search(all_code))
            col_dropped = bool(
                re.search(r"\.drop\s*\(.*['\"]" + col_esc + r"['\"]", all_code)
            )

            if col_filled or col_dropped:
                handled_cols.append(cm.column)

                # 2a) Column with >80% NaN is filled instead of dropped — questionable
                if cm.missing_frac > 0.80 and col_filled and not col_dropped:
                    issues.append(
                        f"Column '{cm.column}' has {cm.missing_frac:.0%} missing — "
                        f"filling may be unreliable, consider dropping"
                    )
                    penalty += self.bad_strategy_penalty
            else:
                unhandled_cols.append(cm.column)

        # 3) dropna should not lose >15% of rows
        uses_dropna = bool(_DROPNA_PATTERN.search(all_code))
        if uses_dropna:
            ns = session.sandbox_namespace
            current_train = ns.get("train_df")
            if current_train is not None and insights.n_rows > 0:
                rows_lost_frac = 1.0 - len(current_train) / insights.n_rows
                if rows_lost_frac > self.max_dropna_loss_frac:
                    issues.append(
                        f"dropna lost {rows_lost_frac:.0%} of rows "
                        f"(>{self.max_dropna_loss_frac:.0%} threshold) — "
                        f"consider column-level drop or filling instead"
                    )
                    penalty += self.excessive_dropna_penalty

        # 4) Summary
        total = len(insights.missing_columns)
        n_handled = len(handled_cols)

        if unhandled_cols and mentions_missing:
            issues.append(
                f"Unhandled missing columns: {', '.join(unhandled_cols)} "
                f"({n_handled}/{total} handled)"
            )
            # Global imputation (e.g. df.fillna) may not name each column explicitly.
            if _FILL_PATTERNS.search(all_code):
                penalty += 0.02
            else:
                penalty += self.bad_strategy_penalty * min(len(unhandled_cols), 3)

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        if passed:
            details = f"Missing values handled: {n_handled}/{total} columns addressed."
        else:
            details = "; ".join(issues)

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
