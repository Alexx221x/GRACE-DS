""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_EDA_PATTERNS = re.compile(
    r"(\.describe\s*\(|\.hist\s*\(|\.plot\s*\(|\.boxplot\s*\(|"
    r"\.value_counts\s*\(|\.info\s*\(|\.skew\s*\(|\.kurt\s*\(|"
    r"sns\.|seaborn|matplotlib|\.nunique\s*\(|\.dtype|\.shape|"
    r"\.isnull\(\)\.sum|\.isna\(\)\.sum|\.profile_report|"
    r"pandas_profiling|ydata_profiling|\.quantile\s*\(|np\.percentile\s*\(|"
    r"np\.quantile\s*\(|\.kurtosis\s*\(|IQR|iqr|describe_percentiles)",
    re.IGNORECASE,
)

_EDA_PLAN = re.compile(
    r"(EDA|explor|исследов|анализ.данн|distribut|распредел|статистик|describe|"
    r"визуализ|visualiz|overview|обзор)",
    re.IGNORECASE,
)

_OUTLIER_HANDLING = re.compile(
    r"(\.clip\s*\(|IQR|iqr|z.?score|zscore|winsoriz|"
    r"RobustScaler|quantile.*filter|percentile|"
    r"outlier|выброс|remove_outlier|clip_outlier|"
    r"LocalOutlierFactor|IsolationForest|isolation_forest)",
    re.IGNORECASE,
)

_OUTLIER_PLAN = re.compile(
    r"(outlier|выброс|аномал|anomal|clip|robust|IQR|iqr|winsor)",
    re.IGNORECASE,
)

_SIGNIFICANT_OUTLIER_FRAC = 0.05


class DistributionValidator(BaseValidator):
    """Checks that the agent performs EDA and handles outliers."""

    name = "distribution"

    def __init__(
        self,
        no_eda_penalty: float = 0.10,
        no_outlier_handling_penalty: float = 0.08,
    ) -> None:
        self.no_eda_penalty = no_eda_penalty
        self.no_outlier_handling_penalty = no_outlier_handling_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        insights = session.data_insights

        all_code = ""
        plan_text = session.plan_text or ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (
                ActionType.EDA,
                ActionType.CODE,
                ActionType.FEATURE_ENGINEERING,
                ActionType.MODEL,
                ActionType.CODE_FIX,
            ):
                all_code += "\n" + text

        issues: list[str] = []
        penalty = 0.0

        # 1) EDA: expect at least some data exploration
        did_eda = bool(_EDA_PATTERNS.search(all_code) or _EDA_PLAN.search(plan_text))
        if not did_eda:
            issues.append(
                "No EDA detected (expected .describe(), .hist(), .value_counts(), "
                "or mention of data exploration in plan)"
            )
            penalty += self.no_eda_penalty

        # 2) Outliers: if the environment detected significant outliers — expect a response
        if insights is not None and insights.has_outliers:
            significant = [
                o
                for o in insights.outlier_columns
                if o.outlier_frac >= _SIGNIFICANT_OUTLIER_FRAC
            ]

            if significant:
                did_outlier_work = bool(
                    _OUTLIER_HANDLING.search(all_code)
                    or _OUTLIER_PLAN.search(plan_text)
                )
                if not did_outlier_work:
                    cols_str = ", ".join(o.column for o in significant[:5])
                    issues.append(
                        f"Significant outliers detected in [{cols_str}] "
                        f"but no outlier handling found (clip, IQR filter, "
                        f"RobustScaler, etc.)"
                    )
                    penalty += self.no_outlier_handling_penalty

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        if passed:
            parts = ["EDA present"]
            if insights is not None and insights.has_outliers:
                parts.append("outlier handling present")
            details = "; ".join(parts) + "."
        else:
            details = "; ".join(issues)

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
