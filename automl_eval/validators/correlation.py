""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_CORR_CODE = re.compile(
    r"\.corr\s*\(|corrwith|corrcoef|pearsonr|spearmanr|kendalltau|"
    r"correlation|heatmap|VIF|vif|variance_inflation",
    re.IGNORECASE,
)
_CORR_PLAN = re.compile(
    r"(correlat|коррел|multicoll|мультиколлин|VIF|collinear)", re.IGNORECASE
)
_PCA_PATTERN = re.compile(
    r"(PCA|pca|principal.component|decomposition|VarianceThreshold|"
    r"SelectFromModel|regulariz|Lasso|ElasticNet)",
    re.IGNORECASE,
)
_DROP_PATTERN = re.compile(r"\.drop\s*\(", re.IGNORECASE)


class CorrelationValidator(BaseValidator):
    """Checks that the agent analyzes and handles correlations between features."""

    name = "correlation"

    def __init__(
        self, penalty_per_missed_pair: float = 0.05, no_analysis_penalty: float = 0.1
    ) -> None:
        self.penalty_per_missed_pair = penalty_per_missed_pair
        self.no_analysis_penalty = no_analysis_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        insights = session.data_insights
        if insights is None or not insights.has_high_correlation:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No high correlations in dataset or insights not available.",
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
            if rec.action_type == ActionType.PLAN:
                plan_text = text

        issues: list[str] = []
        penalty = 0.0

        did_corr_analysis = bool(
            _CORR_CODE.search(all_code) or _CORR_PLAN.search(plan_text)
        )
        if not did_corr_analysis:
            issues.append(
                f"No correlation analysis found (.corr() or mention in plan) — "
                f"dataset has {len(insights.high_corr_pairs)} highly correlated pair(s)"
            )
            penalty += self.no_analysis_penalty

        used_pca = bool(_PCA_PATTERN.search(all_code))

        if not used_pca:
            for pair in insights.high_corr_pairs:
                col_a_dropped = bool(
                    re.search(
                        r"\.drop\s*\(.*['\"]" + re.escape(pair.col_a) + r"['\"]",
                        all_code,
                    )
                )
                col_b_dropped = bool(
                    re.search(
                        r"\.drop\s*\(.*['\"]" + re.escape(pair.col_b) + r"['\"]",
                        all_code,
                    )
                )
                if not col_a_dropped and not col_b_dropped:
                    issues.append(
                        f"Correlated pair ({pair.col_a}, {pair.col_b}, r={pair.corr_value}) "
                        f"not addressed (no drop or PCA)"
                    )
                    penalty += self.penalty_per_missed_pair

        total_pairs = len(insights.high_corr_pairs)
        missed = sum(1 for i in issues if "not addressed" in i)
        addressed = total_pairs - missed

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0
        if passed:
            details = (
                f"Correlation handled: {addressed}/{total_pairs} pair(s) addressed, "
                f"analysis present."
            )
        else:
            details = "; ".join(issues)

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
