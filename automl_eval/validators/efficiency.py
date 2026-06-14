""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

# Flag automated/exhaustive model-search APIs. The experiment policy is to
_FORBIDDEN_SEARCH_PATTERNS = re.compile(
    r"(\bGridSearchCV\b|\bRandomizedSearchCV\b|\bHalvingGridSearchCV\b|"
    r"\bHalvingRandomSearchCV\b|\bParameterGrid\b|\bParameterSampler\b|"
    r"\bRidgeCV\b|\bLassoCV\b|\bElasticNetCV\b|"
    r"BayesSearchCV|Optuna|optuna|hyperopt|ray\.tune|tune\.run|"
    r"sklearn\.model_selection\.GridSearch)",
    re.IGNORECASE,
)


class EfficiencyValidator(BaseValidator):
    """Penalizes excessive execution time and inefficient hyperparameter search."""

    name = "efficiency"

    def __init__(
        self,
        hard_time_limit: float = 3600.0,
        gridsearch_penalty: float = 0.10,
        time_penalty_max: float = 0.3,
    ) -> None:
        self.hard_time_limit = hard_time_limit
        self.gridsearch_penalty = gridsearch_penalty
        self.time_penalty_max = time_penalty_max

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        elapsed = session.elapsed_seconds()
        budget = session.task.time_budget_seconds
        issues: list[str] = []
        penalty = 0.0

        if elapsed >= self.hard_time_limit:
            penalty += self.time_penalty_max
            issues.append(
                f"Hard time limit exceeded: {elapsed:.0f}s >= {self.hard_time_limit:.0f}s (max penalty)"
            )
        elif elapsed > budget:
            ratio = min((elapsed - budget) / budget, 1.0)
            time_pen = self.time_penalty_max * ratio
            penalty += time_pen
            issues.append(
                f"Over budget: {elapsed:.0f}s / {budget:.0f}s (penalty={time_pen:.3f})"
            )

        code_steps = [
            rec
            for rec in session.steps
            if rec.action_type in (ActionType.CODE, ActionType.MODEL)
            and rec.execution_success
        ]
        all_code = "\n".join(
            (rec.code_body if rec.code_body else rec.action_text) for rec in code_steps
        )

        has_forbidden_search = bool(_FORBIDDEN_SEARCH_PATTERNS.search(all_code))

        if has_forbidden_search:
            penalty += self.gridsearch_penalty
            issues.append(
                "Automated hyperparameter/model search detected — "
                "choose a small set of explicit hyperparameters manually"
            )

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0
        details = (
            "; ".join(issues)
            if issues
            else f"Efficient: {elapsed:.0f}s / {budget:.0f}s budget."
        )

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
