"""Validate only evaluator-owned validation results produced by ACTION: VALIDATE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult, ValidatorStatus

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class ModelEvalValidator(BaseValidator):
    """Do not independently score a model; consume the gated evaluator metric only."""

    name = "model_eval"

    def __init__(self, min_score_above_baseline: float = -0.08) -> None:
        self.min_score_above_baseline = min_score_above_baseline

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.current_submission_replayable or session.current_metric is None:
            return ValidationResult(
                self.name,
                False,
                0.0,
                "No successful evaluator-owned validation result is available for the current candidate.",
                penalty=0.03,
                status=ValidatorStatus.UNRESOLVED,
            )
        metric_val = session.current_metric
        baseline = session.task.baseline_score or 0.0
        delta = metric_val - baseline
        if delta < self.min_score_above_baseline:
            return ValidationResult(
                self.name,
                False,
                max(0.0, 0.45 + max(delta, -0.5)),
                f"Validated candidate is well below baseline (delta={delta:+.4f}).",
                penalty=0.04,
                status=ValidatorStatus.UNRESOLVED,
            )
        return ValidationResult(
            self.name,
            True,
            min(1.0, 0.5 + delta),
            f"Evaluator-owned validation completed (delta from baseline={delta:+.4f}).",
            status=ValidatorStatus.RESOLVED,
        )
