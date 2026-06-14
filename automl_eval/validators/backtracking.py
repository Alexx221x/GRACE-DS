""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_BASIC_OPS = re.compile(
    r"(\.drop\(|\.dropna\(|\.fillna\(|\.replace\(|\.rename\(|"
    r"\.astype\(|del\s+\w|\.pop\(|\.remove\()",
    re.IGNORECASE,
)

_MODEL_FIT = re.compile(r"\.fit\(", re.IGNORECASE)


class BacktrackingValidator(BaseValidator):
    """Penalizes reverting to basic operations after model training."""

    name = "backtracking"

    def __init__(
        self,
        penalty_per_backtrack: float = 0.05,
        max_backtrack_penalty: float = 0.12,
    ) -> None:
        self.penalty_per_backtrack = penalty_per_backtrack
        self.max_backtrack_penalty = max_backtrack_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        model_trained_at: int | None = None
        backtrack_count = 0
        backtrack_details: list[str] = []

        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text

            if model_trained_at is None:
                if rec.action_type == ActionType.MODEL and rec.execution_success:
                    model_trained_at = rec.step_idx
                elif rec.action_type == ActionType.CODE and rec.execution_success:
                    if _MODEL_FIT.search(text):
                        model_trained_at = rec.step_idx
                continue

            is_backtrack = False
            if rec.action_type == ActionType.FEATURE_ENGINEERING:
                if _BASIC_OPS.search(text):
                    is_backtrack = True
            elif rec.action_type == ActionType.CODE and rec.execution_success:
                if _BASIC_OPS.search(text) and not _MODEL_FIT.search(text):
                    is_backtrack = True

            if is_backtrack:
                backtrack_count += 1
                backtrack_details.append(
                    f"step {rec.step_idx}: basic op after model (trained at step {model_trained_at})"
                )

        penalty = min(
            self.max_backtrack_penalty,
            self.penalty_per_backtrack * backtrack_count,
        )
        if backtrack_count == 0:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No backtracking detected.",
            )

        return ValidationResult(
            validator_name=self.name,
            passed=False,
            score=max(0.0, 1.0 - penalty),
            details=f"{backtrack_count} backtrack(s): {'; '.join(backtrack_details)}",
            penalty=penalty,
        )
