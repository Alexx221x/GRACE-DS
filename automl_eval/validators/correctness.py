""""""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class CorrectnessValidator(BaseValidator):
    name = "correctness"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        preds = session.predictions

        if preds is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=0.5,
                details="No predictions submitted yet (neutral).",
            )

        issues: list[str] = []

        test_size = len(session.test_df)  # type: ignore[arg-type]
        if len(preds) != test_size:
            issues.append(
                f"Prediction length mismatch: got {len(preds)}, expected {test_size}"
            )

        if np.isnan(preds).any():
            nan_count = int(np.isnan(preds).sum())
            issues.append(f"Predictions contain {nan_count} NaN values")

        if not np.isfinite(preds).all():
            inf_count = int(~np.isfinite(preds).sum())
            issues.append(f"Predictions contain {inf_count} non-finite values")

        if issues:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues),
                penalty=0.2,
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            score=1.0,
            details="Predictions are valid.",
        )
