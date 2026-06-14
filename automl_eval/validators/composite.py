""""""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class AndValidator(BaseValidator):
    """Passes only if ALL sub-validators pass."""

    name = "and"

    def __init__(self, validators: list[BaseValidator], label: str = "and") -> None:
        self.sub_validators = validators
        self.name = label

    def validate(self, session: RuntimeSession) -> ValidationResult:
        results = [v.validate(session) for v in self.sub_validators]
        all_passed = all(r.passed for r in results)
        total_penalty = sum(r.penalty for r in results)
        min_score = min(r.score for r in results) if results else 1.0

        failed = [r for r in results if not r.passed]
        if failed:
            details = "; ".join(
                f"[FAIL {r.validator_name}] {r.details}" for r in failed
            )
        else:
            details = "All sub-validators passed."

        return ValidationResult(
            validator_name=self.name,
            passed=all_passed,
            score=min_score,
            details=details,
            penalty=total_penalty if not all_passed else 0.0,
        )


class OrValidator(BaseValidator):
    """Passes if ANY sub-validator passes."""

    name = "or"

    def __init__(self, validators: list[BaseValidator], label: str = "or") -> None:
        self.sub_validators = validators
        self.name = label

    def validate(self, session: RuntimeSession) -> ValidationResult:
        results = [v.validate(session) for v in self.sub_validators]
        any_passed = any(r.passed for r in results)
        max_score = max(r.score for r in results) if results else 0.0

        passed_names = [r.validator_name for r in results if r.passed]
        if any_passed:
            details = f"Passed via: {', '.join(passed_names)}"
            penalty = 0.0
        else:
            details = "; ".join(
                f"[FAIL {r.validator_name}] {r.details}" for r in results
            )
            penalty = min(r.penalty for r in results) if results else 0.0

        return ValidationResult(
            validator_name=self.name,
            passed=any_passed,
            score=max_score,
            details=details,
            penalty=penalty,
        )
