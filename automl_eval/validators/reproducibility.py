""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_SEED_PATTERNS = re.compile(
    r"(random_state\s*=|seed\s*=|np\.random\.seed\(|random\.seed\(|"
    r"torch\.manual_seed\(|set_seed\(|SEED\s*=)",
    re.IGNORECASE,
)


class ReproducibilityValidator(BaseValidator):
    """Checks whether the agent fixes seeds and produces reproducible results."""

    name = "reproducibility"

    def __init__(
        self,
        seed_penalty: float = 0.08,
    ) -> None:
        self.seed_penalty = seed_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        code_steps = [
            rec
            for rec in session.steps
            if rec.action_type
            in (ActionType.CODE, ActionType.MODEL, ActionType.FEATURE_ENGINEERING)
            and rec.execution_success
        ]

        if not code_steps:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No code steps to check.",
            )

        def _executable(rec):
            return rec.code_body if rec.code_body else rec.action_text

        all_code = "\n".join(_executable(rec) for rec in code_steps)
        has_model_fit = bool(re.search(r"\.fit\(", all_code))

        has_seed = bool(_SEED_PATTERNS.search(all_code))

        issues: list[str] = []
        penalty = 0.0

        if has_model_fit and not has_seed:
            issues.append("No seed/random_state found in code that trains a model")
            penalty += self.seed_penalty

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0
        details = (
            "; ".join(issues) if issues else "Reproducibility OK (seed check only)."
        )

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
