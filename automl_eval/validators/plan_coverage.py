""""""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class PlanCoverageValidator(BaseValidator):
    name = "plan_coverage"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        checklist = session.task.plan_checklist

        if not checklist:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No checklist defined for this task.",
            )

        if not session.plan_text:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details="Agent has not submitted a plan yet.",
            )

        total_weight = sum(item.weight for item in checklist)
        covered_weight = 0.0
        covered_items: list[str] = []
        missed_items: list[str] = []
        missed_required: list[str] = []

        for item in checklist:
            if item.check(session.plan_text):
                covered_weight += item.weight
                covered_items.append(item.id)
            else:
                missed_items.append(item.id)
                if item.required:
                    missed_required.append(item.id)

        coverage = covered_weight / total_weight if total_weight > 0 else 0.0

        penalty = 0.0
        if missed_required:
            penalty = 0.2 * len(missed_required)

        details_parts = [
            f"Coverage: {coverage:.0%} ({len(covered_items)}/{len(checklist)} items)",
        ]
        if covered_items:
            details_parts.append(f"Covered: {', '.join(covered_items)}")
        if missed_items:
            details_parts.append(f"Missed: {', '.join(missed_items)}")
        if missed_required:
            details_parts.append(f"Missed REQUIRED: {', '.join(missed_required)}")

        return ValidationResult(
            validator_name=self.name,
            passed=len(missed_required) == 0 and coverage > 0.5,
            score=coverage,
            details="; ".join(details_parts),
            penalty=penalty,
        )
