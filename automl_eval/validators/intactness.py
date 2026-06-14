"""Protect immutable raw snapshots while allowing mutations of the working state."""

from __future__ import annotations
from typing import TYPE_CHECKING
from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class IntactnessValidator(BaseValidator):
    name = "intactness"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        intact = session.check_data_intact()
        if intact:
            return ValidationResult(
                self.name, True, 1.0, "Protected raw snapshots remain unchanged."
            )
        return ValidationResult(
            self.name,
            False,
            0.0,
            "Protected raw snapshots were modified; work only on mutable working frames.",
            penalty=0.3,
        )
