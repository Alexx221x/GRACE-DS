""""""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class CrashCategory(str, Enum):
    """DSEval-compatible error categories for Crash verdict."""

    NONE = "none"
    SYNTAX_ERROR = "syntax_error"
    MODULE_NOT_FOUND = "module_not_found"
    KEY_ERROR = "key_error"
    NAME_ERROR = "name_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    TIMEOUT = "timeout"
    OTHER = "other"


_ERROR_PATTERNS: list[tuple[str, CrashCategory]] = [
    (r"SyntaxError", CrashCategory.SYNTAX_ERROR),
    (r"ModuleNotFoundError|No module named", CrashCategory.MODULE_NOT_FOUND),
    (r"ImportError.*forbidden", CrashCategory.MODULE_NOT_FOUND),
    (r"KeyError", CrashCategory.KEY_ERROR),
    (r"NameError", CrashCategory.NAME_ERROR),
    (r"AttributeError", CrashCategory.ATTRIBUTE_ERROR),
    (r"TypeError", CrashCategory.TYPE_ERROR),
    (r"ValueError", CrashCategory.VALUE_ERROR),
    (r"timed out|Timeout", CrashCategory.TIMEOUT),
]


def classify_error(error_message: str | None) -> CrashCategory:
    """Classify an error message into a DSEval crash category."""
    if not error_message:
        return CrashCategory.NONE
    for pattern, category in _ERROR_PATTERNS:
        if re.search(pattern, error_message, re.IGNORECASE):
            return category
    return CrashCategory.OTHER


class ExecutionValidator(BaseValidator):
    name = "execution"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.steps:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No steps yet.",
            )

        last = session.steps[-1]

        if last.execution_success:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Last step executed successfully.",
            )

        category = classify_error(last.error_message)

        return ValidationResult(
            validator_name=self.name,
            passed=False,
            score=0.0,
            details=f"Crash [{category.value}]: {last.error_message}",
            penalty=0.1,
        )
