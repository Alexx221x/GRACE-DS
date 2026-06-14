""""""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_LEAKAGE_PATTERNS = [
    re.compile(r"\.fit\s*\(.*(?:valid_df|test_df)", re.IGNORECASE),
    re.compile(r"\.fit_transform\s*\(.*(?:valid_df|test_df)", re.IGNORECASE),
]

_TARGET_AS_FEATURE = re.compile(
    r"(?:X_train|X_val|X_test|features)\s*\[.*['\"]{}['\"]\]"
)

_TARGET_SAFE_OPS = re.compile(r"(\.drop\s*\(|\.pop\s*\(|\.remove\s*\(|y\s*=\s*.*\[)")

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*(?:\r?\n)?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code_only(text: str) -> str:
    """Return only the Python code from *text*."""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return "\n".join(b.strip() for b in blocks if b.strip())
    return ""


class LeakageValidator(BaseValidator):
    name = "leakage"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        issues: list[str] = []

        for step in session.steps:
            if step.action_type.value == "FINAL_SUBMIT":
                continue

            raw = step.code_body if step.code_body else step.action_text
            text = _extract_code_only(raw)

            if self._unsafe_test_df_usage(text):
                issues.append(
                    f"Step {step.step_idx}: possible leakage — unsafe usage of 'test_df'"
                )

            for pattern in _LEAKAGE_PATTERNS:
                if pattern.search(text):
                    issues.append(
                        f"Step {step.step_idx}: possible leakage — "
                        f"matched '{pattern.pattern}'"
                    )

            target_col = session.task.target_column
            target_re = re.compile(
                _TARGET_AS_FEATURE.pattern.format(re.escape(target_col))
            )

            for line in text.splitlines():
                if target_re.search(line) and not _TARGET_SAFE_OPS.search(line):
                    issues.append(
                        f"Step {step.step_idx}: target column '{target_col}' "
                        f"may be used as a feature"
                    )

        if issues:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues),
                penalty=0.5,
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            score=1.0,
            details="No data leakage detected.",
        )

    @staticmethod
    def _unsafe_test_df_usage(text: str) -> bool:
        """Flag only evaluator-reserved held-out namespaces."""
        return bool(re.search(r"\b(?:test_df|test_features|hidden_test_df)\b", text))
