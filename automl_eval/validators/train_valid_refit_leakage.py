""""""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


# Merge expressions: ``pd.concat([... train ... valid ...])`` or
_TRAIN_NAMES = r"(?:train_df|train_df_original)"
_VALID_NAMES = r"(?:valid_df|valid_df_original|validation_df)"

# Combined "concat([train, valid])" or "concat([valid, train])".
_CONCAT_MERGE = re.compile(
    rf"(?:pd\.)?concat\s*\(\s*\[\s*"
    rf"(?:{_TRAIN_NAMES}|{_VALID_NAMES})"
    rf"[^\]]*?"
    rf"(?:{_VALID_NAMES}|{_TRAIN_NAMES})"
    rf"[^\]]*?\]",
    re.IGNORECASE | re.DOTALL,
)
# Append shape: train_df.append(valid_df) or valid_df.append(train_df).
_APPEND_MERGE = re.compile(
    rf"\b(?:{_TRAIN_NAMES}|{_VALID_NAMES})\s*\.\s*append\s*\(\s*"
    rf"(?:{_TRAIN_NAMES}|{_VALID_NAMES})\b",
    re.IGNORECASE,
)
# A subsequent fit/fit_transform/fit_predict.  We do not require the merge
_FIT_CALL = re.compile(r"\.fit(?:_transform|_predict)?\s*\(", re.IGNORECASE)

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*(?:\r?\n)?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code_only(text: str) -> str:
    """Mirror LeakageValidator._extract_code_only — keep only Python code."""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return "\n".join(b.strip() for b in blocks if b.strip())
    return ""


class TrainValidRefitLeakageValidator(BaseValidator):
    """Flag the train+valid refit-leakage pattern as a CRITICAL error."""

    name = "train_valid_refit_leakage"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        issues: list[str] = []

        for step in session.steps:
            # FINAL_SUBMIT itself never executes new code (it just terminates).
            if step.action_type.value == "FINAL_SUBMIT":
                continue

            raw = step.code_body if step.code_body else step.action_text
            text = _extract_code_only(raw)
            if not text:
                continue

            has_concat = bool(_CONCAT_MERGE.search(text))
            has_append = bool(_APPEND_MERGE.search(text))
            if not (has_concat or has_append):
                continue
            if not _FIT_CALL.search(text):
                # Merge without subsequent fit is not leakage; it's just data
                continue
            shape = "concat" if has_concat else "append"
            issues.append(
                f"Step {step.step_idx}: train + valid frames merged via "
                f"{shape}(...) and then .fit*() was called -- this overfits "
                f"the final model to the validation labels and is treated as "
                f"a critical methodological error."
            )

        if issues:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues),
                penalty=0.0,
            )
        return ValidationResult(
            validator_name=self.name,
            passed=True,
            score=1.0,
            details="No train+valid refit leakage detected.",
        )
