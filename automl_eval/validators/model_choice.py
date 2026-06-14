""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_ANN_PATTERNS = re.compile(
    r"(torch\.nn|nn\.Module|nn\.Linear|nn\.Sequential|"
    r"keras\.|tensorflow\.|tf\.keras|Dense\(|Conv[12]d\(|LSTM\(|"
    r"MLPClassifier|MLPRegressor|neural.net)",
    re.IGNORECASE,
)

_GOOD_TABULAR_MODELS = re.compile(
    r"(RandomForest|GradientBoosting|XGB|LightGBM|lgb\.|CatBoost|"
    r"ExtraTrees|AdaBoost|LogisticRegression|LinearRegression|"
    r"Ridge|Lasso|ElasticNet|SVM|SVC|SVR|KNeighbors|"
    r"DecisionTree|BaggingClassifier|BaggingRegressor|"
    r"VotingClassifier|VotingRegressor|StackingClassifier|StackingRegressor|"
    r"HistGradientBoosting)",
    re.IGNORECASE,
)

_ENSEMBLE_PATTERNS = re.compile(
    r"(VotingClassifier|VotingRegressor|StackingClassifier|StackingRegressor|"
    r"ensemble|blending|stacking|averaging.*model)",
    re.IGNORECASE,
)

_CUSTOM_MODEL = re.compile(
    r"(class\s+\w+.*:\s*\n.*def\s+(predict|forward|fit)\b)",
    re.DOTALL,
)

_SKLEARN_IMPORT = re.compile(
    r"(from\s+sklearn|import\s+sklearn|from\s+xgboost|from\s+lightgbm|"
    r"from\s+catboost|import\s+xgboost|import\s+lightgbm|import\s+catboost)",
    re.IGNORECASE,
)


class ModelChoiceValidator(BaseValidator):
    """Checks the adequacy of model choice for the task."""

    name = "model_choice"

    def __init__(
        self,
        ann_on_tabular_penalty: float = 0.12,
        custom_model_penalty: float = 0.08,
        no_model_penalty: float = 0.15,
        ensemble_bonus: float = 0.03,
    ) -> None:
        self.ann_on_tabular_penalty = ann_on_tabular_penalty
        self.custom_model_penalty = custom_model_penalty
        self.no_model_penalty = no_model_penalty
        self.ensemble_bonus = ensemble_bonus

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        all_code = ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (ActionType.CODE, ActionType.MODEL):
                all_code += "\n" + text

        if not re.search(r"\.fit\s*\(", all_code):
            if session.done:
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    score=0.0,
                    details="No model trained at all.",
                    penalty=self.no_model_penalty,
                )
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No model code yet.",
            )

        issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0

        uses_ann = bool(_ANN_PATTERNS.search(all_code))
        uses_good_model = bool(_GOOD_TABULAR_MODELS.search(all_code))
        uses_ensemble = bool(_ENSEMBLE_PATTERNS.search(all_code))
        writes_custom = bool(_CUSTOM_MODEL.search(all_code))
        uses_library = bool(_SKLEARN_IMPORT.search(all_code))

        # ── 1. ANN on tabular data ───────────────────────────────────
        if uses_ann and not uses_good_model:
            issues.append(
                "Using neural network on tabular data without trying "
                "tree-based models first — gradient boosting usually works better"
            )
            penalty += self.ann_on_tabular_penalty
        elif uses_ann and uses_good_model:
            bonuses.append("Tried both NN and tree-based models")

        # ── 2. Custom model from scratch ──────────────────────────────
        if writes_custom and not uses_library:
            issues.append(
                "Writing model from scratch instead of using established libraries — "
                "sklearn/xgboost/lightgbm are well-tested and typically outperform"
            )
            penalty += self.custom_model_penalty

        # ── 3. Good model choice ──────────────────────────────────────
        if uses_good_model:
            bonuses.append("Using proven tabular model")
        if uses_library:
            bonuses.append("Using established ML library")

        # ── 4. Ensemble ──────────────────────────────────────────────
        if uses_ensemble:
            bonuses.append("Ensemble/stacking attempted")

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if issues:
            parts.append("Issues: " + "; ".join(issues))
        details = ". ".join(parts) if parts else "Model choice looks reasonable."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
