""""""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

logger = logging.getLogger(__name__)

_IMPORTANCE_PATTERNS = re.compile(
    r"(feature_importances_|permutation_importance|\.coef_|"
    r"shap\.|SHAP|eli5|\.feature_importance|plot_importance|"
    r"важност|feature.ranking)",
    re.IGNORECASE,
)

_FEATURE_DF_NAMES = ["X_train", "X", "features", "df_train", "train_processed"]


class FeatureImportanceValidator(BaseValidator):
    """Checks feature importance analysis and model value."""

    name = "feature_importance"

    def __init__(
        self,
        no_inspection_penalty: float = 0.04,
        model_value_bonus: float = 0.05,
        new_feature_bonus: float = 0.03,
    ) -> None:
        self.no_inspection_penalty = no_inspection_penalty
        self.model_value_bonus = model_value_bonus
        self.new_feature_bonus = new_feature_bonus

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.done:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Feature importance check runs at FINAL_SUBMIT.",
            )

        from automl_eval.core.session import ActionType

        issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0
        bonus = 0.0

        all_code = ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (
                ActionType.CODE,
                ActionType.MODEL,
                ActionType.FEATURE_ENGINEERING,
            ):
                all_code += "\n" + text

        plan_text = session.plan_text or ""

        # ── 1. Does agent inspect feature importance? ─────────────────
        inspects_importance = bool(
            _IMPORTANCE_PATTERNS.search(all_code)
            or _IMPORTANCE_PATTERNS.search(plan_text)
        )
        if not inspects_importance:
            issues.append("No feature importance inspection found")
            penalty += self.no_inspection_penalty

        # ── 2. Feature value: new features in top importances ─────────
        fi_result = self._check_feature_importances(session)
        if fi_result:
            bonuses.append(fi_result)
            bonus += self.new_feature_bonus

        # ── 3. Model value test (universal) ───────────────────────────
        mv_result = self._model_value_test(session)
        if mv_result is not None and mv_result > 0.005:
            bonuses.append(
                f"Model value test: +{mv_result:.4f} metric gain with agent predictions"
            )
            bonus += self.model_value_bonus

        score = max(0.0, min(1.0, 1.0 - penalty + bonus))
        passed = len(issues) == 0

        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if issues:
            parts.append("Issues: " + "; ".join(issues))
        details = ". ".join(parts) if parts else "Feature importance analysis adequate."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _check_feature_importances(self, session: RuntimeSession) -> str | None:
        """If the model has feature_importances_, check newly created features."""
        ns = session.sandbox_namespace
        model = None
        for _n in ("model", "clf", "estimator"):
            _obj = ns.get(_n)
            if _obj is not None and hasattr(_obj, "predict"):
                model = _obj
                break
        if model is None:
            return None

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None

        X_df = self._find_features(ns)
        if X_df is None:
            return None

        if len(importances) != len(X_df.columns):
            return None

        insights = session.data_insights
        if insights is None:
            return None

        original_cols = set(insights.numeric_columns + insights.categorical_columns)
        current_cols = list(X_df.columns)
        new_cols = [c for c in current_cols if c not in original_cols]

        if not new_cols:
            return None

        fi_series = pd.Series(importances, index=current_cols).sort_values(
            ascending=False
        )
        top_k = max(3, len(fi_series) // 3)
        top_features = set(fi_series.head(top_k).index)

        important_new = [c for c in new_cols if c in top_features]
        if important_new:
            return f"New features in top importance: {', '.join(important_new)}"

        return None

    def _model_value_test(self, session: RuntimeSession) -> float | None:
        """"""
        if session.predictions is None or session.test_df is None:
            return None

        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )
            from automl_eval.evaluation.metrics import compute_metric
            from automl_eval.domain.task import TaskType

            target = session.task.target_column
            train_df = session.train_df
            valid_df = session.valid_df
            if train_df is None or valid_df is None:
                return None

            num_cols = (
                train_df.drop(columns=[target])
                .select_dtypes(include="number")
                .columns.tolist()
            )
            if len(num_cols) < 2:
                return None

            X_tr = train_df[num_cols].fillna(0).values
            y_tr = train_df[target].values
            X_val = valid_df[num_cols].fillna(0).values
            y_val = valid_df[target].values

            is_clf = session.task.task_type != TaskType.REGRESSION
            if is_clf:
                baseline_model = GradientBoostingClassifier(
                    n_estimators=30, max_depth=3, random_state=42
                )
            else:
                baseline_model = GradientBoostingRegressor(
                    n_estimators=30, max_depth=3, random_state=42
                )

            baseline_model.fit(X_tr, y_tr)
            if is_clf and hasattr(baseline_model, "predict_proba"):
                base_preds = baseline_model.predict_proba(X_val)
                if base_preds.shape[1] == 2:
                    base_preds = base_preds[:, 1]
            else:
                base_preds = baseline_model.predict(X_val)

            base_score = compute_metric(session.task.metric, y_val, base_preds)

            ns = session.sandbox_namespace
            agent_model = None
            for _n in ("model", "clf", "estimator"):
                _obj = ns.get(_n)
                if _obj is not None and hasattr(_obj, "predict"):
                    agent_model = _obj
                    break
            if agent_model is None:
                return None

            try:
                if is_clf and hasattr(agent_model, "predict_proba"):
                    agent_tr_preds = agent_model.predict_proba(
                        train_df[num_cols].fillna(0).values
                    )
                    agent_val_preds = agent_model.predict_proba(
                        valid_df[num_cols].fillna(0).values
                    )
                    if agent_tr_preds.shape[1] == 2:
                        agent_tr_preds = agent_tr_preds[:, 1]
                        agent_val_preds = agent_val_preds[:, 1]
                else:
                    agent_tr_preds = agent_model.predict(
                        train_df[num_cols].fillna(0).values
                    )
                    agent_val_preds = agent_model.predict(
                        valid_df[num_cols].fillna(0).values
                    )
            except Exception:
                return None

            if agent_tr_preds.ndim > 1:
                agent_tr_preds = agent_tr_preds.ravel()
                agent_val_preds = agent_val_preds.ravel()

            X_tr_aug = np.column_stack([X_tr, agent_tr_preds[: len(X_tr)]])
            X_val_aug = np.column_stack([X_val, agent_val_preds[: len(X_val)]])

            if is_clf:
                stacked_model = GradientBoostingClassifier(
                    n_estimators=30, max_depth=3, random_state=42
                )
            else:
                stacked_model = GradientBoostingRegressor(
                    n_estimators=30, max_depth=3, random_state=42
                )
            stacked_model.fit(X_tr_aug, y_tr)

            if is_clf and hasattr(stacked_model, "predict_proba"):
                stacked_preds = stacked_model.predict_proba(X_val_aug)
                if stacked_preds.shape[1] == 2:
                    stacked_preds = stacked_preds[:, 1]
            else:
                stacked_preds = stacked_model.predict(X_val_aug)

            stacked_score = compute_metric(session.task.metric, y_val, stacked_preds)
            return stacked_score - base_score

        except Exception as e:
            logger.debug("Model value test failed: %s", e)
            return None

    def _find_features(self, ns: dict) -> pd.DataFrame | None:
        for name in _FEATURE_DF_NAMES:
            obj = ns.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                return obj
        return None
