"""Evaluator-side heuristic diagnostics for target leakage and severe overfit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from automl_eval.evaluation.submission import resolve_submission, score_bundle
from automl_eval.domain.task import TaskType
from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

logger = logging.getLogger(__name__)
_FEATURE_DF_NAMES = ["X_train", "X", "features", "df_train", "train_processed"]
_TARGET_NAMES = ["y_train", "y"]


class TargetLeakageModelValidator(BaseValidator):
    name = "target_leakage_model"

    def __init__(
        self,
        single_feature_threshold: float = 0.99,
        train_valid_gap_threshold: float = 0.30,
        leakage_penalty: float = 0.25,
        overfit_penalty: float = 0.10,
    ) -> None:
        self.single_feature_threshold = single_feature_threshold
        self.train_valid_gap_threshold = train_valid_gap_threshold
        self.leakage_penalty = leakage_penalty
        self.overfit_penalty = overfit_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.done or not session.final_submitted:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Model-based leakage diagnostics run after successful terminal submission only.",
            )
        issues: list[str] = []
        penalty = 0.0
        leak_features = self._single_feature_check(session)
        if leak_features:
            issues.append(
                "Suspicious single-feature predictiveness: "
                + ", ".join(
                    f"{name} (score={score:.3f})" for name, score in leak_features
                )
            )
            penalty += self.leakage_penalty
        gap_issue = self._train_valid_gap_check(session)
        if gap_issue:
            issues.append(gap_issue)
            penalty += self.overfit_penalty
        return ValidationResult(
            validator_name=self.name,
            passed=not issues,
            score=max(0.0, 1.0 - penalty),
            details="; ".join(issues)
            if issues
            else "No model-based leakage or severe overfit signal detected.",
            penalty=penalty,
        )

    def _single_feature_check(self, session: RuntimeSession) -> list[tuple[str, float]]:
        from sklearn.metrics import r2_score, roc_auc_score
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        X_df = self._find_features(session.sandbox_namespace, session)
        y = self._find_target(session.sandbox_namespace, session)
        if X_df is None or y is None:
            return []
        X_numeric = X_df.select_dtypes(include="number").dropna(axis=1)
        if X_numeric.empty or len(X_numeric) < 10:
            return []
        common_index = X_numeric.index.intersection(y.index)
        X_numeric, y = X_numeric.loc[common_index], y.loc[common_index]
        leaky: list[tuple[str, float]] = []
        for column in X_numeric.columns:
            x = X_numeric[[column]].values
            mask = ~np.isnan(x.ravel())
            if mask.sum() < 10:
                continue
            try:
                if session.task.task_type != TaskType.REGRESSION:
                    model = DecisionTreeClassifier(
                        max_depth=2, random_state=session.seed
                    ).fit(x[mask], y.values[mask])
                    proba = model.predict_proba(x[mask])
                    score = (
                        roc_auc_score(y.values[mask], proba[:, 1])
                        if proba.shape[1] == 2
                        else roc_auc_score(
                            y.values[mask], proba, multi_class="ovr", average="macro"
                        )
                    )
                else:
                    model = DecisionTreeRegressor(
                        max_depth=2, random_state=session.seed
                    ).fit(x[mask], y.values[mask])
                    score = r2_score(y.values[mask], model.predict(x[mask]))
                if score >= self.single_feature_threshold:
                    leaky.append((column, float(score)))
            except Exception:
                continue
        return leaky

    def _train_valid_gap_check(self, session: RuntimeSession) -> str | None:
        """Evaluate the formal raw-input artefact on evaluator-owned train/dev labels."""
        bundle = resolve_submission(session.sandbox_namespace)
        if bundle is None or session.train_df is None or session.private_dev_df is None:
            return None
        try:
            train_score, _ = score_bundle(bundle, session.train_df, session.task)
            valid_score, _ = score_bundle(bundle, session.private_dev_df, session.task)
            gap = train_score - valid_score
            if train_score > 0.95 and gap > self.train_valid_gap_threshold:
                return f"Train/private-validation gap: train_score={train_score:.3f}, valid_score={valid_score:.3f} (gap={gap:.3f}) — possible overfit or leakage"
        except Exception as exc:
            logger.debug("Evaluator-side train/valid gap check failed: %s", exc)
        return None

    def _find_features(
        self, namespace: dict, session: RuntimeSession
    ) -> pd.DataFrame | None:
        target = session.task.target_column
        for name in _FEATURE_DF_NAMES:
            obj = namespace.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                return obj.drop(columns=[target], errors="ignore")
        train = namespace.get("train_df")
        return (
            train.drop(columns=[target], errors="ignore")
            if isinstance(train, pd.DataFrame)
            else None
        )

    def _find_target(
        self, namespace: dict, session: RuntimeSession
    ) -> pd.Series | None:
        for name in _TARGET_NAMES:
            obj = namespace.get(name)
            if isinstance(obj, np.ndarray) and len(obj) > 0:
                return pd.Series(obj)
            if isinstance(obj, pd.Series) and len(obj) > 0:
                return obj
        train = namespace.get("train_df")
        target = session.task.target_column
        return (
            train[target]
            if isinstance(train, pd.DataFrame) and target in train.columns
            else None
        )
