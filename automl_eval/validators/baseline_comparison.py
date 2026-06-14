"""Terminal same-split comparison against an evaluator-owned simple baseline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np

from automl_eval.domain.task import MetricName, TaskType
from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

logger = logging.getLogger(__name__)
_PLATEAU_WINDOW = 3
_PLATEAU_THRESHOLD = 0.008


class BaselineComparisonValidator(BaseValidator):
    name = "baseline_comparison"

    def __init__(
        self,
        worse_than_baseline_penalty: float = 0.08,
        better_bonus: float = 0.05,
        plateau_penalty: float = 0.03,
    ) -> None:
        self.worse_than_baseline_penalty = worse_than_baseline_penalty
        self.better_bonus = better_bonus
        self.plateau_penalty = plateau_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if (
            not session.done
            or not session.final_submitted
            or session.hidden_test_metric is None
        ):
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Baseline comparison runs after successful terminal hidden-test scoring only.",
            )
        baseline_score = self._compute_baseline(session)
        agent_score = session.hidden_test_metric
        issues: list[str] = []
        positives: list[str] = []
        penalty = 0.0
        bonus = 0.0
        if baseline_score is None:
            issues.append("Same-split terminal baseline score is unavailable")
            penalty += self.worse_than_baseline_penalty
        else:
            diff = agent_score - baseline_score
            if diff > _PLATEAU_THRESHOLD:
                positives.append(
                    f"terminal agent ({agent_score:.4f}) beats same-split baseline ({baseline_score:.4f}) by +{diff:.4f}"
                )
                bonus += self.better_bonus
            elif diff < -_PLATEAU_THRESHOLD:
                issues.append(
                    f"terminal agent ({agent_score:.4f}) is worse than same-split baseline ({baseline_score:.4f}) by {diff:.4f}"
                )
                penalty += self.worse_than_baseline_penalty
            else:
                positives.append(
                    f"terminal agent ({agent_score:.4f}) approximately matches same-split baseline ({baseline_score:.4f})"
                )
        plateau_msg = self._check_plateau(session)
        if plateau_msg:
            issues.append(plateau_msg)
            penalty += self.plateau_penalty
        details = []
        if positives:
            details.append("Good: " + ", ".join(positives))
        if issues:
            details.append("Issues: " + "; ".join(issues))
        return ValidationResult(
            validator_name=self.name,
            passed=not issues,
            score=max(0.0, min(1.0, 1.0 - penalty + bonus)),
            details=". ".join(details)
            if details
            else "Baseline comparison not applicable.",
            penalty=penalty,
        )

    def _compute_baseline(self, session: RuntimeSession) -> float | None:
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )
            from automl_eval.evaluation.metrics import compute_metric

            train_df, test_df = session.train_df, session.hidden_test_df
            if train_df is None or test_df is None:
                return None
            target = session.task.target_column
            num_cols = (
                train_df.drop(columns=[target])
                .select_dtypes(include="number")
                .columns.tolist()
            )
            if not num_cols:
                return None
            X_tr = train_df[num_cols].fillna(0).values
            y_tr = train_df[target].values
            X_te = test_df[num_cols].fillna(0).values
            y_te = test_df[target].values
            if session.task.task_type != TaskType.REGRESSION:
                model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=session.seed
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=session.seed
                )
            model.fit(X_tr, y_tr)
            if (
                session.task.task_type != TaskType.REGRESSION
                and session.task.metric in {MetricName.ROC_AUC, MetricName.LOG_LOSS}
            ):
                proba = np.asarray(model.predict_proba(X_te))
                pred = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            else:
                pred = np.asarray(model.predict(X_te))
            return compute_metric(session.task.metric, np.asarray(y_te), pred)
        except Exception as exc:
            logger.debug("Terminal baseline computation failed: %s", exc)
            return None

    def _check_plateau(self, session: RuntimeSession) -> str | None:
        if len(session.metric_history) < _PLATEAU_WINDOW:
            return None
        recent = [metric for _, metric in session.metric_history[-_PLATEAU_WINDOW:]]
        spread = max(recent) - min(recent)
        if spread < _PLATEAU_THRESHOLD and session.cycle_count > 2:
            return f"Validation plateau: last {_PLATEAU_WINDOW} values spread={spread:.4f} (<{_PLATEAU_THRESHOLD}) after {session.cycle_count} cycles"
        return None
