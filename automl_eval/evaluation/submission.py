"""Strict replayable submission bundles evaluated only on evaluator-owned raw frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from automl_eval.evaluation.metrics import compute_metric
from automl_eval.domain.task import MetricName, Task

_PIPELINE_NAMES = ("submission_pipeline", "pipeline")
_PROBABILITY_METRICS = frozenset({MetricName.ROC_AUC, MetricName.LOG_LOSS})


@dataclass
class SubmissionBundle:
    kind: str
    name: str
    predict: Callable[[pd.DataFrame], Any]
    estimator: Any | None = None


def resolve_submission(namespace: dict[str, Any]) -> SubmissionBundle | None:
    """Accept only the formal callback or a fitted sklearn raw-input pipeline."""
    callback = namespace.get("predict_fn")
    if callable(callback):
        return SubmissionBundle(kind="predict_fn", name="predict_fn", predict=callback)
    for name in _PIPELINE_NAMES:
        candidate = namespace.get(name)
        if isinstance(candidate, Pipeline) and hasattr(candidate, "predict"):
            return SubmissionBundle(
                kind="raw_input_pipeline",
                name=name,
                predict=lambda frame, fitted=candidate: fitted.predict(frame),
                estimator=candidate,
            )
    return None


def diagnose_missing_submission(namespace: dict[str, Any]) -> str:
    """Return a concrete reason why ``resolve_submission`` returned ``None``."""
    callback = namespace.get("predict_fn")
    if callback is not None and not callable(callback):
        return (
            f"`predict_fn` is defined but is not callable (got {type(callback).__name__}). "
            "Bind it to a function that takes a raw DataFrame and returns predictions, e.g. "
            "`def predict_fn(raw_dataframe): return pipeline.predict_proba(raw_dataframe)[:, 1]`."
        )
    for name in _PIPELINE_NAMES:
        candidate = namespace.get(name)
        if candidate is None:
            continue
        if not isinstance(candidate, Pipeline):
            return (
                f"`{name}` is defined but is not an sklearn Pipeline (got "
                f"{type(candidate).__name__}). Wrap your preprocessing and "
                "estimator inside `Pipeline([...])` and assign it to `pipeline`."
            )
        # Two specific Pipeline pathologies:
        # 1. Last step is a transformer (no predict).
        if not hasattr(candidate, "predict"):
            last_step = candidate.steps[-1][1] if candidate.steps else None
            last_step_kind = (
                type(last_step).__name__ if last_step is not None else "<empty>"
            )
            return (
                f"`{name}` ends with `{last_step_kind}` which has no `predict` method. "
                "Append a fitted classifier or regressor as the final step, e.g. "
                "`('classifier', RandomForestClassifier(random_state=42))`, then refit on training data."
            )
        # 2. Pipeline has a predict step but was never fit (final estimator missing fitted attrs).
        final_estimator = candidate.steps[-1][1] if candidate.steps else None
        if final_estimator is not None and not _looks_fitted(final_estimator):
            return (
                f"`{name}` is constructed but does not look fit (its final estimator has no fitted "
                "attributes). Call `pipeline.fit(X_train, y_train)` before validation."
            )
    if any(name in namespace for name in _PIPELINE_NAMES) or "predict_fn" in namespace:
        return "A submission-shaped name exists but is not usable; expose `predict_fn(raw_dataframe)` or a fitted `pipeline`."
    return (
        "No `predict_fn` callable and no fitted `pipeline` in the workspace. "
        "Create either: a fitted sklearn `Pipeline` named `pipeline` that includes "
        "preprocessing + a final estimator with `predict`, or a function `predict_fn(raw_dataframe)`."
    )


def _looks_fitted(estimator: Any) -> bool:
    """Heuristic: an sklearn estimator is considered fit if it has any trailing-underscore attribute."""
    try:
        attrs = vars(estimator)
    except TypeError:
        return True  # Cannot inspect; do not flag false positives.
    return any(name.endswith("_") and not name.startswith("_") for name in attrs)


def predict_for_metric(
    bundle: SubmissionBundle, raw_features: pd.DataFrame, task: Task
) -> np.ndarray:
    if (
        bundle.estimator is not None
        and task.metric in _PROBABILITY_METRICS
        and hasattr(bundle.estimator, "predict_proba")
    ):
        output = np.asarray(bundle.estimator.predict_proba(raw_features))
        if output.ndim == 2 and output.shape[1] == 2:
            return output[:, 1]
        return output
    output = np.asarray(bundle.predict(raw_features))
    if (
        output.ndim == 2
        and output.shape[1] == 2
        and task.metric in _PROBABILITY_METRICS
    ):
        output = output[:, 1]
    return output.reshape(-1) if output.ndim == 1 else output


def score_bundle(
    bundle: SubmissionBundle, frame: pd.DataFrame, task: Task
) -> tuple[float, np.ndarray]:
    raw_features = frame.drop(columns=[task.target_column])
    truth = frame[task.target_column].to_numpy()
    predictions = predict_for_metric(bundle, raw_features, task)
    if len(predictions) != len(truth):
        raise ValueError(
            f"prediction length mismatch: got {len(predictions)}, expected {len(truth)}"
        )
    score = compute_metric(task.metric, truth, predictions)
    return score, np.asarray(predictions)
