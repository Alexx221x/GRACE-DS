"""
Metrics — computation of ML metrics for a task.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from automl_eval.domain.task import MetricName


def compute_metric(
    metric: MetricName,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the metric; returns float (higher is better for all except loss).

    "higher = better"
    """
    if metric == MetricName.ROC_AUC:
        return float(roc_auc_score(y_true, y_pred))
    if metric == MetricName.ACCURACY:
        return float(
            accuracy_score(
                y_true, (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            )
        )
    if metric == MetricName.F1:
        return float(
            f1_score(
                y_true,
                (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred,
                average="macro",
            )
        )
    if metric == MetricName.LOG_LOSS:
        return float(-log_loss(y_true, y_pred))
    if metric == MetricName.RMSE:
        return float(-np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == MetricName.MAE:
        return float(-mean_absolute_error(y_true, y_pred))
    if metric == MetricName.R2:
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")


def normalize_score(
    raw: float,
    baseline: float | None,
    oracle: float | None,
) -> float:
    """Normalize the metric to [0, 1] relative to baseline and oracle."""
    if baseline is None or oracle is None:
        return max(0.0, min(1.0, raw))

    if oracle == baseline:
        return 1.0 if raw >= oracle else 0.0

    normalized = (raw - baseline) / (oracle - baseline)
    return max(0.0, min(1.0, normalized))
