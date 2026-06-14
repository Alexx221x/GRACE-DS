"""Compute baseline_score and oracle_score for a prepared dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReferenceScores:
    baseline_score: float
    oracle_score: float
    metric: str
    n_rows: int
    n_features: int
    note: str = ""


def _split(
    df: pd.DataFrame, target: str, *, test_frac: float, stratify: bool, seed: int
):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=strat
    )


def _build_reference_model(X: pd.DataFrame, task_type: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=50,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    if task_type == "regression":
        est = HistGradientBoostingRegressor(random_state=0)
    else:
        est = HistGradientBoostingClassifier(random_state=0)
    return Pipeline([("pre", pre), ("est", est)])


def compute_reference_scores(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    metric: str,
    test_frac: float = 0.2,
    seed: int = 0,
) -> ReferenceScores:
    """Compute (baseline, oracle) reference scores for normalization."""
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        f1_score,
        log_loss,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    is_reg = task_type == "regression"
    X_tr, X_te, y_tr, y_te = _split(
        df,
        target_column,
        test_frac=test_frac,
        stratify=(not is_reg),
        seed=seed,
    )

    if is_reg:
        const = float(np.mean(y_tr))
        y_pred_base = np.full(len(y_te), const)
        baseline = _score_regression(
            metric,
            y_te,
            y_pred_base,
            roc=None,
            rmse=mean_squared_error,
            mae=mean_absolute_error,
            r2=r2_score,
        )
    else:
        majority = y_tr.value_counts().idxmax()
        if metric == "roc_auc":
            baseline = 0.5
        elif metric == "accuracy":
            baseline = float((y_te == majority).mean())
        elif metric == "f1":
            y_pred_base = np.full(len(y_te), majority)
            baseline = float(
                f1_score(y_te, y_pred_base, average="macro", zero_division=0)
            )
        elif metric == "log_loss":
            classes = sorted(y_tr.unique())
            rates = np.array([float((y_tr == c).mean()) for c in classes])
            proba = np.tile(rates, (len(y_te), 1))
            baseline = float(log_loss(y_te, proba, labels=classes))
        else:
            baseline = float("nan")

    model = _build_reference_model(X_tr, task_type)
    model.fit(X_tr, y_tr)
    if is_reg:
        y_pred = model.predict(X_te)
        oracle = _score_regression(
            metric,
            y_te,
            y_pred,
            roc=None,
            rmse=mean_squared_error,
            mae=mean_absolute_error,
            r2=r2_score,
        )
    else:
        if metric == "roc_auc":
            if len(np.unique(y_tr)) == 2:
                proba = model.predict_proba(X_te)[:, 1]
                oracle = float(roc_auc_score(y_te, proba))
            else:
                proba = model.predict_proba(X_te)
                oracle = float(roc_auc_score(y_te, proba, multi_class="ovr"))
        elif metric == "accuracy":
            oracle = float(accuracy_score(y_te, model.predict(X_te)))
        elif metric == "f1":
            oracle = float(
                f1_score(y_te, model.predict(X_te), average="macro", zero_division=0)
            )
        elif metric == "log_loss":
            classes = sorted(y_tr.unique())
            oracle = float(log_loss(y_te, model.predict_proba(X_te), labels=classes))
        else:
            oracle = float("nan")

    if metric in ("rmse", "mae", "log_loss"):
        if np.isfinite(baseline):
            baseline = -baseline
        if np.isfinite(oracle):
            oracle = -oracle

    return ReferenceScores(
        baseline_score=round(float(baseline), 4),
        oracle_score=round(float(oracle), 4),
        metric=metric,
        n_rows=len(df),
        n_features=df.shape[1] - 1,
        note="oracle=untuned HistGradientBoosting on raw features; baseline=trivial predictor",
    )


def _score_regression(metric, y_true, y_pred, *, roc, rmse, mae, r2) -> float:
    if metric == "rmse":
        return float(np.sqrt(rmse(y_true, y_pred)))
    if metric == "mae":
        return float(mae(y_true, y_pred))
    if metric == "r2":
        return float(r2(y_true, y_pred))
    return float("nan")
