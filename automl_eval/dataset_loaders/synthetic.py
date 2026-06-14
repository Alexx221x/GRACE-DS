"""Synthetic dataset loaders with a known data-generating process."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_classification_task(
    *,
    n_samples: int = 10000,
    n_features: int = 20,
    n_informative: int = 5,
    n_redundant: int = 5,
    flip_y: float = 0.02,
    class_sep: float = 0.8,
    seed: int = 42,
    out_data_dir: str | Path = "data/synthetic",
    out_task_dir: str | Path = "automl_eval/tasks",
    compute_reference: bool = True,
) -> Path:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=None,
        flip_y=flip_y,
        class_sep=class_sep,
        shuffle=False,
        random_state=seed,
    )
    cols = [f"feat_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    gt_informative = cols[:n_informative]
    gt_redundant = cols[n_informative : n_informative + n_redundant]

    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_data_dir / f"synthetic_classification_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    baseline_score = oracle_score = None
    if compute_reference:
        from automl_eval.dataset_loaders.reference import compute_reference_scores

        ref = compute_reference_scores(
            df,
            target_column="target",
            task_type="binary_classification",
            metric="roc_auc",
        )
        baseline_score, oracle_score = ref.baseline_score, ref.oracle_score

    task = {
        "task_id": "synthetic_classification_known_dgp",
        "dataset_path": str(csv_path),
        "target_column": "target",
        "task_type": "binary_classification",
        "metric": "roc_auc",
        "description": (
            "Synthetic binary classification with a known data-generating process "
            f"({n_informative} informative + {n_redundant} redundant + "
            f"{n_features - n_informative - n_redundant} noise features). Memorisation is "
            "impossible by construction; ground-truth informative features are recorded for "
            "feature-relevance evaluation."
        ),
        "plan_checklist": [],
        "time_budget_seconds": 300.0,
        "max_steps": 30,
        "oracle_score": oracle_score,
        "baseline_score": baseline_score,
        "metadata": {
            "dataset_source": "sklearn.make_classification",
            "suite": "synthetic",
            "dgp": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": n_informative,
                "n_redundant": n_redundant,
                "flip_y": flip_y,
                "class_sep": class_sep,
                "seed": seed,
            },
            "ground_truth_informative_features": gt_informative,
            "ground_truth_redundant_features": gt_redundant,
            "time_series_task": False,
            "contamination_note": "Generated on-the-fly; zero memorisation risk.",
        },
        "split_strategy": {
            "method": "random",
            "train_size": 0.7,
            "valid_size": 0.15,
            "test_size": 0.15,
            "stratify": True,
            "drop_split_columns": True,
            "rationale": "Stratified holdout for synthetic-DGP classification.",
        },
    }
    out_task_dir = Path(out_task_dir)
    out_task_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_task_dir / "synthetic_classification_known_dgp.json"
    task_path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_path


def build_regression_task(
    *,
    n_samples: int = 10000,
    n_features: int = 15,
    n_informative: int = 3,
    noise: float = 0.5,
    seed: int = 42,
    out_data_dir: str | Path = "data/synthetic",
    out_task_dir: str | Path = "automl_eval/tasks",
    compute_reference: bool = True,
) -> Path:
    from sklearn.datasets import make_regression

    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        shuffle=False,
        random_state=seed,
    )
    cols = [f"feat_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    informative_idx = [i for i, c in enumerate(coef) if abs(c) > 1e-8]
    gt_informative = [cols[i] for i in informative_idx]

    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_data_dir / f"synthetic_regression_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    baseline_score = oracle_score = None
    if compute_reference:
        from automl_eval.dataset_loaders.reference import compute_reference_scores

        ref = compute_reference_scores(
            df,
            target_column="target",
            task_type="regression",
            metric="rmse",
        )
        baseline_score, oracle_score = ref.baseline_score, ref.oracle_score

    task = {
        "task_id": "synthetic_regression_known_dgp",
        "dataset_path": str(csv_path),
        "target_column": "target",
        "task_type": "regression",
        "metric": "rmse",
        "description": (
            "Synthetic regression with a known linear data-generating process "
            f"({n_informative} informative features with known coefficients out of {n_features}). "
            "Memorisation is impossible by construction; ground-truth informative features are recorded."
        ),
        "plan_checklist": [],
        "time_budget_seconds": 300.0,
        "max_steps": 30,
        "oracle_score": oracle_score,
        "baseline_score": baseline_score,
        "metadata": {
            "dataset_source": "sklearn.make_regression",
            "suite": "synthetic",
            "dgp": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": n_informative,
                "noise": noise,
                "seed": seed,
                "true_coefficients": [float(c) for c in coef],
            },
            "ground_truth_informative_features": gt_informative,
            "time_series_task": False,
            "contamination_note": "Generated on-the-fly; zero memorisation risk.",
        },
        "split_strategy": {
            "method": "random",
            "train_size": 0.7,
            "valid_size": 0.15,
            "test_size": 0.15,
            "stratify": False,
            "drop_split_columns": True,
            "rationale": "Random holdout for synthetic-DGP regression.",
        },
    }
    out_task_dir = Path(out_task_dir)
    out_task_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_task_dir / "synthetic_regression_known_dgp.json"
    task_path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_path
