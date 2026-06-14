"""TabReD loader: github.com/yandex-research/tabred."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TABRED_DATASETS: dict[str, str] = {
    "homesite-insurance": "Predict whether an insurance quote converts to a purchase (Homesite).",
    "ecom-offers": "Predict whether a customer redeems an e-commerce offer (Acquire Valued Shoppers).",
    "homecredit-default": "Predict loan default from Home Credit application + bureau features.",
    "sberbank-housing": "Predict (log) residential price per square metre (Sberbank).",
    "cooking-time": "Predict recipe cooking time (log-transformed).",
    "delivery-eta": "Predict delivery ETA (log-transformed).",
    "maps-routing": "Predict route travel time from routing features.",
    "weather": "Predict temperature from meteorological features.",
}

_TASKTYPE_MAP = {
    "binclass": "binary_classification",
    "multiclass": "multiclass_classification",
    "regression": "regression",
}
_SCORE_MAP = {
    "roc-auc": "roc_auc",
    "accuracy": "accuracy",
    "rmse": "rmse",
    "mae": "mae",
    "r2": "r2",
    "cross-entropy": "log_loss",
}


def available_datasets() -> list[str]:
    return list(TABRED_DATASETS.keys())


@dataclass
class _Loaded:
    df: pd.DataFrame
    target_col: str
    task_type: str
    metric: str


def _dataset_dir(tabred_root: Path, name: str) -> Path:
    """Resolve <root>/data/<name>, with a couple of graceful fallbacks."""
    for cand in (tabred_root / "data" / name, tabred_root / name):
        if (cand / "info.json").exists() or (cand / "Y.npy").exists():
            return cand
    return tabred_root / "data" / name


def _load_split_indices(folder: Path, split: str = "default") -> dict[str, np.ndarray]:
    sdir = folder / f"split-{split}"
    out = {}
    for part in ("train", "val", "test"):
        p = sdir / f"{part}_idx.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"TabReD split index missing: {p}. Run TabReD preprocessing "
                f"(python preprocessing/<script>.py) first; see docs/DATASETS.md."
            )
        out[part] = np.load(p, allow_pickle=False)
    return out


def _load_folder(folder: Path) -> _Loaded:
    info_path = folder / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"TabReD dataset not prepared at {folder} (no info.json). Run "
            f"`python preprocessing/<script>.py` in the TabReD repo first."
        )
    info = json.loads(info_path.read_text())
    raw_task = info.get("task_type", "regression")
    task_type = _TASKTYPE_MAP.get(raw_task, "regression")
    raw_score = info.get("score") or ("roc-auc" if raw_task == "binclass" else "rmse")
    metric = _SCORE_MAP.get(raw_score, "rmse")

    # Assemble the feature matrix from whichever blocks exist (NOT X_meta).
    columns: dict[str, np.ndarray] = {}
    for key, prefix in (("X_num", "num"), ("X_bin", "bin"), ("X_cat", "cat")):
        p = folder / f"{key}.npy"
        if p.exists():
            arr = np.load(p, allow_pickle=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            for j in range(arr.shape[1]):
                columns[f"{prefix}_{j:04d}"] = arr[:, j]
    if not columns:
        raise FileNotFoundError(
            f"TabReD {folder} has no X_num/X_bin/X_cat feature arrays."
        )

    y = np.load(folder / "Y.npy", allow_pickle=False).squeeze()
    df = pd.DataFrame(columns)
    df["target"] = y

    # Build the canonical (default) temporal split labels.
    idxs = _load_split_indices(folder, "default")
    labels = np.empty(len(df), dtype=object)
    labels[:] = None
    labels[idxs["train"]] = "train"
    labels[idxs["val"]] = "valid"
    labels[idxs["test"]] = "test"
    df["__split__"] = labels
    # Keep only rows that belong to the canonical split (some rows may be unused).
    df = df[df["__split__"].notna()].reset_index(drop=True)
    return _Loaded(df=df, target_col="target", task_type=task_type, metric=metric)


def build_task_json(
    tabred_root: str | Path,
    dataset_name: str,
    *,
    out_data_dir: str | Path = "data/tabred",
    out_task_dir: str | Path = "automl_eval/tasks",
    compute_reference: bool = True,
    max_rows: int | None = 60000,
) -> Path:
    """Prepare one TabReD dataset into a GRACE task JSON + CSV. Returns task JSON path."""
    if dataset_name not in TABRED_DATASETS:
        raise KeyError(
            f"Unknown TabReD dataset '{dataset_name}'. Available: {available_datasets()}"
        )
    tabred_root = Path(tabred_root)
    folder = _dataset_dir(tabred_root, dataset_name)
    loaded = _load_folder(folder)
    df = loaded.df

    if max_rows is not None and len(df) > max_rows:
        df = (
            df.groupby("__split__", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, round(len(g) * max_rows / len(df))), random_state=0
                )
            )
            .reset_index(drop=True)
        )

    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    safe = dataset_name.replace("-", "_")
    csv_path = out_data_dir / f"{safe}.csv"
    df.to_csv(csv_path, index=False)

    baseline_score = oracle_score = None
    if compute_reference:
        from automl_eval.dataset_loaders.reference import compute_reference_scores

        ref_df = df[df["__split__"].isin(["train", "valid"])].drop(
            columns=["__split__"]
        )
        ref = compute_reference_scores(
            ref_df,
            target_column=loaded.target_col,
            task_type=loaded.task_type,
            metric=loaded.metric,
        )
        baseline_score, oracle_score = ref.baseline_score, ref.oracle_score

    task = {
        "task_id": f"tabred_{safe}",
        "dataset_path": str(csv_path),
        "target_column": loaded.target_col,
        "task_type": loaded.task_type,
        "metric": loaded.metric,
        "description": TABRED_DATASETS[dataset_name]
        + " Industry tabular data with an official "
        "temporal train/valid/test split (TabReD, NeurIPS 2024 D&B).",
        "plan_checklist": [],
        "time_budget_seconds": 1800.0,
        "max_steps": 30,
        "oracle_score": oracle_score,
        "baseline_score": baseline_score,
        "metadata": {
            "dataset_source": f"tabred/{dataset_name}",
            "suite": "TabReD",
            "citation": "Rubachev et al., TabReD, NeurIPS 2024 Datasets & Benchmarks (arXiv 2406.19380)",
            "time_series_task": False,
            "evaluation_mode": "predefined_temporal_split",
            "temporal_split_note": "Rows are i.i.d. tabular records; the official split is temporal (distribution shift).",
            "contamination_note": "Industry dataset, far less common than UCI/OpenML; low LLM-memorisation risk.",
            "n_rows_used": int(len(df)),
            "n_features": int(df.shape[1] - 2),
        },
        "split_strategy": {
            "method": "predefined",
            "split_column": "__split__",
            "train_values": ["train"],
            "valid_values": ["valid"],
            "test_values": ["test"],
            "drop_split_columns": True,
            "rationale": "Honour TabReD's official temporal train/valid/test split.",
        },
    }
    out_task_dir = Path(out_task_dir)
    out_task_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_task_dir / f"tabred_{safe}.json"
    task_path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_path


def build_all(
    tabred_root: str | Path, *, only: list[str] | None = None, **kwargs
) -> list[Path]:
    """Prepare every prepared TabReD dataset found under ``tabred_root``."""
    names = only or available_datasets()
    written: list[Path] = []
    for name in names:
        try:
            written.append(build_task_json(tabred_root, name, **kwargs))
        except FileNotFoundError as exc:
            print(f"  [tabred] SKIP {name}: {exc}")
    return written
