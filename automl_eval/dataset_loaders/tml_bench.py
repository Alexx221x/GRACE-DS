"""TML-bench loader: github.com/mykolapinchuk/tml-bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_TASKTYPE_MAP = {
    "binary": "binary_classification",
    "binclass": "binary_classification",
    "binary_classification": "binary_classification",
    "multiclass": "multiclass_classification",
    "regression": "regression",
}
_METRIC_MAP = {
    "auc": "roc_auc",
    "roc_auc": "roc_auc",
    "roc-auc": "roc_auc",
    "rmse": "rmse",
    "mae": "mae",
    "r2": "r2",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "f1": "f1",
    "logloss": "log_loss",
    "log_loss": "log_loss",
}


@dataclass
class _Spec:
    competition_id: str
    task_type: str
    metric: str
    target_column: str
    id_column: str


def _load_spec(comp_dir: Path) -> _Spec:
    import yaml

    spec_path = comp_dir / "spec.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"TML-bench competition has no spec.yaml: {spec_path}")
    s = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw_task = str(s.get("task_type", "regression")).lower()
    metric_block = s.get("metric", {})
    raw_metric = (
        str(metric_block.get("name", "rmse")).lower()
        if isinstance(metric_block, dict)
        else str(metric_block).lower()
    )
    return _Spec(
        competition_id=s["id"],
        task_type=_TASKTYPE_MAP.get(raw_task, "regression"),
        metric=_METRIC_MAP.get(raw_metric, "rmse"),
        target_column=s["target_column"],
        id_column=s.get("id_column", "id"),
    )


def discover_competitions(
    tml_root: str | Path, *, include_toy: bool = False
) -> list[str]:
    """Return competition ids that are PREPARED (have public/train_public.csv)."""
    tml_root = Path(tml_root)
    comp_root = tml_root / "competitions"
    if not comp_root.exists():
        raise FileNotFoundError(
            f"No competitions/ under {tml_root}. Is this the TML-bench repo root?"
        )
    ids = []
    for sub in sorted(comp_root.iterdir()):
        if not sub.is_dir():
            continue
        if not include_toy and sub.name.startswith("toy"):
            continue
        if (sub / "spec.yaml").exists() and (
            sub / "public" / "train_public.csv"
        ).exists():
            ids.append(sub.name)
    return ids


def build_task_json(
    tml_root: str | Path,
    competition_id: str,
    *,
    out_data_dir: str | Path = "data/tml_bench",
    out_task_dir: str | Path = "automl_eval/tasks",
    valid_frac_of_train: float = 0.15,
    compute_reference: bool = True,
    seed: int = 0,
) -> Path:
    """Prepare one TML-bench competition into a GRACE task JSON + CSV. Returns task JSON path."""
    tml_root = Path(tml_root)
    comp_dir = tml_root / "competitions" / competition_id
    spec = _load_spec(comp_dir)

    public = comp_dir / "public"
    private = comp_dir / "private"
    train_public_csv = public / "train_public.csv"
    test_public_csv = public / "test_public.csv"
    holdout_labels_pq = private / "holdout_labels.parquet"
    holdout_labels_csv = private / "holdout_labels.csv"
    holdout_labels_path = (
        holdout_labels_pq if holdout_labels_pq.exists() else holdout_labels_csv
    )
    for p in (train_public_csv, test_public_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"TML-bench competition '{competition_id}' not prepared (missing {p}). "
                f"Run: python competitions/{competition_id}/prepare_competition.py --download"
            )
    if not holdout_labels_path.exists():
        raise FileNotFoundError(
            f"TML-bench competition '{competition_id}' missing private holdout labels "
            f"({holdout_labels_pq.name} or {holdout_labels_csv.name}). "
            f"Run: python competitions/{competition_id}/prepare_competition.py --download"
        )

    train = pd.read_csv(train_public_csv)
    test_feats = pd.read_csv(test_public_csv)
    if holdout_labels_path.suffix == ".parquet":
        holdout_labels = pd.read_parquet(holdout_labels_path)
    else:
        holdout_labels = pd.read_csv(holdout_labels_path)

    tgt, idc = spec.target_column, spec.id_column
    if tgt not in train.columns:
        raise ValueError(f"target '{tgt}' not in train_public.csv for {competition_id}")

    holdout = test_feats.merge(holdout_labels[[idc, tgt]], on=idc, how="inner")

    from sklearn.model_selection import train_test_split

    strat = train[tgt] if spec.task_type != "regression" else None
    tr_idx, va_idx = train_test_split(
        np.arange(len(train)),
        test_size=valid_frac_of_train,
        random_state=seed,
        stratify=strat,
    )
    labels = np.array(["train"] * len(train), dtype=object)
    labels[va_idx] = "valid"
    train = train.copy()
    train["__split__"] = labels
    holdout = holdout.copy()
    holdout["__split__"] = "test"

    # Align columns across train + holdout, drop the identifier column from features.
    common = [c for c in train.columns if c in holdout.columns]
    df = pd.concat([train[common], holdout[common]], ignore_index=True)
    if idc in df.columns:
        df = df.drop(columns=[idc])

    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    safe = competition_id.replace("-", "_")
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
            target_column=tgt,
            task_type=spec.task_type,
            metric=spec.metric,
        )
        baseline_score, oracle_score = ref.baseline_score, ref.oracle_score

    task = {
        "task_id": f"tml_{safe}",
        "dataset_path": str(csv_path),
        "target_column": tgt,
        "task_type": spec.task_type,
        "metric": spec.metric,
        "description": f"TML-bench Kaggle competition '{competition_id}'. Strict tabular task with a "
        f"deterministic public/private-holdout split reproduced as a predefined GRACE split.",
        "plan_checklist": [],
        "time_budget_seconds": 1800.0,
        "max_steps": 30,
        "oracle_score": oracle_score,
        "baseline_score": baseline_score,
        "metadata": {
            "dataset_source": f"tml-bench/{competition_id}",
            "suite": "TML-bench",
            "citation": "Pinchuk, TML-bench, 2026 (arXiv 2603.05764)",
            "contamination_note": "Recent Kaggle competition selected for contamination control "
            "(release post-dates evaluated-model cutoffs).",
            "time_series_task": False,
            "evaluation_mode": "private_holdout",
            "n_rows": int(len(df)),
            "n_features": int(df.shape[1] - 2),
        },
        "split_strategy": {
            "method": "predefined",
            "split_column": "__split__",
            "train_values": ["train"],
            "valid_values": ["valid"],
            "test_values": ["test"],
            "drop_split_columns": True,
            "rationale": "Reproduce TML-bench private-holdout scoring via a predefined split.",
        },
    }
    out_task_dir = Path(out_task_dir)
    out_task_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_task_dir / f"tml_{safe}.json"
    task_path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_path


def build_all(
    tml_root: str | Path, *, include_toy: bool = False, **kwargs
) -> list[Path]:
    """Prepare every PREPARED TML-bench competition under ``tml_root``."""
    ids = discover_competitions(tml_root, include_toy=include_toy)
    if not ids:
        raise FileNotFoundError(
            f"No prepared TML-bench competitions under {tml_root}/competitions. "
            f"Run each competition's prepare_competition.py --download first (see docs/DATASETS.md)."
        )
    written = []
    for cid in ids:
        try:
            written.append(build_task_json(tml_root, cid, **kwargs))
            print(f"  [tml-bench] {cid} -> tml_{cid.replace('-', '_')}.json")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [tml-bench] SKIP {cid}: {exc}")
    return written
