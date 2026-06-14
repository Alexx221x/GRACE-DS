from __future__ import annotations

import pandas as pd

from automl_eval.core.session import RuntimeSession
from automl_eval.domain.task import MetricName, SplitStrategy, Task, TaskType


def test_ordered_split_strategy_uses_sorted_contiguous_blocks(tmp_path):
    path = tmp_path / "ordered.csv"
    pd.DataFrame(
        {
            "row_id": list(reversed(range(12))),
            "feature": list(range(12)),
            "target": list(range(12)),
        }
    ).to_csv(path, index=False)
    task = Task(
        task_id="ordered",
        dataset_path=str(path),
        target_column="target",
        task_type=TaskType.REGRESSION,
        metric=MetricName.R2,
        description="ordered split test",
        split_strategy=SplitStrategy(
            method="ordered",
            train_size=0.5,
            valid_size=0.25,
            test_size=0.25,
            stratify=False,
            sort_by=["row_id"],
        ),
    )

    session = RuntimeSession(task)
    session.initialize()

    assert session.train_df["row_id"].tolist() == list(range(6))
    assert session.valid_df["row_id"].tolist() == [6, 7, 8]
    assert session.hidden_test_df["row_id"].tolist() == [9, 10, 11]


def test_predefined_split_strategy_uses_manifest_labels_and_drops_split_column(
    tmp_path,
):
    path = tmp_path / "predefined.csv"
    pd.DataFrame(
        {
            "split": ["train"] * 4 + ["valid"] * 2 + ["test"] * 2,
            "feature": list(range(8)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    task = Task(
        task_id="predefined",
        dataset_path=str(path),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="predefined split test",
        split_strategy=SplitStrategy(method="predefined", split_column="split"),
    )

    session = RuntimeSession(task)
    session.initialize()

    assert len(session.train_df) == 4
    assert len(session.valid_df) == 2
    assert len(session.hidden_test_df) == 2
    assert "split" not in session.train_df.columns
    assert "split" not in session.visible_valid_df.columns


def test_group_split_strategy_keeps_groups_disjoint(tmp_path):
    path = tmp_path / "grouped.csv"
    rows = []
    for group_idx in range(12):
        for row_idx in range(2):
            rows.append(
                {
                    "group": f"g{group_idx}",
                    "feature": group_idx * 10 + row_idx,
                    "target": group_idx % 2,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    task = Task(
        task_id="grouped",
        dataset_path=str(path),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="group split test",
        split_strategy=SplitStrategy(
            method="group",
            train_size=0.5,
            valid_size=0.25,
            test_size=0.25,
            group_column="group",
            drop_split_columns=False,
        ),
    )

    session = RuntimeSession(task)
    session.initialize()

    train_groups = set(session.train_df["group"])
    valid_groups = set(session.valid_df["group"])
    test_groups = set(session.hidden_test_df["group"])
    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)
