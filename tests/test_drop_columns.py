from __future__ import annotations

import pandas as pd

from automl_eval.core.session import RuntimeSession
from automl_eval.domain.task import MetricName, Task, TaskType


def test_manifest_drop_columns_are_hidden_from_agent_frames(tmp_path):
    path = tmp_path / "drop_columns.csv"
    pd.DataFrame(
        {
            "safe_feature": [0, 1, 2, 3, 4, 5, 6, 7],
            "leaky_feature": [0, 1, 0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    task = Task(
        task_id="drop_columns",
        dataset_path=str(path),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="drop columns test",
        drop_columns=["leaky_feature"],
    )

    session = RuntimeSession(task)
    session.initialize()

    assert "leaky_feature" not in session.train_df.columns
    assert "leaky_feature" not in session.visible_valid_df.columns
    assert "leaky_feature" not in session.hidden_test_df.columns
    assert "safe_feature" in session.train_df.columns
