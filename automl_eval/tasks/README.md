# Adding a Dataset Task

GRACE-DS defines each benchmark as one JSON file in `automl_eval/tasks/`. A task file is the reproducible evaluation unit: it binds a CSV dataset to a supervised target, problem type, metric, and evaluator-owned split protocol.

## Required Fields

| Field | Meaning |
|---|---|
| `task_id` | Unique identifier used in configs and `--tasks`. |
| `dataset_path` | CSV path relative to the repository root. |
| `target_column` | Supervised target column in the CSV. |
| `task_type` | One of `binary_classification`, `multiclass_classification`, `regression`. |
| `metric` | Primary metric; see supported metrics below. |
| `description` | Public task description shown to the AutoML agent. |

## Supported Metrics

| Task type | Valid metrics |
|---|---|
| `binary_classification` | `roc_auc`, `accuracy`, `f1`, `log_loss` |
| `multiclass_classification` | `accuracy`, `f1`, `log_loss` |
| `regression` | `r2`, `rmse`, `mae` |

All GRACE-DS scores use a higher-is-better convention. Therefore `log_loss`, `rmse`, and `mae` are stored and reported as negative values. If you provide `baseline_score` and `oracle_score` for these metrics, store them as negative numbers and ensure `oracle_score > baseline_score`.

## Recommended Fields

Recommended for reproducible benchmark reporting:

| Field | Meaning |
|---|---|
| `time_budget_seconds` | Per-task time budget exposed to the agent. |
| `max_steps` | Maximum number of agent actions. |
| `baseline_score` | Trivial-model reference score for normalization. |
| `oracle_score` | Strong-model reference score for normalization. |
| `metadata` | Provenance, suite, row/feature counts, contamination notes. |
| `split_strategy` | Evaluator-owned train/validation/test split policy. |

Reference scores can be computed with `automl_eval.dataset_loaders.reference.compute_reference_scores`.

## Split Strategy

Default behavior is random `70/15/15`. For benchmark reporting, prefer an explicit evaluator-owned split policy.

| Method | Use case |
|---|---|
| `random` | IID tabular tasks; use `stratify: true` for classification and `false` for regression. |
| `ordered` | Ordered or temporal holdout protocols where row order is meaningful. |
| `group` | Group-disjoint evaluation; requires `group_column`. |
| `predefined` | Existing benchmark, competition, or public/private split; requires `split_column`. |

## Template

```json
{
  "task_id": "my_new_binary_task",
  "dataset_path": "data/my_dataset.csv",
  "target_column": "target",
  "task_type": "binary_classification",
  "metric": "roc_auc",
  "description": "Predict the binary target from tabular customer features under an evaluator-owned holdout protocol.",
  "time_budget_seconds": 1800.0,
  "max_steps": 30,  
  "baseline_score": 0.5,
  "oracle_score": 0.85,
  "metadata": {"dataset_source": "custom/source", "suite": "custom", "time_series_task": false, "evaluation_mode": "iid_tabular_holdout", "n_rows": 10000, "n_features": 20},
  "split_strategy": {"method": "random", "train_size": 0.7, "valid_size": 0.15, "test_size": 0.15, "stratify": true, "drop_split_columns": true}
}
```

## Registration

Add the task id to `configs/final.yaml` under `task_ids`, or pass it directly:

```bash
bash scripts/run_experiments.sh --config configs/final.yaml --tasks my_new_binary_task --dry-run
```
