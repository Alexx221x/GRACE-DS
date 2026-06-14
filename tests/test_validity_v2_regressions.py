"""Regression tests for issues revealed by the fourth Titanic mode output."""

from __future__ import annotations

from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.domain.task_registry import TaskRegistry
from automl_eval.validators.baseline_comparison import BaselineComparisonValidator
from automl_eval.validators.target_leakage_model import TargetLeakageModelValidator


def _registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")
    return registry


def _model_using_local_x_test() -> str:
    return """ACTION: MODEL
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
X = train_df_original.drop(columns=[target_column]).copy()
y = train_df_original[target_column]
# Local variable name only: this is transformed training data, not held-out data.
X_test = X.select_dtypes(include="number").fillna(0).head(3)
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipeline = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=20, random_state=42))])
pipeline.fit(X, y)
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe)[:, 1]
```"""


def _terminal_env() -> AutoMLEnvironment:
    env = AutoMLEnvironment(_registry(), seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary", max_actions=3)
    env.step(_model_using_local_x_test())
    env.step("ACTION: VALIDATE")
    env.step("ACTION: FINAL_SUBMIT")
    return env


def test_local_x_test_variable_is_not_a_hidden_test_leak() -> None:
    env = _terminal_env()
    final = env.observe()
    assert env._session is not None and env._session.hidden_test_metric is not None
    leakage = next(
        v.validate(env._session) for v in env.validators if v.name == "leakage"
    )
    assert leakage.passed, leakage.details


def test_terminal_reward_is_not_zeroed_by_local_x_test_variable() -> None:
    env = AutoMLEnvironment(_registry(), seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary", max_actions=3)
    env.step(_model_using_local_x_test())
    env.step("ACTION: VALIDATE")
    output = env.step("ACTION: FINAL_SUBMIT")
    assert output.reward > 0.0, output.state
    assert "[UNRESOLVED] leakage" not in output.state


def test_terminal_baseline_uses_hidden_test_score_not_best_validation_score() -> None:
    env = _terminal_env()
    assert env._session is not None
    hidden = env._session.hidden_test_metric
    assert hidden is not None
    env._session.best_metric = 0.9999
    result = BaselineComparisonValidator().validate(env._session)
    assert f"terminal agent ({hidden:.4f})" in result.details
    assert "0.9999" not in result.details


def test_target_column_is_excluded_from_model_based_feature_probe() -> None:
    env = _terminal_env()
    assert env._session is not None
    result = TargetLeakageModelValidator().validate(env._session)
    assert "Survived" not in result.details
