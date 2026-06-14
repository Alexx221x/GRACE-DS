from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.domain.task_registry import TaskRegistry


def _registry():
    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")
    return registry


def _model_action():
    return """ACTION: MODEL
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
X = train_df.drop(columns=[target_column]).select_dtypes(include='number')
y = train_df[target_column]
pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', RandomForestClassifier(n_estimators=20, random_state=42))])
pipeline.fit(X, y)
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe.select_dtypes(include='number'))[:, 1]
```"""


def test_forced_budget_termination_does_not_score_hidden_test_when_suppressed():
    env = AutoMLEnvironment(
        _registry(), seed=42, allow_forced_terminal_evaluation=False
    )
    env.reset("titanic_binary", max_actions=1)
    output = env.step(_model_action())
    assert output.done
    assert env._session is not None
    assert env._session.hidden_test_metric is None
    assert env._session.test_evaluation_count == 0
    assert (
        "hidden-test evaluation suppressed by protocol"
        in env._session.finalization_reason
    )


def test_explicit_terminal_submission_still_scores_once_when_forced_scoring_suppressed():
    env = AutoMLEnvironment(
        _registry(), seed=42, allow_forced_terminal_evaluation=False
    )
    env.reset("titanic_binary", max_actions=2)
    env.step(_model_action())
    output = env.step("ACTION: FINAL_SUBMIT")
    assert output.done
    assert env._session is not None
    assert env._session.hidden_test_metric is not None
    assert env._session.test_evaluation_count == 1
