"""Acceptance tests for the feedback-driven stage-aware environment refactor."""

from __future__ import annotations

from automl_eval.core.action_parser import ActionParser
from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.domain.hidden_checklists import compile_hidden_checklist
from automl_eval.core.session import ActionType, RuntimeSession
from automl_eval.domain.task import StageLimit
from automl_eval.domain.task_registry import TaskRegistry


def registry() -> TaskRegistry:
    reg = TaskRegistry()
    reg.load_directory("automl_eval/tasks")
    return reg


def pipeline_code() -> str:
    return """ACTION: MODEL
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']
num = X.select_dtypes(include='number').columns
cat = X.select_dtypes(exclude='number').columns
pre = ColumnTransformer([('num', SimpleImputer(strategy='median'), num), ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), cat)])
pipeline = Pipeline([('pre', pre), ('model', RandomForestClassifier(n_estimators=20, random_state=42))])
pipeline.fit(X, y)
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe)[:, 1]
```"""


def test_parser_supports_first_class_eda_and_validate() -> None:
    parser = ActionParser()
    assert (
        parser.parse("ACTION: EDA\nprint(train_df.shape)").action_type == ActionType.EDA
    )
    assert parser.parse("ACTION: VALIDATE").action_type == ActionType.VALIDATE


def test_mutable_working_frames_do_not_violate_raw_snapshot_protection() -> None:
    session = RuntimeSession(registry().get("titanic_binary"))
    session.initialize()
    session.sandbox_namespace["train_df"]["Age"] = session.sandbox_namespace[
        "train_df"
    ]["Age"].fillna(0)
    assert session.check_data_intact()
    session.sandbox_namespace["train_df_original"]["Age"] = 0
    assert not session.check_data_intact()


def test_dataset_specific_hidden_checklists_activate_relevant_conditions() -> None:
    titanic = RuntimeSession(registry().get("titanic_binary"))
    titanic.initialize()
    bike = RuntimeSession(registry().get("bike_sharing_regression"))
    bike.initialize()
    titanic_keys = {
        criterion.key for criterion in compile_hidden_checklist(titanic).criteria
    }
    bike_keys = {criterion.key for criterion in compile_hidden_checklist(bike).criteria}
    assert "fe_missing" in titanic_keys and "plan_categories" in titanic_keys
    assert "fe_missing" not in bike_keys


def test_validation_is_scored_only_after_explicit_validate_action() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    model_out = env.step(pipeline_code())
    assert "Evaluator validation roc_auc:" not in model_out.state
    assert "Current validation roc_auc:" not in model_out.state
    assert env._session.current_metric is None
    assert env._session.metric_history == []

    validate_out = env.step("ACTION: VALIDATE")
    assert "Evaluator validation roc_auc:" in validate_out.state
    assert "Current validation roc_auc:" in validate_out.state
    assert env._session.current_metric is not None
    assert len(env._session.metric_history) == 1


def test_validation_and_terminal_scoring_require_replayable_submission() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step("""ACTION: MODEL
```python
from sklearn.ensemble import RandomForestClassifier
X = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)
y = train_df['Survived']
model = RandomForestClassifier(random_state=42).fit(X, y)
```""")
    failed = env.step("ACTION: VALIDATE")
    assert "not replayable" in failed.state
    terminal_failed = env.step("ACTION: FINAL_SUBMIT")
    assert terminal_failed.done and env._session.test_evaluation_count == 0

    env.reset("titanic_binary")
    model_out = env.step(pipeline_code())
    assert "Evaluator validation roc_auc:" not in model_out.state
    validate_out = env.step("ACTION: VALIDATE")
    assert "Evaluator validation roc_auc:" in validate_out.state
    terminal = env.step("ACTION: FINAL_SUBMIT")
    assert "Evaluator validation roc_auc:" not in terminal.state
    assert terminal.done and env._session.hidden_test_metric is not None
    assert env._session.test_evaluation_count == 1


def test_standard_feedback_hides_internal_validator_identifiers() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    output = env.step("ACTION: EDA\n```python\nprint(train_df.shape)\n```")
    assert "Missingness has not been examined" in output.state
    assert "missing_values:" not in output.state
    assert "Internal diagnostics" not in output.state


def test_repeated_stage_budget_exceedance_autofinalises_latest_bundle() -> None:
    reg = registry()
    task = reg.get("titanic_binary")
    task.stage_limits["EDA"] = StageLimit(
        max_steps=0, max_seconds=90.0, max_consecutive_steps=10
    )
    env = AutoMLEnvironment(reg, seed=42)
    env.reset("titanic_binary")
    env.step(pipeline_code())
    warning = env.step("ACTION: EDA\n```python\nprint(train_df.shape)\n```")
    assert not warning.done and "Stage budget exceeded" in warning.state
    finalised = env.step("ACTION: EDA\n```python\nprint(train_df.shape)\n```")
    assert (
        finalised.done
        and "latest replayable solution will now be finalised" in finalised.state
    )
    assert env._session.hidden_test_metric is not None


def test_validation_labels_are_not_in_agent_namespace() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    assert "Survived" in env._session.sandbox_namespace["train_df"].columns
    assert "Survived" not in env._session.sandbox_namespace["valid_df"].columns
    assert "Survived" not in env._session.sandbox_namespace["valid_df_original"].columns
    assert "validation target labels are evaluator-private" in env.observe()


def test_executable_prose_is_rejected_without_running() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    out = env.step("ACTION: EDA\nWe'll inspect missing values next.")
    assert "requires exactly one fenced Python code block" in out.state
    assert "SyntaxError" not in out.state


def test_protected_snapshot_mutation_is_rejected_and_rolled_back() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    out = env.step("""ACTION: FEATURE_ENGINEERING
```python
train_df_original['Injected'] = 1
```""")
    assert "ProtectedSnapshotViolation" in out.state
    assert "Injected" not in env._session.sandbox_namespace["train_df_original"].columns
    assert env._session.check_data_intact()


def test_validation_metric_bypass_is_blocked_outside_validate() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    out = env.step("""ACTION: CODE
```python
from sklearn.metrics import roc_auc_score
roc_auc_score([0, 1], [0.2, 0.8])
```""")
    assert "bypasses evaluator-owned validation" in out.state


def test_validate_and_submit_are_code_free_triggers() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    out = env.step("""ACTION: VALIDATE
```python
print('not allowed')
```""")
    assert "VALIDATE is an evaluator-owned scoring trigger" in out.state
    assert env._session.current_metric is None


def test_transformer_fit_is_not_counted_as_replayable_model() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step("""ACTION: FEATURE_ENGINEERING
```python
from sklearn.impute import SimpleImputer
preprocessor = SimpleImputer(strategy='median')
preprocessor.fit(train_df[['Age']])
```""")
    assert env._session.trained_models == []


def test_reset_accepts_max_action_override_without_mutating_registry_default() -> None:
    reg = registry()
    original = reg.get("titanic_binary").max_steps
    env = AutoMLEnvironment(reg, seed=42)
    env.reset("titanic_binary", max_action=12)
    observation = env.observe()
    assert "Maximum actions: 12" in observation
    assert "Step: 0 / 12" in observation
    assert env._session.task.max_steps == 12
    assert reg.get("titanic_binary").max_steps == original


def test_reset_rejects_conflicting_action_budget_aliases() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    try:
        env.reset("titanic_binary", max_actions=12, max_action=10)
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("Conflicting action budget aliases should be rejected")


def test_feature_feedback_flags_dropped_ordinary_predictors() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    limited = env.step("""ACTION: FEATURE_ENGINEERING
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
X_train = train_df.drop(columns=[target_column, 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y_train = train_df[target_column]
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), ['Age', 'Fare']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), ['Sex', 'Embarked']),
])
preprocessor.fit(X_train)
```""")
    assert "Potentially useful ordinary predictors are excluded" in limited.state

    env.reset("titanic_binary")
    complete = env.step("""ACTION: FEATURE_ENGINEERING
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
X_train = train_df.drop(columns=[target_column, 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y_train = train_df[target_column]
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), ['Sex', 'Embarked']),
])
preprocessor.fit(X_train)
```""")
    assert "Potentially useful ordinary predictors are excluded" not in complete.state


def test_second_validation_reports_unresolved_feedback_response() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step("""ACTION: EDA
```python
print(train_df.info())
print(train_df[target_column].value_counts(normalize=True))
print(train_df.isna().sum())
```""")
    env.step(pipeline_code())
    env.step("ACTION: VALIDATE")
    revised = pipeline_code().replace("n_estimators=20", "n_estimators=30")
    env.step(revised)
    output = env.step("ACTION: VALIDATE")
    assert "--- Iteration feedback ---" in output.state
    assert (
        "Previously surfaced workflow concerns have not been demonstrably addressed"
        in output.state
    )
