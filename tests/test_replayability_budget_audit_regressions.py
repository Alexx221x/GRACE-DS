"""Regression tests for issues discovered in live router episodes."""

from __future__ import annotations

import json
from pathlib import Path

from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.llm.router_client import OpenAICompatibleRouterClient
from automl_eval.domain.task_registry import TaskRegistry


def registry() -> TaskRegistry:
    reg = TaskRegistry()
    reg.load_directory("automl_eval/tasks")
    return reg


NON_REPLAYABLE_MODEL = """ACTION: MODEL
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
X_train = train_df_original.drop(columns=[target_column]).copy()
X_train["FamilySize"] = X_train["SibSp"] + X_train["Parch"] + 1
y_train = train_df_original[target_column]
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["Age", "Fare", "FamilySize"])])
pipeline = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
```"""


REPLAYABLE_FIX = """ACTION: CODE_FIX
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
X_raw = train_df_original.drop(columns=[target_column]).copy()
y_train = train_df_original[target_column]
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["Age", "Fare", "SibSp", "Parch"])])
pipeline = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
pipeline.fit(X_raw, y_train)
```"""


def test_model_replayability_failure_is_reported_before_validate() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    output = env.step(NON_REPLAYABLE_MODEL)
    assert "Candidate raw-input smoke check (no metric): failed" in output.state
    assert "Evaluator validation roc_auc" not in output.state
    assert env._session.candidate_raw_input_compatible is False
    assert env._session.trained_models == []


def test_self_repair_counts_only_evaluator_verified_candidate_fixes() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step(NON_REPLAYABLE_MODEL)

    still_broken = env.step("""ACTION: CODE_FIX
```python
# Execution succeeds but the previously registered raw-input artefact is unchanged.
repair_note = "attempted"
```""")
    assert "Candidate raw-input smoke check (no metric): failed" in still_broken.state
    assert "Self-repair: 0/1 fixes evaluator-verified" in still_broken.state

    repaired = env.step(REPLAYABLE_FIX)
    assert "Candidate raw-input smoke check (no metric): passed" in repaired.state
    assert "Self-repair: 1/2 fixes evaluator-verified" in repaired.state
    assert len(env._session.trained_models) == 1


def test_terminal_state_keeps_metrics_private_but_session_preserves_best_for_operator_reporting() -> (
    None
):
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step(REPLAYABLE_FIX.replace("ACTION: CODE_FIX", "ACTION: MODEL"))
    validated = env.step("ACTION: VALIDATE")
    assert "Best validation roc_auc:" in validated.state
    final = env.step("ACTION: FINAL_SUBMIT")
    assert final.done
    assert "Best validation roc_auc:" not in final.state
    assert env._session.best_metric is not None


class _FakeResponse:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": f"ACTION: PLAN\\nturn {self.idx}"}}]
        }


def test_router_jsonl_audit_is_immutable_and_turn_faithful(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(calls["n"])

    monkeypatch.setattr("automl_eval.llm.router_client.requests.post", fake_post)
    log_path = tmp_path / "router.jsonl"
    router = OpenAICompatibleRouterClient(
        "https://router.invalid/v1/chat/completions",
        "token",
        "model",
        audit_jsonl_path=log_path,
    )
    messages = [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "turn 0"},
    ]
    router.complete(messages)
    messages.extend(
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    )
    router.complete(messages)
    messages.append({"role": "assistant", "content": "late mutation"})

    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [len(row["request"]["messages"]) for row in records] == [2, 4]
    assert [len(row["request"]["messages"]) for row in router.communication_log] == [
        2,
        4,
    ]
    assert all("late mutation" not in str(row["request"]) for row in records)


STRONG_REPLAYABLE_MODEL = """ACTION: MODEL
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
X_raw = train_df_original.drop(columns=[target_column]).copy()
y_train = train_df_original[target_column]
num = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
cat = ["Sex", "Embarked"]
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipeline = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, random_state=42))])
pipeline.fit(X_raw, y_train)
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe)[:, 1]
```"""


WEAKER_REPLAYABLE_MODEL = """ACTION: MODEL
```python
import numpy as np
def predict_fn(raw_dataframe):
    return np.full(len(raw_dataframe), 0.5)
```"""


RESTORE_STRONG_PREDICTOR = """ACTION: CODE_FIX
```python
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe)[:, 1]
```"""


def test_final_submit_blocks_a_validated_candidate_below_the_best_until_restored() -> (
    None
):
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step(STRONG_REPLAYABLE_MODEL)
    first = env.step("ACTION: VALIDATE")
    first_metric = env._session.current_metric
    assert first_metric is not None and first_metric > 0.5

    env.step(WEAKER_REPLAYABLE_MODEL)
    env.step("ACTION: VALIDATE")
    assert env._session.current_metric == 0.5
    assert env._session.best_metric == first_metric

    blocked = env.step("ACTION: FINAL_SUBMIT")
    assert not blocked.done
    assert "FINAL_SUBMIT blocked" in blocked.state
    assert "latest candidate only" in blocked.state
    assert env._session.test_evaluation_count == 0
    assert env._session.hidden_test_metric is None

    env.step(RESTORE_STRONG_PREDICTOR)
    restored = env.step("ACTION: VALIDATE")
    assert "Evaluator validation roc_auc:" in restored.state
    assert abs(env._session.current_metric - first_metric) < 1e-12
    final = env.step("ACTION: FINAL_SUBMIT")
    assert final.done
    assert env._session.test_evaluation_count == 1


def test_final_submit_blocks_changed_unvalidated_candidate_when_a_best_exists() -> None:
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step(STRONG_REPLAYABLE_MODEL)
    env.step("ACTION: VALIDATE")
    env.step(WEAKER_REPLAYABLE_MODEL)

    blocked = env.step("ACTION: FINAL_SUBMIT")
    assert not blocked.done
    assert "changed since the latest successful validation" in blocked.state
    assert env._session.test_evaluation_count == 0


def test_model_comparison_is_recognised_as_productive_revision_even_if_score_decreases() -> (
    None
):
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    env.step(STRONG_REPLAYABLE_MODEL)
    env.step("ACTION: VALIDATE")
    env.step(WEAKER_REPLAYABLE_MODEL)
    second = env.step("ACTION: VALIDATE")
    assert "No evidence of productive revision is available" not in second.state
    assert "did not improve on the best prior scored candidate" in second.state


def test_generic_drop_transformer_does_not_count_as_structured_high_cardinality_derivation() -> (
    None
):
    env = AutoMLEnvironment(registry(), seed=42)
    env.reset("titanic_binary")
    output = env.step("""ACTION: FEATURE_ENGINEERING
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.iloc[:, 0:0]
preprocessor = ColumnTransformer([("drop", DropColumns(), ["Name", "Ticket", "Cabin"])])
```""")
    assert (
        "No useful structured derivation from complex columns is evident"
        in output.state
    )
