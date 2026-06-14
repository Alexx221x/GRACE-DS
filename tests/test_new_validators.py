""""""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np

from automl_eval.domain.task import Task, TaskType, MetricName
from automl_eval.core.session import RuntimeSession, StepRecord, ActionType
from automl_eval.validators.backtracking import BacktrackingValidator
from automl_eval.validators.reproducibility import ReproducibilityValidator
from automl_eval.validators.efficiency import EfficiencyValidator

SEPARATOR = "=" * 60


def make_dummy_task(time_budget: float = 300.0) -> Task:
    return Task(
        task_id="test_validators",
        dataset_path="data/titanic.csv",
        target_column="Survived",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="Test task for validator checks.",
        time_budget_seconds=time_budget,
        max_steps=20,
        oracle_score=0.87,
        baseline_score=0.5,
        plan_checklist=[],
    )


def make_step(
    idx: int, action_type: ActionType, text: str, success: bool = True
) -> StepRecord:
    return StepRecord(
        step_idx=idx,
        action_type=action_type,
        action_text=text,
        state_before="",
        state_after="",
        reward=0.0,
        execution_success=success,
    )


def test_backtracking_no_penalty():
    """No penalty: FE -> CODE(.fit) is the normal order."""
    print(f"\n{SEPARATOR}")
    print("TEST: BacktrackingValidator — no penalty (normal order)")
    print(SEPARATOR)

    task = make_dummy_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(0, ActionType.FEATURE_ENGINEERING, "df = df.drop(columns=['Cabin'])"),
        make_step(
            1,
            ActionType.FEATURE_ENGINEERING,
            "df['Age'] = df['Age'].fillna(df['Age'].median())",
        ),
        make_step(
            2, ActionType.CODE, "model = RandomForestClassifier()\nmodel.fit(X, y)"
        ),
    ]

    v = BacktrackingValidator()
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, "Expected PASS: normal pipeline order"
    assert result.penalty == 0.0
    print("  -> OK")


def test_backtracking_with_penalty():
    """Penalty: model was trained, then the agent goes back to .drop()/.fillna()."""
    print(f"\n{SEPARATOR}")
    print("TEST: BacktrackingValidator — penalty (backtrack after model)")
    print(SEPARATOR)

    task = make_dummy_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0, ActionType.CODE, "model = RandomForestClassifier()\nmodel.fit(X, y)"
        ),
        make_step(
            1, ActionType.FEATURE_ENGINEERING, "df = df.drop(columns=['Ticket'])"
        ),
        make_step(2, ActionType.CODE, "df['Age'] = df['Age'].fillna(0)"),
    ]

    v = BacktrackingValidator(
        penalty_per_backtrack=0.15,
        max_backtrack_penalty=1.0,
    )
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: backtracking detected"
    assert result.penalty == 0.30, (
        f"Expected 0.30 penalty for 2 backtracks, got {result.penalty}"
    )
    print("  -> OK")


def test_reproducibility_with_seed():
    """Code sets random_state — no penalty."""
    print(f"\n{SEPARATOR}")
    print("TEST: ReproducibilityValidator — seed present, deterministic")
    print(SEPARATOR)

    task = make_dummy_task()
    session = RuntimeSession(task)
    session.initialize()

    code = (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X = train_df.drop(columns=['Survived'])\n"
        "y = train_df['Survived']\n"
        "X = X.select_dtypes(include='number').fillna(0)\n"
        "model = RandomForestClassifier(n_estimators=10, random_state=42)\n"
        "model.fit(X, y)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
    )

    session.steps = [make_step(0, ActionType.CODE, code, success=True)]

    from automl_eval.core.sandbox import Sandbox

    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)
    preds = session.sandbox_namespace.get("predictions")
    if preds is not None:
        session.predictions = np.asarray(preds)

    v = ReproducibilityValidator()
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS: seed is fixed. Details: {result.details}"
    assert result.penalty == 0.0
    print("  -> OK")


def test_reproducibility_no_seed():
    """Code does not set random_state — penalty."""
    print(f"\n{SEPARATOR}")
    print("TEST: ReproducibilityValidator — no seed, penalty expected")
    print(SEPARATOR)

    task = make_dummy_task()
    session = RuntimeSession(task)
    session.initialize()

    code = (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X = train_df.drop(columns=['Survived'])\n"
        "y = train_df['Survived']\n"
        "X = X.select_dtypes(include='number').fillna(0)\n"
        "model = RandomForestClassifier(n_estimators=10)\n"
        "model.fit(X, y)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
    )

    session.steps = [make_step(0, ActionType.CODE, code, success=True)]

    from automl_eval.core.sandbox import Sandbox

    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)
    preds = session.sandbox_namespace.get("predictions")
    if preds is not None:
        session.predictions = np.asarray(preds)

    v = ReproducibilityValidator()
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: no seed in code"
    assert result.penalty > 0.0
    print("  -> OK")


def test_efficiency_ok():
    """Normal runtime, no GridSearch — no penalty."""
    print(f"\n{SEPARATOR}")
    print("TEST: EfficiencyValidator — within budget, no gridsearch")
    print(SEPARATOR)

    task = make_dummy_task(time_budget=300.0)
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0, ActionType.CODE, "model = RandomForestClassifier()\nmodel.fit(X, y)"
        ),
    ]

    v = EfficiencyValidator(hard_time_limit=3600.0)
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, "Expected PASS: within budget, no gridsearch"
    assert result.penalty == 0.0
    print("  -> OK")


def test_efficiency_gridsearch_penalty():
    """Code contains GridSearchCV — penalty."""
    print(f"\n{SEPARATOR}")
    print("TEST: EfficiencyValidator — GridSearchCV penalty")
    print(SEPARATOR)

    task = make_dummy_task(time_budget=300.0)
    session = RuntimeSession(task)
    session.initialize()

    gridsearch_code = (
        "from sklearn.model_selection import GridSearchCV\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, 10]}\n"
        "gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)\n"
        "gs.fit(X, y)\n"
    )
    session.steps = [make_step(0, ActionType.CODE, gridsearch_code)]

    v = EfficiencyValidator(gridsearch_penalty=0.15)
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: GridSearchCV detected"
    assert result.penalty >= 0.15
    print("  -> OK")


def test_efficiency_randomized_penalty():
    """RandomizedSearchCV is disallowed by the manual-configuration policy."""
    print(f"\n{SEPARATOR}")
    print("TEST: EfficiencyValidator — RandomizedSearchCV penalty")
    print(SEPARATOR)

    task = make_dummy_task(time_budget=300.0)
    session = RuntimeSession(task)
    session.initialize()

    code = (
        "from sklearn.model_selection import RandomizedSearchCV\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "param_dist = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}\n"
        "rs = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=5, cv=3)\n"
        "rs.fit(X, y)\n"
    )
    session.steps = [make_step(0, ActionType.CODE, code)]

    v = EfficiencyValidator()
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: automated search detected"
    assert result.penalty > 0.0
    print("  -> OK")


def test_efficiency_hard_time_limit():
    """Artificially set start_time in the past to exceed the hard limit."""
    print(f"\n{SEPARATOR}")
    print("TEST: EfficiencyValidator — hard time limit exceeded")
    print(SEPARATOR)

    task = make_dummy_task(time_budget=300.0)
    session = RuntimeSession(task)
    session.initialize()

    session.execution_time_seconds = 4000.0

    session.steps = [make_step(0, ActionType.CODE, "model.fit(X, y)")]

    v = EfficiencyValidator(hard_time_limit=3600.0, time_penalty_max=0.3)
    result = v.validate(session)

    print(f"  passed:  {result.passed}")
    print(f"  score:   {result.score}")
    print(f"  penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: hard time limit exceeded"
    assert result.penalty >= 0.3
    print("  -> OK")


def test_full_pipeline_with_new_validators():
    """"""
    print(f"\n{SEPARATOR}")
    print("TEST: Full pipeline — new validators visible in feedback")
    print(SEPARATOR)

    from automl_eval.domain.task_registry import TaskRegistry
    from automl_eval.core.environment import AutoMLEnvironment

    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")

    env = AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary")

    plan = (
        "ACTION: PLAN\n"
        "I will handle missing values, encode categoricals, engineer features "
        "and train a Random Forest model with manually chosen hyperparameters, then use evaluator validation."
    )
    out1 = env.step(plan)
    print(f"  PLAN reward={out1.reward:.4f}, done={out1.done}")

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X = train_df.drop(columns=['Survived'])\n"
        "y = train_df['Survived']\n"
        "X = X.select_dtypes(include='number').fillna(0)\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X, y)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "```"
    )
    out2 = env.step(code)
    print(f"  CODE reward={out2.reward:.4f}, done={out2.done}")

    submit = "ACTION: FINAL_SUBMIT"
    out3 = env.step(submit)
    print(f"  SUBMIT reward={out3.reward:.4f}, done={out3.done}")

    print("\n  --- Full step response ---")
    print(out3.state)

    expected_validators = ["backtracking", "reproducibility", "efficiency"]
    for vname in expected_validators:
        assert vname in out3.state, f"Validator '{vname}' not found in step response!"
        print(f"  -> Validator '{vname}' is present in output")

    env.close()
    print("  -> OK")


def main():
    print("=" * 60)
    print("  TESTING NEW VALIDATORS: backtracking, reproducibility, efficiency")
    print("=" * 60)

    test_backtracking_no_penalty()
    test_backtracking_with_penalty()
    test_reproducibility_with_seed()
    test_reproducibility_no_seed()
    test_efficiency_ok()
    test_efficiency_gridsearch_penalty()
    test_efficiency_randomized_ok()
    test_efficiency_hard_time_limit()
    test_full_pipeline_with_new_validators()

    print(f"\n{'=' * 60}")
    print("  ALL 9 TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
