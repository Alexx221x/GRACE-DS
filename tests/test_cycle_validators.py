""""""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from automl_eval.domain.task import Task, TaskType, MetricName
from automl_eval.core.session import RuntimeSession, StepRecord, ActionType
from automl_eval.validators.iterative_cycle import (
    IterativeCycleValidator,
    cycle_error_multiplier,
)
from automl_eval.validators.baseline_comparison import BaselineComparisonValidator

SEP = "=" * 60


def make_task() -> Task:
    return Task(
        task_id="test_cycle",
        dataset_path="data/titanic.csv",
        target_column="Survived",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="Test task.",
        time_budget_seconds=300.0,
        max_steps=20,
        oracle_score=0.87,
        baseline_score=0.5,
    )


def make_step(idx, action_type, text, success=True, metric=None):
    return StepRecord(
        step_idx=idx,
        action_type=action_type,
        action_text=text,
        state_before="",
        state_after="",
        reward=0.0,
        execution_success=success,
        code_body=text,
        metric_value=metric,
    )


# ═══════════════════════════════════════════════════════════════


def test_cycle_single_pass():
    """Agent does not return to EDA — single pass — PASS."""
    print(f"\n{SEP}")
    print("TEST: IterativeCycleValidator — single pass (no cycles)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.cycle_count = 0

    v = IterativeCycleValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed
    assert result.penalty == 0.0
    print("  -> OK")


def test_cycle_one_free():
    """One cycle — within free limit — PASS."""
    print(f"\n{SEP}")
    print("TEST: IterativeCycleValidator — 1 cycle (free)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.cycle_count = 1
    session.metric_history = [(0, 0.70), (3, 0.75)]

    v = IterativeCycleValidator(max_free_cycles=1)
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed
    print("  -> OK")


def test_cycle_escalating_penalty():
    """Three cycles without metric improvement — escalating penalty."""
    print(f"\n{SEP}")
    print("TEST: IterativeCycleValidator — 3 cycles, no metric gain")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.cycle_count = 3
    session.metric_history = [(0, 0.70), (3, 0.70), (6, 0.701)]

    v = IterativeCycleValidator(max_free_cycles=1)
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty:.4f}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: 3 cycles with no gain"
    assert result.penalty > 0.03
    print("  -> OK")


def test_cycle_with_improvement():
    """Three cycles but metric improves substantially — partially offset."""
    print(f"\n{SEP}")
    print("TEST: IterativeCycleValidator — 3 cycles with metric improvement")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.cycle_count = 3
    session.metric_history = [(0, 0.65), (3, 0.72), (6, 0.80)]

    v = IterativeCycleValidator(max_free_cycles=1)
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty:.4f}")
    print(f"  score: {result.score:.4f}")
    print(f"  details: {result.details}")
    assert result.penalty < 0.10, (
        f"Expected low penalty with good gains, got {result.penalty}"
    )
    print("  -> OK")


def test_cycle_regression():
    """Metric dropped after a cycle — stronger penalty."""
    print(f"\n{SEP}")
    print("TEST: IterativeCycleValidator — metric regression after cycle")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.cycle_count = 2
    session.metric_history = [(0, 0.75), (3, 0.68)]

    v = IterativeCycleValidator(max_free_cycles=1)
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty:.4f}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: metric regression"
    print("  -> OK")


def test_cycle_error_multiplier():
    """Verify the error severity multiplier function."""
    print(f"\n{SEP}")
    print("TEST: cycle_error_multiplier function")
    print(SEP)

    m0 = cycle_error_multiplier(0)
    m1 = cycle_error_multiplier(1)
    m2 = cycle_error_multiplier(2)
    m3 = cycle_error_multiplier(3)
    m5 = cycle_error_multiplier(5)

    print(f"  cycle 0: {m0}x")
    print(f"  cycle 1: {m1}x")
    print(f"  cycle 2: {m2}x")
    print(f"  cycle 3: {m3}x")
    print(f"  cycle 5: {m5}x")

    assert m0 == 1.0
    assert m1 == 1.0
    assert m2 == 1.5
    assert m3 == 2.0
    assert m5 == 3.0
    print("  -> OK")


def test_cycle_tracking_in_session():
    """Verify that session correctly tracks cycles via record_step."""
    print(f"\n{SEP}")
    print("TEST: Session cycle tracking via record_step")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    steps = [
        make_step(0, ActionType.FEATURE_ENGINEERING, "df['Age'] = df['Age'].fillna(0)"),
        make_step(1, ActionType.CODE, "model = RF()\nmodel.fit(X, y)", metric=0.72),
        make_step(
            2,
            ActionType.FEATURE_ENGINEERING,
            "df['FamilySize'] = df['SibSp'] + df['Parch']",
        ),
        make_step(3, ActionType.CODE, "model2 = GBT()\nmodel2.fit(X, y)", metric=0.76),
        make_step(
            4, ActionType.FEATURE_ENGINEERING, "df = df.drop(columns=['Ticket'])"
        ),
        make_step(5, ActionType.CODE, "model3 = XGB()\nmodel3.fit(X, y)", metric=0.78),
    ]

    for s in steps:
        session.record_step(s)

    print(f"  cycle_count: {session.cycle_count}")
    print(f"  metric_history: {session.metric_history}")
    assert session.cycle_count == 3, f"Expected 3 cycles, got {session.cycle_count}"
    assert len(session.metric_history) == 3
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_baseline_better_than():
    """A high terminal hidden-test score beats the same-split baseline."""
    print(f"\n{SEP}")
    print("TEST: BaselineComparisonValidator — terminal agent better than baseline")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True
    session.final_submitted = True
    session.hidden_test_metric = 0.95

    v = BaselineComparisonValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, result.details
    assert "terminal agent (0.9500)" in result.details
    print("  -> OK")


def test_baseline_worse_than():
    """Agent worse than baseline — penalty."""
    print(f"\n{SEP}")
    print("TEST: BaselineComparisonValidator — agent worse (random preds)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True
    session.final_submitted = True

    np.random.seed(99)
    session.predictions = np.random.rand(len(session.test_df))
    session.hidden_test_metric = 0.48

    v = BaselineComparisonValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: worse than baseline"
    assert result.penalty >= 0.08
    print("  -> OK")


def test_baseline_plateau():
    """Metric plateaus with more than two cycles — warning."""
    print(f"\n{SEP}")
    print("TEST: BaselineComparisonValidator — plateau detection")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True
    session.final_submitted = True
    session.hidden_test_metric = 0.80
    session.predictions = np.random.rand(len(session.test_df))
    session.cycle_count = 4
    session.metric_history = [
        (0, 0.799),
        (3, 0.800),
        (6, 0.801),
        (9, 0.800),
    ]

    v = BaselineComparisonValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert "plateau" in result.details.lower(), (
        f"Expected plateau mention. Got: {result.details}"
    )
    print("  -> OK")


def test_baseline_not_done():
    """Before FINAL_SUBMIT — skip."""
    print(f"\n{SEP}")
    print("TEST: BaselineComparisonValidator — not done (skip)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = False

    v = BaselineComparisonValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}")
    assert result.passed
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_full_pipeline():
    """End-to-end: both validators appear in feedback."""
    print(f"\n{SEP}")
    print("TEST: Full pipeline — cycle validators visible")
    print(SEP)

    from automl_eval.domain.task_registry import TaskRegistry
    from automl_eval.core.environment import AutoMLEnvironment

    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")

    env = AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary")

    plan = "ACTION: PLAN\nTrain GradientBoosting, evaluate, iterate if needed.\n"
    env.step(plan)

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "print(model.feature_importances_)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "```"
    )
    env.step(code)

    out = env.step("ACTION: FINAL_SUBMIT")
    print(f"  SUBMIT reward={out.reward:.4f}")

    for line in out.state.split("\n"):
        for key in ["iterative_cycles", "baseline_comparison"]:
            if key in line:
                print(f"  {line.strip()}")

    for vname in ["iterative_cycles", "baseline_comparison"]:
        assert vname in out.state, f"'{vname}' not in response!"
        print(f"  -> '{vname}' present")

    env.close()
    print("  -> OK")


def main():
    print(SEP)
    print("  TESTING CYCLE VALIDATORS")
    print(SEP)

    test_cycle_single_pass()
    test_cycle_one_free()
    test_cycle_escalating_penalty()
    test_cycle_with_improvement()
    test_cycle_regression()
    test_cycle_error_multiplier()
    test_cycle_tracking_in_session()

    test_baseline_better_than()
    test_baseline_worse_than()
    test_baseline_plateau()
    test_baseline_not_done()

    test_full_pipeline()

    print(f"\n{SEP}")
    print("  ALL 12 TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
