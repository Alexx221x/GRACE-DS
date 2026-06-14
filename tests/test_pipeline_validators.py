""""""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from automl_eval.domain.task import Task, TaskType, MetricName
from automl_eval.core.session import RuntimeSession, StepRecord, ActionType
from automl_eval.evaluation.data_insights import analyze_dataset
from automl_eval.core.sandbox import Sandbox
from automl_eval.validators.feature_pipeline import FeaturePipelineValidator
from automl_eval.validators.duplicate import DuplicateValidator
from automl_eval.validators.target_leakage_model import TargetLeakageModelValidator

SEP = "=" * 60


def make_task() -> Task:
    return Task(
        task_id="test_fp",
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


def make_step(idx, action_type, text, success=True):
    return StepRecord(
        step_idx=idx,
        action_type=action_type,
        action_text=text,
        state_before="",
        state_after="",
        reward=0.0,
        execution_success=success,
        code_body=text,
    )


# ═══════════════════════════════════════════════════════════════


def test_fp_raw_data_issues():
    """Sandbox still has raw data: NaN, object columns — FAIL."""
    print(f"\n{SEP}")
    print("TEST: FeaturePipelineValidator — raw data (NaN + object cols)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [make_step(0, ActionType.CODE, "x = 1")]

    v = FeaturePipelineValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: raw data has NaN + string columns"
    assert result.penalty > 0
    print("  -> OK")


def test_fp_clean_data():
    """Agent correctly processed data: no NaN, no string columns, new features."""
    print(f"\n{SEP}")
    print("TEST: FeaturePipelineValidator — clean transformed data")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    code = (
        "X_train = train_df.drop(columns=['Survived'])\n"
        "X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())\n"
        "X_train = X_train.drop(columns=['Cabin', 'Name', 'Ticket'])\n"
        "X_train['Embarked'] = X_train['Embarked'].fillna('S')\n"
        "X_train['Sex'] = X_train['Sex'].map({'male': 0, 'female': 1})\n"
        "X_train['Embarked'] = X_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n"
        "X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1\n"
        "X_train = X_train.fillna(0)\n"
    )

    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)

    session.steps = [make_step(0, ActionType.CODE, code)]

    v = FeaturePipelineValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS for clean data. Got: {result.details}"
    assert "missing values handled" in result.details.lower(), (
        f"Missing values should be handled. Got: {result.details}"
    )
    assert "categoricals encoded" in result.details.lower(), (
        f"Categoricals should be encoded. Got: {result.details}"
    )
    assert (
        "new feature" in result.details.lower() or "created" in result.details.lower()
    )
    assert result.penalty <= 0.10, (
        f"Expected low penalty for clean data, got {result.penalty}"
    )
    print("  -> OK")


def test_fp_no_feature_engineering():
    """Agent did not create new features — soft penalty."""
    print(f"\n{SEP}")
    print("TEST: FeaturePipelineValidator — no FE (same columns)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    code = (
        "X_train = train_df.drop(columns=['Survived'])\n"
        "X_train = X_train.select_dtypes(include='number').fillna(0)\n"
    )
    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)

    session.steps = [make_step(0, ActionType.CODE, code)]

    v = FeaturePipelineValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    # Soft penalty for missing FE, not a hard FAIL
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_dup_no_duplicates():
    """No duplicate rows — automatic PASS."""
    print(f"\n{SEP}")
    print("TEST: DuplicateValidator — no duplicates")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    if not session.data_insights.has_duplicates:
        v = DuplicateValidator()
        result = v.validate(session)
        print(f"  passed: {result.passed}, score: {result.score}")
        print(f"  details: {result.details}")
        assert result.passed
        print("  -> OK")
        return

    print("  -> SKIP (Titanic has duplicates)")


def test_dup_synthetic_not_handled():
    """Synthetic data: duplicates present, agent did not remove them — FAIL."""
    print(f"\n{SEP}")
    print("TEST: DuplicateValidator — duplicates not handled (synthetic)")
    print(SEP)

    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    insights = analyze_dataset(df, "target")
    print(f"  duplicate_count: {insights.duplicate_count}")
    assert insights.has_duplicates

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights

    # Duplicates are not in sandbox code path, but train_df was already loaded
    session.sandbox_namespace["train_df"] = df.copy()

    session.steps = [make_step(0, ActionType.CODE, "model.fit(X, y)")]

    v = DuplicateValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: duplicates not handled"
    print("  -> OK")


def test_dup_handled():
    """Agent removed duplicates — PASS."""
    print(f"\n{SEP}")
    print("TEST: DuplicateValidator — duplicates handled")
    print(SEP)

    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )
    df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)

    insights = analyze_dataset(df_with_dups, "target")

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights

    # Agent removed duplicates -> clean df in sandbox
    session.sandbox_namespace["train_df"] = df.copy()

    session.steps = [
        make_step(0, ActionType.CODE, "train_df = train_df.drop_duplicates()"),
    ]

    v = DuplicateValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS: duplicates handled. Got: {result.details}"
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_leakage_model_no_leak():
    """Normal features, no leakage — PASS."""
    print(f"\n{SEP}")
    print("TEST: TargetLeakageModelValidator — no leakage")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True
    session.final_submitted = True

    # Build normal features in sandbox
    code = (
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
    )
    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)

    v = TargetLeakageModelValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS: normal features. Got: {result.details}"
    print("  -> OK")


def test_leakage_model_with_leak():
    """Feature = target + noise → detector catches leakage."""
    print(f"\n{SEP}")
    print("TEST: TargetLeakageModelValidator — leakage via perfect feature")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True
    session.final_submitted = True

    code = (
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
        "import numpy as np\n"
        "X_train['leaked_feature'] = y_train.values + np.random.randn(len(y_train)) * 0.001\n"
    )
    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)

    v = TargetLeakageModelValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, f"Expected FAIL: leaked feature. Got: {result.details}"
    assert result.penalty >= 0.25
    assert "leaked_feature" in result.details
    print("  -> OK")


def test_leakage_model_not_done():
    """Before FINAL_SUBMIT — validator does not run."""
    print(f"\n{SEP}")
    print("TEST: TargetLeakageModelValidator — not done yet (skip)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = False

    v = TargetLeakageModelValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, score: {result.score}")
    print(f"  details: {result.details}")
    assert result.passed
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_full_pipeline():
    """End-to-end: environment with a good agent; all validators appear in output."""
    print(f"\n{SEP}")
    print("TEST: Full pipeline — all new validators visible")
    print(SEP)

    from automl_eval.domain.task_registry import TaskRegistry
    from automl_eval.core.environment import AutoMLEnvironment

    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")

    env = AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary")

    plan = (
        "ACTION: PLAN\n"
        "1. EDA: describe, check distributions, outliers (clip Fare).\n"
        "2. Handle missing: fill Age median, drop Cabin, fill Embarked mode.\n"
        "3. Encode Sex/Embarked, create FamilySize feature.\n"
        "4. Check correlations. Remove duplicates if any.\n"
        "5. Train RandomForest with random_state=42, predict on valid.\n"
    )
    out1 = env.step(plan)
    print(f"  PLAN reward={out1.reward:.4f}")

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "\n"
        "X_train = train_df.drop(columns=['Survived'])\n"
        "y_train = train_df['Survived']\n"
        "\n"
        "print(X_train.describe())\n"
        "print(X_train.isnull().sum())\n"
        "X_train = X_train.drop_duplicates()\n"
        "y_train = y_train.loc[X_train.index]\n"
        "\n"
        "X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())\n"
        "X_train = X_train.drop(columns=['Cabin', 'Name', 'Ticket'])\n"
        "X_train['Embarked'] = X_train['Embarked'].fillna('S')\n"
        "X_train['Sex'] = X_train['Sex'].map({'male': 0, 'female': 1})\n"
        "X_train['Embarked'] = X_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n"
        "X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1\n"
        "X_train = X_train.fillna(0)\n"
        "\n"
        "corr_matrix = X_train.corr()\n"
        "\n"
        "q_low, q_high = X_train['Fare'].quantile([0.01, 0.99])\n"
        "X_train['Fare'] = X_train['Fare'].clip(lower=q_low, upper=q_high)\n"
        "\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X_train.drop(columns=['PassengerId']), y_train)\n"
        "\n"
        "X_val = valid_df.drop(columns=['Survived'])\n"
        "X_val['Age'] = X_val['Age'].fillna(X_val['Age'].median())\n"
        "X_val = X_val.drop(columns=['Cabin', 'Name', 'Ticket'])\n"
        "X_val['Embarked'] = X_val['Embarked'].fillna('S')\n"
        "X_val['Sex'] = X_val['Sex'].map({'male': 0, 'female': 1})\n"
        "X_val['Embarked'] = X_val['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n"
        "X_val['FamilySize'] = X_val['SibSp'] + X_val['Parch'] + 1\n"
        "X_val = X_val.fillna(0)\n"
        "q_low, q_high = X_val['Fare'].quantile([0.01, 0.99])\n"
        "X_val['Fare'] = X_val['Fare'].clip(lower=q_low, upper=q_high)\n"
        "\n"
        "predictions = model.predict_proba(X_val.drop(columns=['PassengerId']))[:, 1]\n"
        "```"
    )
    out2 = env.step(code)
    print(f"  CODE reward={out2.reward:.4f}")

    submit = "ACTION: FINAL_SUBMIT"
    out3 = env.step(submit)
    print(f"  SUBMIT reward={out3.reward:.4f}, done={out3.done}")

    print("\n  --- Validator feedback ---")
    for line in out3.state.split("\n"):
        if "[PASS]" in line or "[FAIL]" in line:
            print(f"  {line.strip()}")

    for vname in ["feature_pipeline", "duplicates", "target_leakage_model"]:
        assert vname in out3.state, f"Validator '{vname}' not in response!"
        print(f"  -> '{vname}' present")

    env.close()
    print("  -> OK")


def main():
    print(SEP)
    print("  TESTING PIPELINE VALIDATORS")
    print(SEP)

    test_fp_raw_data_issues()
    test_fp_clean_data()
    test_fp_no_feature_engineering()

    test_dup_no_duplicates()
    test_dup_synthetic_not_handled()
    test_dup_handled()

    test_leakage_model_no_leak()
    test_leakage_model_with_leak()
    test_leakage_model_not_done()

    test_full_pipeline()

    print(f"\n{SEP}")
    print("  ALL 10 TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
