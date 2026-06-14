""""""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from automl_eval.domain.task import Task, TaskType, MetricName
from automl_eval.core.session import RuntimeSession, StepRecord, ActionType
from automl_eval.core.sandbox import Sandbox
from automl_eval.validators.feature_importance import FeatureImportanceValidator
from automl_eval.validators.hyperparam import HyperparamValidator
from automl_eval.validators.model_choice import ModelChoiceValidator
from automl_eval.validators.split_quality import SplitValidator

SEP = "=" * 60


def make_task() -> Task:
    return Task(
        task_id="test_model",
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


def test_fi_no_inspection():
    """Agent does not inspect importance — penalty."""
    print(f"\n{SEP}")
    print("TEST: FeatureImportanceValidator — no importance inspection")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier(n_estimators=10, random_state=42)\n"
            "model.fit(X, y)",
        ),
    ]

    v = FeatureImportanceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: no importance inspection"
    assert result.penalty >= 0.04
    print("  -> OK")


def test_fi_with_inspection():
    """Agent inspects feature_importances_ — PASS."""
    print(f"\n{SEP}")
    print("TEST: FeatureImportanceValidator — inspects importances")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier(n_estimators=10, random_state=42)\n"
            "model.fit(X, y)\n"
            "print(model.feature_importances_)",
        ),
    ]

    v = FeatureImportanceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS. Got: {result.details}"
    print("  -> OK")


def test_fi_model_value():
    """Model value test: train a model in the sandbox for real, check stacking."""
    print(f"\n{SEP}")
    print("TEST: FeatureImportanceValidator — model value test (real execution)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True

    code = (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "import numpy as np\n"
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "print('feature_importances_:', model.feature_importances_)\n"
    )

    sb = Sandbox(timeout_seconds=30)
    sb.execute(code, session.sandbox_namespace)
    preds = session.sandbox_namespace.get("predictions")
    if preds is not None:
        session.predictions = np.asarray(preds)

    session.steps = [make_step(0, ActionType.CODE, code)]

    v = FeatureImportanceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_hp_all_defaults():
    """Agent set no hyperparameters — penalty."""
    print(f"\n{SEP}")
    print("TEST: HyperparamValidator — all defaults")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X, y)",
        ),
    ]

    v = HyperparamValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: all defaults"
    print("  -> OK")


def test_hp_explicit_params():
    """Agent sets hyperparameters — PASS."""
    print(f"\n{SEP}")
    print("TEST: HyperparamValidator — explicit params")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\n"
            "model.fit(X, y)",
        ),
    ]

    v = HyperparamValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS. Got: {result.details}"
    print("  -> OK")


def test_hp_bad_params():
    """n_estimators=1 — anti-pattern."""
    print(f"\n{SEP}")
    print("TEST: HyperparamValidator — bad params (n_estimators=1)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier(n_estimators=1, max_depth=5)\n"
            "model.fit(X, y)",
        ),
    ]

    v = HyperparamValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: bad params"
    assert result.penalty >= 0.05
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_mc_good_model():
    """Agent uses sklearn RandomForest — PASS."""
    print(f"\n{SEP}")
    print("TEST: ModelChoiceValidator — good model (RandomForest)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier()\nmodel.fit(X, y)",
        ),
    ]

    v = ModelChoiceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS. Got: {result.details}"
    assert "proven tabular model" in result.details.lower()
    print("  -> OK")


def test_mc_ann_only():
    """Agent uses only a neural net on tabular data — penalty."""
    print(f"\n{SEP}")
    print("TEST: ModelChoiceValidator — ANN only on tabular")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "import torch.nn as nn\n"
            "model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))\n"
            "model.fit(X, y)",
        ),
    ]

    v = ModelChoiceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: ANN on tabular"
    assert result.penalty >= 0.12
    print("  -> OK")


def test_mc_both_nn_and_tree():
    """Agent tried both a tree model and NN — no penalty, bonus."""
    print(f"\n{SEP}")
    print("TEST: ModelChoiceValidator — both NN and tree")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "model = GradientBoostingClassifier()\nmodel.fit(X, y)",
        ),
        make_step(
            1,
            ActionType.CODE,
            "from sklearn.neural_network import MLPClassifier\n"
            "nn = MLPClassifier(hidden_layer_sizes=(64,))\nnn.fit(X, y)",
        ),
    ]

    v = ModelChoiceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS (both tried). Got: {result.details}"
    print("  -> OK")


def test_mc_no_model():
    """Agent did not train a model when done=True — penalty."""
    print(f"\n{SEP}")
    print("TEST: ModelChoiceValidator — no model at all (done)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.done = True

    session.steps = [
        make_step(0, ActionType.CODE, "x = train_df.describe()"),
    ]

    v = ModelChoiceValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: no model"
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_split_no_cv():
    """No CV but using environment split — PASS (CV optional with env split)."""
    print(f"\n{SEP}")
    print("TEST: SplitValidator — no cross-validation (env split)")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(0, ActionType.CODE, "model.fit(X_train, y_train)"),
    ]

    v = SplitValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, "Expected PASS: env split is acceptable without CV"
    print("  -> OK")


def test_split_with_cv():
    """Uses StratifiedKFold — PASS."""
    print(f"\n{SEP}")
    print("TEST: SplitValidator — with StratifiedKFold")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
            "skf = StratifiedKFold(n_splits=5)\n"
            "scores = cross_val_score(model, X, y, cv=skf)\n"
            "model.fit(X, y)",
        ),
    ]

    v = SplitValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS. Got: {result.details}"
    print("  -> OK")


def test_split_timeseries_shuffle():
    """Time series with shuffle=True — penalty."""
    print(f"\n{SEP}")
    print("TEST: SplitValidator — time series with shuffle=True")
    print(SEP)

    task = make_task()
    task.metadata["is_time_series"] = True
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.model_selection import train_test_split\n"
            "X_tr, X_val = train_test_split(X, test_size=0.2, shuffle=True)\n"
            "model.fit(X_tr, y_tr)",
        ),
    ]

    v = SplitValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: shuffle on time series"
    assert result.penalty >= 0.15
    print("  -> OK")


def test_split_tiny_test():
    """test_size=0.01 — test set too small."""
    print(f"\n{SEP}")
    print("TEST: SplitValidator — tiny test_size")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(
            0,
            ActionType.CODE,
            "from sklearn.model_selection import train_test_split, cross_val_score\n"
            "X_tr, X_val = train_test_split(X, test_size=0.01)\n"
            "scores = cross_val_score(model, X_tr, y_tr, cv=3)\n"
            "model.fit(X_tr, y_tr)",
        ),
    ]

    v = SplitValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: tiny test"
    print("  -> OK")


# ═══════════════════════════════════════════════════════════════


def test_full_pipeline():
    """End-to-end: all four validators appear in feedback."""
    print(f"\n{SEP}")
    print("TEST: Full pipeline — all model validators visible")
    print(SEP)

    from automl_eval.domain.task_registry import TaskRegistry
    from automl_eval.core.environment import AutoMLEnvironment

    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")

    env = AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary")

    plan = (
        "ACTION: PLAN\n"
        "1. EDA + handle missing + encode categoricals.\n"
        "2. Feature importance analysis.\n"
        "3. Train GradientBoosting with tuned hyperparameters.\n"
        "4. Cross-validation for reliable estimates.\n"
    )
    env.step(plan)

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "from sklearn.model_selection import cross_val_score\n"
        "\n"
        "X_train = train_df.drop(columns=['Survived'])\n"
        "y_train = train_df['Survived']\n"
        "X_train = X_train.select_dtypes(include='number').fillna(0)\n"
        "\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, "
        "learning_rate=0.1, random_state=42)\n"
        "scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')\n"
        "model.fit(X_train, y_train)\n"
        "print('feature_importances_:', model.feature_importances_)\n"
        "\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "```"
    )
    env.step(code)

    out = env.step("ACTION: FINAL_SUBMIT")
    print(f"  SUBMIT reward={out.reward:.4f}, done={out.done}")

    print("\n  --- Validator feedback (model block) ---")
    for line in out.state.split("\n"):
        for key in [
            "feature_importance",
            "hyperparameters",
            "model_choice",
            "split_quality",
        ]:
            if key in line:
                print(f"  {line.strip()}")

    for vname in [
        "feature_importance",
        "hyperparameters",
        "model_choice",
        "split_quality",
    ]:
        assert vname in out.state, f"Validator '{vname}' not in response!"
        print(f"  -> '{vname}' present")

    env.close()
    print("  -> OK")


def main():
    print(SEP)
    print("  TESTING MODEL VALIDATORS")
    print(SEP)

    test_fi_no_inspection()
    test_fi_with_inspection()
    test_fi_model_value()

    test_hp_all_defaults()
    test_hp_explicit_params()
    test_hp_bad_params()

    test_mc_good_model()
    test_mc_ann_only()
    test_mc_both_nn_and_tree()
    test_mc_no_model()

    test_split_no_cv()
    test_split_with_cv()
    test_split_timeseries_shuffle()
    test_split_tiny_test()

    test_full_pipeline()

    print(f"\n{SEP}")
    print("  ALL 15 TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
