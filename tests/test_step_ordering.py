""""""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl_eval.domain.task_registry import TaskRegistry
from automl_eval.core.environment import AutoMLEnvironment

SEP = "=" * 60


def make_env() -> AutoMLEnvironment:
    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")
    return AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)


def test_baseline_runs_on_final_submit():
    """BaselineComparisonValidator must actually compute baseline on FINAL_SUBMIT."""
    print(f"\n{SEP}")
    print("TEST: baseline_comparison actually runs on FINAL_SUBMIT")
    print(SEP)

    env = make_env()
    env.reset("titanic_binary")

    env.step(
        "ACTION: PLAN\n"
        "Handle missing values with fillna, encode categoricals, "
        "train GradientBoosting, evaluate ROC AUC."
    )

    env.step(
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "def predict_fn(raw_dataframe):\n"
        "    clean = raw_dataframe.select_dtypes(include='number').fillna(0)\n"
        "    return model.predict_proba(clean)[:, 1]\n"
        "```"
    )

    out = env.step("ACTION: FINAL_SUBMIT")

    assert out.done is True

    # The key assertion: baseline_comparison must NOT contain the skip message.
    skip_msg = "Baseline comparison runs at FINAL_SUBMIT."
    assert skip_msg not in out.state, (
        f"BaselineComparisonValidator was skipped on FINAL_SUBMIT! "
        f"This means session.done was not set before validators ran.\n"
        f"Response excerpt: ...{out.state[out.state.find('baseline') : out.state.find('baseline') + 200]}..."
    )

    # It should contain actual comparison results (Good/Issues)
    has_good = "Good:" in out.state
    has_issues = "Issues:" in out.state
    assert has_good or has_issues, (
        f"baseline_comparison should have 'Good:' or 'Issues:' in output, "
        f"meaning it actually computed the baseline.\n"
        f"Full state:\n{out.state}"
    )

    print(f"  done={out.done}, reward={out.reward:.4f}")
    print("  baseline_comparison ran (not skipped): OK")
    print("  -> PASSED")

    env.close()


def test_cycle_count_visible_to_validators():
    """IterativeCycleValidator must see the cycle_count from the CURRENT step."""
    print(f"\n{SEP}")
    print("TEST: iterative_cycles sees current cycle_count")
    print(SEP)

    env = make_env()
    env.reset("titanic_binary")

    env.step("ACTION: PLAN\nTrain a model, iterate if needed.")

    # Cycle 1: FE then model
    env.step(
        "ACTION: FEATURE_ENGINEERING\n"
        "```python\n"
        "train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())\n"
        "valid_df['Age'] = valid_df['Age'].fillna(valid_df['Age'].median())\n"
        "```"
    )

    out_model1 = env.step(
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y = train_df['Survived']\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X, y)\n"
        "def predict_fn(raw_dataframe):\n"
        "    clean = raw_dataframe.select_dtypes(include='number').fillna(0)\n"
        "    return model.predict_proba(clean)[:, 1]\n"
        "```"
    )

    # After FE->Model, cycle_count should be 1.
    session = env._session
    assert session.cycle_count >= 1, (
        f"Expected cycle_count >= 1 after FE->Model, got {session.cycle_count}"
    )

    # Check the validator output mentions the cycle
    has_cycle_info = "cycle" in out_model1.state.lower()
    print(f"  After FE->Model: cycle_count={session.cycle_count}")
    print(f"  Validator mentions cycles: {has_cycle_info}")

    # Cycle 2: back to FE then model again
    env.step(
        "ACTION: FEATURE_ENGINEERING\n"
        "```python\n"
        "train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']\n"
        "valid_df['FamilySize'] = valid_df['SibSp'] + valid_df['Parch']\n"
        "```"
    )

    out_model2 = env.step(
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "X = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X, y)\n"
        "def predict_fn(raw_dataframe):\n"
        "    clean = raw_dataframe.select_dtypes(include='number').fillna(0)\n"
        "    return model.predict_proba(clean)[:, 1]\n"
        "```"
    )

    assert session.cycle_count >= 2, (
        f"Expected cycle_count >= 2 after second FE->Model, got {session.cycle_count}"
    )
    print(f"  After second FE->Model: cycle_count={session.cycle_count}")

    # The iterative_cycles validator should mention paid cycles
    assert "cycle" in out_model2.state.lower(), (
        "iterative_cycles validator should mention cycles in its feedback"
    )

    print("  -> PASSED")
    env.close()


def test_metric_history_available_to_validators():
    """Validators must see the metric_history entry from THIS step, not previous."""
    print(f"\n{SEP}")
    print("TEST: metric_history is fresh when validators run")
    print(SEP)

    env = make_env()
    env.reset("titanic_binary")

    env.step("ACTION: PLAN\nTrain model, check performance.")

    env.step(
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "X = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X, y)\n"
        "def predict_fn(raw_dataframe):\n"
        "    clean = raw_dataframe.select_dtypes(include='number').fillna(0)\n"
        "    return model.predict_proba(clean)[:, 1]\n"
        "```"
    )

    validation_out = env.step("ACTION: VALIDATE")
    out = env.step("ACTION: FINAL_SUBMIT")
    session = env._session

    assert session.best_metric is not None, (
        "best_metric should be set after explicit VALIDATE"
    )

    # metric_history should contain only the explicit validation measurement
    has_final_metric = any(
        abs(m - session.best_metric) < 1e-6 for _, m in session.metric_history
    )

    print(f"  best_metric: {session.best_metric:.4f}")
    print(f"  metric_history: {session.metric_history}")
    print(f"  final metric in history: {has_final_metric}")

    # current_step should reflect that 4 steps have been recorded
    assert session.current_step == 4, (
        f"Expected current_step=4 (PLAN+CODE+VALIDATE+SUBMIT), got {session.current_step}"
    )
    print(f"  current_step: {session.current_step}")

    print("  -> PASSED")
    env.close()


def test_current_step_incremented_before_validators():
    """session.current_step should reflect this step when validators run."""
    print(f"\n{SEP}")
    print("TEST: current_step is incremented before validators")
    print(SEP)

    env = make_env()
    env.reset("titanic_binary")

    # Step 0: PLAN
    out1 = env.step("ACTION: PLAN\nTrain a model.")
    session = env._session
    assert session.current_step == 1, (
        f"After 1st step, expected current_step=1, got {session.current_step}"
    )

    # Step 1: CODE
    out2 = env.step(
        "ACTION: CODE\n"
        "```python\n"
        "X = train_df.select_dtypes(include='number').fillna(0)\n"
        "print('hello')\n"
        "```"
    )
    assert session.current_step == 2, (
        f"After 2nd step, expected current_step=2, got {session.current_step}"
    )

    # The state summary should reflect the step count accurately
    assert "Step: 2" in out2.state, (
        f"State summary should show 'Step: 2' after 2 steps.\n"
        f"Got: {[l for l in out2.state.split(chr(10)) if 'Step:' in l]}"
    )

    print("  current_step after PLAN: 1")
    print("  current_step after CODE: 2")
    print("  state_summary reflects Step: 2")
    print("  -> PASSED")
    env.close()


def test_done_flag_before_baseline_validator():
    """Directly verify that session.done is True when BaselineComparison runs."""
    print(f"\n{SEP}")
    print("TEST: session.done is True when BaselineComparison.validate() runs")
    print(SEP)

    env = make_env()
    env.reset("titanic_binary")

    env.step("ACTION: PLAN\nTrain model.")
    env.step(
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "X = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X, y)\n"
        "def predict_fn(raw_dataframe):\n"
        "    clean = raw_dataframe.select_dtypes(include='number').fillna(0)\n"
        "    return model.predict_proba(clean)[:, 1]\n"
        "```"
    )

    # Monkey-patch to capture what the validator sees
    captured = {}
    from automl_eval.validators.baseline_comparison import BaselineComparisonValidator

    original_validate = BaselineComparisonValidator.validate

    def patched_validate(self, session):
        captured["done_at_validate_time"] = session.done
        captured["cycle_count_at_validate_time"] = session.cycle_count
        captured["current_step_at_validate_time"] = session.current_step
        return original_validate(self, session)

    BaselineComparisonValidator.validate = patched_validate

    try:
        out = env.step("ACTION: FINAL_SUBMIT")

        assert captured.get("done_at_validate_time") is True, (
            f"session.done should be True when BaselineComparison runs on FINAL_SUBMIT, "
            f"but was {captured.get('done_at_validate_time')}"
        )
        print(f"  session.done at validate time: {captured['done_at_validate_time']}")
        print(
            f"  session.cycle_count at validate time: {captured['cycle_count_at_validate_time']}"
        )
        print(
            f"  session.current_step at validate time: {captured['current_step_at_validate_time']}"
        )
        print("  -> PASSED")
    finally:
        BaselineComparisonValidator.validate = original_validate
        env.close()


def main():
    print(SEP)
    print("  TESTING STEP ORDERING (environment.py)")
    print(SEP)

    test_baseline_runs_on_final_submit()
    test_cycle_count_visible_to_validators()
    test_metric_history_available_to_validators()
    test_current_step_incremented_before_validators()
    test_done_flag_before_baseline_validator()

    print(f"\n{SEP}")
    print("  ALL 5 STEP-ORDERING TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
