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
from automl_eval.validators.correlation import CorrelationValidator
from automl_eval.validators.missing_values import MissingValuesValidator
from automl_eval.validators.distribution import DistributionValidator

SEP = "=" * 60


def make_task(time_budget: float = 300.0) -> Task:
    return Task(
        task_id="test_dq",
        dataset_path="data/titanic.csv",
        target_column="Survived",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ROC_AUC,
        description="Test task for data quality validators.",
        time_budget_seconds=time_budget,
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


# ── DataInsightsAnalyzer ────────────────────────────────────────


def test_data_insights_titanic():
    """Check that DataInsights finds missing values and outliers on Titanic."""
    print(f"\n{SEP}")
    print("TEST: DataInsightsAnalyzer on Titanic")
    print(SEP)

    df = pd.read_csv("data/titanic.csv")
    insights = analyze_dataset(df, "Survived")

    print(f"  n_rows={insights.n_rows}, n_cols={insights.n_cols}")
    print(f"  numeric: {insights.numeric_columns}")
    print(f"  categorical: {insights.categorical_columns}")
    print(f"  has_missing: {insights.has_missing}")
    for cm in insights.missing_columns:
        print(f"    {cm.column}: {cm.missing_frac:.1%} missing -> {cm.recommend}")
    print(f"  has_high_correlation: {insights.has_high_correlation}")
    for cp in insights.high_corr_pairs:
        print(f"    {cp.col_a} <-> {cp.col_b}: r={cp.corr_value}")
    print(f"  has_outliers: {insights.has_outliers}")
    for oi in insights.outlier_columns:
        print(f"    {oi.column}: {oi.outlier_count} outliers ({oi.outlier_frac:.1%})")
    print(f"  class_imbalance_ratio: {insights.class_imbalance_ratio}")

    assert insights.has_missing, "Titanic should have missing values"
    assert any(c.column == "Age" for c in insights.missing_columns), (
        "Age should have missing"
    )
    assert insights.n_rows > 100
    print("  -> OK")


# ── CorrelationValidator ────────────────────────────────────────


def test_correlation_no_high_corr():
    """When the data has no high correlations — PASS."""
    print(f"\n{SEP}")
    print("TEST: CorrelationValidator — no high correlations")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    if not session.data_insights.has_high_correlation:
        print(
            "  Titanic has no pairs with corr > 0.9 — validator should PASS by default"
        )
        v = CorrelationValidator()
        result = v.validate(session)
        print(f"  passed: {result.passed}, score: {result.score}")
        print(f"  details: {result.details}")
        assert result.passed
        print("  -> OK")
        return

    print("  Titanic has high corr pairs — skipping this test variant")
    print("  -> SKIP (dataset has correlations)")


def test_correlation_with_synthetic_data():
    """Synthetic dataset: two features are almost identical. Agent does not drop — FAIL."""
    print(f"\n{SEP}")
    print("TEST: CorrelationValidator — high corr not addressed (synthetic)")
    print(SEP)

    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01
    x3 = np.random.randn(n)
    y = (x1 + x3 > 0).astype(int)
    df = pd.DataFrame({"feat_a": x1, "feat_b": x2, "feat_c": x3, "target": y})

    insights = analyze_dataset(df, "target", corr_threshold=0.90)
    print(
        f"  high_corr_pairs: {[(p.col_a, p.col_b, p.corr_value) for p in insights.high_corr_pairs]}"
    )
    assert insights.has_high_correlation, "Expected high corr between feat_a and feat_b"

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights

    session.steps = [
        make_step(0, ActionType.CODE, "model.fit(X, y)"),
    ]

    v = CorrelationValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: no corr analysis, no drop"
    assert result.penalty > 0
    print("  -> OK")


def test_correlation_addressed():
    """Agent calls .corr() and drops the correlated feature — PASS."""
    print(f"\n{SEP}")
    print("TEST: CorrelationValidator — corr addressed (drop + .corr())")
    print(SEP)

    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01
    x3 = np.random.randn(n)
    y = (x1 + x3 > 0).astype(int)
    df = pd.DataFrame({"feat_a": x1, "feat_b": x2, "feat_c": x3, "target": y})

    insights = analyze_dataset(df, "target", corr_threshold=0.90)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights

    session.steps = [
        make_step(0, ActionType.CODE, "corr_matrix = df.corr()\nprint(corr_matrix)"),
        make_step(
            1, ActionType.FEATURE_ENGINEERING, "df = df.drop(columns=['feat_b'])"
        ),
    ]

    v = CorrelationValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, (
        f"Expected PASS: corr analysis done + feat_b dropped. Got: {result.details}"
    )
    print("  -> OK")


# ── MissingValuesValidator ──────────────────────────────────────


def test_missing_no_handling():
    """Agent ignores missing values — FAIL."""
    print(f"\n{SEP}")
    print("TEST: MissingValuesValidator — no handling at all")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(0, ActionType.CODE, "model.fit(X, y)"),
    ]

    v = MissingValuesValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: missing values not handled"
    assert result.penalty >= 0.15
    print("  -> OK")


def test_missing_handled_fillna():
    """Agent correctly fills Age and drops Cabin."""
    print(f"\n{SEP}")
    print("TEST: MissingValuesValidator — fillna + drop Cabin")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.plan_text = (
        "Handle missing values: fill Age with median, drop Cabin column."
    )

    session.steps = [
        make_step(
            0,
            ActionType.FEATURE_ENGINEERING,
            "df['Age'] = df['Age'].fillna(df['Age'].median())",
        ),
        make_step(1, ActionType.FEATURE_ENGINEERING, "df = df.drop(columns=['Cabin'])"),
        make_step(
            2,
            ActionType.FEATURE_ENGINEERING,
            "df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])",
        ),
    ]

    v = MissingValuesValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.penalty < 0.15, f"Expected low penalty, got {result.penalty}"
    print("  -> OK")


def test_missing_bad_strategy_fill_mostly_null():
    """Column with >80% NaN is filled instead of dropped — bad strategy (expect failure)."""
    print(f"\n{SEP}")
    print("TEST: MissingValuesValidator — filling column with >80% missing")
    print(SEP)

    np.random.seed(42)
    n = 100
    col_vals = np.random.randn(n)
    col_vals[:85] = np.nan
    df = pd.DataFrame(
        {
            "bad_col": col_vals,
            "good_col": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )

    insights = analyze_dataset(df, "target")
    print(
        f"  missing columns: {[(m.column, f'{m.missing_frac:.0%}', m.recommend) for m in insights.missing_columns]}"
    )

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights
    session.plan_text = "Handle missing values."

    session.steps = [
        make_step(0, ActionType.CODE, "df['bad_col'] = df['bad_col'].fillna(0)"),
    ]

    v = MissingValuesValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: bad strategy for heavily-null column"
    print("  -> OK")


# ── DistributionValidator ───────────────────────────────────────


def test_distribution_no_eda():
    """Agent does no EDA — FAIL."""
    print(f"\n{SEP}")
    print("TEST: DistributionValidator — no EDA at all")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(0, ActionType.CODE, "model.fit(X, y)"),
    ]

    v = DistributionValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: no EDA"
    assert result.penalty >= 0.10
    print("  -> OK")


def test_distribution_with_eda():
    """Agent runs describe + clips outliers — PASS."""
    print(f"\n{SEP}")
    print("TEST: DistributionValidator — describe + outlier clip")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()

    session.steps = [
        make_step(0, ActionType.CODE, "print(df.describe())\nprint(df.info())"),
        make_step(
            1,
            ActionType.FEATURE_ENGINEERING,
            "q_low, q_high = df['Fare'].quantile([0.01, 0.99])\n"
            "df['Fare'] = df['Fare'].clip(lower=q_low, upper=q_high)",
        ),
    ]

    v = DistributionValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, f"Expected PASS. Got: {result.details}"
    print("  -> OK")


def test_distribution_eda_in_plan():
    """Agent mentions EDA and outliers in the plan — PASS."""
    print(f"\n{SEP}")
    print("TEST: DistributionValidator — EDA mentioned in plan")
    print(SEP)

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.plan_text = (
        "Step 1: Exploratory Data Analysis — investigate distributions and outliers.\n"
        "Step 2: Clip outliers using IQR.\n"
        "Step 3: Train model."
    )

    session.steps = []

    v = DistributionValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert result.passed, (
        f"Expected PASS (plan mentions EDA + outlier). Got: {result.details}"
    )
    print("  -> OK")


def test_distribution_outliers_not_handled():
    """Data has outliers; agent runs describe but does not handle them — partial FAIL."""
    print(f"\n{SEP}")
    print("TEST: DistributionValidator — outliers present but not handled")
    print(SEP)

    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    x[:20] = x[:20] * 100
    df = pd.DataFrame({"feature": x, "target": np.random.randint(0, 2, n)})
    insights = analyze_dataset(df, "target")
    print(
        f"  outlier_columns: {[(o.column, o.outlier_frac) for o in insights.outlier_columns]}"
    )

    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights

    session.steps = [
        make_step(0, ActionType.CODE, "print(df.describe())"),
    ]

    v = DistributionValidator()
    result = v.validate(session)

    print(f"  passed: {result.passed}, penalty: {result.penalty}")
    print(f"  details: {result.details}")
    assert not result.passed, "Expected FAIL: outliers not handled"
    assert result.penalty >= 0.08
    print("  -> OK")


def test_correlation_spearman_analysis_in_explicit_eda_is_accepted():
    """Explicit EDA plus an equivalent correlation method should satisfy the diagnostic."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01
    df = pd.DataFrame({"feat_a": x1, "feat_b": x2, "target": (x1 > 0).astype(int)})
    insights = analyze_dataset(df, "target", corr_threshold=0.90)
    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.data_insights = insights
    session.steps = [
        make_step(
            0,
            ActionType.EDA,
            "from scipy.stats import spearmanr\nrho = spearmanr(df['feat_a'], df['feat_b'])",
        ),
        make_step(
            1,
            ActionType.FEATURE_ENGINEERING,
            "from sklearn.decomposition import PCA\npca = PCA(n_components=1)",
        ),
    ]
    result = CorrelationValidator().validate(session)
    assert result.passed, result.details


def test_distribution_quantile_eda_in_explicit_stage_is_accepted():
    """Explicit EDA may use quantiles instead of describe/hist, with robust handling."""
    task = make_task()
    session = RuntimeSession(task)
    session.initialize()
    session.steps = [
        make_step(
            0, ActionType.EDA, "bounds = train_df['Fare'].quantile([0.01, 0.99])"
        ),
        make_step(
            1,
            ActionType.FEATURE_ENGINEERING,
            "from sklearn.preprocessing import RobustScaler\nscale = RobustScaler()",
        ),
    ]
    result = DistributionValidator().validate(session)
    assert result.passed, result.details


# ── Full pipeline ───────────────────────────────────────────────


def test_full_pipeline_data_quality():
    """End-to-end: run the environment and assert new validators appear in feedback."""
    print(f"\n{SEP}")
    print("TEST: Full pipeline — data quality validators in feedback")
    print(SEP)

    from automl_eval.domain.task_registry import TaskRegistry
    from automl_eval.core.environment import AutoMLEnvironment

    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")

    env = AutoMLEnvironment(registry, seed=42, reveal_internal_feedback=True)
    env.reset("titanic_binary")

    plan = (
        "ACTION: PLAN\n"
        "1. Exploratory data analysis: describe, check distributions and outliers.\n"
        "2. Handle missing values: fill Age with median, drop Cabin, fill Embarked with mode.\n"
        "3. Encode categoricals, compute correlation matrix.\n"
        "4. Train Random Forest with random_state=42, evaluate ROC AUC."
    )
    out1 = env.step(plan)
    print(f"  PLAN reward={out1.reward:.4f}")

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "\n"
        "X = train_df.drop(columns=['Survived'])\n"
        "y = train_df['Survived']\n"
        "\n"
        "print(X.describe())\n"
        "print(X.isnull().sum())\n"
        "corr_matrix = X.select_dtypes(include='number').corr()\n"
        "\n"
        "X['Age'] = X['Age'].fillna(X['Age'].median())\n"
        "X = X.drop(columns=['Cabin', 'Name', 'Ticket'])\n"
        "X['Embarked'] = X['Embarked'].fillna('S')\n"
        "X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})\n"
        "X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n"
        "X = X.fillna(0)\n"
        "\n"
        "q_low, q_high = X['Fare'].quantile([0.01, 0.99])\n"
        "X['Fare'] = X['Fare'].clip(lower=q_low, upper=q_high)\n"
        "\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X, y)\n"
        "\n"
        "X_val = valid_df.drop(columns=['Survived'])\n"
        "X_val['Age'] = X_val['Age'].fillna(X_val['Age'].median())\n"
        "X_val = X_val.drop(columns=['Cabin', 'Name', 'Ticket'])\n"
        "X_val['Embarked'] = X_val['Embarked'].fillna('S')\n"
        "X_val['Sex'] = X_val['Sex'].map({'male': 0, 'female': 1})\n"
        "X_val['Embarked'] = X_val['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n"
        "X_val = X_val.fillna(0)\n"
        "q_low, q_high = X_val['Fare'].quantile([0.01, 0.99])\n"
        "X_val['Fare'] = X_val['Fare'].clip(lower=q_low, upper=q_high)\n"
        "\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "```"
    )
    out2 = env.step(code)
    print(f"  CODE reward={out2.reward:.4f}")

    submit = "ACTION: FINAL_SUBMIT"
    out3 = env.step(submit)
    print(f"  SUBMIT reward={out3.reward:.4f}, done={out3.done}")

    print("\n  --- Full step response ---")
    print(out3.state)

    for vname in ["correlation", "missing_values", "distribution"]:
        assert vname in out3.state, f"Validator '{vname}' not in response!"
        print(f"  -> Validator '{vname}' present")

    assert out3.reward >= 0.0, f"Expected non-negative reward, got {out3.reward}"
    print(f"  -> Reward {out3.reward:.4f} >= 0.0")

    env.close()
    print("  -> OK")


def main():
    print(SEP)
    print("  TESTING DATA QUALITY VALIDATORS")
    print(SEP)

    test_data_insights_titanic()

    test_correlation_no_high_corr()
    test_correlation_with_synthetic_data()
    test_correlation_addressed()

    test_missing_no_handling()
    test_missing_handled_fillna()
    test_missing_bad_strategy_fill_mostly_null()

    test_distribution_no_eda()
    test_distribution_with_eda()
    test_distribution_eda_in_plan()
    test_distribution_outliers_not_handled()

    test_full_pipeline_data_quality()

    print(f"\n{SEP}")
    print("  ALL 12 TESTS PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
