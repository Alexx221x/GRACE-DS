import nbformat
from pathlib import Path


MODEL_CODE = """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
X = train_df.drop(columns=[target_column])
y = train_df[target_column]
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipeline = Pipeline([("prep", prep), ("model", DecisionTreeClassifier(max_depth=3, random_state=42))])
pipeline.fit(X, y)
def predict_fn(raw_dataframe):
    return pipeline.predict_proba(raw_dataframe)[:, 1]
"""


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 10}


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.request_log = []
        self.local_total_tokens = 100
        self.token_budget_valid = True
        self.budget_exhausted = False

    def invoke(self, messages, context=None):
        self.request_log.append({"context": context})
        return FakeResponse(self.responses.pop(0))


def _namespace():
    project = Path(__file__).resolve().parents[1]
    nb = nbformat.read(project / "titanic_paper_experiment_suite.ipynb", as_version=4)
    ns = {}
    for idx in [3, 5, 7, 9, 11, 13]:
        exec(compile(nb.cells[idx].source, f"cell_{idx}", "exec"), ns, ns)
    return ns


def test_unstructured_inline_validation_marker_no_longer_rolls_back_candidate():
    ns = _namespace()
    llm = FakeLLM([f"```python\n{MODEL_CODE}\nREQUEST_VALIDATION\n```", "STOP_WORKING"])
    result = ns["run_unstructured_agent"](llm)
    working = [
        step for step in result.steps if step.phase in {"working", "auto_validation"}
    ]
    assert result.eligible_for_terminal_comparison
    assert result.execution_failure_count == 0
    assert any(
        step.phase == "auto_validation" and step.validation_metric is not None
        for step in working
    )
    assert any(
        step.constrained_response_format
        == "wrapped_fenced_python_inline_validation_normalized"
        for step in working
    )


def test_primary_restart_is_candidate_matched_and_upper_bound_is_separate():
    ns = _namespace()
    primary = FakeLLM([MODEL_CODE] * ns["N_RESTARTS"])
    primary_result = ns["run_n_restarts_from_scratch"](primary)
    assert primary_result.working_llm_calls == ns["PRIMARY_VALIDATED_CANDIDATE_BUDGET"]
    assert (
        sum(step.phase == "auto_validation" for step in primary_result.steps)
        == ns["PRIMARY_VALIDATED_CANDIDATE_BUDGET"]
    )
    upper = FakeLLM([MODEL_CODE] * ns["N_RESTARTS_CALL_MATCHED_UPPER_BOUND"])
    upper_result = ns["run_n_restarts_call_matched_upper_bound"](upper)
    assert upper_result.working_llm_calls == ns["WORKING_LLM_CALL_BUDGET"]
    assert upper_result.working_llm_calls > primary_result.working_llm_calls


def test_fixed_stage_auto_validation_does_not_consume_llm_validate_turns():
    ns = _namespace()
    fe = "ACTION: FEATURE_ENGINEERING\n```python\nprint('feature step')\n```"
    eda = "ACTION: EDA\n```python\nprint(train_df.shape)\n```"
    plan = (
        "ACTION: PLAN\nBuild replayable candidates and use automatic evaluator scores."
    )
    model = f"ACTION: MODEL\n```python\n{MODEL_CODE}\n```"
    responses = [plan, eda, fe, model, fe, model, model, model]
    result = ns["run_fixed_stage_iterative"](FakeLLM(responses))
    assert result.eligible_for_terminal_comparison
    assert result.working_llm_calls == ns["WORKING_LLM_CALL_BUDGET"]
    result = ns["annotate_result"](
        result, FakeLLM([]), experiment_family="test", temperature=0.2, repeat_index=1
    )
    assert result.validated_candidate_count == ns["PRIMARY_VALIDATED_CANDIDATE_BUDGET"]
    assert result.automatic_validation_count == ns["PRIMARY_VALIDATED_CANDIDATE_BUDGET"]
    assert (
        sum(step.phase == "auto_validation" for step in result.steps)
        == ns["PRIMARY_VALIDATED_CANDIDATE_BUDGET"]
    )
    llm_generated_validate = [
        step
        for step in result.steps
        if step.source == "llm" and step.actual_action == "VALIDATE"
    ]
    assert llm_generated_validate == []


def test_baseline_first_starts_with_model_and_completes_without_generated_validate_action():
    ns = _namespace()
    fe = "ACTION: FEATURE_ENGINEERING\n```python\nprint('feature refinement')\n```"
    eda = "ACTION: EDA\n```python\nprint(train_df.shape)\n```"
    model = f"ACTION: MODEL\n```python\n{MODEL_CODE}\n```"
    responses = [model, eda, fe, model, model, model, model, model]
    result = ns["run_baseline_first_structured"](FakeLLM(responses))
    assert result.eligible_for_terminal_comparison
    assert result.agent_protocol_violation_count == 0
    assert result.steps[0].actual_action == "MODEL"
    assert any(step.phase == "auto_validation" for step in result.steps)
    assert not any(
        step.source == "llm" and step.actual_action == "VALIDATE"
        for step in result.steps
    )


class BudgetStoppingLLM(FakeLLM):
    def __init__(self, responses, exception_type):
        super().__init__(responses)
        self.exception_type = exception_type
        self.budget_exhausted = False

    def invoke(self, messages, context=None):
        if not self.responses:
            self.budget_exhausted = True
            raise self.exception_type(
                "Total local token budget exhausted before the next call: no next request fits."
            )
        return super().invoke(messages, context=context)


def test_fixed_stage_finalizes_validated_candidate_when_no_next_token_budget_call_fits():
    ns = _namespace()
    plan = "ACTION: PLAN\nBuild and validate a replayable model."
    eda = "ACTION: EDA\n```python\nprint('x' * 4000)\n```"
    fe = "ACTION: FEATURE_ENGINEERING\n```python\nprint('feature step')\n```"
    model = f"ACTION: MODEL\n```python\n{MODEL_CODE}\n```"
    llm = BudgetStoppingLLM([plan, eda, fe, model], ns["EpisodeTokenBudgetExceeded"])
    result = ns["run_fixed_stage_iterative"](llm)
    result = ns["annotate_result"](
        result, llm, experiment_family="test", temperature=0.2, repeat_index=1
    )
    assert result.eligible_for_terminal_comparison
    assert result.final_hidden_test_metric is not None
    assert result.finalized_after_token_budget_exhaustion
    assert result.token_budget_exhausted_before_new_call
    assert result.token_budget_valid
    assert result.error is None
    assert any(step.actual_action == "TOKEN_BUDGET_STOP" for step in result.steps)
    eda_step = next(step for step in result.steps if step.actual_action == "EDA")
    assert len(eda_step.audit_feedback) > len(eda_step.public_feedback)
    assert "full output retained in audit log" in eda_step.public_feedback


def test_upper_bound_exports_its_actual_eight_candidate_cap():
    ns = _namespace()
    llm = FakeLLM([MODEL_CODE] * ns["N_RESTARTS_CALL_MATCHED_UPPER_BOUND"])
    result = ns["run_n_restarts_call_matched_upper_bound"](llm)
    result = ns["annotate_result"](
        result, llm, experiment_family="test", temperature=0.2, repeat_index=1
    )
    assert result.validated_candidate_count == ns["N_RESTARTS_CALL_MATCHED_UPPER_BOUND"]
    assert (
        result.validated_candidate_budget_cap
        == ns["N_RESTARTS_CALL_MATCHED_UPPER_BOUND"]
    )
    assert result.candidate_budget_policy == "call_matched_upper_resource_baseline"
