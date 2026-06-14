import nbformat
from pathlib import Path


MODEL_CODE = """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
X = train_df.drop(columns=[target_column])
y = train_df[target_column]
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipeline = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=10, random_state=42))])
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
    nb = nbformat.read(
        project / "titanic_all_approaches_comparison.ipynb", as_version=4
    )
    ns = {}
    for idx in [3, 5, 7, 9, 11, 13]:
        exec(compile(nb.cells[idx].source, f"cell_{idx}", "exec"), ns, ns)
    return ns


def test_single_shot_has_private_validation_and_single_terminal_test():
    ns = _namespace()
    llm = FakeLLM([MODEL_CODE])
    result = ns["run_single_shot"](llm)
    result = ns["annotate_result"](
        result, llm, experiment_family="test", temperature=0.2, repeat_index=1
    )
    assert result.selected_validation_metric is not None
    assert result.final_hidden_test_metric is not None
    assert result.terminal_hidden_test_evaluation_count == 1
    assert result.working_hidden_test_evaluation_count == 0
    assert result.validation_request_count == 1
    assert result.split_id is not None


def test_unstructured_mode_gets_code_feedback_and_can_validate():
    ns = _namespace()
    llm = FakeLLM(
        [f"```python\n{MODEL_CODE}\n```", "REQUEST_VALIDATION", "STOP_WORKING"]
    )
    result = ns["run_unstructured_agent"](llm)
    assert result.final_hidden_test_metric is not None
    assert result.eligible_for_terminal_comparison
    assert any(step.actual_action == "VALIDATE" for step in result.steps)
