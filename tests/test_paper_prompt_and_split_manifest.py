from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.llm.prompts import build_system_prompt
from automl_eval.domain.task_registry import TaskRegistry


def _env(seed: int) -> AutoMLEnvironment:
    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")
    env = AutoMLEnvironment(registry, seed=seed)
    env.reset("titanic_binary", max_actions=2)
    return env


def test_prompt_exposes_public_versions_and_print_instruction():
    prompt = build_system_prompt(8)
    assert "Public runtime versions available for this episode" in prompt
    assert "scikit-learn=" in prompt
    assert "use print(...)" in prompt
    observation = _env(42).observe()
    assert "Public runtime versions:" in observation
    assert "use print(...)" in observation


def test_split_manifest_reproducible_and_label_free():
    first = _env(42).evaluator_split_manifest(include_indices=True)
    second = _env(42).evaluator_split_manifest(include_indices=True)
    third = _env(73).evaluator_split_manifest(include_indices=True)
    assert first == second
    assert first["split_id"] != third["split_id"]
    assert first["contains_hidden_test_labels"] is False
    assert first["split_strategy"]["stratify"] is True
    assert first["split_strategy_resolved"]["stratification_applied"] is True
    assert first["split_strategy_resolved"]["stratification_values_exported"] is False
    assert "hidden_test" in first["indices"]
    assert "target_values" not in first
    assert "hidden_test_targets" not in first


def test_model_search_policy_in_main_prompt():
    prompt = build_system_prompt(8)
    assert "Model search and hyperparameter policy" in prompt
    assert "no `GridSearchCV`, `RandomizedSearchCV`" in prompt
    assert "Choose a small number of sensible hyperparameters yourself" in prompt
    assert "LogisticRegression" in prompt


def test_model_search_policy_in_regime_prompts():
    from automl_eval.experiment import _regimes_extracted as R

    constrained = R.constrained_model_only_system_prompt(R.StudyMode.SINGLE_SHOT)
    unstructured = R.unstructured_agent_system_prompt(working_calls=4)
    direct = R.direct_model_task_request()

    assert "Model search and hyperparameter policy" in constrained
    assert "Model search and hyperparameter policy" in unstructured
    assert "do not run grid/randomized/automated tuning" in direct
