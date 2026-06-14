"""Regression tests for the non-stage-aware autonomous coding-agent baseline."""

from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.domain.task_registry import TaskRegistry


def _registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.load_directory("automl_eval/tasks")
    return registry


def _code_cell(idx: int) -> str:
    return f"""ACTION: CODE
```python
print('arbitrary code cell {idx}')
scratch_{idx} = {idx}
```"""


def test_unstructured_execution_can_disable_hidden_stage_local_governance() -> None:
    env = AutoMLEnvironment(
        _registry(),
        seed=42,
        allow_forced_terminal_evaluation=False,
        enforce_stage_budgets=False,
    )
    env.reset("titanic_binary", max_actions=6)
    outputs = [env.step(_code_cell(idx)) for idx in range(1, 5)]
    assert all(not output.done for output in outputs)
    assert all("Stage budget exceeded" not in output.state for output in outputs)
    assert env._session is not None
    assert env._session.hidden_test_metric is None
    assert env._session.current_step == 4


def test_structured_default_still_enforces_stage_local_governance() -> None:
    env = AutoMLEnvironment(
        _registry(), seed=42, allow_forced_terminal_evaluation=False
    )
    env.reset("titanic_binary", max_actions=6)
    outputs = [env.step(_code_cell(idx)) for idx in range(1, 4)]
    assert "Stage budget exceeded" not in outputs[-1].state
    warning = env.step(_code_cell(4))
    assert "Stage budget exceeded for CODE" in warning.state
