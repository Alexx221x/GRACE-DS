"""Gymnasium-compatible wrapper for the GRACE environment"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False

from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.domain.task_registry import TaskRegistry


def gym_available() -> bool:
    return _GYM_AVAILABLE


if _GYM_AVAILABLE:

    class GraceGymEnv(gym.Env):  # type: ignore[misc]
        """Gymnasium wrapper around a single-task GRACE episode."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            registry: TaskRegistry,
            task_id: str,
            *,
            max_actions: int = 8,
            seed: int = 42,
            max_text_len: int = 65536,
        ) -> None:
            super().__init__()
            self._registry = registry
            self._task_id = task_id
            self._max_actions = max_actions
            self._seed = seed
            self._inner = AutoMLEnvironment(registry, seed=seed)
            self.action_space = spaces.Text(max_length=max_text_len)
            self.observation_space = spaces.Text(max_length=max_text_len)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self._inner = AutoMLEnvironment(self._registry, seed=seed)
            self._inner.reset(self._task_id, max_actions=self._max_actions)
            obs = self._inner.observe()
            info: dict[str, Any] = {"task_id": self._task_id}
            return obs, info

        def step(self, action: str):
            out = self._inner.step(action)
            session = self._inner._session
            breakdown = None
            if session is not None and session.steps:
                breakdown = getattr(session.steps[-1], "reward_breakdown", None)
            terminated = bool(out.done)
            truncated = False
            info: dict[str, Any] = {
                "reward_breakdown": breakdown,
                "validation_metric": getattr(session, "current_metric", None)
                if session
                else None,
                "hidden_test_metric": getattr(session, "hidden_test_metric", None)
                if session
                else None,
            }
            return out.state, float(out.reward), terminated, truncated, info

        def close(self):
            self._inner = None

    def make_grace_env(
        registry: TaskRegistry, task_id: str, **kwargs: Any
    ) -> "GraceGymEnv":
        """Convenience constructor mirroring ``gym.make`` ergonomics."""
        return GraceGymEnv(registry, task_id, **kwargs)

else:  # pragma: no cover - only hit when gymnasium is absent

    class GraceGymEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium is not installed. Install it with `pip install gymnasium` "
                "to use the GRACE Gym wrapper. The core evaluator does not require it."
            )

    def make_grace_env(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        raise ImportError(
            "gymnasium is not installed. Install it with `pip install gymnasium`."
        )
