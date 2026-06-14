""""""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

# Threshold: if gain < this value — considered insignificant
_MIN_SIGNIFICANT_GAIN = 0.005


class IterativeCycleValidator(BaseValidator):
    """Validator for iterative cycles with growing penalty and decreasing reward."""

    name = "iterative_cycles"

    def __init__(
        self,
        base_cycle_penalty: float = 0.03,
        penalty_exponent: float = 1.5,
        gain_reward_factor: float = 0.10,
        regression_penalty: float = 0.08,
        max_free_cycles: int = 1,
    ) -> None:
        self.base_cycle_penalty = base_cycle_penalty
        self.penalty_exponent = penalty_exponent
        self.gain_reward_factor = gain_reward_factor
        self.regression_penalty = regression_penalty
        self.max_free_cycles = max_free_cycles

    def validate(self, session: RuntimeSession) -> ValidationResult:
        n_cycles = session.cycle_count
        history = session.metric_history

        if n_cycles <= self.max_free_cycles:
            if n_cycles == 0:
                details = "Single-pass pipeline (no iteration cycles)."
            else:
                details = f"{n_cycles} cycle(s) — within free allowance."
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details=details,
            )

        penalty = 0.0
        reward = 0.0
        cycle_log: list[str] = []

        paid_cycles = n_cycles - self.max_free_cycles

        for i in range(1, paid_cycles + 1):
            actual_cycle = i + self.max_free_cycles
            cycle_cost = self.base_cycle_penalty * (i**self.penalty_exponent)

            gain = self._metric_gain_for_cycle(history, i)

            if gain is None:
                cycle_log.append(
                    f"Cycle {actual_cycle}: cost={cycle_cost:.4f}, no metric data"
                )
                penalty += cycle_cost
            elif gain < -_MIN_SIGNIFICANT_GAIN:
                extra = self.regression_penalty * abs(gain)
                penalty += cycle_cost + extra
                cycle_log.append(
                    f"Cycle {actual_cycle}: REGRESSION gain={gain:.4f}, "
                    f"cost={cycle_cost:.4f}+{extra:.4f}"
                )
            elif gain < _MIN_SIGNIFICANT_GAIN:
                penalty += cycle_cost
                cycle_log.append(
                    f"Cycle {actual_cycle}: negligible gain={gain:.4f}, "
                    f"full cost={cycle_cost:.4f}"
                )
            else:
                compensated = min(cycle_cost, self.gain_reward_factor * gain)
                net = cycle_cost - compensated
                penalty += max(0.0, net)
                cycle_log.append(
                    f"Cycle {actual_cycle}: gain={gain:.4f}, "
                    f"cost={cycle_cost:.4f}, compensated={compensated:.4f}, net={net:.4f}"
                )

        total_gain = self._total_gain(history)
        if total_gain is not None and total_gain > 0.02:
            reward = min(0.05, total_gain * 0.3)

        score = max(0.0, min(1.0, 1.0 - penalty + reward))
        passed = penalty <= 0.01

        if cycle_log:
            details = f"{n_cycles} cycle(s) ({paid_cycles} paid). " + "; ".join(
                cycle_log
            )
        else:
            details = f"{n_cycles} cycle(s)."

        if total_gain is not None:
            details += f" | Total metric gain: {total_gain:+.4f}"

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _metric_gain_for_cycle(
        self,
        history: list[tuple[int, float]],
        paid_idx: int,
    ) -> float | None:
        """Metric gain for the paid_idx-th paid cycle (1-indexed)."""
        if len(history) < 2:
            return None

        after_idx = self.max_free_cycles + paid_idx - 1
        before_idx = after_idx - 1

        if before_idx < 0:
            before_idx = 0
        if after_idx >= len(history):
            return history[-1][1] - history[-2][1]

        return history[after_idx][1] - history[before_idx][1]

    def _total_gain(self, history: list[tuple[int, float]]) -> float | None:
        if len(history) < 2:
            return None
        return history[-1][1] - history[0][1]


def cycle_error_multiplier(cycle_count: int) -> float:
    """"""
    if cycle_count <= 1:
        return 1.0
    return min(3.0, 1.0 + 0.5 * (cycle_count - 1))
