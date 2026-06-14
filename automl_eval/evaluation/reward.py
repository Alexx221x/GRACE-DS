"""
RewardCalculator — assembles the final reward from validator results
and the model metric.

Formula
-------
  weighted = w_perf * r_perf + w_plan * r_plan + w_code * r_code
  capped_penalty = min(total_penalty, max_penalty_frac * weighted)
  final = max(progress_floor, weighted - capped_penalty)

Weight hierarchy (by design):
  plan (0.15) < code quality / FE (0.30) < model metric (0.55)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from automl_eval.validators.base import ValidationResult
from automl_eval.validators.base import ValidatorStatus


class CriticalErrorCategory(str, Enum):
    """Hard methodological failures that zero the reward.
    NONE
        No critical error this turn.
    TARGET_LEAKAGE_FROM_CODE_PATTERN
        Static analysis of the executed code matched a known leakage shape:
        e.g. fitting on validation/test rows, or using the target column as
        a feature directly.  Detected by ``LeakageValidator``.
    TRAIN_VALID_REFIT_LEAKAGE
        Agent concatenated training and validation frames (``pd.concat([
        train_df, valid_df])`` or ``train_df.append(valid_df)``) and then
        fit/transformed the resulting frame.  This is the classic
        "refit on train+valid before final submit" mistake that inflates
        evaluator metrics if the agent later attempts to re-validate.
        Detected by ``TrainValidRefitLeakageValidator``
    EVALUATOR_PRIVATE_ACCESS_ATTEMPT
        The agent tried to read or import an evaluator-private object such
        as ``test_df``, ``private_dev_df`` or ``hidden_test_df``.  The
        sandbox already blocks this via policy, but it is also logged here
        so the reward analysis sees the attempt.
    PROTECTED_SNAPSHOT_TAMPERING
        ``train_df_original`` or ``valid_df_original`` was mutated.  The
        environment rolls back the workspace; we also zero the reward
        for that turn to prevent the agent from getting credit for the
        attempt itself.
    """

    NONE = "none"
    TARGET_LEAKAGE_FROM_CODE_PATTERN = "target_leakage_from_code_pattern"
    TRAIN_VALID_REFIT_LEAKAGE = "train_valid_refit_leakage"
    EVALUATOR_PRIVATE_ACCESS_ATTEMPT = "evaluator_private_access_attempt"
    PROTECTED_SNAPSHOT_TAMPERING = "protected_snapshot_tampering"


_CRITICAL_VALIDATORS: dict[str, CriticalErrorCategory] = {
    "leakage": CriticalErrorCategory.TARGET_LEAKAGE_FROM_CODE_PATTERN,
    "train_valid_refit_leakage": CriticalErrorCategory.TRAIN_VALID_REFIT_LEAKAGE,
    "evaluator_private_access": CriticalErrorCategory.EVALUATOR_PRIVATE_ACCESS_ATTEMPT,
    "protected_snapshot_intactness": CriticalErrorCategory.PROTECTED_SNAPSHOT_TAMPERING,
}


@dataclass
class RewardWeights:
    performance: float = 0.55
    plan_coverage: float = 0.15
    code_quality: float = 0.30


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown.

    The five "additive" terms below sum to ``weighted_reward`` (before the
    capped penalty and the progress floor).  The signed contribution of each
    term is exposed as a separate field so per-step trajectory analysis can
    decompose iteration-to-iteration reward growth into its components.
    """

    raw_performance: float = 0.0
    normalized_performance: float = 0.0
    plan_coverage_score: float = 0.0
    code_quality_score: float = 0.0
    performance_contribution: float = 0.0
    plan_contribution: float = 0.0
    code_quality_contribution: float = 0.0
    weighted_reward: float = 0.0
    total_penalty: float = 0.0
    capped_penalty: float = 0.0
    progress_floor: float = 0.0
    leakage_detected: bool = False
    critical_error_category: CriticalErrorCategory = CriticalErrorCategory.NONE
    critical_error_details: list[tuple[CriticalErrorCategory, str]] = field(
        default_factory=list
    )
    final_reward: float = 0.0
    validator_details: dict[str, ValidationResult] = field(default_factory=dict)


class RewardCalculator:
    """Compute the total reward for a step or episode."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        max_penalty_frac: float = 0.6,
    ) -> None:
        self.weights = weights or RewardWeights()
        self.max_penalty_frac = max_penalty_frac

    def compute(
        self,
        perf_score: float,
        validation_results: list[ValidationResult],
        raw_metric: float | None = None,
    ) -> RewardBreakdown:
        plan_score = 0.0
        code_scores: list[float] = []
        total_penalty = 0.0
        details: dict[str, ValidationResult] = {}
        critical_errors: list[tuple[CriticalErrorCategory, str]] = []

        for vr in validation_results:
            if vr.status == ValidatorStatus.UNRESOLVED:
                vr.status = (
                    ValidatorStatus.RESOLVED
                    if vr.passed
                    else ValidatorStatus.UNRESOLVED
                )
            details[vr.validator_name] = vr
            total_penalty += vr.penalty

            if vr.validator_name in _CRITICAL_VALIDATORS and not vr.passed:
                critical_errors.append(
                    (_CRITICAL_VALIDATORS[vr.validator_name], vr.details or "")
                )

            if vr.validator_name == "plan_coverage":
                plan_score = 0.0 if vr.status == ValidatorStatus.INACTIVE else vr.score
            else:
                if vr.status != ValidatorStatus.INACTIVE:
                    code_scores.append(vr.score)

        code_quality = sum(code_scores) / len(code_scores) if code_scores else 1.0

        perf_contribution = self.weights.performance * perf_score
        plan_contribution = self.weights.plan_coverage * plan_score
        code_contribution = self.weights.code_quality * code_quality
        weighted = perf_contribution + plan_contribution + code_contribution

        capped_penalty = min(total_penalty, self.max_penalty_frac * weighted)

        progress_floor = self._compute_progress_floor(
            perf_score,
            plan_score,
            code_quality,
            details,
        )

        leakage_detected = any(
            cat == CriticalErrorCategory.TARGET_LEAKAGE_FROM_CODE_PATTERN
            for cat, _ in critical_errors
        )

        if critical_errors:
            final = 0.0
            critical_category = critical_errors[0][0]
        else:
            final = max(progress_floor, weighted - capped_penalty)
            critical_category = CriticalErrorCategory.NONE

        return RewardBreakdown(
            raw_performance=(raw_metric if raw_metric is not None else perf_score),
            normalized_performance=perf_score,
            plan_coverage_score=plan_score,
            code_quality_score=code_quality,
            performance_contribution=perf_contribution,
            plan_contribution=plan_contribution,
            code_quality_contribution=code_contribution,
            weighted_reward=weighted,
            total_penalty=total_penalty,
            capped_penalty=capped_penalty,
            progress_floor=progress_floor,
            leakage_detected=leakage_detected,
            critical_error_category=critical_category,
            critical_error_details=critical_errors,
            final_reward=final,
            validator_details=details,
        )

    @staticmethod
    def _compute_progress_floor(
        perf_score: float,
        plan_score: float,
        code_quality: float,
        details: dict[str, ValidationResult],
    ) -> float:
        """Minimum reward based on actual pipeline progress.

        Guarantees that real work is never rewarded with zero:
          - Plan submitted with some coverage      -> floor 0.02
          - Code executed successfully at least once -> floor 0.05
          - Model trained (has predictions)         -> floor 0.08
          - Positive metric (model actually works)  -> floor 0.10
        """
        floor = 0.0

        if plan_score > 0.3:
            floor = 0.02

        exec_vr = details.get("execution")
        if exec_vr and exec_vr.passed:
            floor = max(floor, 0.05)

        model_vr = details.get("model_choice")
        if model_vr and "proven tabular model" in (model_vr.details or ""):
            floor = max(floor, 0.08)

        if perf_score > 0.0:
            floor = max(floor, 0.10)

        return floor


LEAKAGE_VALIDATOR_NAMES = frozenset({"leakage"})
