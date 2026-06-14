"""Stage-aware AutoML environment with gated validation and isolated terminal evaluation."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from automl_eval.core.action_parser import ActionParser, ParsedAction
from automl_eval.domain.hidden_checklists import (
    Stage,
    compile_hidden_checklist,
    stage_for_action,
)
from automl_eval.evaluation.metrics import normalize_score
from automl_eval.domain.runtime_info import (
    PRINT_FEEDBACK_INSTRUCTION,
    approved_library_versions_text,
)
from automl_eval.evaluation.reward import (
    RewardBreakdown,
    RewardCalculator,
    RewardWeights,
)
from automl_eval.core.sandbox import ExecutionResult, Sandbox
from automl_eval.evaluation.debug_trace import (
    context_stem,
    log_executable_code_enabled,
    save_text_artifact,
    sha256_text,
    trace_event,
)
from automl_eval.core.session import ActionType, RuntimeSession, StepRecord
from automl_eval.evaluation.submission import (
    diagnose_missing_submission,
    predict_for_metric,
    resolve_submission,
    score_bundle,
)
from automl_eval.domain.task import StageLimit
from automl_eval.domain.task_registry import TaskRegistry
from automl_eval.validators.base import BaseValidator, ValidationResult, ValidatorStatus
from automl_eval.validators.backtracking import BacktrackingValidator
from automl_eval.validators.baseline_comparison import BaselineComparisonValidator
from automl_eval.validators.correctness import CorrectnessValidator
from automl_eval.validators.correlation import CorrelationValidator
from automl_eval.validators.distribution import DistributionValidator
from automl_eval.validators.duplicate import DuplicateValidator
from automl_eval.validators.efficiency import EfficiencyValidator
from automl_eval.validators.execution import ExecutionValidator
from automl_eval.validators.feature_importance import FeatureImportanceValidator
from automl_eval.validators.feature_pipeline import FeaturePipelineValidator
from automl_eval.validators.hyperparam import HyperparamValidator
from automl_eval.validators.intactness import IntactnessValidator
from automl_eval.validators.iterative_cycle import IterativeCycleValidator
from automl_eval.validators.leakage import LeakageValidator
from automl_eval.validators.missing_values import MissingValuesValidator
from automl_eval.validators.model_choice import ModelChoiceValidator
from automl_eval.validators.model_eval import ModelEvalValidator
from automl_eval.validators.namespace_check import NamespaceCheckValidator
from automl_eval.validators.plan_coverage import PlanCoverageValidator
from automl_eval.validators.reproducibility import ReproducibilityValidator
from automl_eval.validators.split_quality import SplitValidator
from automl_eval.validators.target_leakage_model import TargetLeakageModelValidator
from automl_eval.validators.train_valid_refit_leakage import (
    TrainValidRefitLeakageValidator,
)

logger = logging.getLogger(__name__)

VALIDATOR_MIN_PHASE: dict[str, int] = {
    "execution": 0,
    "intactness": 0,
    "leakage": 0,
    "efficiency": 0,
    "backtracking": 0,
    "iterative_cycles": 0,
    "plan_coverage": 1,
    "missing_values": 3,
    "feature_pipeline": 3,
    "distribution": 2,
    "correlation": 2,
    "duplicates": 2,
    "model_choice": 4,
    "hyperparameters": 4,
    "split_quality": 4,
    "reproducibility": 4,
    "namespace_check": 4,
    "model_eval": 4,
    "target_leakage_model": 5,
    "feature_importance": 5,
    "correctness": 5,
    "baseline_comparison": 5,
}
EVALUATOR_OWNED_VALIDATORS = frozenset(
    {
        "model_eval",
        "target_leakage_model",
        "feature_importance",
        "correctness",
        "baseline_comparison",
    }
)
_CODE_REQUIRED_ACTIONS = frozenset(
    {
        ActionType.EDA,
        ActionType.FEATURE_ENGINEERING,
        ActionType.MODEL,
        ActionType.CODE,
        ActionType.CODE_FIX,
    }
)
_MODEL_VAR_NAMES = (
    "submission_pipeline",
    "pipeline",
    "model",
    "clf",
    "classifier",
    "regressor",
    "estimator",
)
_FE_VAR_NAMES = ("X_train", "X_valid", "X_val", "features", "train_processed")


def _session_phase(session: RuntimeSession) -> int:
    if session.done or session.final_submitted:
        return 5
    actions = {record.action_type for record in session.steps}
    if (
        ActionType.MODEL in actions
        or ActionType.VALIDATE in actions
        or any(
            hasattr(session.sandbox_namespace.get(name), "predict")
            for name in _MODEL_VAR_NAMES
        )
        or callable(session.sandbox_namespace.get("predict_fn"))
    ):
        return 4
    if ActionType.FEATURE_ENGINEERING in actions or any(
        isinstance(session.sandbox_namespace.get(name), pd.DataFrame)
        for name in _FE_VAR_NAMES
    ):
        return 3
    if ActionType.EDA in actions:
        return 2
    if session.plan_text is not None:
        return 1
    return 0


def _neutral_result(
    name: str, detail: str = "Inactive at the current stage."
) -> ValidationResult:
    return ValidationResult(name, True, 0.0, detail, 0.0, ValidatorStatus.INACTIVE)


def _targeted_repair_hint(exc: BaseException | str) -> str:
    """Actionable, public repair hints for common replay/preprocessing failures."""
    text = str(exc)
    msg = text.lower()
    if "cannot use median strategy with non-numeric data" in msg:
        return (
            "REPAIR HINT: use a ColumnTransformer and split preprocessing by dtype inside the submitted raw-input Pipeline: "
            "numeric columns -> SimpleImputer(strategy='median'); categorical/object columns -> "
            "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')."
        )
    if "could not convert string to float" in msg or "unknown categories" in msg:
        return (
            "REPAIR HINT: categorical columns are reaching a numeric estimator or encoder state is not replayable. "
            "Use a ColumnTransformer inside `pipeline` with OneHotEncoder(handle_unknown='ignore') for categorical columns."
        )
    if (
        "not a column of the dataframe" in msg
        or "columns are missing" in msg
        or "feature names" in msg
    ):
        return (
            "REPAIR HINT: train-time and raw validation/test transformations are inconsistent. Put feature creation "
            "inside `predict_fn(raw_dataframe)` or as the first step of a sklearn Pipeline, so the same columns are "
            "created from raw rows during validation and terminal scoring."
        )
    if "pipeline is not defined" in msg or "name 'pipeline' is not defined" in msg:
        return (
            "REPAIR HINT: expose a fitted raw-input sklearn Pipeline named `pipeline`, or define "
            "`predict_fn(raw_dataframe)`. Re-import Pipeline/ColumnTransformer/estimators inside the same successful code block."
        )
    if "no formal replayable submission" in msg or "no replayable" in msg:
        return (
            "REPAIR HINT: after fitting, leave either `pipeline` as a fitted raw-input sklearn Pipeline or "
            "`predict_fn(raw_dataframe)` in the global namespace. Do not rely on local transformed arrays only."
        )
    return (
        "REPAIR HINT: make the submitted artefact replay all preprocessing from raw rows: fit transformations on training "
        "data inside a single Pipeline/ColumnTransformer, expose it as `pipeline`, then run ACTION: VALIDATE."
    )


class StepOutput:
    def __init__(self, state: str, reward: float, done: bool) -> None:
        self.state = state
        self.reward = reward
        self.done = done


class AutoMLEnvironment:
    """Multi-turn environment with hidden criteria and evaluator-owned scoring."""

    def __init__(
        self,
        registry: TaskRegistry,
        reward_weights: RewardWeights | None = None,
        sandbox_timeout: int = 60,
        seed: int = 42,
        reveal_internal_feedback: bool = False,
        allow_forced_terminal_evaluation: bool = True,
        enforce_stage_budgets: bool = True,
        subsample_factor: int = 1,
        task_time_budget_override_seconds: float | None = None,
        stage_time_budget_multiplier: float = 1.0,
    ) -> None:
        self.registry = registry
        self.reward_calc = RewardCalculator(reward_weights)
        self.sandbox = Sandbox(timeout_seconds=sandbox_timeout)
        self.parser = ActionParser()
        self.seed = seed

        self.subsample_factor = int(subsample_factor) if subsample_factor else 1
        self.task_time_budget_override_seconds = task_time_budget_override_seconds
        self.stage_time_budget_multiplier = float(stage_time_budget_multiplier or 1.0)
        self.reveal_internal_feedback = reveal_internal_feedback

        self.allow_forced_terminal_evaluation = allow_forced_terminal_evaluation

        self.enforce_stage_budgets = enforce_stage_budgets
        self.validators: list[BaseValidator] = [
            ExecutionValidator(),
            CorrectnessValidator(),
            IntactnessValidator(),
            LeakageValidator(),
            PlanCoverageValidator(),
            NamespaceCheckValidator(),
            ModelEvalValidator(),
            BacktrackingValidator(),
            ReproducibilityValidator(),
            EfficiencyValidator(),
            CorrelationValidator(),
            MissingValuesValidator(),
            DistributionValidator(),
            FeaturePipelineValidator(),
            DuplicateValidator(),
            TargetLeakageModelValidator(),
            TrainValidRefitLeakageValidator(),
            FeatureImportanceValidator(),
            HyperparamValidator(),
            ModelChoiceValidator(),
            SplitValidator(),
            IterativeCycleValidator(),
            BaselineComparisonValidator(),
        ]
        self._session: RuntimeSession | None = None
        self._task = None

    def reset(
        self,
        task_id: str,
        max_actions: int | None = None,
        *,
        max_action: int | None = None,
    ) -> None:
        """Start an episode, optionally overriding the task action budget."""
        if (
            max_actions is not None
            and max_action is not None
            and max_actions != max_action
        ):
            raise ValueError(
                "Use one action-budget override or provide matching max_actions/max_action values."
            )
        override = max_actions if max_actions is not None else max_action
        if override is not None:
            if (
                isinstance(override, bool)
                or not isinstance(override, int)
                or override <= 0
            ):
                raise ValueError("max_actions must be a positive integer.")
        self._task = copy.deepcopy(self.registry.get(task_id))
        if self.task_time_budget_override_seconds is not None:
            self._task.time_budget_seconds = float(
                self.task_time_budget_override_seconds
            )
        if self.stage_time_budget_multiplier != 1.0:
            self._task.stage_limits = {
                name: StageLimit(
                    max_steps=limit.max_steps,
                    max_seconds=float(limit.max_seconds)
                    * self.stage_time_budget_multiplier,
                    max_consecutive_steps=limit.max_consecutive_steps,
                )
                for name, limit in self._task.stage_limits.items()
            }
        if override is not None:
            self._task.max_steps = override
        self._session = RuntimeSession(
            self._task, seed=self.seed, subsample_factor=self.subsample_factor
        )
        self._session.initialize()
        self._session.hidden_checklist = compile_hidden_checklist(self._session)
        logger.info(
            "Environment reset for task '%s' with max_actions=%s",
            task_id,
            self._task.max_steps,
        )

    def evaluator_split_manifest(
        self, *, include_indices: bool = False
    ) -> dict[str, Any]:
        """Expose reproducibility metadata to the trusted experiment harness only."""
        self._check_active()
        assert self._session is not None
        return self._session.evaluator_split_manifest(include_indices=include_indices)

    def observe(self) -> str:
        self._check_active()
        session = self._session
        assert session is not None and self._task is not None
        visible_train = session.sandbox_namespace["train_df"]
        visible_valid = session.sandbox_namespace["valid_df"]
        parts = [
            self._task.observation_text(),
            "",
            "=== Agent-visible sandbox objects ===",
            "`train_df`: working training dataframe; includes the target column.",
            "`valid_df`: working validation-feature dataframe; validation target labels are evaluator-private.",
            "`train_df_original` and `valid_df_original`: protected raw snapshots; never modify them.",
            "Use in-memory objects directly; no dataset file path exists for agent code.",
            "Public runtime versions: " + approved_library_versions_text(),
            PRINT_FEEDBACK_INSTRUCTION,
            "",
            self._describe_dataframe(visible_train, "train_df"),
            self._describe_dataframe(visible_valid, "valid_df (features only)"),
            "",
            "The validation target is not visible. Select ACTION: VALIDATE to request evaluator-owned scoring.",
            "The test split is isolated and is not available during exploration.",
            "A terminal solution must expose predict_fn(raw_dataframe) or a fitted raw-input sklearn Pipeline named pipeline.",
            "",
            "=== Session state ===",
            session.state_summary(),
            "",
            "Available actions: PLAN, EDA, FEATURE_ENGINEERING, MODEL, VALIDATE, CODE, CODE_FIX, FINAL_SUBMIT",
            "Executable actions EDA/FEATURE_ENGINEERING/MODEL/CODE/CODE_FIX require exactly one fenced Python block.",
            "VALIDATE and FINAL_SUBMIT are evaluator triggers and must not contain Python code.",
        ]
        return "\n".join(parts)

    def step(self, content: str) -> StepOutput:
        self._check_active()
        session = self._session
        assert session is not None
        if session.done:
            return StepOutput(
                "Episode already finished; no further actions are accepted.", 0.0, True
            )

        session.final_submit_blocked = False
        session.final_submit_block_reason = None
        trace_event(
            "before_environment_step",
            step_index=session.current_step,
            raw_action_characters=len(content or ""),
            raw_action_sha256=sha256_text(content or ""),
        )
        parsed = self.parser.parse(content)
        trace_event(
            "after_action_parse",
            step_index=session.current_step,
            action_type=parsed.action_type.value,
            code_block_count=parsed.code_block_count,
            parsed_body_characters=len(parsed.body or ""),
            parsed_body_sha256=sha256_text(parsed.body or ""),
            has_code_block=parsed.has_code_block,
        )
        state_before = session.state_summary()
        started = time.perf_counter()
        exec_result = self._execute_action(parsed, session)
        duration = time.perf_counter() - started
        trace_event(
            "after_environment_step",
            step_index=session.current_step,
            action_type=parsed.action_type.value,
            duration_seconds=round(duration, 3),
            execution_success=exec_result.success,
            error=(exec_result.error or "")[:500],
        )
        record = StepRecord(
            step_idx=session.current_step,
            action_type=parsed.action_type,
            action_text=parsed.raw_text,
            state_before=state_before,
            state_after="",
            reward=0.0,
            execution_success=exec_result.success,
            error_message=exec_result.error,
            metric_value=session.current_metric
            if parsed.action_type == ActionType.VALIDATE and exec_result.success
            else None,
            code_body=parsed.body if parsed.has_code_block else "",
            duration_seconds=duration,
        )
        session.record_step(record)
        record.state_after = session.state_summary()

        if self.enforce_stage_budgets:
            budget_message, forced_by_stage, budget_penalty = (
                session.stage_budget_status(parsed.action_type.value)
            )
        else:
            budget_message, forced_by_stage, budget_penalty = None, False, 0.0
        session.stage_budget_message = budget_message
        valid_final_request = (
            parsed.action_type == ActionType.FINAL_SUBMIT
            and parsed.code_block_count == 0
            and not session.final_submit_blocked
        )
        termination_reason: str | None = None
        if valid_final_request:
            termination_reason = "agent submission"
        elif forced_by_stage:
            termination_reason = "repeated stage-budget exceedance"
        elif session.is_over_steps():
            termination_reason = "maximum action budget reached"
        elif session.is_over_budget():
            termination_reason = "sandbox execution-time budget reached"

        if termination_reason and not valid_final_request:
            if (
                termination_reason != "agent submission"
                and not self.allow_forced_terminal_evaluation
            ):
                session.finalization_reason = (
                    termination_reason
                    + " (hidden-test evaluation suppressed by protocol)"
                )
                terminal_result = ExecutionResult(
                    True,
                    "Working episode terminated without hidden-test evaluation; "
                    "terminal scoring is reserved for evaluator-selected replay.",
                    "",
                )
            else:
                terminal_result = self._finalize_current_solution(
                    session, termination_reason
                )
            exec_result = _merge_results(exec_result, terminal_result)
        session.done = termination_reason is not None

        phase = _session_phase(session)
        scoring_action = (
            parsed.action_type in {ActionType.VALIDATE, ActionType.FINAL_SUBMIT}
            or session.done
        )
        validation_results: list[ValidationResult] = []
        for validator in self.validators:
            if validator.name in EVALUATOR_OWNED_VALIDATORS and not scoring_action:
                validation_results.append(
                    _neutral_result(
                        validator.name,
                        "Evaluator-owned scoring is gated until VALIDATE or FINAL_SUBMIT.",
                    )
                )
            elif validator.name in EVALUATOR_OWNED_VALIDATORS and not (
                session.current_submission_replayable or session.final_submitted
            ):
                validation_results.append(
                    _neutral_result(
                        validator.name,
                        "No successfully replayed candidate is available for evaluator-owned diagnostics.",
                    )
                )
            elif VALIDATOR_MIN_PHASE.get(validator.name, 0) <= phase:
                validation_results.append(validator.validate(session))
            else:
                validation_results.append(_neutral_result(validator.name))
        validation_results.extend(
            session.hidden_checklist.stage_results_for_reward(session)
        )
        if budget_penalty:
            validation_results.append(
                ValidationResult(
                    validator_name="hidden_stage_governance",
                    passed=False,
                    score=0.0,
                    details="Stage-local budget exceeded.",
                    penalty=budget_penalty,
                    status=ValidatorStatus.UNRESOLVED,
                )
            )
        breakdown = self.reward_calc.compute(
            self._performance_for_action(session, parsed.action_type),
            validation_results,
            raw_metric=self._raw_metric_for_action(session, parsed.action_type),
        )
        record.reward = breakdown.final_reward
        record.reward_breakdown = breakdown
        return StepOutput(
            self._format_step_response(session, parsed, exec_result, breakdown),
            breakdown.final_reward,
            session.done,
        )

    def close(self) -> None:
        self._session = None
        self._task = None

    def _execute_action(
        self, parsed: ParsedAction, session: RuntimeSession
    ) -> ExecutionResult:
        if parsed.action_type == ActionType.PLAN:
            session.plan_text = parsed.body
            return ExecutionResult(
                True,
                "Plan recorded. You may now inspect data, transform features, or model.",
                "",
            )
        if parsed.action_type in _CODE_REQUIRED_ACTIONS:
            if parsed.code_block_count != 1:
                return ExecutionResult(
                    False,
                    "",
                    "",
                    error=(
                        f"Action format error: {parsed.action_type.value} requires exactly one fenced Python code block. "
                        "Prose outside a code block is not executed.\n\n"
                        "HINT: Keep any brief rationale outside the block, then add one ```python ... ``` block."
                    ),
                )
            return self._execute_transactional_code(parsed, session)
        if parsed.action_type == ActionType.VALIDATE:
            if parsed.code_block_count:
                return ExecutionResult(
                    False,
                    "",
                    "",
                    error=(
                        "Action format error: VALIDATE is an evaluator-owned scoring trigger and must not include Python code.\n\n"
                        "HINT: Train and register `pipeline` or `predict_fn` in MODEL, then send only `ACTION: VALIDATE`."
                    ),
                )
            session.validation_requests += 1
            scored, message = self._evaluate_current_solution(session)
            if scored:
                return ExecutionResult(True, message, "")
            return ExecutionResult(False, "", "", error=message)
        if parsed.action_type == ActionType.FINAL_SUBMIT:
            if parsed.code_block_count:
                return ExecutionResult(
                    False,
                    "",
                    "",
                    error=(
                        "Action format error: FINAL_SUBMIT must not include Python code.\n\n"
                        "HINT: Register the replayable artefact during MODEL, validate it, then submit with only ACTION: FINAL_SUBMIT."
                    ),
                )
            quality_gate = self._latest_candidate_submission_gate(session)
            if quality_gate is not None:
                session.final_submit_blocked = True
                session.final_submit_block_reason = quality_gate
                return ExecutionResult(False, "", "", error=quality_gate)
            return self._finalize_current_solution(session, "agent submission")
        return ExecutionResult(
            False, "", "", error=f"Unknown action type: {parsed.action_type}"
        )

    def _execute_transactional_code(
        self, parsed: ParsedAction, session: RuntimeSession
    ) -> ExecutionResult:
        snapshot = session.capture_workspace()
        repairing_replayability = (
            bool(session.validation_error)
            or session.candidate_raw_input_compatible is False
        )
        code_path = None
        code_hash = sha256_text(parsed.body or "")
        if log_executable_code_enabled():
            stem = context_stem(
                phase="environment",
                turn=session.current_step,
                action=parsed.action_type.value,
                suffix="exec",
            )
            code_path, code_hash = save_text_artifact(
                "executable_code", parsed.body or "", stem=stem, suffix=".py"
            )
        trace_event(
            "before_sandbox_exec",
            step_index=session.current_step,
            action_type=parsed.action_type.value,
            code_path=code_path,
            code_sha256=code_hash,
            code_characters=len(parsed.body or ""),
            sandbox_timeout_seconds=getattr(self.sandbox, "timeout_seconds", None),
        )
        exec_started = time.perf_counter()
        result = self.sandbox.execute(
            parsed.body, session.sandbox_namespace, allow_validation_metrics=False
        )
        trace_event(
            "after_sandbox_exec",
            step_index=session.current_step,
            action_type=parsed.action_type.value,
            duration_seconds=round(time.perf_counter() - exec_started, 3),
            success=result.success,
            error=(result.error or "")[:500],
        )
        if not result.success:
            session.restore_workspace(snapshot)
            session.pending_repair_verified = False
            repair_hint = _targeted_repair_hint(result.error or "")
            return _append_stdout(
                result,
                "Workspace changes from the failed action were rolled back.\n"
                + repair_hint,
            )
        if not session.check_data_intact():
            session.restore_workspace(snapshot)
            session.pending_repair_verified = False
            return ExecutionResult(
                False,
                "Workspace changes from the rejected action were rolled back.",
                "",
                error=(
                    "ProtectedSnapshotViolation: train_df_original or valid_df_original was modified.\n\n"
                    "HINT: Create local copies or put transformations inside a raw-input pipeline/predict_fn; "
                    "protected snapshots must remain unchanged."
                ),
            )
        self._sync_session_from_sandbox(session)
        session.refresh_current_profile()
        candidate_changing = parsed.action_type in {
            ActionType.FEATURE_ENGINEERING,
            ActionType.MODEL,
            ActionType.CODE,
            ActionType.CODE_FIX,
        }
        if candidate_changing:
            session.invalidate_candidate_validation()

        probe_note = ""
        if parsed.action_type in {
            ActionType.MODEL,
            ActionType.CODE,
            ActionType.CODE_FIX,
        }:
            probe_note = self._probe_raw_input_candidate(session)
            if parsed.action_type == ActionType.CODE_FIX:
                session.pending_repair_verified = (
                    repairing_replayability
                    and session.candidate_raw_input_compatible is True
                )

        self._record_progress(parsed, session)
        return _append_stdout(result, probe_note) if probe_note else result

    def _evaluate_current_solution(self, session: RuntimeSession) -> tuple[bool, str]:
        bundle = resolve_submission(session.sandbox_namespace)
        if bundle is None:
            session.current_submission_replayable = False
            session.current_submission_kind = None
            session.candidate_raw_input_compatible = False
            diagnostic = diagnose_missing_submission(session.sandbox_namespace)
            session.candidate_probe_error = diagnostic
            session.validation_error = diagnostic
            return False, "Validation not scored: " + diagnostic
        try:
            assert session.private_dev_df is not None
            metric, _ = score_bundle(bundle, session.private_dev_df, session.task)
        except Exception as exc:
            session.current_submission_replayable = False
            session.current_submission_kind = bundle.kind
            session.candidate_raw_input_compatible = False
            session.candidate_probe_kind = bundle.kind
            session.candidate_probe_error = str(exc)
            session.validation_error = str(exc)
            return False, (
                "Validation not scored: the registered artefact cannot replay preprocessing on raw validation rows. "
                "Package all feature creation, encoding and imputation inside `predict_fn` or `pipeline`.\n"
                + _targeted_repair_hint(exc)
            )
        previous = session.current_metric
        session.current_metric = metric
        session.current_validated_candidate_version = session.candidate_version
        if session.best_metric is None or metric > session.best_metric + 1e-12:
            session.best_metric = metric
            session.best_candidate_version = session.candidate_version
            # The visible step number after this action is recorded is one-based.
            session.best_metric_step = session.current_step + 1
        session.current_submission_replayable = True
        session.current_submission_kind = bundle.kind
        session.candidate_raw_input_compatible = True
        session.candidate_probe_kind = bundle.kind
        session.candidate_probe_error = None
        session.validation_error = None
        direction = (
            ""
            if previous is None
            else f"; change from preceding scored solution: {metric - previous:+.4f}"
        )
        return (
            True,
            f"Evaluator validation {session.task.metric.value}: {metric:.4f}{direction}.",
        )

    @staticmethod
    def _latest_candidate_submission_gate(session: RuntimeSession) -> str | None:
        """Block explicit submission when the active candidate is not the best validated one."""
        if session.best_metric is None:
            return None
        metric_name = session.task.metric.value
        best_where = (
            f" at step {session.best_metric_step}"
            if session.best_metric_step is not None
            else ""
        )
        if (
            session.current_metric is None
            or session.current_validated_candidate_version != session.candidate_version
        ):
            return (
                "FINAL_SUBMIT blocked: the active candidate has been changed since the latest "
                "successful validation while a validated candidate already exists. "
                "No private-test evaluation was performed.\n\n"
                f"HINT: Terminal scoring uses the latest candidate only. Restore or retrain the "
                f"strongest prior design (best validated {metric_name}: {session.best_metric:.4f}"
                f"{best_where}), run ACTION: VALIDATE for that active candidate, then submit again."
            )
        if session.current_metric < session.best_metric - 1e-12:
            return (
                f"FINAL_SUBMIT blocked: current validated {metric_name}: {session.current_metric:.4f} "
                f"is below best validated {metric_name}: {session.best_metric:.4f}{best_where}. "
                "No private-test evaluation was performed.\n\n"
                "HINT: Terminal scoring uses the latest candidate only. Restore or retrain the "
                "stronger prior design, run ACTION: VALIDATE to verify it is active, then submit again."
            )
        return None

    def _finalize_current_solution(
        self, session: RuntimeSession, reason: str
    ) -> ExecutionResult:
        if session.final_submitted:
            return ExecutionResult(
                False, "", "", error="Terminal scoring has already been performed."
            )
        bundle = resolve_submission(session.sandbox_namespace)
        if bundle is None:
            session.current_submission_replayable = False
            session.current_submission_kind = None
            session.candidate_raw_input_compatible = False
            session.candidate_probe_error = (
                "No formal replayable submission artefact is available."
            )
            session.finalization_reason = reason
            return ExecutionResult(
                False,
                "No replayable terminal artefact is available.",
                "",
                error="Terminal evaluation requires `predict_fn(raw_dataframe)` or a fitted raw-input sklearn Pipeline named `pipeline`.",
            )
        try:
            assert session.hidden_test_df is not None
            hidden_metric, predictions = score_bundle(
                bundle, session.hidden_test_df, session.task
            )
        except Exception as exc:
            session.current_submission_replayable = False
            session.current_submission_kind = bundle.kind
            session.candidate_raw_input_compatible = False
            session.candidate_probe_kind = bundle.kind
            session.candidate_probe_error = str(exc)
            session.finalization_reason = reason
            return ExecutionResult(
                False,
                "Terminal artefact could not be replayed.",
                "",
                error=f"Terminal replay failed: {exc}",
            )
        session.current_submission_replayable = True
        session.current_submission_kind = bundle.kind
        session.candidate_raw_input_compatible = True
        session.candidate_probe_kind = bundle.kind
        session.candidate_probe_error = None
        session.hidden_test_metric = hidden_metric
        session.predictions = predictions
        session.final_submitted = True
        session.finalization_reason = reason
        session.test_evaluation_count += 1
        return ExecutionResult(
            True, f"Terminal isolated evaluation completed ({reason}).", ""
        )

    def _record_progress(self, parsed: ParsedAction, session: RuntimeSession) -> None:
        if parsed.action_type == ActionType.FEATURE_ENGINEERING:
            session.applied_transforms.append({"code": parsed.body})
        registers_candidate = parsed.action_type == ActionType.MODEL or any(
            token in parsed.body.lower()
            for token in (
                "pipeline",
                "predict_fn",
                ".fit(",
                "gridsearch",
                "randomizedsearch",
            )
        )
        if registers_candidate and session.candidate_raw_input_compatible:
            already_recorded = any(
                item.get("candidate_version") == session.candidate_version
                for item in session.trained_models
            )
            if not already_recorded:
                session.trained_models.append(
                    {
                        "code": parsed.body,
                        "candidate_version": session.candidate_version,
                    }
                )

    @staticmethod
    def _raw_probe_features(session: RuntimeSession) -> pd.DataFrame:
        """Return evaluator-owned raw feature rows without labels for compatibility checking."""
        raw = session.sandbox_namespace.get("valid_df_original")
        if not isinstance(raw, pd.DataFrame):
            raise ValueError("Raw validation feature snapshot is unavailable.")
        return raw.head(min(3, len(raw))).copy(deep=True)

    def _probe_raw_input_candidate(self, session: RuntimeSession) -> str:
        """Check raw-row replayability without calculating or revealing a metric."""
        bundle = resolve_submission(session.sandbox_namespace)
        session.candidate_probe_step = session.current_step
        if bundle is None:
            session.candidate_raw_input_compatible = False
            session.candidate_probe_kind = None
            diagnostic = diagnose_missing_submission(session.sandbox_namespace)
            session.candidate_probe_error = diagnostic
            return "Candidate raw-input smoke check (no metric): failed. " + diagnostic
        try:
            raw_features = self._raw_probe_features(session)
            preds = np.asarray(predict_for_metric(bundle, raw_features, session.task))
            if (
                len(preds) != len(raw_features)
                or not np.isfinite(preds.astype(float)).all()
            ):
                raise ValueError(
                    "prediction output has invalid length or non-finite values"
                )
        except Exception as exc:
            session.candidate_raw_input_compatible = False
            session.candidate_probe_kind = bundle.kind
            session.candidate_probe_error = str(exc)
            return (
                "Candidate raw-input smoke check (no metric): failed. "
                "Current artefact cannot replay all feature creation and preprocessing from raw rows; "
                "package those steps inside `pipeline` or `predict_fn`.\n"
                + _targeted_repair_hint(exc)
            )
        session.candidate_raw_input_compatible = True
        session.candidate_probe_kind = bundle.kind
        session.candidate_probe_error = None
        return "Candidate raw-input smoke check (no metric): passed; the candidate can now be sent to VALIDATE."

    @staticmethod
    def _sync_session_from_sandbox(session: RuntimeSession) -> None:
        working_train = session.sandbox_namespace.get("train_df")
        working_valid = session.sandbox_namespace.get("valid_df")
        if isinstance(working_train, pd.DataFrame):
            session.train_df = working_train
        if isinstance(working_valid, pd.DataFrame):
            session.visible_valid_df = working_valid

    def _performance_for_action(
        self, session: RuntimeSession, action_type: ActionType
    ) -> float:
        if session.done and session.hidden_test_metric is not None:
            return normalize_score(
                session.hidden_test_metric,
                session.task.baseline_score,
                session.task.oracle_score,
            )
        if action_type == ActionType.VALIDATE and session.current_metric is not None:
            return normalize_score(
                session.current_metric,
                session.task.baseline_score,
                session.task.oracle_score,
            )
        return 0.0

    def _raw_metric_for_action(
        self, session: RuntimeSession, action_type: ActionType
    ) -> float | None:
        if session.done and session.hidden_test_metric is not None:
            return session.hidden_test_metric
        if action_type == ActionType.VALIDATE and session.current_metric is not None:
            return session.current_metric
        return None

    def _format_step_response(
        self,
        session: RuntimeSession,
        parsed: ParsedAction,
        exec_result: ExecutionResult,
        breakdown: RewardBreakdown,
    ) -> str:
        stage = stage_for_action(parsed.action_type)
        assessment, messages = session.hidden_checklist.public_feedback(session, stage)
        lines = [
            "Execution: OK"
            if exec_result.success
            else f"Execution: FAILED — {exec_result.error}"
        ]
        if exec_result.stdout.strip():
            lines.append(f"Output: {exec_result.stdout.strip()}")
        lines.extend(
            [
                "",
                f"--- {stage.value.replace('_', ' ').title()} feedback ---",
                f"Stage score: {assessment.score:.3f}",
            ]
        )
        if messages:
            lines.extend(f"- {message}" for message in messages)
        else:
            lines.append(
                "- No unresolved checklist signals for this stage at this time. "
                "This is not a stopping recommendation: if candidate diversity is still low and budget remains, "
                "validate one small, meaningfully different manual model family before finalising."
            )
        if parsed.action_type == ActionType.VALIDATE and (
            len(session.metric_history) >= 2
            or session.validation_requests >= 2
            or session.repair_attempts > 0
        ):
            iteration_assessment, iteration_messages = (
                session.hidden_checklist.public_feedback(session, Stage.ITERATION)
            )
            lines.extend(
                [
                    "",
                    "--- Iteration feedback ---",
                    f"Stage score: {iteration_assessment.score:.3f}",
                ]
            )
            if iteration_messages:
                lines.extend(f"- {message}" for message in iteration_messages)
            else:
                lines.append(
                    "- No unresolved iteration checklist signals at this time. "
                    "If you have fewer than three validated candidates or only one model family, use the remaining budget for one targeted manual alternative rather than stopping early."
                )
        if session.stage_budget_message:
            lines.append(f"- {session.stage_budget_message}")
        lines.extend(f"- {message}" for message in session.profile_delta_messages())
        lines.extend(
            [
                "",
                "--- Progress signal ---",
                f"Reward: {breakdown.final_reward:.4f}",
                session.state_summary(
                    show_validation_metrics=parsed.action_type == ActionType.VALIDATE
                ),
            ]
        )
        if self.reveal_internal_feedback:
            lines.extend(["", "--- Internal diagnostics (debug mode only) ---"])
            for name, result in breakdown.validator_details.items():
                lines.append(
                    f"[{result.status.value.upper()}] {name}: {result.details}"
                )
        if session.done:
            lines.extend(
                [
                    "",
                    f"=== Episode finished: {session.finalization_reason or 'terminated'} ===",
                ]
            )
            if session.hidden_test_metric is not None:
                lines.append(
                    f"Final hidden test {session.task.metric.value}: {session.hidden_test_metric:.4f}"
                )
            else:
                lines.append(
                    "Final hidden test metric unavailable because no replayable terminal artefact succeeded."
                )
        return "\n".join(lines)

    def _check_active(self) -> None:
        if self._session is None or self._task is None:
            raise RuntimeError("No active session. Call reset(task_id) first.")

    @staticmethod
    def _describe_dataframe(df: Any, name: str) -> str:
        if df is None:
            return f"{name}: not loaded"
        return (
            f"{name}: {df.shape[0]} rows, {df.shape[1]} columns\n"
            f"  Columns: {', '.join(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}\n"
            f"  Dtypes: {dict(df.dtypes.value_counts())}\n"
            f"  Missing: {int(df.isnull().sum().sum())} total nulls"
        )


def _append_stdout(result: ExecutionResult, note: str) -> ExecutionResult:
    stdout = (result.stdout.rstrip() + "\n" + note).strip() if result.stdout else note
    return ExecutionResult(result.success, stdout, result.stderr, result.error)


def _merge_results(first: ExecutionResult, second: ExecutionResult) -> ExecutionResult:
    stdout = "\n".join(
        part for part in (first.stdout.strip(), second.stdout.strip()) if part
    )
    error = first.error or second.error
    return ExecutionResult(
        first.success and second.success, stdout, first.stderr + second.stderr, error
    )
