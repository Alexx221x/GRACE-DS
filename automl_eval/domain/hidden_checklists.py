"""Dataset-compiled hidden criteria with diverse abstract feedback and broad code recognition."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd

from automl_eval.core.session import ActionType, RuntimeSession
from automl_eval.evaluation.submission import predict_for_metric, resolve_submission
from automl_eval.domain.task import TaskType
from automl_eval.validators.base import ValidationResult, ValidatorStatus
from automl_eval.evaluation.candidate_diversity import (
    candidate_diversity_score,
    primary_model_family,
)


class Stage(str, Enum):
    PLAN = "PLAN"
    EDA = "EDA"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL = "MODEL"
    VALIDATE = "VALIDATE"
    SUBMISSION = "SUBMISSION"
    ITERATION = "ITERATION"


@dataclass(frozen=True)
class Criterion:
    key: str
    stage: Stage
    feedback: str
    hint: str = ""
    weight: float = 1.0


@dataclass
class CriterionResult:
    criterion: Criterion
    passed: bool


@dataclass
class StageAssessment:
    stage: Stage
    score: float
    results: list[CriterionResult] = field(default_factory=list)

    @property
    def unresolved_feedback(self) -> list[str]:
        messages: list[str] = []
        for result in self.results:
            if not result.passed:
                message = result.criterion.feedback
                if result.criterion.hint:
                    message += " Hint: " + result.criterion.hint
                messages.append(message)
        return messages


@dataclass
class HiddenChecklist:
    criteria: list[Criterion]
    previous_passed: dict[str, bool] = field(default_factory=dict)

    def stage_criteria(self, stage: Stage) -> list[Criterion]:
        return [criterion for criterion in self.criteria if criterion.stage == stage]

    def assess(self, session: RuntimeSession, stage: Stage) -> StageAssessment:
        results = [
            self._evaluate(criterion, session)
            for criterion in self.stage_criteria(stage)
        ]
        total = sum(result.criterion.weight for result in results)
        score = (
            sum(result.criterion.weight for result in results if result.passed) / total
            if total
            else 1.0
        )
        return StageAssessment(stage=stage, score=score, results=results)

    def stage_results_for_reward(
        self, session: RuntimeSession
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for stage in Stage:
            if not self._stage_attempted(session, stage):
                continue
            assessment = self.assess(session, stage)
            passed = assessment.score >= 0.80
            results.append(
                ValidationResult(
                    validator_name=f"hidden_{stage.value.lower()}",
                    passed=passed,
                    score=assessment.score,
                    details="Internal stage checklist aggregate.",
                    penalty=0.0 if passed else 0.02 * (1.0 - assessment.score),
                    status=ValidatorStatus.RESOLVED
                    if passed
                    else ValidatorStatus.UNRESOLVED,
                )
            )
        diversity = self._candidate_diversity_reward_result(session)
        if diversity is not None:
            results.append(diversity)
        return results

    def _candidate_diversity_reward_result(
        self, session: RuntimeSession
    ) -> ValidationResult | None:
        families = _validated_candidate_families(session)
        if not families:
            return None
        score = candidate_diversity_score(families)
        passed = score >= 0.75
        details = (
            f"Validated candidate families: {', '.join(families)}. "
            "Small manual diversity is sufficient when at least two distinct model families have been validated."
        )
        return ValidationResult(
            validator_name="candidate_diversity",
            passed=passed,
            score=score,
            details=details,
            penalty=0.0 if passed else 0.015 * (1.0 - score),
            status=ValidatorStatus.RESOLVED if passed else ValidatorStatus.UNRESOLVED,
        )

    def public_feedback(
        self, session: RuntimeSession, current_stage: Stage
    ) -> tuple[StageAssessment, list[str]]:
        assessment = self.assess(session, current_stage)
        transitions: list[str] = []
        for result in assessment.results:
            before = self.previous_passed.get(result.criterion.key)
            if before is False and result.passed:
                transitions.append("Resolved: " + result.criterion.feedback.rstrip("."))
            self.previous_passed[result.criterion.key] = result.passed
        return assessment, transitions + assessment.unresolved_feedback[:8]

    def _evaluate(
        self, criterion: Criterion, session: RuntimeSession
    ) -> CriterionResult:
        corpus = _stage_corpus(session, criterion.stage)
        ns = session.sandbox_namespace
        working_train = ns.get("train_df")
        working_valid = ns.get("valid_df")
        processed_train = ns.get("X_train")
        processed_valid = ns.get("X_valid", ns.get("X_val"))
        key = criterion.key
        passed = False

        # Planning checks recognise natural-language and common modelling spellings.
        if key == "plan_metric":
            passed = _has(
                corpus,
                r"roc[ _-]?auc|auc|r2|r[- ]?squared|rmse|mae|log[ _-]?loss|f1|accuracy|metric",
            )
        elif key == "plan_validation":
            passed = _has(
                corpus,
                r"valid|cross[ _-]?val|hold[ _-]?out|stratif|split|out[ _-]?of[ _-]?fold",
            )
        elif key == "plan_iteration":
            passed = _has(
                corpus, r"iterat|revise|feedback|improv|tune|compare|baseline|candidate"
            )
        elif key == "plan_reproducible":
            passed = _has(
                corpus, r"pipeline|predict_fn|reproduc|random_state|seed|raw[ -]?input"
            )
        elif key == "plan_missing":
            passed = _has(corpus, r"missing|null|nan|imput|fillna|dropna")
        elif key == "plan_categories":
            passed = _has(
                corpus, r"categor|encod|one[ _-]?hot|ordinal|get_dummies|catboost"
            )
        elif key == "plan_imbalance":
            passed = _has(
                corpus, r"imbalan|class_weight|stratif|resampl|oversampl|smote|balanced"
            )
        elif key == "plan_suspicious":
            passed = _has(
                corpus,
                r"identifier|high[ _-]?card|cardinal|drop.*id|leak|passengerid|ticket|name|cabin",
            )
        # EDA recognises pandas, numpy and common visualization/statistical alternatives.
        elif key == "eda_schema":
            passed = _has(
                corpus,
                r"\.info\s*\(|\.dtypes|\.shape|\.columns|describe\s*\(|select_dtypes|head\s*\(",
            )
        elif key == "eda_target":
            passed = _has(
                corpus,
                r"value_counts|groupby.*target|target.*describe|survived.*mean|hist|crosstab",
            )
        elif key == "eda_missing":
            passed = _has(corpus, r"isna|isnull|missing|null|nan|notna")
        elif key == "eda_duplicates":
            passed = _has(corpus, r"duplicat|drop_duplicates")
        elif key == "eda_correlation":
            passed = _has(
                corpus,
                r"\.corr\s*\(|correlat|heatmap|pearsonr|spearmanr|kendall|vif|variance_inflation",
            )
        elif key == "eda_distribution":
            passed = _has(
                corpus,
                r"outlier|skew|kurt|boxplot|quantile|percentile|iqr|hist|winsor|clip|describe",
            )
        elif key == "eda_balance":
            passed = _has(
                corpus,
                r"value_counts|class.*distribut|imbalan|proportion|normalize\s*=\s*true|crosstab",
            )
        elif key == "eda_identifier":
            passed = _has(
                corpus,
                r"unique|nunique|identifier|\bid\b|cardinal|leak|passengerid|ticket|name|cabin",
            )
        elif key == "eda_scale":
            passed = _has(corpus, r"std\s*\(|standard|scale|range|describe|variance")
        # Feature engineering supports materialised dataframes and reusable transformer objects.
        elif key == "fe_missing":
            passed = (
                (
                    _no_feature_missing(working_train, session)
                    and _no_missing(working_valid)
                )
                or (_no_missing(processed_train) and _no_missing(processed_valid))
                or _has_component(
                    ns, {"SimpleImputer", "KNNImputer", "IterativeImputer"}
                )
                or _has(
                    corpus,
                    r"fillna\s*\(|dropna\s*\(|simpleimputer|knnimputer|iterativeimputer",
                )
            )
        elif key == "fe_categories":
            passed = (
                (_all_numeric(processed_train) and _all_numeric(processed_valid))
                or _has_component(
                    ns, {"OneHotEncoder", "OrdinalEncoder", "TargetEncoder"}
                )
                or _has(
                    corpus,
                    r"onehot|ordinalencod|targetencod|get_dummies|catboost|columntransformer",
                )
            )
        elif key == "fe_alignment":
            if isinstance(processed_train, pd.DataFrame) and isinstance(
                processed_valid, pd.DataFrame
            ):
                passed = list(processed_train.columns) == list(processed_valid.columns)
            passed = (
                passed
                or _has_component(ns, {"ColumnTransformer", "Pipeline"})
                or _has(
                    corpus,
                    r"pipeline|columntransformer|\.transform\s*\(|reindex\s*\(|align\s*\(",
                )
            )
        elif key == "fe_candidate_replayability":
            # Only activate after a formal candidate has actually been registered.
            passed = resolve_submission(
                ns
            ) is None or _raw_candidate_accepts_training_rows(session)
        elif key == "fe_predictor_coverage":
            expected = _expected_low_risk_columns(session)
            selected = _selected_transform_input_columns(ns)
            if len(expected) < 3:
                passed = True
            elif selected is None:
                passed = _has(
                    corpus,
                    r"remainder\s*=\s*['\"]passthrough|select_dtypes|make_column_selector|all[_ ]?features|feature_columns",
                )
            else:
                passed = len(expected.intersection(selected)) / len(expected) >= 0.70
        elif key == "fe_duplicates":
            passed = _has(corpus, r"drop_duplicates|duplicat")
            if isinstance(working_train, pd.DataFrame):
                passed = (
                    passed
                    or working_train.drop(
                        columns=[session.task.target_column], errors="ignore"
                    )
                    .duplicated()
                    .sum()
                    == 0
                )
        elif key == "fe_suspicious":
            passed = _has(
                corpus,
                r"drop\s*\(|remainder\s*=\s*['\"]drop|identifier|cardinal|passengerid|ticket|cabin|name|feature.*select|columntransformer",
            )
        elif key == "fe_feature_derivation":
            passed = _has(
                corpus,
                r"famil(?:y|ysize)|title|isalonen|cabin.*known|ticket.*prefix|deck|surname|"
                r"log1p|interaction|\.str\.extract|functiontransformer\s*\(",
            )
        elif key == "fe_scale":
            passed = _has_component(
                ns, {"StandardScaler", "RobustScaler", "MinMaxScaler"}
            ) or _has(
                corpus,
                r"standardscaler|robustscaler|minmaxscaler|powertransformer|quantiletransformer",
            )
        # Modelling checks verify the formal raw-input contract without touching validation labels.
        elif key == "model_fitted":
            passed = _raw_candidate_accepts_training_rows(session)
        elif key == "model_replayable":
            passed = _raw_candidate_accepts_training_rows(session)
        elif key == "model_reproducible":
            passed = _has(corpus, r"random_state|seed|random\.seed|np\.random\.seed")
        elif key == "model_tuning":
            passed = _has(
                corpus,
                r"n_estimators\s*=|max_depth\s*=|min_samples_leaf\s*=|"
                r"min_samples_split\s*=|max_features\s*=|class_weight\s*=|"
                r"max_iter\s*=|C\s*=|alpha\s*=|fit_intercept\s*=|random_state\s*=",
            ) and not _has(
                corpus,
                r"GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|"
                r"HalvingRandomSearchCV|ParameterGrid|ParameterSampler|"
                r"RidgeCV|LassoCV|ElasticNetCV|Optuna|optuna|BayesSearchCV|hyperopt",
            )
        elif key == "model_parameters":
            passed = _has(
                corpus,
                r"n_estimators\s*=|max_depth\s*=|min_samples|learning_rate\s*=|c\s*=|max_iter\s*=|class_weight\s*=",
            )
        elif key == "model_candidate_diversity":
            families = _validated_candidate_families(session)
            passed = (
                candidate_diversity_score(families) >= 0.75
                if families
                else len(set(_candidate_families_from_stage_corpus(session))) >= 2
            )
        elif key == "model_metric_output":
            if session.task.metric.value in {"roc_auc", "log_loss"}:
                passed = _has(
                    corpus, r"predict_proba|decision_function"
                ) or _candidate_supports_probability(ns)
            else:
                passed = True
        elif key == "validate_scored":
            passed = session.current_metric is not None
        elif key == "validate_replayable":
            passed = session.current_submission_replayable
        elif key == "validate_protocol":
            passed = bool(
                session.steps
                and session.steps[-1].action_type == ActionType.VALIDATE
                and session.steps[-1].execution_success
            )
        elif key == "iterate_metric":
            passed = len(session.metric_history) >= 2
        elif key == "iterate_candidate_diversity":
            passed = (
                candidate_diversity_score(_validated_candidate_families(session))
                >= 0.75
            )
        elif key == "iterate_response":
            # A model-family comparison or a tuning pass is a meaningful
            passed = (
                session.cycle_count >= 2
                or session.repair_successes > 0
                or len(session.trained_models) >= 2
                or _has(
                    _stage_corpus(session, Stage.MODEL),
                    r"gridsearch|randomizedsearch|param_grid|cross_val|cv\s*=",
                )
            )
        elif key == "iterate_improvement":
            history = [metric for _, metric in session.metric_history]
            passed = len(history) >= 2 and history[-1] > max(history[:-1]) + 1e-12
        elif key == "iterate_feedback_resolution":
            surfaced_failures = [
                item_key
                for item_key, was_passed in self.previous_passed.items()
                if was_passed is False
                and not item_key.startswith(("iterate_", "validate_", "submit_"))
            ]
            passed = not surfaced_failures or any(
                self._surfaced_issue_addressed(item_key, session)
                for item_key in surfaced_failures
            )
        elif key == "submit_bundle":
            passed = (
                session.current_submission_replayable
                and resolve_submission(ns) is not None
            )
        elif key == "submit_current_best":
            passed = session.best_metric is None or (
                session.current_metric is not None
                and session.current_validated_candidate_version
                == session.candidate_version
                and session.current_metric >= session.best_metric - 1e-12
            )
        elif key == "submit_test":
            passed = (
                session.final_submitted
                and session.hidden_test_metric is not None
                and session.test_evaluation_count == 1
            )
        elif key == "submit_integrity":
            passed = session.check_data_intact()
        return CriterionResult(criterion=criterion, passed=bool(passed))

    def _surfaced_issue_addressed(self, key: str, session: RuntimeSession) -> bool:
        """Accept direct revisits or later-stage mitigations for surfaced concerns."""
        criteria_by_key = {item.key: item for item in self.criteria}
        criterion = criteria_by_key.get(key)
        if criterion is not None and self._evaluate(criterion, session).passed:
            return True
        alternatives = {
            "eda_missing": ("fe_missing",),
            "eda_scale": ("fe_scale",),
            "eda_identifier": ("fe_suspicious", "fe_feature_derivation"),
            "plan_suspicious": ("fe_suspicious", "fe_feature_derivation"),
            "eda_duplicates": ("fe_duplicates",),
        }.get(key, ())
        if any(
            alt in criteria_by_key
            and self._evaluate(criteria_by_key[alt], session).passed
            for alt in alternatives
        ):
            return True
        model_corpus = _stage_corpus(session, Stage.MODEL)
        feature_corpus = _stage_corpus(session, Stage.FEATURE_ENGINEERING)
        if key == "plan_imbalance":
            return _has(
                model_corpus + "\n" + feature_corpus,
                r"class_weight|sample_weight|stratif|resampl|oversampl|smote|balanced",
            )
        if key == "eda_distribution":
            return _has(
                feature_corpus + "\n" + model_corpus,
                r"log1p|powertransformer|quantiletransformer|robustscaler|winsor|clip\s*\(",
            )
        if key == "eda_correlation":
            return _has(
                feature_corpus + "\n" + model_corpus,
                r"pca|variance_inflation|vif|drop.*corr|correlat.*drop",
            )
        return False

    @staticmethod
    def _stage_attempted(session: RuntimeSession, stage: Stage) -> bool:
        mappings = {
            Stage.PLAN: {ActionType.PLAN},
            Stage.EDA: {ActionType.EDA},
            Stage.FEATURE_ENGINEERING: {
                ActionType.FEATURE_ENGINEERING,
                ActionType.CODE_FIX,
            },
            Stage.MODEL: {ActionType.MODEL, ActionType.CODE},
            Stage.VALIDATE: {ActionType.VALIDATE},
            Stage.SUBMISSION: {ActionType.FINAL_SUBMIT},
            Stage.ITERATION: set(),
        }
        if stage == Stage.ITERATION:
            return len(session.metric_history) >= 2 or session.cycle_count >= 2
        return any(record.action_type in mappings[stage] for record in session.steps)


def _candidate_families_from_stage_corpus(session: RuntimeSession) -> list[str]:
    families: list[str] = []
    for record in session.steps:
        if record.action_type in {
            ActionType.MODEL,
            ActionType.CODE,
            ActionType.CODE_FIX,
        }:
            family = primary_model_family(record.action_text)
            if family != "unknown":
                families.append(family)
    return families


def _validated_candidate_families(session: RuntimeSession) -> list[str]:
    by_version: dict[int, str] = {}
    for item in session.trained_models:
        version = item.get("candidate_version")
        if version is None:
            continue
        by_version[int(version)] = primary_model_family(item.get("code") or "")
    families: list[str] = []
    for version, _metric in session.metric_history:
        families.append(by_version.get(int(version), "unknown"))
    return families


def compile_hidden_checklist(session: RuntimeSession) -> HiddenChecklist:
    insights = session.data_insights
    assert insights is not None
    C = Criterion
    criteria = [
        C(
            "plan_metric",
            Stage.PLAN,
            "Evaluation metric has not been acknowledged.",
            "Name the metric you will optimise through validation.",
        ),
        C(
            "plan_validation",
            Stage.PLAN,
            "Validation discipline has not been planned.",
            "Describe how candidate revisions will be assessed.",
        ),
        C(
            "plan_iteration",
            Stage.PLAN,
            "An iteration strategy has not been stated.",
            "Plan at least one evidence-driven model comparison or revision.",
        ),
        C(
            "plan_reproducible",
            Stage.PLAN,
            "Replayable submission packaging has not been planned.",
            "Plan a raw-input pipeline or predict_fn before submission.",
        ),
        C(
            "eda_schema",
            Stage.EDA,
            "Data structure has not been examined.",
            "Inspect dimensions, column types and representative summaries.",
        ),
        C(
            "eda_target",
            Stage.EDA,
            "Target behaviour has not been examined.",
            "Inspect the training target distribution or summary.",
        ),
        C(
            "fe_alignment",
            Stage.FEATURE_ENGINEERING,
            "Training and unseen-row transformations are not demonstrably aligned.",
            "Prefer one reusable transformer or pipeline for all raw inputs.",
        ),
        C(
            "fe_candidate_replayability",
            Stage.FEATURE_ENGINEERING,
            "A repair or feature update still leaves the registered candidate unable to accept raw held-out rows.",
            "Derived-feature creation must run before downstream selection inside one submitted pipeline or predict_fn.",
        ),
        C(
            "fe_predictor_coverage",
            Stage.FEATURE_ENGINEERING,
            "Potentially useful ordinary predictors are excluded from the reusable feature pipeline.",
            "Review retained low-cardinality and numeric feature coverage before discarding columns.",
        ),
        C(
            "model_fitted",
            Stage.MODEL,
            "No callable prediction candidate accepts raw feature rows yet.",
            "Fit and register pipeline or predict_fn on raw-input features.",
        ),
        C(
            "model_replayable",
            Stage.MODEL,
            "Current model artefacts are not replayable from raw rows.",
            "Put engineered feature creation inside the submitted artefact.",
        ),
        C(
            "model_reproducible",
            Stage.MODEL,
            "Reproducibility has not been demonstrated.",
            "Set fixed random seeds on stochastic estimators.",
        ),
        C(
            "model_tuning",
            Stage.MODEL,
            "Manual model configuration has not been demonstrated.",
            "Choose a small number of explicit, sensible hyperparameters without automated search.",
        ),
        C(
            "model_parameters",
            Stage.MODEL,
            "Model configuration is not explicitly controlled.",
            "Specify sensible non-default parameters manually.",
        ),
        C(
            "model_candidate_diversity",
            Stage.MODEL,
            "Small manual candidate diversity has not been demonstrated.",
            "Before stopping, validate at least one meaningfully different model family, such as tree ensemble vs linear/logistic baseline, without automated search.",
        ),
        C(
            "model_metric_output",
            Stage.MODEL,
            "Prediction output is not suited to the evaluation metric.",
            "For probability metrics, expose probabilities or decision scores.",
        ),
        C(
            "validate_protocol",
            Stage.VALIDATE,
            "Evaluator-owned validation has not run successfully.",
            "Register a replayable candidate, then call VALIDATE without code.",
        ),
        C(
            "validate_scored",
            Stage.VALIDATE,
            "A validation score has not been obtained.",
            "Repair replayability before requesting another score.",
        ),
        C(
            "validate_replayable",
            Stage.VALIDATE,
            "Current artefacts are not replayable on raw held-out rows.",
            "Bundle all preprocessing and feature generation inside submission code.",
        ),
        C(
            "iterate_metric",
            Stage.ITERATION,
            "No metric comparison across revisions is available.",
            "Validate a materially revised candidate to measure change.",
        ),
        C(
            "iterate_candidate_diversity",
            Stage.ITERATION,
            "Validated candidates do not yet cover meaningfully different model families.",
            "Try one small manual alternative family before stopping, e.g. tree ensemble vs linear/logistic baseline.",
        ),
        C(
            "iterate_response",
            Stage.ITERATION,
            "No evidence of productive revision is available.",
            "Use feedback to change a concrete modelling decision.",
        ),
        C(
            "iterate_improvement",
            Stage.ITERATION,
            "The latest validated revision did not improve on the best prior scored candidate.",
            "Keep or restore the strongest validated candidate before submission.",
        ),
        C(
            "iterate_feedback_resolution",
            Stage.ITERATION,
            "Previously surfaced workflow concerns have not been demonstrably addressed.",
            "Revisit an unresolved stage concern or implement an equivalent corrective change.",
        ),
        C(
            "submit_bundle",
            Stage.SUBMISSION,
            "The final submission is not replayable from raw features.",
            "Submit only a validated raw-input artefact.",
        ),
        C(
            "submit_current_best",
            Stage.SUBMISSION,
            "The active candidate is not the strongest validated candidate.",
            "Restore or retrain the stronger candidate, validate it again, then submit.",
        ),
        C(
            "submit_test",
            Stage.SUBMISSION,
            "Terminal isolated evaluation was not completed.",
            "Submit once a valid artefact is registered.",
        ),
        C(
            "submit_integrity",
            Stage.SUBMISSION,
            "Protected raw snapshots were not preserved.",
            "Never modify original snapshot objects.",
        ),
    ]
    if insights.has_missing:
        criteria.extend(
            [
                C(
                    "plan_missing",
                    Stage.PLAN,
                    "Missing-value handling has not been planned.",
                    "Plan training-fitted imputation or justified column removal.",
                ),
                C(
                    "eda_missing",
                    Stage.EDA,
                    "Missingness has not been examined.",
                    "Inspect null counts or missing fractions by feature.",
                ),
                C(
                    "fe_missing",
                    Stage.FEATURE_ENGINEERING,
                    "Missing values remain unresolved for modelling.",
                    "Place appropriate imputers in the reusable preprocessing artefact.",
                ),
            ]
        )
    if insights.categorical_columns:
        criteria.extend(
            [
                C(
                    "plan_categories",
                    Stage.PLAN,
                    "Categorical-variable handling has not been planned.",
                    "Plan safe encoding with unseen-category handling.",
                ),
                C(
                    "fe_categories",
                    Stage.FEATURE_ENGINEERING,
                    "Categorical variables are not demonstrably model-ready.",
                    "Use a reusable encoder or a compatible native categorical model.",
                ),
            ]
        )
    if insights.has_duplicates:
        criteria.extend(
            [
                C(
                    "eda_duplicates",
                    Stage.EDA,
                    "Duplicate structure has not been examined.",
                    "Check feature-row duplication before modelling.",
                ),
                C(
                    "fe_duplicates",
                    Stage.FEATURE_ENGINEERING,
                    "Duplicate observations remain unresolved.",
                    "Handle duplicates only when justified by the data audit.",
                ),
            ]
        )
    if insights.has_high_correlation:
        criteria.append(
            C(
                "eda_correlation",
                Stage.EDA,
                "Correlation structure has not been examined.",
                "Assess numeric dependence or redundant predictors.",
            )
        )
    if insights.has_outliers or insights.has_high_skew:
        criteria.append(
            C(
                "eda_distribution",
                Stage.EDA,
                "Distribution anomalies have not been examined.",
                "Inspect skew or extreme-value behaviour in numeric features.",
            )
        )
    if insights.scale_range_ratio > 20:
        criteria.extend(
            [
                C(
                    "eda_scale",
                    Stage.EDA,
                    "Potential feature-scale disparity has not been examined.",
                    "Review numeric scales before linear or distance-based models.",
                ),
                C(
                    "fe_scale",
                    Stage.FEATURE_ENGINEERING,
                    "Scale-sensitive preprocessing is not addressed.",
                    "Consider a training-fitted scaler when model family requires it.",
                ),
            ]
        )
    if session.task.task_type != TaskType.REGRESSION:
        criteria.append(
            C(
                "eda_balance",
                Stage.EDA,
                "Class distribution has not been examined.",
                "Review target class proportions on training data.",
            )
        )
        if (
            insights.class_imbalance_ratio is not None
            and insights.class_imbalance_ratio < 0.67
        ):
            criteria.append(
                C(
                    "plan_imbalance",
                    Stage.PLAN,
                    "Class imbalance has not been planned for.",
                    "Consider stratification, class weights or robust metrics.",
                )
            )
    if session.initial_profile and (
        session.initial_profile.identifier_like_columns
        or session.initial_profile.high_cardinality_columns
    ):
        criteria.extend(
            [
                C(
                    "plan_suspicious",
                    Stage.PLAN,
                    "Suspicious high-cardinality features are absent from the plan.",
                    "Plan how identifiers or free-text-like columns will be treated.",
                ),
                C(
                    "eda_identifier",
                    Stage.EDA,
                    "Potential identifier or high-cardinality features have not been examined.",
                    "Inspect uniqueness and leakage-like columns.",
                ),
                C(
                    "fe_suspicious",
                    Stage.FEATURE_ENGINEERING,
                    "Suspicious high-cardinality features remain unaddressed.",
                    "Drop or safely derive signal from identifier-like fields.",
                ),
                C(
                    "fe_feature_derivation",
                    Stage.FEATURE_ENGINEERING,
                    "No useful structured derivation from complex columns is evident.",
                    "Where justified, derive compact reusable features rather than raw identifiers.",
                ),
            ]
        )
    return HiddenChecklist(criteria)


def stage_for_action(action: ActionType) -> Stage:
    return {
        ActionType.PLAN: Stage.PLAN,
        ActionType.EDA: Stage.EDA,
        ActionType.FEATURE_ENGINEERING: Stage.FEATURE_ENGINEERING,
        ActionType.MODEL: Stage.MODEL,
        ActionType.VALIDATE: Stage.VALIDATE,
        ActionType.FINAL_SUBMIT: Stage.SUBMISSION,
        ActionType.CODE_FIX: Stage.FEATURE_ENGINEERING,
        ActionType.CODE: Stage.MODEL,
    }[action]


def _stage_corpus(session: RuntimeSession, stage: Stage) -> str:
    mapping = {
        Stage.PLAN: {ActionType.PLAN},
        Stage.EDA: {ActionType.EDA},
        Stage.FEATURE_ENGINEERING: {
            ActionType.FEATURE_ENGINEERING,
            ActionType.CODE_FIX,
        },
        Stage.MODEL: {ActionType.MODEL, ActionType.CODE},
        Stage.VALIDATE: {ActionType.VALIDATE},
        Stage.ITERATION: set(ActionType),
        Stage.SUBMISSION: {ActionType.FINAL_SUBMIT},
    }
    return "\n".join(
        (record.code_body or record.action_text).lower()
        for record in session.steps
        if record.action_type in mapping[stage]
    )


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _no_missing(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and int(frame.isna().sum().sum()) == 0


def _no_feature_missing(frame: Any, session: RuntimeSession) -> bool:
    return (
        isinstance(frame, pd.DataFrame)
        and int(
            frame.drop(columns=[session.task.target_column], errors="ignore")
            .isna()
            .sum()
            .sum()
        )
        == 0
    )


def _all_numeric(frame: Any) -> bool:
    return (
        isinstance(frame, pd.DataFrame)
        and len(frame.select_dtypes(exclude="number").columns) == 0
    )


def _walk_estimators(obj: Any, seen: set[int] | None = None) -> Iterable[Any]:
    seen = seen or set()
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if hasattr(obj, "steps"):
        for _, child in getattr(obj, "steps", []):
            yield from _walk_estimators(child, seen)
    if hasattr(obj, "transformers"):
        for _, child, _ in getattr(obj, "transformers", []):
            yield from _walk_estimators(child, seen)
    if isinstance(obj, dict):
        for child in obj.values():
            yield from _walk_estimators(child, seen)
    elif isinstance(obj, (list, tuple)):
        for child in obj:
            yield from _walk_estimators(child, seen)


def _has_component(namespace: dict[str, Any], class_names: set[str]) -> bool:
    for name in (
        "pipeline",
        "submission_pipeline",
        "preprocessor",
        "transformer",
        "model",
        "search",
    ):
        for item in _walk_estimators(namespace.get(name)):
            if type(item).__name__ in class_names:
                return True
    return False


def _expected_low_risk_columns(session: RuntimeSession) -> set[str]:
    """Infer ordinary candidate predictors without revealing task-specific answers."""
    frame = session._protected_train_snapshot
    if not isinstance(frame, pd.DataFrame):
        return set()
    features = frame.drop(columns=[session.task.target_column], errors="ignore")
    expected: set[str] = set()
    for column in features.columns:
        non_null_unique = int(features[column].nunique(dropna=True))
        uniqueness = non_null_unique / max(len(features), 1)
        lower = column.lower()
        suspicious_name = any(
            token in lower
            for token in ("id", "identifier", "ticket", "name", "uuid", "index")
        )
        if suspicious_name or uniqueness >= 0.95:
            continue
        if pd.api.types.is_numeric_dtype(features[column]) or non_null_unique <= max(
            20, int(0.10 * len(features))
        ):
            expected.add(str(column))
    return expected


def _selected_transform_input_columns(namespace: dict[str, Any]) -> set[str] | None:
    """Return columns consumed by the active ColumnTransformer, when inspectable."""
    for root_name in ("pipeline", "submission_pipeline", "preprocessor", "transformer"):
        for item in _walk_estimators(namespace.get(root_name)):
            if type(item).__name__ != "ColumnTransformer":
                continue
            selected: set[str] = set()
            specs = (
                getattr(item, "transformers", None)
                or getattr(item, "transformers_", None)
                or []
            )
            for name, transformer, columns in specs:
                if name == "remainder" or transformer == "drop":
                    continue
                if isinstance(columns, str):
                    selected.add(columns)
                elif isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                    selected.update(str(column) for column in columns)
                else:
                    return None
            remainder = getattr(item, "remainder", "drop")
            if remainder == "passthrough":
                return None
            return selected
    processed = namespace.get("X_train")
    return (
        set(map(str, processed.columns))
        if isinstance(processed, pd.DataFrame)
        else None
    )


def _candidate_supports_probability(namespace: dict[str, Any]) -> bool:
    bundle = resolve_submission(namespace)
    if bundle is None:
        return False
    return (
        callable(namespace.get("predict_fn"))
        or hasattr(bundle.estimator, "predict_proba")
        or hasattr(bundle.estimator, "decision_function")
    )


def _raw_candidate_accepts_training_rows(session: RuntimeSession) -> bool:
    """Assess raw-row compatibility using untouched, target-free evaluator features."""
    if session.candidate_raw_input_compatible is not None:
        return bool(session.candidate_raw_input_compatible)
    bundle = resolve_submission(session.sandbox_namespace)
    raw = session.sandbox_namespace.get("valid_df_original")
    if bundle is None or not isinstance(raw, pd.DataFrame):
        return False
    raw = raw.head(min(3, len(raw))).copy(deep=True)
    try:
        prediction = predict_for_metric(bundle, raw, session.task)
        return (
            len(prediction) == len(raw)
            and np.isfinite(np.asarray(prediction, dtype=float)).all()
        )
    except Exception:
        return False
