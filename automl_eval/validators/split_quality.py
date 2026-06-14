""""""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

_CV_PATTERNS = re.compile(
    r"(cross_val_score|KFold|StratifiedKFold|RepeatedKFold|"
    r"RepeatedStratifiedKFold|LeaveOneOut|ShuffleSplit|"
    r"GroupKFold|cross_validate|\.cv\s*=|cv\s*=\s*\d)",
    re.IGNORECASE,
)

_TIMESERIES_SPLIT = re.compile(
    r"(TimeSeriesSplit|time_series_split|expanding.*window|"
    r"rolling.*window|walk.forward)",
    re.IGNORECASE,
)

_SHUFFLE_TRUE = re.compile(r"shuffle\s*=\s*True", re.IGNORECASE)
_SHUFFLE_FALSE = re.compile(r"shuffle\s*=\s*False", re.IGNORECASE)

_STRATIFY = re.compile(r"stratif", re.IGNORECASE)

_TRAIN_TEST_SPLIT = re.compile(r"train_test_split", re.IGNORECASE)

_TEST_SIZE = re.compile(r"test_size\s*=\s*([\d.]+)")

_TIMESERIES_HINTS = re.compile(
    r"(time.?series|temporal|дата|date|timestamp|time_col|"
    r"datetime|year|month|day|period|forecast)",
    re.IGNORECASE,
)


class SplitValidator(BaseValidator):
    """Checks data split correctness and CV usage."""

    name = "split_quality"

    def __init__(
        self,
        no_cv_penalty: float = 0.05,
        bad_split_penalty: float = 0.10,
        timeseries_shuffle_penalty: float = 0.15,
        no_stratify_penalty: float = 0.05,
        min_test_frac: float = 0.10,
        max_test_frac: float = 0.50,
    ) -> None:
        self.no_cv_penalty = no_cv_penalty
        self.bad_split_penalty = bad_split_penalty
        self.timeseries_shuffle_penalty = timeseries_shuffle_penalty
        self.no_stratify_penalty = no_stratify_penalty
        self.min_test_frac = min_test_frac
        self.max_test_frac = max_test_frac

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType
        from automl_eval.domain.task import TaskType

        all_code = ""
        plan_text = session.plan_text or ""
        for rec in session.steps:
            text = rec.code_body if rec.code_body else rec.action_text
            if rec.action_type in (
                ActionType.CODE,
                ActionType.MODEL,
                ActionType.FEATURE_ENGINEERING,
                ActionType.CODE_FIX,
            ):
                all_code += "\n" + text

        has_fit = bool(re.search(r"\.fit\s*\(", all_code))
        if not has_fit:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No model training detected yet.",
            )

        issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0

        # Agent uses fixed train_df / valid_df from the environment (no own split)
        agent_defines_split = bool(_TRAIN_TEST_SPLIT.search(all_code))

        is_ts = self._is_timeseries_task(session, all_code, plan_text)

        # ── 1. Cross-validation ───────────────────────────────────────
        uses_cv = bool(_CV_PATTERNS.search(all_code))
        if uses_cv:
            bonuses.append("Cross-validation used")
        elif agent_defines_split:
            issues.append(
                "No cross-validation — custom train_test_split without CV may be unreliable"
            )
            penalty += self.no_cv_penalty
        else:
            bonuses.append(
                "Using environment train/valid split (CV optional; not required)"
            )

        # ── 2. Time series: no shuffle ────────────────────────────────
        if is_ts:
            uses_ts_split = bool(_TIMESERIES_SPLIT.search(all_code))
            uses_shuffle = bool(_SHUFFLE_TRUE.search(all_code))
            explicitly_no_shuffle = bool(_SHUFFLE_FALSE.search(all_code))

            if uses_ts_split:
                bonuses.append("TimeSeriesSplit used for temporal data")
            elif uses_shuffle and not explicitly_no_shuffle:
                issues.append(
                    "Temporal data shuffled during split — "
                    "use TimeSeriesSplit or shuffle=False"
                )
                penalty += self.timeseries_shuffle_penalty

        # ── 3. Test size check ────────────────────────────────────────
        size_matches = _TEST_SIZE.findall(all_code)
        for size_str in size_matches:
            try:
                size = float(size_str)
                if 0 < size < 1:
                    if size < self.min_test_frac:
                        issues.append(
                            f"test_size={size} is very small (<{self.min_test_frac})"
                        )
                        penalty += self.bad_split_penalty
                    elif size > self.max_test_frac:
                        issues.append(
                            f"test_size={size} is very large (>{self.max_test_frac})"
                        )
                        penalty += self.bad_split_penalty
            except ValueError:
                pass

        # ── 4. Stratification for imbalanced classification ───────────
        is_clf = session.task.task_type != TaskType.REGRESSION
        if is_clf and session.data_insights and agent_defines_split:
            imbalance = session.data_insights.class_imbalance_ratio
            if imbalance is not None and imbalance < 0.3:
                uses_stratify = bool(_STRATIFY.search(all_code))
                if not uses_stratify and not uses_cv:
                    issues.append(
                        f"Imbalanced classes (ratio={imbalance:.2f}) "
                        f"but no stratification in agent's split"
                    )
                    penalty += self.no_stratify_penalty

        # ── 5. Sandbox split check: actual sizes ─────────────────────
        ns = session.sandbox_namespace
        train_df = ns.get("train_df")
        valid_df = ns.get("valid_df")
        if train_df is not None and valid_df is not None:
            try:
                total = len(train_df) + len(valid_df)
                valid_frac = len(valid_df) / total if total > 0 else 0
                if valid_frac < 0.05:
                    issues.append(
                        f"Validation set is only {valid_frac:.0%} of data — too small"
                    )
                    penalty += self.bad_split_penalty
                elif valid_frac > 0.05:
                    bonuses.append(f"Valid set is {valid_frac:.0%} of data")
            except Exception:
                pass

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if issues:
            parts.append("Issues: " + "; ".join(issues))
        details = ". ".join(parts) if parts else "Split quality looks good."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _is_timeseries_task(
        self,
        session: RuntimeSession,
        code: str,
        plan: str,
    ) -> bool:
        """Determine if this is a time-series task (from metadata or heuristics)."""
        meta = session.task.metadata
        if meta.get("is_time_series"):
            return True
        if meta.get("time_column"):
            return True

        if session.data_insights and session.data_insights.datetime_like_columns:
            return True

        if _TIMESERIES_HINTS.search(code) or _TIMESERIES_HINTS.search(plan):
            return True

        return False
