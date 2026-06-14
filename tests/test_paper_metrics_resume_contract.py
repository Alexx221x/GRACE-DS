"""Contract tests for paper-metric aggregation and resume support."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _step(action: str | None = None, *, validation_metric: float | None = None) -> dict:
    row = {
        "phase": "working",
        "source": "llm",
        "turn": 1,
        "reward": 0.1,
        "reward_weighted": 0.1,
        "reward_performance_contribution": 0.05,
        "reward_plan_contribution": 0.02,
        "reward_code_quality_contribution": 0.03,
        "reward_critical_error_category": "none",
    }
    if action is not None:
        row["actual_action"] = action
    if validation_metric is not None:
        row["validation_metric"] = validation_metric
    return row


def _episode(
    *,
    unit_id: str,
    regime: str,
    metric: float | None,
    protocol_valid: bool,
    reward: float = 0.5,
    task_id: str = "task_a",
    split_seed: int = 42,
    repeat_index: int = 0,
    temperature: float = 0.2,
    agent_violations: int = 0,
    execution_failures: int = 0,
    steps: list[dict] | None = None,
) -> dict:
    step_rows = steps if steps is not None else [_step(validation_metric=metric)]
    # aggregate._df_from_episodes derives final_reward from the terminal/last
    for row in step_rows:
        row.setdefault("reward", reward)
        row.setdefault("reward_weighted", reward)
        row["reward"] = reward
        row["reward_weighted"] = reward

    return {
        "unit_id": unit_id,
        "model": "model_a",
        "task_id": task_id,
        "regime": regime,
        "repeat_index": repeat_index,
        "split_seed": split_seed,
        "temperature": temperature,
        "ok": True,
        "payload": {
            "final_hidden_test_metric": metric,
            "final_reward": reward,
            "selected_validation_metric": metric,
            "protocol_valid": protocol_valid,
            "hidden_test_protocol_valid": protocol_valid,
            "agent_protocol_violation_count": agent_violations,
            "execution_failure_count": execution_failures,
            "llm_calls": 3,
            "working_llm_calls": 2,
            "selection_llm_calls": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_tokens": 120,
            "steps": step_rows,
        },
    }


def _aggregate(records: list[dict]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from automl_eval.experiment.aggregate import aggregate_run

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        raw = out_dir / "episodes_raw.jsonl"
        raw.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        paths = aggregate_run(raw, out_dir)
        flat = pd.read_csv(paths["episodes_flat.csv"])
        tables = {}
        for name, path in paths.items():
            if not name.endswith(".csv"):
                continue
            try:
                tables[name] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                tables[name] = pd.DataFrame()
        return flat, tables


class TestProtocolValidPaperMetricContract(unittest.TestCase):
    def test_invalid_hidden_metric_is_raw_only_not_headline(self):
        records = [
            _episode(
                unit_id="valid",
                regime="flexible_iterative",
                metric=0.80,
                protocol_valid=True,
                reward=0.50,
            ),
            _episode(
                unit_id="invalid",
                regime="flexible_iterative",
                metric=0.99,
                protocol_valid=False,
                reward=0.90,
                agent_violations=1,
            ),
            _episode(
                unit_id="no_metric",
                regime="flexible_iterative",
                metric=None,
                protocol_valid=False,
                reward=0.20,
                execution_failures=2,
            ),
        ]
        flat, tables = _aggregate(records)

        invalid = flat.loc[flat["raw_final_hidden_test_metric"] == 0.99].iloc[0]
        self.assertTrue(math.isnan(float(invalid["final_hidden_test_metric"])))
        self.assertTrue(math.isnan(float(invalid["main_hidden_test_metric"])))
        self.assertEqual(invalid["terminal_path"], "agent_violated_protocol")

        main = tables["table_main_performance.csv"].iloc[0]
        self.assertEqual(int(main["n_episodes"]), 3)
        self.assertEqual(int(main["n_observed"]), 1)
        self.assertEqual(main["observed_over_episodes"], "1/3")
        self.assertAlmostEqual(float(main["success_rate"]), 1 / 3)
        self.assertEqual(int(main["n_raw_observed"]), 2)
        self.assertEqual(main["raw_observed_over_episodes"], "2/3")
        self.assertAlmostEqual(float(main["raw_success_rate"]), 2 / 3)
        self.assertEqual(int(main["n_protocol_valid"]), 1)
        self.assertEqual(int(main["n_protocol_invalid"]), 2)
        self.assertEqual(int(main["n_no_hidden_metric"]), 1)
        self.assertEqual(int(main["n_agent_violations"]), 1)
        self.assertEqual(int(main["n_execution_failures"]), 2)
        self.assertAlmostEqual(float(main["median_hidden_test"]), 0.80)
        self.assertAlmostEqual(float(main["mean_hidden_test"]), 0.80)
        self.assertAlmostEqual(
            float(main["mean_final_reward_all"]), (0.50 + 0.90 + 0.20) / 3
        )
        self.assertAlmostEqual(float(main["mean_final_reward_protocol_valid"]), 0.50)
        self.assertAlmostEqual(float(main["mean_final_reward_successful"]), 0.50)
        self.assertAlmostEqual(
            float(main["mean_final_reward_failed"]), (0.90 + 0.20) / 2
        )
        self.assertAlmostEqual(
            float(main["reward_success_gap"]), 0.50 - ((0.90 + 0.20) / 2)
        )

        terminal = tables["table_terminal_path_breakdown.csv"].iloc[0]
        self.assertEqual(int(terminal["n_agent_final_submit"]), 1)
        self.assertEqual(int(terminal["n_agent_violated_protocol"]), 1)
        self.assertEqual(int(terminal["n_no_hidden_metric"]), 1)
        self.assertAlmostEqual(
            float(terminal["mean_hidden_test_honest"]), 0.80, places=4
        )
        self.assertAlmostEqual(
            float(terminal["mean_hidden_test_rescued"]), 0.99, places=4
        )

    def test_all_protocol_valid_cell_preserves_legacy_semantics(self):
        records = [
            _episode(
                unit_id="a",
                regime="single_shot",
                metric=0.60,
                protocol_valid=True,
                reward=0.40,
                repeat_index=0,
            ),
            _episode(
                unit_id="b",
                regime="single_shot",
                metric=0.80,
                protocol_valid=True,
                reward=0.60,
                repeat_index=1,
            ),
        ]
        _, tables = _aggregate(records)
        main = tables["table_main_performance.csv"].iloc[0]
        self.assertEqual(int(main["n_episodes"]), 2)
        self.assertEqual(int(main["n_observed"]), 2)
        self.assertEqual(int(main["n_raw_observed"]), 2)
        self.assertEqual(int(main["n_protocol_valid"]), 2)
        self.assertAlmostEqual(float(main["success_rate"]), 1.0)
        self.assertAlmostEqual(float(main["raw_success_rate"]), 1.0)
        self.assertAlmostEqual(float(main["protocol_valid_rate"]), 1.0)
        self.assertAlmostEqual(float(main["median_hidden_test"]), 0.70)
        self.assertAlmostEqual(float(main["mean_hidden_test"]), 0.70)
        self.assertAlmostEqual(
            float(main["std_hidden_test"]), np.std([0.60, 0.80], ddof=1)
        )
        self.assertAlmostEqual(float(main["mean_final_reward"]), 0.50)
        self.assertAlmostEqual(float(main["mean_final_reward_all"]), 0.50)
        self.assertAlmostEqual(float(main["mean_final_reward_protocol_valid"]), 0.50)
        self.assertAlmostEqual(float(main["mean_final_reward_successful"]), 0.50)
        self.assertTrue(pd.isna(main["mean_final_reward_failed"]))

    def test_task_normalisation_uses_main_metrics_for_headline_and_raw_for_diagnostics(
        self,
    ):
        records = [
            _episode(
                unit_id="valid_low",
                regime="r1",
                metric=0.60,
                protocol_valid=True,
                repeat_index=0,
            ),
            _episode(
                unit_id="valid_high",
                regime="r2",
                metric=0.80,
                protocol_valid=True,
                repeat_index=1,
            ),
            _episode(
                unit_id="invalid_best",
                regime="r3",
                metric=1.00,
                protocol_valid=False,
                agent_violations=1,
                repeat_index=2,
            ),
        ]
        flat, tables = _aggregate(records)
        by_uid = flat.set_index("repeat_index")
        self.assertAlmostEqual(
            float(by_uid.loc[0, "task_normalized_hidden_test_metric"]), 0.0
        )
        self.assertAlmostEqual(
            float(by_uid.loc[1, "task_normalized_hidden_test_metric"]), 1.0
        )
        self.assertTrue(pd.isna(by_uid.loc[2, "task_normalized_hidden_test_metric"]))
        self.assertAlmostEqual(
            float(by_uid.loc[0, "raw_task_normalized_hidden_test_metric"]), 0.0
        )
        self.assertAlmostEqual(
            float(by_uid.loc[1, "raw_task_normalized_hidden_test_metric"]), 0.5
        )
        self.assertAlmostEqual(
            float(by_uid.loc[2, "raw_task_normalized_hidden_test_metric"]), 1.0
        )

        main = tables["table_main_performance.csv"].set_index("regime")
        self.assertAlmostEqual(
            float(main.loc["r1", "mean_task_normalized_hidden_test"]), 0.0
        )
        self.assertAlmostEqual(
            float(main.loc["r2", "mean_task_normalized_hidden_test"]), 1.0
        )
        self.assertTrue(pd.isna(main.loc["r3", "mean_task_normalized_hidden_test"]))

    def test_significance_pairs_drop_protocol_invalid_rescue_scores(self):
        records = [
            _episode(
                unit_id="s1",
                regime="single_shot",
                metric=0.50,
                protocol_valid=True,
                split_seed=1,
                repeat_index=0,
            ),
            _episode(
                unit_id="f1",
                regime="flexible_iterative",
                metric=0.99,
                protocol_valid=False,
                agent_violations=1,
                split_seed=1,
                repeat_index=0,
            ),
            _episode(
                unit_id="s2",
                regime="single_shot",
                metric=0.60,
                protocol_valid=True,
                split_seed=2,
                repeat_index=0,
            ),
            _episode(
                unit_id="f2",
                regime="flexible_iterative",
                metric=0.70,
                protocol_valid=True,
                split_seed=2,
                repeat_index=0,
            ),
            _episode(
                unit_id="s3",
                regime="single_shot",
                metric=0.80,
                protocol_valid=False,
                agent_violations=1,
                split_seed=3,
                repeat_index=0,
            ),
            _episode(
                unit_id="f3",
                regime="flexible_iterative",
                metric=0.90,
                protocol_valid=True,
                split_seed=3,
                repeat_index=0,
            ),
        ]
        _, tables = _aggregate(records)
        sig = tables["table_significance.csv"]
        row = sig[sig["regime"] == "flexible_iterative"].iloc[0]
        self.assertEqual(int(row["n_pairs"]), 1)
        self.assertAlmostEqual(float(row["mean_delta"]), 0.10, places=4)

    def test_forbidden_ablation_action_attempts_are_reported_not_reinterpreted_as_quality(
        self,
    ):
        records = [
            _episode(
                unit_id="eda",
                regime="flexible_without_eda",
                metric=0.70,
                protocol_valid=False,
                agent_violations=1,
                steps=[_step("EDA"), _step("MODEL")],
            ),
            _episode(
                unit_id="plan",
                regime="fixed_without_plan",
                metric=0.70,
                protocol_valid=False,
                agent_violations=1,
                steps=[_step("PLAN"), _step("PLAN"), _step("MODEL")],
            ),
            _episode(
                unit_id="fe",
                regime="flexible_without_feature_engineering",
                metric=0.70,
                protocol_valid=True,
                steps=[_step("MODEL")],
            ),
        ]
        flat, tables = _aggregate(records)
        by_regime = flat.set_index("regime")
        self.assertEqual(
            int(
                by_regime.loc["flexible_without_eda", "forbidden_action_attempt_count"]
            ),
            1,
        )
        self.assertEqual(
            by_regime.loc["flexible_without_eda", "forbidden_actions_attempted"], "EDA"
        )
        self.assertEqual(
            int(by_regime.loc["fixed_without_plan", "forbidden_action_attempt_count"]),
            2,
        )
        self.assertEqual(
            by_regime.loc["fixed_without_plan", "forbidden_actions_attempted"], "PLAN"
        )
        self.assertEqual(
            int(
                by_regime.loc[
                    "flexible_without_feature_engineering",
                    "forbidden_action_attempt_count",
                ]
            ),
            0,
        )

        main = tables["table_main_performance.csv"].set_index("regime")
        self.assertAlmostEqual(
            float(main.loc["flexible_without_eda", "forbidden_action_attempt_rate"]),
            1.0,
        )
        self.assertEqual(
            int(main.loc["fixed_without_plan", "n_forbidden_action_attempts"]), 2
        )
        self.assertAlmostEqual(
            float(
                main.loc[
                    "flexible_without_feature_engineering",
                    "forbidden_action_attempt_rate",
                ]
            ),
            0.0,
        )


class TestResumeCheckpointContract(unittest.TestCase):
    def test_resume_records_merge_raw_and_checkpoints_with_checkpoint_precedence(self):
        from automl_eval.experiment.parallel_runner import (
            _load_resume_records,
            _rewrite_raw_jsonl,
            _write_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "episodes_raw.jsonl"
            checkpoints = root / "checkpoints"
            raw.write_text(
                json.dumps({"unit_id": "unit_a", "payload": {"value": 1}})
                + "\n"
                + json.dumps({"unit_id": "unit_a", "payload": {"value": 2}})
                + "\n"
                + json.dumps({"payload": {"value": 99}})
                + "\n"
                + "{torn json line\n",
                encoding="utf-8",
            )
            _write_checkpoint(
                checkpoints, {"unit_id": "unit_a", "payload": {"value": 4}}
            )
            _write_checkpoint(
                checkpoints, {"unit_id": "unit_b", "payload": {"value": 3}}
            )

            loaded = _load_resume_records(raw, checkpoints)
            self.assertEqual(set(loaded), {"unit_a", "unit_b"})
            self.assertEqual(loaded["unit_a"]["payload"]["value"], 4)
            self.assertEqual(loaded["unit_b"]["payload"]["value"], 3)

            _rewrite_raw_jsonl(raw, [loaded["unit_a"], loaded["unit_b"]])
            lines = [
                json.loads(line)
                for line in raw.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([r["unit_id"] for r in lines], ["unit_a", "unit_b"])

    def test_aggregate_loads_last_record_per_unit_id_for_idempotent_resume(self):
        from automl_eval.experiment.aggregate import _load_episodes

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "episodes_raw.jsonl"
            raw.write_text(
                json.dumps({"unit_id": "unit_a", "payload": {"metric": 0.1}})
                + "\n"
                + json.dumps({"unit_id": "unit_a", "payload": {"metric": 0.2}})
                + "\n"
                + json.dumps({"unit_id": "unit_b", "payload": {"metric": 0.3}})
                + "\n",
                encoding="utf-8",
            )
            loaded = _load_episodes(raw)
        by_uid = {r["unit_id"]: r for r in loaded}
        self.assertEqual(set(by_uid), {"unit_a", "unit_b"})
        self.assertEqual(by_uid["unit_a"]["payload"]["metric"], 0.2)
        self.assertEqual(by_uid["unit_b"]["payload"]["metric"], 0.3)

    def test_cli_exposes_resume_without_changing_default(self):
        from automl_eval.experiment.run_grace_experiments import _parse_args

        base = _parse_args(["--config", "config.yaml"])
        self.assertFalse(base.resume)
        resumed = _parse_args(["--config", "config.yaml", "--resume"])
        self.assertTrue(resumed.resume)


class TestNoGridPolicyIntegrationContract(unittest.TestCase):
    def test_policy_reaches_all_prompt_families(self):
        from automl_eval import prompts
        from automl_eval.experiment import _regimes_extracted as regimes

        prompt_texts = [
            prompts.build_system_prompt(6),
            regimes.constrained_model_only_system_prompt(regimes.StudyMode.SINGLE_SHOT),
            regimes.unstructured_agent_system_prompt(working_calls=6),
        ]
        for text in prompt_texts:
            self.assertIn(
                "Do NOT use automated or exhaustive hyperparameter/model-search approaches",
                text,
            )
            self.assertIn("GridSearchCV", text)
            self.assertIn("RandomizedSearchCV", text)
            self.assertIn(
                "Choose a small number of sensible hyperparameters yourself", text
            )

        direct = regimes.direct_model_task_request()
        self.assertIn("do not run grid/randomized/automated tuning", direct)
        self.assertIn("choose a few explicit hyperparameters yourself", direct)

    def test_model_tuning_checklist_now_means_manual_configuration(self):
        import inspect
        import automl_eval.domain.hidden_checklists as hc

        src = inspect.getsource(hc.compile_hidden_checklist)
        self.assertIn(
            'C("model_tuning", Stage.MODEL, "Manual model configuration has not been demonstrated."',
            src,
        )
        self.assertIn("without automated search", src)
        self.assertNotIn("Model search or tuning has not been demonstrated", src)


if __name__ == "__main__":
    unittest.main()


class TestFlexibleCandidateFirstAndDiversityContract(unittest.TestCase):
    def test_candidate_diversity_helpers_score_small_manual_diversity(self):
        from automl_eval.evaluation.candidate_diversity import (
            candidate_diversity_feedback,
            candidate_diversity_score,
            primary_model_family,
        )

        self.assertEqual(
            primary_model_family("RandomForestClassifier(n_estimators=120)"),
            "RandomForest",
        )
        self.assertEqual(
            primary_model_family("LogisticRegression(max_iter=1000)"),
            "LogisticRegression",
        )
        self.assertLess(candidate_diversity_score(["LogisticRegression"]), 0.75)
        self.assertLess(
            candidate_diversity_score(["LogisticRegression", "LogisticRegression"]),
            0.75,
        )
        self.assertGreaterEqual(
            candidate_diversity_score(["LogisticRegression", "RandomForest"]), 0.75
        )
        self.assertIn(
            "different manual model family",
            candidate_diversity_feedback(["LogisticRegression"], remaining_turns=2),
        )

    def test_aggregate_reports_candidate_family_diagnostics_and_stop_redirects(self):
        def model_step(family: str) -> dict:
            return _step("MODEL") | {
                "actual_action": "MODEL",
                "execution_success": True,
                "action_text": f"ACTION: MODEL\n```python\nfrom sklearn.ensemble import {family}\npipeline = {family}(random_state=42)\n```",
            }

        records = [
            _episode(
                unit_id="diverse",
                regime="flexible_iterative",
                metric=0.8,
                protocol_valid=True,
                steps=[
                    model_step("RandomForestClassifier"),
                    _step("VALIDATE", validation_metric=0.7),
                    _step("STOP_WORKING_TOO_EARLY"),
                    _step("MODEL")
                    | {
                        "actual_action": "MODEL",
                        "execution_success": True,
                        "action_text": "ACTION: MODEL\n```python\nfrom sklearn.linear_model import LogisticRegression\npipeline = LogisticRegression(max_iter=1000)\n```",
                    },
                    _step("VALIDATE", validation_metric=0.8),
                ],
            )
        ]
        flat, tables = _aggregate(records)
        row = flat.iloc[0]
        self.assertEqual(int(row["n_model_actions"]), 2)
        self.assertEqual(int(row["n_validated_candidates"]), 2)
        self.assertEqual(int(row["n_distinct_model_families"]), 2)
        self.assertGreaterEqual(float(row["candidate_diversity_score"]), 0.75)
        self.assertEqual(int(row["early_stop_redirect_count"]), 1)
        main = tables["table_main_performance.csv"].iloc[0]
        self.assertEqual(int(main["n_early_stop_redirects"]), 1)
        self.assertGreaterEqual(float(main["mean_candidate_diversity_score"]), 0.75)
        self.assertAlmostEqual(float(main["mean_distinct_model_families"]), 2.0)

    def test_flexible_prompt_enforces_candidate_first_and_stop_guard_language(self):
        from automl_eval.experiment._regimes_extracted import (
            StudyMode,
            mode_system_prompt,
        )

        prompt = mode_system_prompt(StudyMode.FLEXIBLE, 8)
        self.assertIn("Flexible candidate-first regime", prompt)
        self.assertIn("PLAN -> MODEL -> EDA -> FEATURE_ENGINEERING", prompt)
        self.assertIn("Do not write a PLAN that jumps straight to EDA", prompt)
        self.assertIn("STOP_WORKING is only accepted", prompt)

    def test_stop_guard_blocks_only_primary_flexible_when_too_early(self):
        from automl_eval.experiment._regimes_extracted import (
            CandidateRecord,
            StudyMode,
            _should_block_flexible_stop,
        )

        candidate = CandidateRecord(
            "C1",
            0.7,
            [
                "ACTION: MODEL\n```python\nfrom sklearn.linear_model import LogisticRegression\npipeline = LogisticRegression(max_iter=1000)\n```",
                "ACTION: VALIDATE",
            ],
        )
        blocked, reason = _should_block_flexible_stop(
            StudyMode.FLEXIBLE, [candidate], 3
        )
        self.assertTrue(blocked)
        self.assertIn("STOP_WORKING rejected as too early", reason)
        blocked_restart, _ = _should_block_flexible_stop(
            StudyMode.FLEXIBLE_COMPACT, [candidate], 3
        )
        self.assertFalse(blocked_restart)

    def test_candidate_first_guard_redirects_only_primary_flexible_opening(self):
        from automl_eval.experiment._regimes_extracted import (
            StudyMode,
            _flexible_candidate_first_block_reason,
        )

        self.assertIn(
            "start with ACTION: PLAN",
            _flexible_candidate_first_block_reason(StudyMode.FLEXIBLE, [], "EDA", 1),
        )
        self.assertIn(
            "after PLAN, use ACTION: MODEL",
            _flexible_candidate_first_block_reason(StudyMode.FLEXIBLE, [], "EDA", 2),
        )
        self.assertIsNone(
            _flexible_candidate_first_block_reason(StudyMode.FLEXIBLE, [], "PLAN", 1)
        )
        self.assertIsNone(
            _flexible_candidate_first_block_reason(
                StudyMode.FLEXIBLE_COMPACT, [], "EDA", 1
            )
        )


class TestTimeoutStandardisationContract(unittest.TestCase):
    def test_config_exposes_standardized_timeout_fields_and_units(self):
        from automl_eval.experiment.config import ExperimentConfig
        from automl_eval.experiment.parallel_runner import build_units

        cfg = ExperimentConfig(
            models=["m"],
            task_ids=["t"],
            regimes=[
                "single_shot",
                "n_restarts_from_scratch",
                "n_restarts_call_matched_upper_bound",
                "unstructured_agent",
                "flexible_iterative",
            ],
            repeats_per_condition=1,
            stateless_sandbox_timeout_sec=111,
            stateful_sandbox_timeout_sec=333,
            stateless_task_time_budget_sec=444.0,
            stateful_task_time_budget_sec=999.0,
            stateful_stage_time_budget_multiplier=4.0,
        )
        cfg.validate()
        units = {u.regime: u for u in build_units(cfg)}
        for regime in [
            "single_shot",
            "n_restarts_from_scratch",
            "n_restarts_call_matched_upper_bound",
            "unstructured_agent",
        ]:
            self.assertEqual(units[regime].stateless_sandbox_timeout_sec, 111)
            self.assertEqual(units[regime].stateful_sandbox_timeout_sec, 333)
            self.assertEqual(units[regime].stateless_task_time_budget_sec, 444.0)
            self.assertEqual(units[regime].stateful_task_time_budget_sec, 999.0)
        self.assertEqual(
            units["flexible_iterative"].stateful_stage_time_budget_multiplier, 4.0
        )

    def test_regime_environment_applies_stateless_vs_stateful_timeout_policy(self):
        import importlib
        import os
        from automl_eval.experiment import _regimes_extracted as regimes

        old_env = {
            k: os.environ.get(k)
            for k in [
                "GRACE_STATELESS_SANDBOX_TIMEOUT_SECONDS",
                "GRACE_STATEFUL_SANDBOX_TIMEOUT_SECONDS",
                "GRACE_STATELESS_TASK_TIME_BUDGET_SECONDS",
                "GRACE_STATEFUL_TASK_TIME_BUDGET_SECONDS",
                "GRACE_STATEFUL_STAGE_TIME_BUDGET_MULTIPLIER",
                "ACTIVE_TASK_ID",
            ]
        }
        try:
            os.environ["ACTIVE_TASK_ID"] = "titanic_binary"
            os.environ["GRACE_STATELESS_SANDBOX_TIMEOUT_SECONDS"] = "111"
            os.environ["GRACE_STATEFUL_SANDBOX_TIMEOUT_SECONDS"] = "333"
            os.environ["GRACE_STATELESS_TASK_TIME_BUDGET_SECONDS"] = "444"
            os.environ["GRACE_STATEFUL_TASK_TIME_BUDGET_SECONDS"] = "999"
            os.environ["GRACE_STATEFUL_STAGE_TIME_BUDGET_MULTIPLIER"] = "4"
            regimes = importlib.reload(regimes)

            stateless = regimes.new_environment(1, mode=regimes.StudyMode.SINGLE_SHOT)
            stateful = regimes.new_environment(1, mode=regimes.StudyMode.FLEXIBLE)
            self.assertEqual(stateless.sandbox.timeout_seconds, 111)
            self.assertEqual(stateful.sandbox.timeout_seconds, 333)
            self.assertEqual(stateless._task.time_budget_seconds, 444.0)
            self.assertEqual(stateful._task.time_budget_seconds, 999.0)
            self.assertAlmostEqual(
                stateful._task.stage_limits["MODEL"].max_seconds,
                180.0 * 4,
            )
            self.assertAlmostEqual(
                stateless._task.stage_limits["MODEL"].max_seconds, 180.0
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(regimes)

    def test_cli_exposes_timeout_overrides_and_rerun_transient_requires_resume(self):
        from automl_eval.experiment.run_grace_experiments import _parse_args

        args = _parse_args(
            [
                "--config",
                "config.yaml",
                "--resume",
                "--rerun-transient",
                "--stateless-sandbox-timeout",
                "111",
                "--stateful-sandbox-timeout",
                "333",
                "--stateless-task-timeout",
                "444",
                "--stateful-task-timeout",
                "999",
            ]
        )
        self.assertTrue(args.resume)
        self.assertTrue(args.rerun_transient)
        self.assertEqual(args.stateless_sandbox_timeout, 111)
        self.assertEqual(args.stateful_sandbox_timeout, 333)
        self.assertEqual(args.stateless_task_timeout, 444.0)
        self.assertEqual(args.stateful_task_timeout, 999.0)

    def test_rerun_transient_filters_only_provider_failures(self):
        from automl_eval.experiment.parallel_runner import (
            _filter_resume_records_for_rerun,
            _is_transient_failure_record,
        )

        transient = {
            "unit_id": "a",
            "ok": False,
            "error": "LLM HTTP error 429: rate limit exceeded",
        }
        timeout = {
            "unit_id": "b",
            "ok": False,
            "error": "LLM read timed out after 300s",
        }
        protocol = {
            "unit_id": "c",
            "ok": False,
            "error": "single-shot protocol violation",
        }
        success = {"unit_id": "d", "ok": True, "error": None}
        self.assertTrue(_is_transient_failure_record(transient))
        self.assertTrue(_is_transient_failure_record(timeout))
        self.assertFalse(_is_transient_failure_record(protocol))
        self.assertFalse(_is_transient_failure_record(success))

        keep, rerun = _filter_resume_records_for_rerun(
            {"a": transient, "b": timeout, "c": protocol, "d": success},
            rerun_transient=True,
        )
        self.assertEqual(set(rerun), {"a", "b"})
        self.assertEqual(set(keep), {"c", "d"})


class TestParserFeedbackAndAblationContracts(unittest.TestCase):
    def test_action_parser_ignores_non_python_fences_around_one_python_block(self):
        from automl_eval.core.action_parser import ActionParser
        from automl_eval.core.session import ActionType

        text = """ACTION: MODEL
Here is the candidate.
```text
not executable output
```
```python
pipeline = None
```
"""
        parsed = ActionParser().parse(text)
        self.assertEqual(parsed.action_type, ActionType.MODEL)
        self.assertEqual(parsed.code_block_count, 1)
        self.assertEqual(parsed.body.strip(), "pipeline = None")

    def test_sandbox_allows_train_local_metric_but_blocks_valid_metric(self):
        from automl_eval.core.sandbox import Sandbox

        ns = {}
        train_only = """
from sklearn.metrics import mean_squared_error
score = mean_squared_error([1, 2], [1, 3])
"""
        self.assertTrue(Sandbox(timeout_seconds=5).execute(train_only, ns).success)
        blocked = """
from sklearn.metrics import mean_squared_error
score = mean_squared_error(valid_df['y'], [1, 2])
"""
        out = Sandbox(timeout_seconds=5).execute(blocked, {"valid_df": {"y": [1, 2]}})
        self.assertFalse(out.success)
        self.assertIn("ACTION: VALIDATE", out.error or "")

    def test_replay_feedback_mentions_dtype_specific_preprocessing(self):
        from automl_eval.core.environment import _targeted_repair_hint

        hint = _targeted_repair_hint(
            "ValueError: Cannot use median strategy with non-numeric data"
        )
        self.assertIn("numeric columns", hint)
        self.assertIn("categorical", hint)
        self.assertIn("ColumnTransformer", hint)

    def test_raw_payload_metric_semantics_and_aggregate_terminal_path(self):
        from automl_eval.experiment.aggregate import (
            _df_from_episodes,
            _derive_terminal_path,
        )

        payload = {
            "final_hidden_test_metric": None,
            "raw_final_hidden_test_metric": 0.99,
            "main_hidden_test_metric": None,
            "protocol_valid": False,
            "agent_protocol_violation_count": 1,
            "steps": [],
        }
        self.assertEqual(_derive_terminal_path(payload), "agent_violated_protocol")
        df = _df_from_episodes(
            [
                {
                    "model": "m",
                    "task_id": "t",
                    "regime": "r",
                    "repeat_index": 0,
                    "split_seed": 42,
                    "temperature": 0.7,
                    "ok": True,
                    "payload": payload,
                }
            ]
        )
        self.assertEqual(float(df.loc[0, "raw_final_hidden_test_metric"]), 0.99)
        self.assertTrue(pd.isna(df.loc[0, "final_hidden_test_metric"]))

    def test_config_and_dispatch_expose_masked_ablation_without_changing_default(self):
        from automl_eval.experiment.config import (
            ExperimentConfig,
            ALL_REGIMES,
            VALID_REGIMES,
        )
        from automl_eval.experiment import _regimes_extracted as regimes

        self.assertIn("flexible_without_eda_masked", VALID_REGIMES)
        self.assertNotIn("flexible_without_eda_masked", ALL_REGIMES)
        cfg = ExperimentConfig(
            models=["m"],
            task_ids=["t"],
            regimes=["flexible_without_eda_masked"],
            repeats_per_condition=1,
        )
        cfg.validate()
        self.assertIn("flexible_without_eda_masked", regimes.RUNNERS)

    def test_aggregate_reports_candidate_first_and_masked_ablation_redirects(self):
        from automl_eval.experiment.aggregate import _df_from_episodes, _agg_main

        ep = {
            "model": "m",
            "task_id": "t",
            "regime": "flexible_without_eda_masked",
            "repeat_index": 0,
            "split_seed": 42,
            "temperature": 0.7,
            "ok": True,
            "payload": {
                "protocol_valid": True,
                "raw_final_hidden_test_metric": 0.8,
                "main_hidden_test_metric": 0.8,
                "final_hidden_test_metric": 0.8,
                "final_reward": 0.5,
                "steps": [
                    {
                        "actual_action": "CANDIDATE_FIRST_REDIRECT",
                        "recoverable_glitch": "candidate_first_redirect",
                        "reward_weighted": 0.1,
                    },
                    {
                        "actual_action": "MASKED_ABLATION_REDIRECT",
                        "recoverable_glitch": "masked_ablation_redirect",
                        "reward_weighted": 0.1,
                    },
                ],
            },
        }
        df = _df_from_episodes([ep])
        self.assertEqual(int(df.loc[0, "candidate_first_redirect_count"]), 1)
        self.assertEqual(int(df.loc[0, "masked_ablation_redirect_count"]), 1)
        main = _agg_main(df)
        self.assertEqual(int(main.loc[0, "n_candidate_first_redirects"]), 1)
        self.assertEqual(int(main.loc[0, "n_masked_ablation_redirects"]), 1)


class TestDebugTraceContract(unittest.TestCase):
    def test_config_builds_trace_paths_for_units(self):
        from automl_eval.experiment.config import ExperimentConfig
        from automl_eval.experiment.parallel_runner import build_units

        cfg = ExperimentConfig(
            models=["m"],
            task_ids=["t"],
            regimes=["single_shot"],
            repeats_per_condition=1,
            output_dir="outputs_trace",
            run_name="r1",
            debug_trace_enabled=True,
            log_executable_code=True,
            log_raw_llm_responses=True,
        )
        cfg.validate()
        unit = build_units(cfg)[0]
        self.assertTrue(unit.debug_trace_enabled)
        self.assertTrue(unit.log_executable_code)
        self.assertTrue(unit.log_raw_llm_responses)
        self.assertTrue(unit.trace_base_dir.endswith("outputs_trace/r1/debug_trace"))

    def test_trace_event_and_artifact_are_fsync_safe_jsonl_files(self):
        from automl_eval.evaluation.debug_trace import save_text_artifact, trace_event

        with tempfile.TemporaryDirectory() as tmp:
            trace_event(
                "before_llm_request",
                unit_id="model::task::regime",
                base_dir=tmp,
                turn=1,
            )
            rel, digest = save_text_artifact(
                "executable_code",
                "print('hello')\n",
                unit_id="model::task::regime",
                base_dir=tmp,
                stem="unit_step1_MODEL",
                suffix=".py",
            )
            trace_path = Path(tmp) / "worker_traces" / "model__task__regime.jsonl"
            self.assertTrue(trace_path.exists())
            rec = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(rec["event"], "before_llm_request")
            self.assertEqual(rec["turn"], 1)
            self.assertIsNotNone(rel)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual((Path(tmp) / rel).read_text(), "print('hello')\n")

    def test_environment_trace_logs_parsing_and_code_before_execution(self):
        import os
        from automl_eval.core.environment import AutoMLEnvironment
        from automl_eval.domain.task_registry import TaskRegistry

        reg = TaskRegistry()
        reg.load_directory("automl_eval/tasks")
        old_env = {
            k: os.environ.get(k)
            for k in [
                "GRACE_TRACE_ENABLED",
                "GRACE_TRACE_DIR",
                "GRACE_UNIT_ID",
                "GRACE_LOG_EXECUTABLE_CODE",
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["GRACE_TRACE_ENABLED"] = "1"
                os.environ["GRACE_TRACE_DIR"] = tmp
                os.environ["GRACE_UNIT_ID"] = "trace_unit"
                os.environ["GRACE_LOG_EXECUTABLE_CODE"] = "1"
                env = AutoMLEnvironment(reg, seed=42, sandbox_timeout=5)
                env.reset("titanic_binary")
                out = env.step("ACTION: EDA\n```python\nprint(train_df.shape)\n```")
                self.assertTrue(out.reward is not None)
                trace_file = Path(tmp) / "worker_traces" / "trace_unit.jsonl"
                events = [
                    json.loads(line)["event"]
                    for line in trace_file.read_text().splitlines()
                ]
                self.assertIn("before_environment_step", events)
                self.assertIn("after_action_parse", events)
                self.assertIn("before_sandbox_exec", events)
                self.assertIn("after_sandbox_exec", events)
                code_files = list((Path(tmp) / "executable_code").glob("*.py"))
                self.assertEqual(len(code_files), 1)
                self.assertIn("print(train_df.shape)", code_files[0].read_text())
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
