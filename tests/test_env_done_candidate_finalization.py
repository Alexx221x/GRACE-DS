"""Regression test for the env-done candidate-discard bug."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "titanic_paper_experiment_suite.ipynb"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EnvDoneDispatchPresenceTest(unittest.TestCase):
    """Notebook-level: each working loop calls the env-done finalizer."""

    @classmethod
    def setUpClass(cls) -> None:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cell9 = "".join(nb["cells"][9]["source"])

    def test_helper_definition_exists(self) -> None:
        self.assertIn("def finalize_validated_candidate_after_env_done(", self.cell9)

    def test_three_dispatch_sites(self) -> None:
        pattern = re.compile(
            r"if done:\s*\n"
            r"\s*if candidates:\s*\n"
            r"\s*return finalize_validated_candidate_after_env_done\("
        )
        matches = pattern.findall(self.cell9)
        self.assertEqual(
            len(matches),
            3,
            msg=f"Expected 3 finalize-after-env-done dispatch sites, found {len(matches)}",
        )

    def test_helper_records_env_budget_stop(self) -> None:
        self.assertIn("ENV_BUDGET_STOP", self.cell9)


class EnvDoneSemanticsTest(unittest.TestCase):
    """End-to-end: confirm the env really suppresses hidden test on done=True"""

    @classmethod
    def setUpClass(cls) -> None:
        from automl_eval.core.environment import AutoMLEnvironment
        from automl_eval.domain.task_registry import TaskRegistry

        cls.AutoMLEnvironment = AutoMLEnvironment
        cls.registry = TaskRegistry()
        cls.registry.load_directory(str(REPO_ROOT / "automl_eval" / "tasks"))

    MODEL_CODE = (
        "ACTION: MODEL\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "X = train_df.drop(columns=['Survived']).copy()\n"
        "y = train_df['Survived']\n"
        "num = X.select_dtypes(include='number').columns.tolist()\n"
        "cat = X.select_dtypes(exclude='number').columns.tolist()\n"
        "pre = ColumnTransformer([\n"
        "    ('num', SimpleImputer(strategy='median'), num),\n"
        "    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),\n"
        "                      ('enc', OneHotEncoder(handle_unknown='ignore'))]), cat),\n"
        "])\n"
        "pipeline = Pipeline([('pre', pre), ('model', RandomForestClassifier(random_state=42, n_estimators=20))])\n"
        "pipeline.fit(X, y)\n"
        "def predict_fn(raw_dataframe):\n"
        "    return pipeline.predict_proba(raw_dataframe)[:, 1]\n"
        "```\n"
    )

    def _drive_until_done(self, env) -> int:
        """Alternate MODEL and VALIDATE until env signals done; return the turn count."""
        for turn in range(1, 30):
            action = self.MODEL_CODE if turn % 2 == 1 else "ACTION: VALIDATE"
            output = env.step(action)
            if output.done:
                return turn
        raise AssertionError("env never signalled done within 30 steps")

    def test_working_env_done_suppresses_hidden_test_but_keeps_candidate(self) -> None:
        env = self.AutoMLEnvironment(
            self.registry,
            seed=42,
            allow_forced_terminal_evaluation=False,
            enforce_stage_budgets=True,
        )
        env.reset("titanic_binary", max_actions=13)
        self._drive_until_done(env)
        self.assertTrue(env._session.done, "session must be marked done")
        self.assertIsNone(
            env._session.hidden_test_metric,
            "hidden test must NOT be evaluated when allow_forced_terminal_evaluation=False",
        )
        self.assertIsNotNone(
            env._session.best_metric,
            "but at least one validated candidate must have been captured",
        )

    def test_fresh_replay_env_finalizes_same_candidate(self) -> None:
        """A fresh env with allow_forced_terminal_evaluation=True scores the candidate."""
        env = self.AutoMLEnvironment(
            self.registry,
            seed=42,
            allow_forced_terminal_evaluation=True,
            enforce_stage_budgets=True,
        )
        env.reset("titanic_binary", max_actions=5)
        env.step(self.MODEL_CODE)
        env.step("ACTION: VALIDATE")
        terminal_output = env.step("ACTION: FINAL_SUBMIT")
        self.assertTrue(terminal_output.done)
        self.assertIsNotNone(env._session.hidden_test_metric)
        self.assertEqual(env._session.test_evaluation_count, 1)


if __name__ == "__main__":
    unittest.main()
