"""Regression test for the compact_feedback EDA-stdout drop bug."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent / "titanic_paper_experiment_suite.ipynb"
)


def _load_compact_feedback() -> callable:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell9_source = "".join(notebook["cells"][9]["source"])
    pattern = re.search(
        r"def _truncate_visible.*?(?=\n\ndef _assistant_history_receipt)",
        cell9_source,
        re.DOTALL,
    )
    assert pattern, "Could not locate the compact_feedback slice in cell 9"
    namespace: dict[str, object] = {"MAX_VISIBLE_FEEDBACK_CHARS": 1800}
    exec(pattern.group(0), namespace)  # noqa: S102 - test-only execution
    return namespace["compact_feedback"]


class CompactFeedbackEDARegression(unittest.TestCase):
    """Compact-feedback must keep stdout from successful EDA executions."""

    SAMPLE_RESPONSE = (
        "Execution: OK\n"
        "Output: rows: 623 cols: 12\n"
        "missing per col:\n"
        "PassengerId      0\n"
        "Pclass           0\n"
        "Age            119\n"
        "Cabin          487\n"
        "target dist:\n"
        "Survived\n"
        "0    0.616\n"
        "1    0.384\n"
        "\n"
        "--- Eda feedback ---\n"
        "Stage score: 0.571\n"
        "- Distribution anomalies have not been examined.\n"
        "- Potential feature-scale disparity has not been examined.\n"
        "\n"
        "--- Progress signal ---\n"
        "Reward: 0.1360\n"
        "Step: 2 / 8"
    )

    def setUp(self) -> None:
        self.compact_feedback = _load_compact_feedback()

    def test_output_line_is_kept(self) -> None:
        result = self.compact_feedback(self.SAMPLE_RESPONSE)
        self.assertIn("Output: rows: 623 cols: 12", result)

    def test_output_continuation_lines_are_kept(self) -> None:
        result = self.compact_feedback(self.SAMPLE_RESPONSE)
        self.assertIn("Age            119", result)
        self.assertIn("Cabin          487", result)

    def test_structured_signals_are_kept(self) -> None:
        result = self.compact_feedback(self.SAMPLE_RESPONSE)
        self.assertIn("Reward: 0.1360", result)
        self.assertIn("- Distribution anomalies have not been examined.", result)

    def test_failed_execution_keeps_error_line(self) -> None:
        """A FAILED execution still surfaces the failure header to the agent."""
        response = (
            "Execution: FAILED - NameError: name 'x' is not defined\n"
            "\n"
            "--- Eda feedback ---\n"
            "Stage score: 0.000\n"
            "- Provide a working EDA query.\n"
            "\n"
            "Reward: 0.0000\n"
            "Step: 2 / 8"
        )
        result = self.compact_feedback(response)
        self.assertIn("Execution: FAILED", result)
        self.assertIn("Reward: 0.0000", result)


if __name__ == "__main__":
    unittest.main()
