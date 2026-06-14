"""Run the deterministic Titanic notebook protocol verifier as a regression test."""

from pathlib import Path
import subprocess
import sys


def test_titanic_notebook_protocol_verifier() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "verify_titanic_modes_protocol.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "protocol/fairness verification: PASS" in completed.stdout
