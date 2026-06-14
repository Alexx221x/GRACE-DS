#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_paper.yaml"
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke_subsampled_grid.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/run_experiments.py",
        description=(
            "Run the existing GRACE experiment runner via uv with repository "
            "defaults and light preflight validation."
        ),
        epilog=(
            "Any unknown arguments are forwarded to "
            "automl_eval.experiment.run_grace_experiments, for example: "
            "--resume, --rerun-transient, --dry-run, --workers, "
            "--models, --tasks, --regimes, --temperatures, --output, "
            "--run-name."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Experiment config path. Default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Use the smoke config by default ({SMOKE_CONFIG.relative_to(REPO_ROOT)})."
        ),
    )
    return parser


def _normalize_args(args: argparse.Namespace) -> tuple[Path, list[str]]:
    config_path = args.config
    if config_path is None:
        config_path = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path.resolve(), []


def _has_flag(argv: list[str], *flags: str) -> bool:
    return any(arg in flags for arg in argv)


def _ensure_prerequisites(config_path: Path, forwarded_args: list[str]) -> None:
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise SystemExit(
            f"Repository root is invalid: missing pyproject.toml in {REPO_ROOT}"
        )

    if (
        subprocess.run(
            ["uv", "--version"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
    ):
        raise SystemExit("`uv` is required but was not found in PATH.")

    import_cmd = [
        "uv",
        "run",
        "python",
        "-c",
        "import automl_eval, numpy, pandas, yaml",
    ]
    if subprocess.run(import_cmd, cwd=REPO_ROOT, check=False).returncode != 0:
        raise SystemExit(
            "Project environment is not ready. Run `uv sync` and try again."
        )

    needs_api_key = not _has_flag(forwarded_args, "--dry-run")
    if needs_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it first, or use `--dry-run`."
        )


def _build_runner_command(config_path: Path, forwarded_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "automl_eval.experiment.run_grace_experiments",
        "--config",
        str(config_path),
        *forwarded_args,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    config_path, extra_args = _normalize_args(args)
    forwarded_args = [*extra_args, *forwarded_args]

    _ensure_prerequisites(config_path, forwarded_args)

    command = _build_runner_command(config_path, forwarded_args)
    rel_config = config_path.relative_to(REPO_ROOT)
    mode = "smoke" if args.smoke and args.config is None else "run"
    print(f"[run_experiments] mode={mode} config={rel_config}", flush=True)
    print(f"[run_experiments] exec: {shlex.join(command)}", flush=True)

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
