"""Single entry point to run GRACE paper experiments."""

from __future__ import annotations

import argparse
import sys

from automl_eval.experiment.config import ExperimentConfig
from automl_eval.experiment.parallel_runner import run_experiment


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_grace_experiments",
        description="Run the GRACE harness-regime experiment grid over OpenRouter LLMs.",
    )
    p.add_argument("--config", required=True, help="Path to a YAML ExperimentConfig.")
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated OpenRouter model slugs (overrides config).",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated GRACE task ids (overrides config).",
    )
    p.add_argument(
        "--regimes", default=None, help="Comma-separated regimes (overrides config)."
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Repeats per (split,temperature) (overrides config).",
    )
    p.add_argument(
        "--split-seeds",
        default=None,
        help="Comma-separated data-split seeds (overrides config).",
    )
    p.add_argument(
        "--temperatures",
        default=None,
        help="Comma-separated temperatures (overrides config).",
    )
    p.add_argument(
        "--llm-seed-base",
        type=int,
        default=None,
        help="Base LLM sampling seed (overrides config).",
    )
    p.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="Per-episode total token budget (overrides config).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process pool size (overrides config).",
    )
    p.add_argument(
        "--output", default=None, help="Output directory (overrides config)."
    )
    p.add_argument(
        "--run-name", default=None, help="Run name subdirectory (overrides config)."
    )
    p.add_argument(
        "--paraphrase",
        action="store_true",
        help="Enable prompt paraphrasing (LLF-Bench-style).",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Enable FeatEng-style performance normalization.",
    )
    p.add_argument(
        "--no-auto-finalize",
        action="store_true",
        help="DISABLE auto-finalize on budget exhaustion (reproduces strict-FINAL_SUBMIT protocol).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build/validate the grid but do not execute episodes.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory: skip completed unit_id records and aggregate all saved checkpoints.",
    )
    p.add_argument(
        "--rerun-transient",
        action="store_true",
        help="With --resume, rerun only transient provider/API failures (429, timeout, 5xx, connection errors) instead of counting them as failed episodes.",
    )
    p.add_argument(
        "--stateless-sandbox-timeout",
        type=int,
        default=None,
        help="Per-action sandbox timeout for single_shot, n_restarts_* and unstructured_agent (seconds; overrides YAML).",
    )
    p.add_argument(
        "--stateful-sandbox-timeout",
        type=int,
        default=None,
        help="Per-action sandbox timeout for stateful structured regimes (seconds; overrides YAML).",
    )
    p.add_argument(
        "--stateless-task-timeout",
        type=float,
        default=None,
        help="Optional cumulative task time budget override for stateless/free-form regimes (seconds; overrides YAML).",
    )
    p.add_argument(
        "--stateful-task-timeout",
        type=float,
        default=None,
        help="Optional cumulative task time budget override for stateful structured regimes (seconds; overrides YAML).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)

    if args.models:
        config.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.tasks:
        config.task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.regimes:
        config.regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    if args.repeats is not None:
        config.repeats_per_condition = args.repeats
    if args.split_seeds:
        config.split_seeds = [
            int(s.strip()) for s in args.split_seeds.split(",") if s.strip()
        ]
    if args.temperatures:
        config.temperature_schedule = [
            float(t.strip()) for t in args.temperatures.split(",") if t.strip()
        ]
        config.temperature = None
    if args.llm_seed_base is not None:
        config.llm_seed_base = args.llm_seed_base
    if args.total_tokens is not None:
        config.total_token_budget = args.total_tokens
    if args.workers is not None:
        config.num_workers = args.workers
    if args.output:
        config.output_dir = args.output
    if args.run_name:
        config.run_name = args.run_name
    if args.paraphrase:
        config.paraphrase_prompts = True
    if args.normalize:
        config.performance_normalization = True
    if args.no_auto_finalize:
        config.auto_finalize_on_exhaustion = False
    if args.stateless_sandbox_timeout is not None:
        config.stateless_sandbox_timeout_sec = args.stateless_sandbox_timeout
    if args.stateful_sandbox_timeout is not None:
        config.stateful_sandbox_timeout_sec = args.stateful_sandbox_timeout
    if args.stateless_task_timeout is not None:
        config.stateless_task_time_budget_sec = args.stateless_task_timeout
    if args.stateful_task_timeout is not None:
        config.stateful_task_time_budget_sec = args.stateful_task_timeout
    if args.rerun_transient and not args.resume:
        raise SystemExit("--rerun-transient requires --resume")

    out_dir = run_experiment(
        config,
        dry_run=args.dry_run,
        resume=args.resume,
        rerun_transient=args.rerun_transient,
    )
    print(f"\n[OK] outputs in: {out_dir}")
    if not args.dry_run:
        print(
            "Paper tables: table_main_performance.csv, table_reward_decomposition.csv, "
            "table_reward_growth_slopes.csv, table_critical_errors.csv, "
            "table_significance.csv"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
