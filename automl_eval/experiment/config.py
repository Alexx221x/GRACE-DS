"""Experiment configuration for the GRACE parallel runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_eval.llm.prompt_paraphrase import PROMPT_VARIANT_IDS

ALL_REGIMES: tuple[str, ...] = (
    # --- core regime comparison (structure vs free-form iteration) ---
    "single_shot",
    "n_restarts_from_scratch",
    "n_restarts_call_matched_upper_bound",
    "unstructured_agent",
    "fixed_stage_iterative",
    "flexible_iterative",
    "baseline_first_structured",
    "flexible_compact_feedback",
    # --- per-stage ablations (each removes exactly one stage) ---
    "fixed_without_plan",
    "fixed_without_eda",
    "fixed_without_feature_engineering",
    "flexible_without_eda",
    "flexible_without_feature_engineering",
    # --- reward-hacking probes (maximize reward, not solve the task) ---
    "reward_maximizer_hidden_hints",
    "reward_maximizer_disclosed_criteria",
    # --- red-team-vs-validators probe (game the validators on purpose) ---
    "red_team_vs_validators",
)

NOVELTY_CRITICAL_REGIMES: tuple[str, ...] = (
    "single_shot",
    "fixed_stage_iterative",
    "flexible_iterative",
)

STAGE_ABLATION_REGIMES: tuple[str, ...] = (
    "fixed_stage_iterative",
    "fixed_without_plan",
    "fixed_without_eda",
    "fixed_without_feature_engineering",
    "flexible_iterative",
    "flexible_without_eda",
    "flexible_without_feature_engineering",
)

OPTIONAL_MASKED_ABLATION_REGIMES: tuple[str, ...] = (
    "flexible_without_eda_masked",
    "flexible_without_feature_engineering_masked",
)

VALID_REGIMES: tuple[str, ...] = ALL_REGIMES + OPTIONAL_MASKED_ABLATION_REGIMES

REWARD_HACKING_REGIMES: tuple[str, ...] = (
    "flexible_iterative",
    "reward_maximizer_hidden_hints",
    "reward_maximizer_disclosed_criteria",
)

RED_TEAM_REGIMES: tuple[str, ...] = (
    "flexible_iterative",
    "red_team_vs_validators",
)

STATELESS_TIMEOUT_REGIMES: tuple[str, ...] = (
    "single_shot",
    "n_restarts_from_scratch",
    "n_restarts_call_matched_upper_bound",
    "unstructured_agent",
)


@dataclass
class ExperimentConfig:
    # --- models & data ---
    models: list[str]
    task_ids: list[str]
    regimes: list[str] = field(default_factory=lambda: list(ALL_REGIMES))

    # --- statistics / replication grid ---
    split_seeds: list[int] = field(default_factory=lambda: [42])
    temperature_schedule: list[float] = field(default_factory=lambda: [0.7])
    repeats_per_condition: int = 3
    # Reproducible LLM sampling: each episode is sent seed = llm_seed_base + repeat_index,
    llm_seed_base: int = 1000
    # Back-compat: a scalar `temperature` and explicit `seeds` from older configs are
    temperature: float | None = None
    seeds: list[int] | None = None

    # --- budgets ---
    max_actions: int = 8
    max_tokens: int = 4096
    total_token_budget: int = 40000
    n_restarts: int = 4

    # --- dataset subsampling ---
    dataset_subsample_factor: int = 1
    # --- parallelism ---
    num_workers: int = 8
    episode_timeout_sec: int = 1800
    # Sandbox/code-execution timeouts are split into two explicit YAML knobs:
    stateless_sandbox_timeout_sec: int = 300
    stateful_sandbox_timeout_sec: int = 900
    stateless_task_time_budget_sec: float | None = None
    stateful_task_time_budget_sec: float | None = None
    stateful_stage_time_budget_multiplier: float = 3.0
    # Timeout for external dataset preparation/download helpers; recorded in
    dataset_download_timeout_sec: int = 14400
    # Spread concurrent requests across different LLMs (round-robin-by-model order).
    interleave_by_model: bool = True
    # Optional HARD cap on concurrent in-flight units per model. 0 disables the cap
    max_concurrent_per_model: int = 0
    # --- auto-finalize hidden-test on budget exhaustion ---
    auto_finalize_on_exhaustion: bool = True

    paraphrase_prompts: bool = False
    # Explicit experimental axis. It is ignored when paraphrase_prompts=False.
    prompt_variants: list[int] = field(
        default_factory=lambda: list(PROMPT_VARIANT_IDS)
    )
    performance_normalization: bool = False

    # --- routing ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    request_timeout_sec: int = 180
    max_retries: int = 5

    # --- per-model OpenRouter extras ---
    provider_preferences: dict = field(default_factory=dict)
    reasoning_preferences: dict = field(default_factory=dict)

    # --- debug tracing ---
    debug_trace_enabled: bool = True
    log_executable_code: bool = True
    log_raw_llm_responses: bool = False

    # --- io ---
    task_dirs: list[str] = field(default_factory=lambda: ["automl_eval/tasks"])
    output_dir: str = "outputs/grace_paper_grid"
    run_name: str = "grace_run"

    def __post_init__(self) -> None:
        # Fold legacy scalar `temperature` into the schedule if the schedule was
        if self.temperature is not None:
            if self.temperature_schedule == [0.7]:
                self.temperature_schedule = [float(self.temperature)]
        # Fold a legacy explicit `seeds` list into split_seeds.
        if self.seeds is not None and self.split_seeds == [42]:
            self.split_seeds = list(self.seeds)

    def validate(self) -> None:
        if not self.models:
            raise ValueError("ExperimentConfig.models is empty.")
        if not self.task_ids:
            raise ValueError("ExperimentConfig.task_ids is empty.")
        unknown = [r for r in self.regimes if r not in VALID_REGIMES]
        if unknown:
            raise ValueError(f"Unknown regimes {unknown}. Valid: {list(VALID_REGIMES)}")
        if not self.split_seeds:
            raise ValueError("split_seeds is empty.")
        if not self.temperature_schedule:
            raise ValueError("temperature_schedule is empty.")
        if self.repeats_per_condition < 1:
            raise ValueError("repeats_per_condition must be >= 1.")
        if not isinstance(self.prompt_variants, list) or not self.prompt_variants:
            raise ValueError("prompt_variants must be a non-empty list.")
        if any(
            isinstance(variant_id, bool) or not isinstance(variant_id, int)
            for variant_id in self.prompt_variants
        ):
            raise ValueError("Every prompt_variants entry must be an integer.")
        if len(set(self.prompt_variants)) != len(self.prompt_variants):
            raise ValueError("prompt_variants must not contain duplicates.")
        unknown_prompt_variants = [
            variant_id
            for variant_id in self.prompt_variants
            if variant_id not in PROMPT_VARIANT_IDS
        ]
        if unknown_prompt_variants:
            raise ValueError(
                f"Unknown prompt variants {unknown_prompt_variants}. "
                f"Valid ids: {list(PROMPT_VARIANT_IDS)}"
            )
        if not isinstance(self.dataset_subsample_factor, int) or isinstance(
            self.dataset_subsample_factor, bool
        ):
            raise ValueError("dataset_subsample_factor must be an integer >= 1.")
        if self.dataset_subsample_factor < 1:
            raise ValueError(
                "dataset_subsample_factor must be >= 1 (1 means no subsampling)."
            )
        for name in (
            "stateless_sandbox_timeout_sec",
            "stateful_sandbox_timeout_sec",
            "episode_timeout_sec",
            "request_timeout_sec",
            "dataset_download_timeout_sec",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive number of seconds.")
        for name in ("stateless_task_time_budget_sec", "stateful_task_time_budget_sec"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or float(value) <= 0):
                raise ValueError(
                    f"{name} must be null or a positive number of seconds."
                )
        if (
            isinstance(self.stateful_stage_time_budget_multiplier, bool)
            or float(self.stateful_stage_time_budget_multiplier) <= 0
        ):
            raise ValueError("stateful_stage_time_budget_multiplier must be positive.")
        for name in (
            "debug_trace_enabled",
            "log_executable_code",
            "log_raw_llm_responses",
            "paraphrase_prompts",
            "performance_normalization",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    def active_prompt_variants(self) -> list[int]:
        """Return the prompt variants that expand into episode units."""
        return list(self.prompt_variants) if self.paraphrase_prompts else [0]

    def episodes_per_cell(self) -> int:
        """Episodes per cell = splits * temperatures * repeats * prompt variants."""
        return (
            len(self.split_seeds)
            * len(self.temperature_schedule)
            * self.repeats_per_condition
            * len(self.active_prompt_variants())
        )

    def replications_per_temperature(self) -> int:
        """Paired observations available per (model, task, regime, temperature)."""
        return (
            len(self.split_seeds)
            * self.repeats_per_condition
            * len(self.active_prompt_variants())
        )

    def total_episodes(self) -> int:
        return (
            len(self.models)
            * len(self.task_ids)
            * len(self.regimes)
            * self.episodes_per_cell()
        )

    def novelty_warnings(self) -> list[str]:
        warnings: list[str] = []
        missing = [r for r in NOVELTY_CRITICAL_REGIMES if r not in self.regimes]
        if missing:
            warnings.append(
                "Novelty-critical regimes missing from config: "
                f"{missing}. The paper's central structure-vs-free-form claim needs these. "
                "Add them unless you are intentionally running a sub-experiment."
            )
        # Significance is computed per temperature, so the relevant n is
        per_temp = self.replications_per_temperature()
        if per_temp < 5:
            warnings.append(
                f"replications per temperature = {per_temp} "
                f"(len(split_seeds)={len(self.split_seeds)} * repeats={self.repeats_per_condition}). "
                "Paired Wilcoxon / bootstrap CIs need >= 5 paired observations per temperature; "
                "increase split_seeds or repeats_per_condition."
            )
        if all(t < 0.5 for t in self.temperature_schedule) and per_temp > 1:
            warnings.append(
                f"all temperatures < 0.5 ({self.temperature_schedule}) with multiple replications: "
                "LLM run-to-run variance will be near-degenerate, so variance-based statistics "
                "will be uninformative. Consider adding a higher temperature."
            )
        if self.paraphrase_prompts and len(self.active_prompt_variants()) < 2:
            warnings.append(
                "paraphrase_prompts is enabled but fewer than two prompt variants "
                "are configured; prompt-robustness variation cannot be estimated."
            )
        return warnings

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
