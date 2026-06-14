import json
from pathlib import Path


def _source() -> str:
    project_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (project_root / "titanic_all_approaches_comparison.ipynb").read_text(
            encoding="utf-8"
        )
    )
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )


def test_full_paper_grid_and_variance_outputs_are_present():
    source = _source()
    assert "RUN_FULL_PAPER_GRID" in source
    assert "PAPER_GRID_MODES" in source
    assert "SPLIT_SEEDS" in source
    assert "TEMPERATURE_SCHEDULE" in source
    assert "REPEATS_PER_CONDITION" in source
    assert "metric_variance_summary" in source
    assert "metric_variance_by_mode.csv" in source
    assert "metric_variance_by_mode_and_temperature.csv" in source
    assert "metric_variance_by_mode_and_split_seed.csv" in source
    assert "paired_hidden_test_deltas_vs_unstructured.csv" in source


def test_token_budget_reward_hacking_state_ablation_and_human_eval_are_present():
    source = _source()
    assert "BudgetedLLM" in source
    assert "MAX_TOTAL_TOKENS_PER_EPISODE" in source
    assert "MAX_VALIDATION_REQUESTS_PER_EPISODE" in source
    assert "RUN_STATE_ABLATION_STUDY" in source
    assert "RUN_REWARD_HACKING_STUDY" in source
    assert "reward_maximizer_disclosed_criteria" in source
    assert "flexible_without_eda" in source
    assert "flexible_without_feature_engineering" in source
    assert "fixed_without_plan" in source
    assert "fixed_without_eda" in source
    assert "fixed_without_feature_engineering" in source
    assert "reward_alignment_episode_summary.csv" in source
    assert "export_human_eval_packets" in source


def test_versions_and_print_feedback_are_in_agent_prompts():
    source = _source()
    assert "approved_library_versions_text" in source
    assert "PRINT_FEEDBACK_INSTRUCTION" in source
    assert "sparse=" in source
