# GRACE-DS

GRACE-DS is a guarded evaluation framework for pre-deployment testing of LLM-based AutoML agents on organization-specific tabular ML tasks. It simulates realistic workflow stages and uses hidden executable validators to measure not just final model quality, but also leakage avoidance, reproducibility, protocol compliance, self-correction, and reward alignment.

## Get the Code

Clone the repository and move into it:

```bash
git clone https://github.com/Alexx221x/GRACE-DS
cd GRACE-DS
```

## Setup

Create the environment with `uv`, then activate it:

```bash
uv sync
source .venv/bin/activate
```

## Data Installation

Before installing data, make sure Kaggle access is configured.

GRACE relies on several datasets that come from Kaggle competitions, so you need:

- Kaggle credentials available either in `~/.kaggle/kaggle.json` or via `KAGGLE_API_TOKEN`
- access to each required competition by joining it on Kaggle first

Required Kaggle competitions:

- `https://www.kaggle.com/competitions/playground-series-s5e10`
- `https://www.kaggle.com/competitions/bank-customer-churn-ict-u-ai`
- `https://www.kaggle.com/competitions/foot-traffic-wuerzburg-retail-forecasting-2-0`
- `https://www.kaggle.com/competitions/sberbank-russian-housing-market`
- `https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability`
- `https://www.kaggle.com/c/acquire-valued-shoppers-challenge`
- `https://www.kaggle.com/competitions/homesite-quote-conversion`

Once Kaggle access is ready, prepare the datasets with:

```bash
bash scripts/install_data.sh
```

The installer will:

- clone and prepare the external TML-bench and TabReD sources under `external/`
- materialize GRACE task files into `automl_eval/tasks/`
- build the synthetic tasks used by the benchmark

## Experiment Requirements

To run real LLM experiments, you need one of the following:

- `OPENROUTER_API_KEY` for the default OpenRouter-based setup
- your own compatible API key and chat-completions endpoint, configured through the experiment YAML

## Running Experiments

Use `scripts/run_experiments.sh` as the main experiment entry point.

A typical resumed run looks like this:

```bash
bash scripts/run_experiments.sh --config configs/final.yaml --resume --rerun-transient
```

Useful variants:

```bash
bash scripts/run_experiments.sh --config configs/final.yaml --dry-run
bash scripts/run_experiments.sh --config configs/final.yaml --regimes single_shot,flexible_iterative
bash scripts/run_experiments.sh --config configs/final.yaml --tasks tml_playground_series_s6e1,tabred_sberbank_housing
bash scripts/run_experiments.sh --config configs/final.yaml --models openai/gpt-5.4
```

## Output Example

See `output_example/` for an example experiment output. You can inspect it to understand the generated run artifacts and result structure.

## Regimes

The current paper config in [configs/final.yaml](configs/final.yaml) includes these regimes:

- `single_shot`
- `n_restarts_from_scratch`
- `n_restarts_call_matched_upper_bound`
- `unstructured_agent`
- `fixed_stage_iterative`
- `flexible_iterative`
- `flexible_compact_feedback`
- `fixed_without_plan`
- `fixed_without_eda`
- `fixed_without_feature_engineering`
- `flexible_without_eda`
- `flexible_without_feature_engineering`
- `reward_maximizer_hidden_hints`
- `reward_maximizer_disclosed_criteria`
- `red_team_vs_validators`

To run only a subset of regimes:

```bash
bash scripts/run_experiments.sh --config configs/final.yaml --regimes fixed_stage_iterative,flexible_iterative
```

## Editing the YAML Configuration

The main experiment configuration lives in [configs/final.yaml](configs/final.yaml). This file controls:

- `models`: model list
- `task_ids`: benchmark tasks included in the run
- `regimes`: evaluation regimes
- `split_seeds`, `temperature_schedule`, `repeats_per_condition`: replication grid
- `max_actions`, `max_tokens`, `total_token_budget`, `n_restarts`: episode budgets
- `num_workers`, `request_timeout_sec`, `max_retries`: execution settings
- `output_dir`, `run_name`: output location
- sandbox and episode timeout values
- `dataset_subsample_factor`: dataset downsampling for the current run

In practice, this YAML is the main place to adjust the experiment setup before launching a run.
