"""Parallel, OpenRouter-driven GRACE experiment runner.
This package turns a config (list of LLMs x list of datasets x list of harness
regimes x repeats) into all the numbers needed for the GRACE paper tables, using
a multiprocessing pool so that many (LLM call + sandboxed code execution)
episodes run concurrently.

Modules:
* config         -- ExperimentConfig dataclass + YAML loading.
* parallel_runner-- the multiprocessing engine (one episode per task unit).
* aggregate      -- collapse raw episode records into the paper tables.
* run_grace_experiments -- CLI entry point.
"""
