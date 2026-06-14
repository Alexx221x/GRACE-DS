"""Multiprocessing parallel experiment runner for GRACE."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_eval.experiment.config import ExperimentConfig
from automl_eval.evaluation.debug_trace import trace_event


@dataclass
class EpisodeUnit:
    model: str
    task_id: str
    regime: str
    repeat_index: int
    split_seed: int
    temperature: float
    llm_seed: int
    max_actions: int
    max_tokens: int
    total_token_budget: int
    paraphrase_prompts: bool
    performance_normalization: bool
    auto_finalize_on_exhaustion: bool
    llm_url: str
    api_key_env: str
    request_timeout_sec: int
    max_retries: int
    stateless_sandbox_timeout_sec: int = 300
    stateful_sandbox_timeout_sec: int = 900
    stateless_task_time_budget_sec: float | None = None
    stateful_task_time_budget_sec: float | None = None
    stateful_stage_time_budget_multiplier: float = 3.0
    task_dirs: list[str] = field(default_factory=list)
    # whole-dataset downsampling factor (1 == full dataset).
    dataset_subsample_factor: int = 1
    # OpenRouter extras for this specific model (forwarded verbatim).
    provider_preferences: dict = field(default_factory=dict)
    reasoning_preferences: dict = field(default_factory=dict)
    # Per-run debug trace controls. trace_base_dir is normally
    debug_trace_enabled: bool = True
    log_executable_code: bool = True
    log_raw_llm_responses: bool = False
    trace_base_dir: str | None = None

    def unit_id(self) -> str:
        safe_model = self.model.replace("/", "__")
        return (
            f"{safe_model}::{self.task_id}::{self.regime}"
            f"::split{self.split_seed}::t{self.temperature}::rep{self.repeat_index}"
        )


@dataclass
class EpisodeOutcome:
    unit_id: str
    model: str
    task_id: str
    regime: str
    repeat_index: int
    split_seed: int
    temperature: float
    llm_seed: int
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_sec: float = 0.0


def _safe_unit_filename(unit_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", unit_id)[:220] + ".json"


def _outcome_record(outcome: EpisodeOutcome) -> dict[str, Any]:
    return {
        "unit_id": outcome.unit_id,
        "model": outcome.model,
        "task_id": outcome.task_id,
        "regime": outcome.regime,
        "repeat_index": outcome.repeat_index,
        "split_seed": outcome.split_seed,
        "temperature": outcome.temperature,
        "llm_seed": outcome.llm_seed,
        "ok": outcome.ok,
        "error": outcome.error,
        "elapsed_sec": round(outcome.elapsed_sec, 2),
        "payload": outcome.payload,
    }


def _write_checkpoint(checkpoint_dir: Path, rec: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / _safe_unit_filename(str(rec["unit_id"]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore a torn line from an interrupted write; checkpoints are
                continue
    return records


def _load_resume_records(
    raw_path: Path, checkpoint_dir: Path
) -> dict[str, dict[str, Any]]:
    """Return last saved record per unit_id from raw JSONL plus checkpoints."""
    by_unit: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl_records(raw_path):
        uid = rec.get("unit_id")
        if uid:
            by_unit[str(uid)] = rec
    if checkpoint_dir.exists():
        for p in sorted(checkpoint_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            uid = rec.get("unit_id")
            if uid:
                by_unit[str(uid)] = rec
    return by_unit


_TRANSIENT_ERROR_RE = re.compile(
    r"(http\s*429|rate limit|too many requests|quota|tpm limit|rpm limit|"
    r"timeout|timed out|connection reset|connection refused|connection aborted|"
    r"remotedisconnected|service unavailable|bad gateway|gateway timeout|"
    r"http\s*5\d\d|urlerror|temporarily unavailable)",
    re.IGNORECASE,
)


def _is_transient_failure_record(rec: dict[str, Any]) -> bool:
    """True only for infrastructure/provider failures safe to rerun."""
    if bool(rec.get("ok", False)):
        return False
    text = " ".join(
        str(part or "")
        for part in (
            rec.get("error"),
            rec.get("payload", {}).get("error")
            if isinstance(rec.get("payload"), dict)
            else None,
        )
    )
    return bool(_TRANSIENT_ERROR_RE.search(text))


def _filter_resume_records_for_rerun(
    records: dict[str, dict[str, Any]],
    *,
    rerun_transient: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split resume records into skipped records and transient records to rerun."""
    if not rerun_transient:
        return dict(records), {}
    keep: dict[str, dict[str, Any]] = {}
    rerun: dict[str, dict[str, Any]] = {}
    for uid, rec in records.items():
        if _is_transient_failure_record(rec):
            rerun[uid] = rec
        else:
            keep[uid] = rec
    return keep, rerun


def _rewrite_raw_jsonl(raw_path: Path, records: list[dict[str, Any]]) -> None:
    tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(raw_path)


def _worker(unit: EpisodeUnit) -> EpisodeOutcome:
    """Run one episode in a fresh process. Configures the regime module via env."""
    t0 = time.time()
    # 1. Configure the extracted regime module BEFORE importing it.
    os.environ["LLM_URL"] = unit.llm_url
    os.environ["LLM_TOKEN"] = os.environ.get(unit.api_key_env, "")
    os.environ["LLM_MODEL"] = unit.model
    os.environ["LLM_TEMPERATURE"] = str(unit.temperature)
    os.environ["ACTIVE_TASK_ID"] = unit.task_id
    os.environ["ACTIVE_DATASET_SEED"] = str(unit.split_seed)
    os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = str(unit.request_timeout_sec)
    os.environ["LLM_MAX_RETRIES"] = str(unit.max_retries)
    os.environ["GRACE_STATELESS_SANDBOX_TIMEOUT_SECONDS"] = str(
        unit.stateless_sandbox_timeout_sec
    )
    os.environ["GRACE_STATEFUL_SANDBOX_TIMEOUT_SECONDS"] = str(
        unit.stateful_sandbox_timeout_sec
    )
    os.environ["GRACE_STATEFUL_STAGE_TIME_BUDGET_MULTIPLIER"] = str(
        unit.stateful_stage_time_budget_multiplier
    )
    if unit.stateless_task_time_budget_sec is None:
        os.environ.pop("GRACE_STATELESS_TASK_TIME_BUDGET_SECONDS", None)
    else:
        os.environ["GRACE_STATELESS_TASK_TIME_BUDGET_SECONDS"] = str(
            unit.stateless_task_time_budget_sec
        )
    if unit.stateful_task_time_budget_sec is None:
        os.environ.pop("GRACE_STATEFUL_TASK_TIME_BUDGET_SECONDS", None)
    else:
        os.environ["GRACE_STATEFUL_TASK_TIME_BUDGET_SECONDS"] = str(
            unit.stateful_task_time_budget_sec
        )
    # Standardised token budget for this run (read by the extracted module).
    os.environ["LLM_MAX_TOKENS_PER_CALL"] = str(unit.max_tokens)
    os.environ["LLM_TOTAL_TOKEN_BUDGET"] = str(unit.total_token_budget)
    # make ExperimentConfig.max_actions actually govern the working
    os.environ["GRACE_WORKING_LLM_CALL_BUDGET"] = str(unit.max_actions)
    # Reproducible LLM sampling seed (forwarded to OpenRouter by the LLM patch).
    os.environ["OPENROUTER_LLM_SEED"] = str(unit.llm_seed)
    # Never auto-run notebook experiment grids on import.
    os.environ.setdefault("RUN_LLM_EXPERIMENTS", "0")
    os.environ.setdefault("RUN_MAIN_MODE_COMPARISON", "0")
    os.environ.setdefault("RUN_FULL_PAPER_GRID", "0")
    os.environ.setdefault("RUN_STATE_ABLATION_STUDY", "0")
    os.environ.setdefault("RUN_REWARD_HACKING_STUDY", "0")
    os.environ.setdefault("RUN_CONNECTION_PROBE", "0")
    # Scenario-B toggles are surfaced for downstream prompt selection / normalization.
    os.environ["GRACE_PARAPHRASE_PROMPTS"] = "1" if unit.paraphrase_prompts else "0"
    os.environ["GRACE_PERF_NORM"] = "1" if unit.performance_normalization else "0"
    # invert auto_finalize -> the SUPPRESS env var the
    os.environ["GRACE_SUPPRESS_FORCED_HIDDEN_TEST"] = (
        "0" if unit.auto_finalize_on_exhaustion else "1"
    )
    # whole-dataset downsampling factor read by the regime module at
    os.environ["GRACE_DATASET_SUBSAMPLE_FACTOR"] = str(
        int(unit.dataset_subsample_factor)
    )
    # OpenRouter per-request extras (provider quantization filter + reasoning mode).
    if unit.provider_preferences:
        os.environ["OPENROUTER_PROVIDER_JSON"] = json.dumps(unit.provider_preferences)
    else:
        os.environ.pop("OPENROUTER_PROVIDER_JSON", None)
    if unit.reasoning_preferences:
        os.environ["OPENROUTER_REASONING_JSON"] = json.dumps(unit.reasoning_preferences)
    else:
        os.environ.pop("OPENROUTER_REASONING_JSON", None)
    os.environ.setdefault("OPENROUTER_X_TITLE", "GRACE-experiments")
    # Lightweight per-episode debug trace. This is deliberately outside the
    if unit.debug_trace_enabled and unit.trace_base_dir:
        os.environ["GRACE_TRACE_ENABLED"] = "1"
        os.environ["GRACE_TRACE_DIR"] = unit.trace_base_dir
        os.environ["GRACE_UNIT_ID"] = unit.unit_id()
        os.environ["GRACE_LOG_EXECUTABLE_CODE"] = (
            "1" if unit.log_executable_code else "0"
        )
        os.environ["GRACE_LOG_RAW_LLM_RESPONSES"] = (
            "1" if unit.log_raw_llm_responses else "0"
        )
    else:
        os.environ["GRACE_TRACE_ENABLED"] = "0"
        os.environ.pop("GRACE_TRACE_DIR", None)
        os.environ.pop("GRACE_UNIT_ID", None)
        os.environ.pop("GRACE_LOG_EXECUTABLE_CODE", None)
        os.environ.pop("GRACE_LOG_RAW_LLM_RESPONSES", None)

    trace_event(
        "worker_start",
        unit_id=unit.unit_id(),
        base_dir=unit.trace_base_dir,
        model=unit.model,
        task_id=unit.task_id,
        regime=unit.regime,
        split_seed=unit.split_seed,
        repeat_index=unit.repeat_index,
        temperature=unit.temperature,
        llm_seed=unit.llm_seed,
        request_timeout_sec=unit.request_timeout_sec,
        max_retries=unit.max_retries,
    )

    # suppress the cosmetic sklearn/joblib "Loky-backed parallel loops
    import warnings as _w

    _w.filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called in a multiprocessing",
        category=UserWarning,
    )
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    try:
        # Suppress the extracted notebook's cosmetic import-time banner (it
        import contextlib
        import io
        import sys

        with contextlib.redirect_stdout(io.StringIO()):
            # Import inside the worker so env config is read fresh per process.
            from automl_eval.experiment import _regimes_extracted as regimes

            # contamination guard. _regimes_extracted reads ACTIVE_TASK_ID
            _module_task_id = getattr(regimes, "TASK_ID", None)
            if _module_task_id is not None and _module_task_id != unit.task_id:
                import importlib

                importlib.reload(regimes)
                _module_task_id = getattr(regimes, "TASK_ID", None)
            if _module_task_id is not None and _module_task_id != unit.task_id:
                raise RuntimeError(
                    "Task-id contamination detected in worker: regime module "
                    f"TASK_ID={_module_task_id!r} != unit.task_id={unit.task_id!r}. "
                    "This indicates the parallel runner is NOT using spawn + "
                    "maxtasksperchild=1 (a forked or reused worker kept a stale "
                    "import). Refusing to score the wrong dataset."
                )
            # Patch the extracted LLM client to honour OpenRouter extras (provider,
            from automl_eval.experiment._llm_patches import patch_openai_compatible_llm

            patch_openai_compatible_llm(regimes.OpenAICompatibleLLM)
            # Optionally register extra task directories beyond the default.
            for d in unit.task_dirs:
                if d and Path(d).exists():
                    try:
                        regimes.registry.load_directory(d)
                    except Exception:
                        pass
        # Concise per-episode progress line (replaces the suppressed banner).
        sys.stdout.write(
            f"[worker] {unit.model} | {unit.task_id} | {unit.regime} | "
            f"split={unit.split_seed} t={unit.temperature} rep={unit.repeat_index}\n"
        )
        sys.stdout.flush()
        trace_event(
            "before_run_one_episode",
            unit_id=unit.unit_id(),
            base_dir=unit.trace_base_dir,
            regime=unit.regime,
        )
        payload = regimes.run_one_episode(unit.regime)
        trace_event(
            "after_run_one_episode",
            unit_id=unit.unit_id(),
            base_dir=unit.trace_base_dir,
        )
        if isinstance(payload, dict):
            payload.setdefault("debug_trace_dir", unit.trace_base_dir)
        return EpisodeOutcome(
            unit_id=unit.unit_id(),
            model=unit.model,
            task_id=unit.task_id,
            regime=unit.regime,
            repeat_index=unit.repeat_index,
            split_seed=unit.split_seed,
            temperature=unit.temperature,
            llm_seed=unit.llm_seed,
            ok=True,
            payload=payload,
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - worker must never crash the pool
        trace_event(
            "worker_exception",
            unit_id=unit.unit_id(),
            base_dir=unit.trace_base_dir,
            error=f"{type(exc).__name__}: {exc}",
        )
        return EpisodeOutcome(
            unit_id=unit.unit_id(),
            model=unit.model,
            task_id=unit.task_id,
            regime=unit.regime,
            repeat_index=unit.repeat_index,
            split_seed=unit.split_seed,
            temperature=unit.temperature,
            llm_seed=unit.llm_seed,
            ok=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:1500]}",
            elapsed_sec=time.time() - t0,
        )


def build_units(config: ExperimentConfig) -> list[EpisodeUnit]:
    """Expand the config into the full (model x task x regime x split x temp x repeat) grid."""
    units: list[EpisodeUnit] = []
    for model in config.models:
        # Per-model preferences are looked up by the OpenRouter slug; missing keys
        prov_prefs = dict(config.provider_preferences.get(model) or {})
        reas_prefs = dict(config.reasoning_preferences.get(model) or {})
        for task_id in config.task_ids:
            for regime in config.regimes:
                for split_seed in config.split_seeds:
                    for temperature in config.temperature_schedule:
                        for rep in range(config.repeats_per_condition):
                            units.append(
                                EpisodeUnit(
                                    model=model,
                                    task_id=task_id,
                                    regime=regime,
                                    repeat_index=rep,
                                    split_seed=split_seed,
                                    temperature=float(temperature),
                                    llm_seed=config.llm_seed_base + rep,
                                    max_actions=config.max_actions,
                                    max_tokens=config.max_tokens,
                                    total_token_budget=config.total_token_budget,
                                    paraphrase_prompts=config.paraphrase_prompts,
                                    performance_normalization=config.performance_normalization,
                                    auto_finalize_on_exhaustion=config.auto_finalize_on_exhaustion,
                                    llm_url=config.openrouter_base_url,
                                    api_key_env=config.api_key_env,
                                    request_timeout_sec=config.request_timeout_sec,
                                    max_retries=config.max_retries,
                                    stateless_sandbox_timeout_sec=config.stateless_sandbox_timeout_sec,
                                    stateful_sandbox_timeout_sec=config.stateful_sandbox_timeout_sec,
                                    stateless_task_time_budget_sec=config.stateless_task_time_budget_sec,
                                    stateful_task_time_budget_sec=config.stateful_task_time_budget_sec,
                                    stateful_stage_time_budget_multiplier=config.stateful_stage_time_budget_multiplier,
                                    task_dirs=config.task_dirs,
                                    dataset_subsample_factor=config.dataset_subsample_factor,
                                    provider_preferences=prov_prefs,
                                    reasoning_preferences=reas_prefs,
                                    debug_trace_enabled=config.debug_trace_enabled,
                                    log_executable_code=config.log_executable_code,
                                    log_raw_llm_responses=config.log_raw_llm_responses,
                                    trace_base_dir=str(
                                        Path(config.output_dir)
                                        / config.run_name
                                        / "debug_trace"
                                    ),
                                )
                            )
    return units


def _interleave_by_model(units: list[EpisodeUnit]) -> list[EpisodeUnit]:
    """Round-robin reorder by model: any window of K consecutive units in the"""
    from collections import defaultdict

    buckets: dict[str, list[EpisodeUnit]] = defaultdict(list)
    for u in units:
        buckets[u.model].append(u)
    # Stable model order = order of first appearance in `units`.
    model_order: list[str] = []
    seen: set[str] = set()
    for u in units:
        if u.model not in seen:
            model_order.append(u.model)
            seen.add(u.model)
    queues = [buckets[m] for m in model_order]
    out: list[EpisodeUnit] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _dispatch_with_timeout(pool, units, episode_timeout_sec: int, num_workers: int):
    """Dispatch via ``pool.apply_async`` with a real per-episode wallclock"""
    import os
    import signal

    pending: dict = {}
    units_iter = iter(units)

    # Prime the pool with ``num_workers`` tasks.
    for _ in range(num_workers):
        try:
            unit = next(units_iter)
        except StopIteration:
            break
        pending[pool.apply_async(_worker, (unit,))] = (unit, time.time())

    while pending:
        # Find first finished OR timed-out task.
        finished_async = None
        for async_res, (unit, t0) in pending.items():
            if async_res.ready():
                finished_async = async_res
                break
            if time.time() - t0 > episode_timeout_sec:
                finished_async = async_res
                break
        if finished_async is None:
            time.sleep(0.5)
            continue

        unit, t0 = pending.pop(finished_async)
        elapsed = time.time() - t0
        if finished_async.ready():
            try:
                yield finished_async.get(timeout=1)
            except Exception as e:
                yield EpisodeOutcome(
                    unit_id=unit.unit_id(),
                    model=unit.model,
                    task_id=unit.task_id,
                    regime=unit.regime,
                    repeat_index=unit.repeat_index,
                    split_seed=unit.split_seed,
                    temperature=unit.temperature,
                    llm_seed=unit.llm_seed,
                    ok=False,
                    error=f"worker_exception: {type(e).__name__}: {e}"[:240],
                    elapsed_sec=elapsed,
                    payload={},
                )
        else:
            # Timeout: kill ALL pool workers (apply_async has no per-task
            trace_event(
                "worker_timeout_kill",
                unit_id=unit.unit_id(),
                base_dir=unit.trace_base_dir,
                elapsed_seconds=round(elapsed, 3),
                episode_timeout_sec=episode_timeout_sec,
            )
            print(
                f"[timeout] killing stuck worker for {unit.unit_id()} "
                f"(>{episode_timeout_sec}s)"
            )
            for proc in list(pool._pool):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            # The remaining `pending` async results are now broken. Re-queue
            for _, (other_unit, _) in pending.items():
                # Insert in front of the iterator (best-effort: append at end
                units = list(units_iter) + [other_unit]
                units_iter = iter(units)
            pending.clear()
            yield EpisodeOutcome(
                unit_id=unit.unit_id(),
                model=unit.model,
                task_id=unit.task_id,
                regime=unit.regime,
                repeat_index=unit.repeat_index,
                split_seed=unit.split_seed,
                temperature=unit.temperature,
                llm_seed=unit.llm_seed,
                ok=False,
                error=f"timeout_exceeded_{episode_timeout_sec}s",
                elapsed_sec=elapsed,
                payload={},
            )

        # Refill the pool with the next unit.
        try:
            new_unit = next(units_iter)
            pending[pool.apply_async(_worker, (new_unit,))] = (new_unit, time.time())
        except StopIteration:
            pass


def _dispatch_with_per_model_cap(
    pool, units: list[EpisodeUnit], cap: int, episode_timeout_sec: int
):
    """Custom apply_async scheduler with per-model concurrency and episode timeouts."""
    from collections import defaultdict
    import time as _time

    pending = list(units)
    inflight: dict[str, int] = defaultdict(int)
    in_progress: list = []

    while pending or in_progress:
        # Dispatch as many as possible: respect the cap AND the pool's worker count
        dispatched_any = True
        while dispatched_any and len(in_progress) < pool._processes:  # type: ignore[attr-defined]
            dispatched_any = False
            for i, u in enumerate(pending):
                if inflight[u.model] < cap:
                    fut = pool.apply_async(_worker, (u,))
                    in_progress.append((fut, u, _time.time()))
                    inflight[u.model] += 1
                    pending.pop(i)
                    dispatched_any = True
                    break

        if not in_progress:
            # Nothing to wait on AND nothing dispatchable -> deadlock guard.
            if pending:
                raise RuntimeError(
                    f"per-model cap deadlock: {len(pending)} units pending but cap={cap} "
                    f"blocks all dispatches. Inflight: {dict(inflight)}"
                )
            break

        # Drain completed without busy-spinning.
        completed_now: list = []
        timed_out_now: tuple | None = None
        now = _time.time()
        for fut, u, started in in_progress:
            if fut.ready():
                completed_now.append((fut, u, started))
            elif now - started > episode_timeout_sec and timed_out_now is None:
                timed_out_now = (fut, u, started)
        if timed_out_now is not None:
            import os as _os
            import signal as _signal

            fut, u, started = timed_out_now
            elapsed = now - started
            trace_event(
                "worker_timeout_kill",
                unit_id=u.unit_id(),
                base_dir=u.trace_base_dir,
                elapsed_seconds=round(elapsed, 3),
                episode_timeout_sec=episode_timeout_sec,
            )
            print(
                f"[timeout] killing stuck worker for {u.unit_id()} (>{episode_timeout_sec}s)"
            )
            for proc in list(pool._pool):  # type: ignore[attr-defined]
                try:
                    _os.kill(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            # Requeue all other in-flight units. The timed-out unit is recorded
            interrupted = [
                other for _, other, _ in in_progress if other.unit_id() != u.unit_id()
            ]
            pending = interrupted + pending
            in_progress.clear()
            inflight.clear()
            yield EpisodeOutcome(
                unit_id=u.unit_id(),
                model=u.model,
                task_id=u.task_id,
                regime=u.regime,
                repeat_index=u.repeat_index,
                split_seed=u.split_seed,
                temperature=u.temperature,
                llm_seed=u.llm_seed,
                ok=False,
                error=f"timeout_exceeded_{episode_timeout_sec}s",
                elapsed_sec=elapsed,
                payload={},
            )
            continue
        if not completed_now:
            _time.sleep(0.05)
            continue
        for fut, u, started in completed_now:
            in_progress.remove((fut, u, started))
            inflight[u.model] -= 1
            yield fut.get()


def run_experiment(
    config: ExperimentConfig,
    *,
    dry_run: bool = False,
    resume: bool = False,
    rerun_transient: bool = False,
) -> Path:
    """Run the full grid in parallel; write raw JSONL + aggregated tables."""
    if rerun_transient and not resume:
        raise ValueError(
            "--rerun-transient requires --resume so only existing transient failures are refined."
        )
    config.validate()
    for w in config.novelty_warnings():
        print(f"[novelty-warning] {w}")

    units = build_units(config)
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config + the unit manifest for provenance.
    (out_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "units.json").write_text(
        json.dumps([u.unit_id() for u in units], indent=2), encoding="utf-8"
    )
    print(
        f"[grid] {len(units)} episode units "
        f"({len(config.models)} models x {len(config.task_ids)} tasks x "
        f"{len(config.regimes)} regimes x {len(config.split_seeds)} split-seeds x "
        f"{len(config.temperature_schedule)} temperatures x {config.repeats_per_condition} repeats)"
    )
    print(
        f"[grid] {config.episodes_per_cell()} episodes per (model,task,regime); "
        f"{config.replications_per_temperature()} paired replications per temperature"
    )

    if dry_run:
        print("[dry-run] not executing; manifest written.")
        return out_dir

    if not os.environ.get(config.api_key_env):
        raise RuntimeError(
            f"Environment variable {config.api_key_env} is not set. "
            f"Export your OpenRouter key: export {config.api_key_env}=sk-or-..."
        )

    raw_path = out_dir / "episodes_raw.jsonl"
    checkpoint_dir = out_dir / "checkpoints"
    resume_records: dict[str, dict[str, Any]] = {}
    if resume:
        current_unit_ids = {u.unit_id() for u in units}
        resume_records = {
            uid: rec
            for uid, rec in _load_resume_records(raw_path, checkpoint_dir).items()
            if uid in current_unit_ids
        }
        if resume_records:
            resume_records, transient_records = _filter_resume_records_for_rerun(
                resume_records, rerun_transient=rerun_transient
            )
            # Rewrite the JSONL once into a de-duplicated, checkpoint-complete
            ordered = [
                resume_records[u.unit_id()]
                for u in units
                if u.unit_id() in resume_records
            ]
            _rewrite_raw_jsonl(raw_path, ordered)
            if transient_records:
                print(
                    f"[resume] found {len(resume_records)} completed episode records; "
                    f"rerunning {len(transient_records)} transient provider failures"
                )
            else:
                print(f"[resume] found {len(resume_records)} completed episode records")
    elif checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    # 'spawn' guarantees each worker re-imports the regime module with fresh env.
    ctx = mp.get_context("spawn")
    completed = len(resume_records)
    t0 = time.time()

    # ---- spread concurrent requests across DIFFERENT models ----
    dispatch_units = (
        _interleave_by_model(units) if config.interleave_by_model else list(units)
    )
    if resume_records:
        done_ids = set(resume_records)
        dispatch_units = [u for u in dispatch_units if u.unit_id() not in done_ids]
        print(
            f"[resume] skipping {len(done_ids)} completed units; {len(dispatch_units)} remain"
        )
    if config.interleave_by_model and len(config.models) > 1:
        head = dispatch_units[: max(config.num_workers, len(config.models))]
        head_models = sorted({u.model for u in head})
        print(
            f"[scheduler] interleave-by-model ON: first {len(head)} units span "
            f"{len(head_models)} distinct models"
        )
    cap = max(0, int(config.max_concurrent_per_model))
    if cap > 0:
        print(f"[scheduler] per-model concurrency cap = {cap}")

    def _emit(outcome) -> None:
        nonlocal completed
        completed += 1
        rec = _outcome_record(outcome)
        trace_base_dir = rec.get("payload", {}).get("debug_trace_dir") or str(
            out_dir / "debug_trace"
        )
        trace_event(
            "before_checkpoint_write",
            unit_id=outcome.unit_id,
            base_dir=trace_base_dir,
            ok=outcome.ok,
        )
        _write_checkpoint(checkpoint_dir, rec)
        trace_event(
            "after_checkpoint_write",
            unit_id=outcome.unit_id,
            base_dir=trace_base_dir,
            ok=outcome.ok,
        )
        raw_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw_handle.flush()
        status = "ok" if outcome.ok else "FAIL"
        print(
            f"[{completed}/{len(units)}] {status} {outcome.unit_id} "
            f"({outcome.elapsed_sec:.0f}s)"
        )

    if not dispatch_units:
        print("[resume] all units already completed; aggregating existing records.")
    else:
        raw_mode = "a" if resume else "w"
        with raw_path.open(raw_mode, encoding="utf-8") as raw_handle:
            with ctx.Pool(processes=config.num_workers, maxtasksperchild=1) as pool:
                if cap > 0:
                    for outcome in _dispatch_with_per_model_cap(
                        pool, dispatch_units, cap, config.episode_timeout_sec
                    ):
                        _emit(outcome)
                else:
                    for outcome in _dispatch_with_timeout(
                        pool,
                        dispatch_units,
                        config.episode_timeout_sec,
                        config.num_workers,
                    ):
                        _emit(outcome)
    print(f"[done] {completed} episodes in {time.time() - t0:.0f}s -> {raw_path}")

    # Aggregate into the paper tables.
    from automl_eval.experiment.aggregate import aggregate_run

    aggregate_run(raw_path, out_dir)
    return out_dir
