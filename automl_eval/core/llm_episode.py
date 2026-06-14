# -*- coding: utf-8 -*-
"""Shared LLM benchmark loop for the stage-aware AutoML environment."""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from automl_eval.core.environment import AutoMLEnvironment
from automl_eval.llm.prompts import build_system_prompt

LLM_TEMPERATURE = 0.38
RETRY_TEMPERATURE = 0.62
MAX_TOKENS = 8192
MAX_CONSECUTIVE_FE_FAILS = 10
MAX_LLM_TURNS = 45
_VALIDATOR_LINE_RE = re.compile(
    r"^\s*\[(PASS|FAIL|RESOLVED|UNRESOLVED|IMPROVED|REGRESSED|BLOCKED|INACTIVE)\]\s+([^:]+):",
    re.IGNORECASE,
)

RETRY_HINT_ERROR = (
    "Your previous code failed. Use a different minimal approach; operate on in-memory `train_df` only "
    "for fitting, treat `valid_df` as feature-only, and package preprocessing inside `predict_fn(raw_dataframe)` "
    "or a fitted raw-input sklearn Pipeline named `pipeline`."
)
RETRY_HINT_STUCK = (
    "You are repeating one stage. Move productively among PLAN, EDA, FEATURE_ENGINEERING, MODEL, "
    "VALIDATE, and FINAL_SUBMIT; only VALIDATE reveals evaluator-owned validation quality."
)
RESTART_HINT = (
    "You have failed {n} consecutive executable steps. Stop mutating shared frames. Build one simple "
    "raw-input sklearn Pipeline from `train_df_original.copy()` and expose `predict_fn(raw_dataframe)` "
    "or `pipeline`; do not attempt validation scoring in code."
)
POST_MODEL_PREDICTION_REMINDER = (
    "\n\n---\nBefore ACTION: VALIDATE or FINAL_SUBMIT, expose a callable predict_fn(raw_dataframe) or "
    "a fitted raw-input sklearn Pipeline named pipeline. VALIDATE must contain no code and performs "
    "evaluator-owned scoring.\n---"
)


def extract_execution_error(state_text: str, max_chars: int = 12000) -> str:
    """Extract full sandbox error (may be multi-line traceback)."""
    if "Execution: FAILED" not in state_text:
        return ""
    marker = "Execution: FAILED — "
    idx = state_text.find(marker)
    if idx < 0:
        return ""
    rest = state_text[idx + len(marker) :]
    for stop in ("\n\nStep:", "\n\n=== Session State", "\n\n--- Validator"):
        if stop in rest:
            rest = rest.split(stop, 1)[0]
            break
    return rest.strip()[:max_chars]


def _sandbox_has_submission_bundle(session: Any) -> bool:
    """True when the sandbox exposes a formal replayable terminal candidate."""
    if session is None:
        return False
    ns = getattr(session, "sandbox_namespace", None) or {}
    if callable(ns.get("predict_fn")):
        return True
    return any(
        hasattr(ns.get(name), "predict") for name in ("pipeline", "submission_pipeline")
    )


def extract_validator_status(state_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in state_text.splitlines():
        m = _VALIDATOR_LINE_RE.match(line)
        if m:
            out[m.group(2).strip()] = m.group(1)
    return out


def _extract_action_type(text: str) -> str:
    line = text.split("\n")[0].strip() if text else ""
    if line.upper().startswith("ACTION:"):
        return line.split(":", 1)[1].strip().split()[0]
    return "UNKNOWN"


@dataclass
class StepLog:
    step: int
    action_type: str
    summary: str
    reward: float
    done: bool
    exec_ok: bool
    exec_error: str
    exec_error_full: str = ""
    fail_validators: list[str] = field(default_factory=list)
    full_response: str = ""
    feedback: str = ""


@dataclass
class EpisodeResult:
    model: str
    task_id: str
    final_reward: float = 0.0
    # Compatibility alias: terminal hidden-test quality only.
    final_metric: float | None = None
    best_validation_metric: float | None = None
    final_hidden_test_metric: float | None = None
    num_steps: int = 0
    elapsed_sec: float = 0.0
    steps: list[StepLog] = field(default_factory=list)
    error: str | None = None
    aborted_early: bool = False


def run_episode(
    env: AutoMLEnvironment,
    llm,
    retry_llm,
    task_id: str,
    model_name: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = MAX_TOKENS,
    max_consecutive_fe_fails: int = MAX_CONSECUTIVE_FE_FAILS,
    max_llm_turns: int = MAX_LLM_TURNS,
    max_action: int | None = None,
    max_actions: int | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """Run one episode. Stops on env done, step budget, or FE fail streak."""
    result = EpisodeResult(model=model_name, task_id=task_id)
    t0 = time.time()
    _summary_re = re.compile(
        r"(?:^|\n)\s*(?:[-*•]\s*)?"
        r"(?:"
        r"\*\*(?:SUMMARY|Summary):\*\*\s*(.+)|"
        r"\*\*(?:SUMMARY|Summary)\*\*\s*:\s*(.+)|"
        r"(?:SUMMARY|Summary)\s*:\s*(.+)"
        r")",
        re.IGNORECASE,
    )

    def _summary(text: str) -> str:
        m = _summary_re.search(text)
        if m:
            g = next((x for x in m.groups() if x is not None), None)
            if g:
                return g.strip()[:400]
        # Fallback: first meaningful line (skip ACTION header and code fences)
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("ACTION"):
                continue
            if stripped.startswith("```"):
                continue
            return stripped[:200]
        return (text.split("\n")[0] if text else "")[:200]

    try:
        if (
            max_action is not None
            and max_actions is not None
            and max_action != max_actions
        ):
            raise ValueError(
                "Provide matching max_action/max_actions values or only one override."
            )
        action_budget = max_actions if max_actions is not None else max_action
        env.reset(task_id, max_actions=action_budget)
        obs = env.observe()
        active_budget = env._session.task.max_steps if env._session else max_llm_turns
        if f"Step: 0 / {active_budget}" not in obs:
            raise RuntimeError(
                f"Reset observation does not confirm active action budget {active_budget}."
            )
        if system_prompt is not None:
            declared = re.search(r"You have at most\s+(\d+)\s+actions", system_prompt)
            if declared and int(declared.group(1)) != int(active_budget):
                raise ValueError(
                    f"System prompt action budget {declared.group(1)} does not match active environment budget {active_budget}."
                )
        effective_prompt = system_prompt or build_system_prompt(active_budget)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_prompt},
            {"role": "user", "content": obs},
        ]

        prev_error_short: str | None = None
        same_error_count = 0
        prev_action_type: str | None = None
        same_action_count = 0
        consecutive_fe_fails = 0
        restart_given = False

        for step_idx in range(max_llm_turns):
            needs_retry = same_error_count >= 1 or same_action_count >= 2
            active_llm = retry_llm if needs_retry else llm

            if same_error_count >= 1:
                messages.append({"role": "user", "content": RETRY_HINT_ERROR})
            elif same_action_count >= 2:
                messages.append({"role": "user", "content": RETRY_HINT_STUCK})

            response = active_llm.invoke(messages, max_tokens=max_tokens)
            action_text = response.content or ""

            if needs_retry:
                messages.pop()

            output = env.step(action_text)

            exec_ok = output.state.startswith("Execution: OK")
            err_full = extract_execution_error(output.state) if not exec_ok else ""
            err_short = err_full.split("\n")[0][:300] if err_full else ""

            if err_full:
                if prev_error_short and err_short == prev_error_short:
                    same_error_count += 1
                else:
                    same_error_count = 0
                    prev_error_short = err_short
            else:
                same_error_count = 0
                prev_error_short = None

            action_type = _extract_action_type(action_text)
            if action_type == prev_action_type:
                same_action_count += 1
            else:
                same_action_count = 0
                prev_action_type = action_type

            if action_type in (
                "EDA",
                "FEATURE_ENGINEERING",
                "MODEL",
                "CODE_FIX",
                "CODE",
            ):
                if not exec_ok:
                    consecutive_fe_fails += 1
                else:
                    consecutive_fe_fails = 0
            else:
                consecutive_fe_fails = 0

            fails = [
                k
                for k, v in extract_validator_status(output.state).items()
                if v.upper() in {"FAIL", "UNRESOLVED", "REGRESSED", "BLOCKED"}
            ]

            log = StepLog(
                step=step_idx,
                action_type=action_type,
                summary=_summary(action_text),
                reward=output.reward,
                done=output.done,
                exec_ok=exec_ok,
                exec_error=err_short,
                exec_error_full=err_full,
                fail_validators=fails,
                full_response=action_text,
                feedback=output.state,
            )
            result.steps.append(log)
            result.final_reward = output.reward
            result.num_steps = step_idx + 1

            if verbose:
                tag = "OK" if exec_ok else "FAIL"
                print(
                    f"  Step {step_idx}: [{action_type}] exec={tag} "
                    f"reward={output.reward:.4f} fails=[{', '.join(fails) or '-'}]"
                )
                print(f"    {log.summary[:200]}")
                if err_short:
                    print(f"    ERROR: {err_short[:220]}")

            # Consecutive-failure handling: restart hint first, abort only on second streak
            if consecutive_fe_fails >= max_consecutive_fe_fails:
                if not restart_given:
                    restart_given = True
                    consecutive_fe_fails = 0
                    same_error_count = 0
                    prev_error_short = None
                    hint = RESTART_HINT.format(n=max_consecutive_fe_fails)
                    messages.append({"role": "assistant", "content": action_text})
                    messages.append({"role": "user", "content": hint})
                    if verbose:
                        print(
                            f"    >> Restart hint injected (was {max_consecutive_fe_fails} consecutive fails)"
                        )
                    continue
                else:
                    result.aborted_early = True
                    result.error = (
                        f"Aborted: {max_consecutive_fe_fails} consecutive failed "
                        f"executable-stage steps after restart hint."
                    )
                    break

            if output.done:
                break

            # User message: full state; cap size to stay within context limits
            fb = output.state
            # Remind whenever MODEL ran but sandbox still has no submission vector
            if action_type == "MODEL" and not _sandbox_has_submission_bundle(
                getattr(env, "_session", None)
            ):
                fb = fb + POST_MODEL_PREDICTION_REMINDER
            if len(fb) > 28000:
                fb = fb[:14000] + "\n\n[... truncated middle ...]\n\n" + fb[-14000:]
            messages.append({"role": "assistant", "content": action_text})
            messages.append({"role": "user", "content": fb})

        if env._session:
            result.best_validation_metric = env._session.best_metric
            result.final_hidden_test_metric = env._session.hidden_test_metric
            result.final_metric = env._session.hidden_test_metric

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc!r}\n{traceback.format_exc()}"

    result.elapsed_sec = time.time() - t0
    env.close()
    return result
