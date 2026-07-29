"""Deterministic, auditable prompt variants for robustness experiments.

The variants deliberately change only natural-language framing. Protocol tokens,
action names, safety constraints, and evaluator semantics remain unchanged.
"""

from __future__ import annotations

import hashlib


PROMPT_VARIANT_IDS: tuple[int, ...] = (0, 1, 2, 3)

# Anchor phrases that MUST appear (verbatim) in every stage-aware variant. These
# are load-bearing instructions; the completeness check enforces their presence.
REQUIRED_ANCHORS: tuple[str, ...] = (
    # Approved-libraries / explicit-imports
    "Only `pd` (pandas) and `np` (numpy) are pre-bound",
    "imported EXPLICITLY",
    "from sklearn.pipeline import Pipeline",
    # Rollback atomicity
    "rolled back atomically",
    # Critical methodological errors
    "Critical methodological errors",
    "Target-leakage code patterns",
    "Train + validation refit leakage",
    "evaluator-private namespaces",
    "protected snapshots",
)


_VARIANT_REPLACEMENTS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (
        (
            "You are an expert AutoML agent solving a non-time-series tabular "
            "machine-learning\ntask in a stage-aware evaluator sandbox.",
            "Act as an expert AutoML agent for a non-time-series tabular "
            "machine-learning\ntask inside a stage-aware evaluator sandbox.",
        ),
        (
            "Your objective is to build the strongest reproducible model possible "
            "within the\navailable action budget and maximise the final evaluation "
            "metric on the private\nheld-out test split. Use evaluator-owned "
            "validation feedback to guide meaningful\nimprovements, but never "
            "attempt to access, infer or optimise directly against\nthe private "
            "test split.",
            "Within the available action budget, build the strongest reproducible "
            "model you\ncan and maximise the final metric on the private held-out "
            "test split. Let only\nevaluator-provided validation feedback guide "
            "substantive improvements; never try\nto access, reconstruct, or tune "
            "directly to the private test split.",
        ),
        (
            "Every response must begin with exactly one stage line:",
            "Begin every response with exactly one of these stage lines:",
        ),
        (
            "You may revisit stages when useful. Prefer evidence-driven iterations:",
            "Return to earlier stages when useful. Favour evidence-driven iteration:",
        ),
        (
            "This is a STRUCTURED EXPERIMENT EPISODE. Terminal hidden-test "
            "evaluation is",
            "This episode uses the STRUCTURED EXPERIMENT protocol. Terminal "
            "hidden-test evaluation is",
        ),
        (
            "You are generating the executable Python body of one fitted AutoML "
            "submission in a constrained",
            "Produce the executable Python body for one fitted AutoML submission "
            "under a constrained",
        ),
        (
            "CRITICAL RESPONSE CONTRACT (you get exactly ONE generation; there is "
            "no second turn):",
            "BINDING RESPONSE CONTRACT (there is exactly ONE generation and no "
            "follow-up turn):",
        ),
        (
            "Build and fit one replayable sklearn raw-input `pipeline` and/or "
            "define `predict_fn(raw_dataframe)`.",
            "Fit and expose one replayable sklearn submission that accepts raw "
            "inputs through `pipeline` and/or `predict_fn(raw_dataframe)`.",
        ),
        (
            "You are an autonomous data-science coding agent solving one tabular "
            "prediction task in a persistent offline Python workspace.",
            "Act as an autonomous data-science coding agent on one tabular "
            "prediction task in a persistent, offline Python workspace.",
        ),
        (
            "You control your own workflow. There are no prescribed PLAN, EDA, "
            "FEATURE_ENGINEERING or MODEL states, and no expert checklist hints "
            "are provided.",
            "Choose your own workflow: no PLAN, EDA, FEATURE_ENGINEERING, or MODEL "
            "state is prescribed, and you receive no expert-checklist hints.",
        ),
        (
            "When a Python block successfully creates a replayable `pipeline` or "
            "`predict_fn`, the evaluator automatically validates it",
            "Whenever a Python block successfully creates a replayable `pipeline` "
            "or `predict_fn`, the evaluator validates it automatically",
        ),
    ),
    2: (
        (
            "You are an expert AutoML agent solving a non-time-series tabular "
            "machine-learning\ntask in a stage-aware evaluator sandbox.",
            "Work as a specialist AutoML agent on a non-time-series tabular "
            "machine-learning\ntask evaluated in a stage-aware sandbox.",
        ),
        (
            "Your objective is to build the strongest reproducible model possible "
            "within the\navailable action budget and maximise the final evaluation "
            "metric on the private\nheld-out test split. Use evaluator-owned "
            "validation feedback to guide meaningful\nimprovements, but never "
            "attempt to access, infer or optimise directly against\nthe private "
            "test split.",
            "Use the action allowance to construct a high-performing, reproducible "
            "model for\nthe final private held-out metric. Make meaningful revisions "
            "from evaluator-owned\nvalidation feedback only, without accessing, "
            "inferring, or directly optimising for\nthe private test split.",
        ),
        (
            "Every response must begin with exactly one stage line:",
            "The first line of every response must be exactly one stage declaration:",
        ),
        (
            "You may revisit stages when useful. Prefer evidence-driven iterations:",
            "Stages can be revisited when beneficial. Use evidence to drive each "
            "iteration:",
        ),
        (
            "This is a STRUCTURED EXPERIMENT EPISODE. Terminal hidden-test "
            "evaluation is",
            "You are in a STRUCTURED EXPERIMENT EPISODE, where terminal hidden-test "
            "evaluation is",
        ),
        (
            "You are generating the executable Python body of one fitted AutoML "
            "submission in a constrained",
            "Your output will be the executable Python body of a fitted AutoML "
            "submission in a constrained",
        ),
        (
            "CRITICAL RESPONSE CONTRACT (you get exactly ONE generation; there is "
            "no second turn):",
            "MANDATORY OUTPUT CONTRACT (only ONE generation is available; no "
            "revision turn follows):",
        ),
        (
            "Build and fit one replayable sklearn raw-input `pipeline` and/or "
            "define `predict_fn(raw_dataframe)`.",
            "Create and fit a single replayable sklearn solution, exposed as a "
            "raw-input `pipeline` and/or `predict_fn(raw_dataframe)`.",
        ),
        (
            "You are an autonomous data-science coding agent solving one tabular "
            "prediction task in a persistent offline Python workspace.",
            "Solve one tabular prediction task as an autonomous data-science coding "
            "agent with a persistent offline Python workspace.",
        ),
        (
            "You control your own workflow. There are no prescribed PLAN, EDA, "
            "FEATURE_ENGINEERING or MODEL states, and no expert checklist hints "
            "are provided.",
            "The workflow is yours to organise. PLAN, EDA, FEATURE_ENGINEERING, and "
            "MODEL states are not imposed, and expert-checklist hints are absent.",
        ),
        (
            "When a Python block successfully creates a replayable `pipeline` or "
            "`predict_fn`, the evaluator automatically validates it",
            "A Python block that successfully creates a replayable `pipeline` or "
            "`predict_fn` is validated automatically by the evaluator",
        ),
    ),
    3: (
        (
            "You are an expert AutoML agent solving a non-time-series tabular "
            "machine-learning\ntask in a stage-aware evaluator sandbox.",
            "Take the role of an expert AutoML practitioner solving a "
            "non-time-series tabular\nmachine-learning task in a stage-aware "
            "evaluation sandbox.",
        ),
        (
            "Your objective is to build the strongest reproducible model possible "
            "within the\navailable action budget and maximise the final evaluation "
            "metric on the private\nheld-out test split. Use evaluator-owned "
            "validation feedback to guide meaningful\nimprovements, but never "
            "attempt to access, infer or optimise directly against\nthe private "
            "test split.",
            "Aim for the best reproducible model achievable within the action "
            "limit, judged by\nthe final metric on a private held-out split. Base "
            "useful changes on validation\nfeedback supplied by the evaluator, and "
            "do not access, deduce, or optimise\nagainst private test data.",
        ),
        (
            "Every response must begin with exactly one stage line:",
            "Start each response using exactly one stage line from this list:",
        ),
        (
            "You may revisit stages when useful. Prefer evidence-driven iterations:",
            "Re-enter a stage if it helps, and make iteration decisions from evidence:",
        ),
        (
            "This is a STRUCTURED EXPERIMENT EPISODE. Terminal hidden-test "
            "evaluation is",
            "The current run is a STRUCTURED EXPERIMENT EPISODE; terminal "
            "hidden-test evaluation is",
        ),
        (
            "You are generating the executable Python body of one fitted AutoML "
            "submission in a constrained",
            "Write the executable Python body for a single fitted AutoML submission "
            "in a constrained",
        ),
        (
            "CRITICAL RESPONSE CONTRACT (you get exactly ONE generation; there is "
            "no second turn):",
            "STRICT RESPONSE RULES (you have ONE generation only and cannot revise "
            "it later):",
        ),
        (
            "Build and fit one replayable sklearn raw-input `pipeline` and/or "
            "define `predict_fn(raw_dataframe)`.",
            "Train one replayable sklearn artefact that handles raw inputs via "
            "`pipeline` and/or `predict_fn(raw_dataframe)`.",
        ),
        (
            "You are an autonomous data-science coding agent solving one tabular "
            "prediction task in a persistent offline Python workspace.",
            "Operate as an autonomous data-science coding agent solving a tabular "
            "prediction task in a persistent offline Python workspace.",
        ),
        (
            "You control your own workflow. There are no prescribed PLAN, EDA, "
            "FEATURE_ENGINEERING or MODEL states, and no expert checklist hints "
            "are provided.",
            "Organise the workflow yourself. No PLAN, EDA, FEATURE_ENGINEERING, or "
            "MODEL stages are required, and no expert-checklist guidance is shown.",
        ),
        (
            "When a Python block successfully creates a replayable `pipeline` or "
            "`predict_fn`, the evaluator automatically validates it",
            "If a Python block successfully produces a replayable `pipeline` or "
            "`predict_fn`, the evaluator performs validation automatically",
        ),
    ),
}


def num_variants() -> int:
    """Return the number of supported prompt variants, including canonical."""
    return len(PROMPT_VARIANT_IDS)


def validate_prompt_variant_id(variant_id: int) -> int:
    """Validate and normalize a configured prompt variant identifier."""
    if isinstance(variant_id, bool) or not isinstance(variant_id, int):
        raise ValueError("prompt variant id must be an integer")
    if variant_id not in PROMPT_VARIANT_IDS:
        raise ValueError(
            f"Unknown prompt variant id {variant_id}; "
            f"valid ids are {list(PROMPT_VARIANT_IDS)}"
        )
    return variant_id


def apply_prompt_variant(
    canonical_prompt: str,
    *,
    variant_id: int = 0,
    enabled: bool = True,
) -> str:
    """Apply one semantics-preserving wording variant to a complete prompt."""
    if not enabled:
        return canonical_prompt
    variant_id = validate_prompt_variant_id(variant_id)
    if variant_id == 0:
        return canonical_prompt

    transformed = canonical_prompt
    replacements_applied = 0
    for source, replacement in _VARIANT_REPLACEMENTS[variant_id]:
        if source in transformed:
            transformed = transformed.replace(source, replacement)
            replacements_applied += 1
    if replacements_applied == 0:
        raise ValueError(
            f"Prompt variant {variant_id} did not match this prompt family. "
            "Refusing to run a silently canonical condition."
        )
    return transformed


def build_system_prompt_variant(
    max_actions: int | str,
    *,
    variant_id: int = 0,
    enabled: bool = True,
) -> str:
    """Build the stage-aware prompt with the real budget, then vary its wording."""
    from automl_eval.llm.prompts import build_system_prompt

    canonical = build_system_prompt(max_actions)
    return apply_prompt_variant(
        canonical,
        variant_id=variant_id,
        enabled=enabled,
    )


def variant_id_for_seed(seed: int | str) -> int:
    """Map a seed to a variant for legacy callers; new grids should use explicit ids."""
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return PROMPT_VARIANT_IDS[int(digest, 16) % num_variants()]


def select_system_prompt(
    seed: int | str | None,
    *,
    enabled: bool = True,
    max_actions: int | str = 8,
) -> str:
    """Backward-compatible seeded selector with a concrete action budget.

    New experiment grids should use explicit ``prompt_variant_id`` values rather
    than deriving prompt wording from the LLM sampling seed.
    """
    variant_id = (
        variant_id_for_seed(seed)
        if enabled and seed is not None
        else PROMPT_VARIANT_IDS[0]
    )
    return build_system_prompt_variant(
        max_actions,
        variant_id=variant_id,
        enabled=enabled,
    )


def prompt_sha256(prompt: str) -> str:
    """Return a stable digest for recording the exact prompt used."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def all_paraphrases(max_actions: int | str = 8) -> list[str]:
    """Return every stage-aware variant for the supplied action budget."""
    return [
        build_system_prompt_variant(max_actions, variant_id=variant_id)
        for variant_id in PROMPT_VARIANT_IDS
    ]


def assert_paraphrase_completeness(max_actions: int | str = 8) -> None:
    """Raise AssertionError if any variant drops a required anchor phrase."""
    for variant_id, prompt in zip(
        PROMPT_VARIANT_IDS, all_paraphrases(max_actions), strict=True
    ):
        normalised = " ".join(prompt.split())
        for anchor in REQUIRED_ANCHORS:
            anchor_norm = " ".join(anchor.split())
            if anchor_norm not in normalised:
                raise AssertionError(
                    f"Prompt variant {variant_id} is missing required anchor: "
                    f"{anchor!r}"
                )
