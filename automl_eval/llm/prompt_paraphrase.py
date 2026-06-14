"""Prompt paraphrasing for robustness"""

from __future__ import annotations

import hashlib

from automl_eval.llm.prompts import SYSTEM_PROMPT

# Anchor phrases that MUST appear (verbatim) in every paraphrase. These are the
# load-bearing instructions; the completeness test enforces their presence.
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


def _canonical() -> str:
    return SYSTEM_PROMPT


def _paraphrase_intro_variant_1() -> str:
    preamble = (
        "You are an expert data-science agent operating inside a sandboxed, "
        "stage-structured evaluation harness. Read the protocol below carefully "
        "and follow it exactly; the same rules are restated in canonical form "
        "immediately after this note.\n\n"
    )
    return preamble + _canonical()


def _paraphrase_intro_variant_2() -> str:
    preamble = (
        "INSTRUCTIONS (paraphrase 2 of 4). You will solve a tabular machine-learning "
        "task by emitting stage-tagged actions. Everything the grader needs is "
        "specified in the protocol that follows; treat it as authoritative.\n\n"
    )
    return preamble + _canonical()


def _paraphrase_intro_variant_3() -> str:
    preamble = (
        "Working context (paraphrase 3 of 4): you act as an autonomous ML engineer. "
        "Your messages drive a deterministic environment that scores each stage. "
        "The complete, binding rules appear below.\n\n"
    )
    return preamble + _canonical()


_VARIANTS = (
    _canonical,
    _paraphrase_intro_variant_1,
    _paraphrase_intro_variant_2,
    _paraphrase_intro_variant_3,
)


def num_variants() -> int:
    return len(_VARIANTS)


def select_system_prompt(seed: int | None, *, enabled: bool = True) -> str:
    """Return a system prompt."""
    if not enabled or seed is None:
        return _canonical()
    idx = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % len(_VARIANTS)
    return _VARIANTS[idx]()


def all_paraphrases() -> list[str]:
    return [fn() for fn in _VARIANTS]


def assert_paraphrase_completeness() -> None:
    """Raise AssertionError if any paraphrase drops a required anchor phrase."""
    for i, prompt in enumerate(all_paraphrases()):
        normalised = " ".join(prompt.split())
        for anchor in REQUIRED_ANCHORS:
            anchor_norm = " ".join(anchor.split())
            if anchor_norm not in normalised:
                raise AssertionError(
                    f"Paraphrase variant {i} is missing required anchor: {anchor!r}"
                )
