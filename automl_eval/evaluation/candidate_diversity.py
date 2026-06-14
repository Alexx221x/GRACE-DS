"""Lightweight diagnostics for small manual candidate diversity."""

from __future__ import annotations

import re
from collections.abc import Iterable

_UNKNOWN = "unknown"

_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("ExtraTrees", r"\bExtraTrees(?:Classifier|Regressor)?\b"),
    ("RandomForest", r"\bRandomForest(?:Classifier|Regressor)?\b"),
    (
        "GradientBoosting",
        r"\b(?:HistGradientBoosting|GradientBoosting)(?:Classifier|Regressor)?\b",
    ),
    ("LogisticRegression", r"\bLogisticRegression\b"),
    (
        "LinearModel",
        r"\b(?:LinearRegression|Ridge|Lasso|ElasticNet|SGDClassifier|SGDRegressor|Perceptron)\b",
    ),
    ("SVM", r"\b(?:SVC|SVR|LinearSVC|LinearSVR)\b"),
    ("KNN", r"\bKNeighbors(?:Classifier|Regressor)?\b"),
    ("NaiveBayes", r"\b(?:GaussianNB|MultinomialNB|BernoulliNB|ComplementNB)\b"),
    ("DecisionTree", r"\bDecisionTree(?:Classifier|Regressor)?\b"),
    ("Dummy", r"\bDummy(?:Classifier|Regressor)?\b"),
]


def detect_model_families(text: str | None) -> list[str]:
    """Return ordered unique model families mentioned in ``text``."""
    if not text:
        return []
    found: list[str] = []
    for family, pattern in _FAMILY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(family)
    return found


def primary_model_family(text: str | None) -> str:
    """Best-effort primary family for one candidate action."""
    families = detect_model_families(text)
    return families[0] if families else _UNKNOWN


def candidate_diversity_score(families: Iterable[str | None]) -> float:
    """Score small manual candidate diversity in [0, 1]."""
    clean = [str(f or _UNKNOWN) for f in families]
    if not clean:
        return 0.0
    recognised = [f for f in clean if f != _UNKNOWN]
    distinct = set(recognised)
    if len(clean) == 1:
        return 0.35 if recognised else 0.20
    if len(distinct) >= 2:
        return 1.0
    if len(recognised) >= 3:
        return 0.65
    if len(recognised) >= 2:
        return 0.55
    return 0.35


def candidate_diversity_feedback(
    families: Iterable[str | None], *, remaining_turns: int | None = None
) -> str | None:
    """Return a concise corrective hint, or None when diversity is sufficient."""
    clean = [str(f or _UNKNOWN) for f in families]
    recognised = [f for f in clean if f != _UNKNOWN]
    distinct = sorted(set(recognised))
    if len(distinct) >= 2:
        return None
    if not clean:
        return "No validated replayable candidate has been scored yet. Create one simple baseline MODEL candidate before stopping."
    suffix = ""
    if remaining_turns is not None:
        suffix = f" ({remaining_turns} working turn{'s' if remaining_turns != 1 else ''} remain)."
    current = distinct[0] if distinct else "an unrecognised model family"
    return (
        f"Candidate diversity is still low: validated candidates so far use {current}. "
        "Before STOP_WORKING, validate one meaningfully different manual model family "
        "such as a tree ensemble vs a linear/logistic baseline; do not use grid/randomized search."
        + suffix
    )


def family_sequence_summary(families: Iterable[str | None]) -> str:
    """Stable compact sequence for CSV diagnostics."""
    return ";".join(str(f or _UNKNOWN) for f in families)
