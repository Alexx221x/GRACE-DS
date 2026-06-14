"""FeatEng-style performance normalization"""

from __future__ import annotations

from dataclasses import dataclass

# Metrics where a larger value is better.
_HIGHER_IS_BETTER = {"roc_auc", "accuracy", "f1", "r2"}
# Metrics where a smaller value is better (error metrics).
_LOWER_IS_BETTER = {"log_loss", "rmse", "mae"}


@dataclass(frozen=True)
class NormalizationResult:
    raw_score: float
    normalized: float
    baseline: float
    oracle: float
    higher_is_better: bool
    degenerate: bool


def metric_is_higher_better(metric: str) -> bool:
    m = metric.lower()
    if m in _HIGHER_IS_BETTER:
        return True
    if m in _LOWER_IS_BETTER:
        return False
    # Unknown metric: assume higher-is-better but callers should set it explicitly.
    return True


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_performance(
    raw_score: float,
    *,
    metric: str,
    baseline: float | None,
    oracle: float | None,
    eps: float = 1e-9,
) -> NormalizationResult:
    """Map a raw metric onto [0, 1] error-reduction / headroom-capture form."""
    higher = metric_is_higher_better(metric)

    if baseline is None or oracle is None:
        return NormalizationResult(
            raw_score=raw_score,
            normalized=_clip01(raw_score),
            baseline=float("nan") if baseline is None else baseline,
            oracle=float("nan") if oracle is None else oracle,
            higher_is_better=higher,
            degenerate=True,
        )

    if higher:
        denom = oracle - baseline
        if abs(denom) < eps:
            return NormalizationResult(
                raw_score, _clip01(raw_score), baseline, oracle, higher, True
            )
        norm = (raw_score - baseline) / denom
    else:
        # Error metric: baseline error is larger, oracle error is smaller.
        denom = baseline - oracle
        if abs(denom) < eps:
            return NormalizationResult(
                raw_score, _clip01(raw_score), baseline, oracle, higher, True
            )
        norm = (baseline - raw_score) / denom

    return NormalizationResult(
        raw_score=raw_score,
        normalized=_clip01(norm),
        baseline=baseline,
        oracle=oracle,
        higher_is_better=higher,
        degenerate=False,
    )
