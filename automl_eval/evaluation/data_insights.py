"""
DataInsights — automatic preliminary analysis of the dataset.

The environment inspects the data BEFORE the agent starts working
and builds a set of expectations:
  - which feature pairs are highly correlated (need drop/PCA),
  - which columns contain missing values (need fillna/imputer/drop),
  - where outliers exist (need describe / clip / IQR filter),
  - basic distribution statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd


@dataclass
class ColumnMissing:
    column: str
    missing_count: int
    missing_frac: float
    is_numeric: bool
    recommend: str


@dataclass
class CorrelationPair:
    col_a: str
    col_b: str
    corr_value: float


@dataclass
class OutlierInfo:
    column: str
    outlier_count: int
    outlier_frac: float
    lower_bound: float
    upper_bound: float


@dataclass
class SkewInfo:
    column: str
    skewness: float


@dataclass
class DataInsights:
    """Result of the automatic dataset analysis by the environment."""

    n_rows: int = 0
    n_cols: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_like_columns: list[str] = field(default_factory=list)

    missing_total_frac: float = 0.0
    missing_columns: list[ColumnMissing] = field(default_factory=list)
    has_missing: bool = False

    high_corr_pairs: list[CorrelationPair] = field(default_factory=list)
    has_high_correlation: bool = False

    outlier_columns: list[OutlierInfo] = field(default_factory=list)
    has_outliers: bool = False

    skewed_columns: list[SkewInfo] = field(default_factory=list)
    has_high_skew: bool = False

    duplicate_count: int = 0
    has_duplicates: bool = False

    scale_range_ratio: float = 0.0

    class_imbalance_ratio: float | None = None


def analyze_dataset(
    df: pd.DataFrame,
    target_column: str,
    *,
    corr_threshold: float = 0.90,
    missing_drop_threshold: float = 0.50,
    outlier_iqr_factor: float = 1.5,
    outlier_min_frac: float = 0.01,
) -> DataInsights:
    """"""
    insights = DataInsights()
    insights.n_rows, insights.n_cols = df.shape

    feature_cols = [c for c in df.columns if c != target_column]
    num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    insights.numeric_columns = num_cols
    insights.categorical_columns = cat_cols

    # ── 1. Missing values ──────────────────────────────────────────────
    total_cells = df[feature_cols].size
    total_missing = int(df[feature_cols].isnull().sum().sum())
    insights.missing_total_frac = total_missing / total_cells if total_cells else 0.0

    for col in feature_cols:
        n_miss = int(df[col].isnull().sum())
        if n_miss == 0:
            continue
        frac = n_miss / len(df)
        is_num = col in num_cols

        if frac > 0.80:
            recommend = "drop_column"
        elif frac > missing_drop_threshold:
            recommend = "fill_or_drop"
        elif is_num:
            recommend = "fill_median"
        else:
            recommend = "fill_mode"

        insights.missing_columns.append(
            ColumnMissing(
                column=col,
                missing_count=n_miss,
                missing_frac=frac,
                is_numeric=is_num,
                recommend=recommend,
            )
        )

    insights.has_missing = len(insights.missing_columns) > 0

    # ── 2. Correlations (feature-feature, excluding target) ────────────
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        seen: set[tuple[str, str]] = set()
        for i, ca in enumerate(num_cols):
            for j, cb in enumerate(num_cols):
                if i >= j:
                    continue
                val = corr_matrix.iloc[i, j]
                if np.isnan(val):
                    continue
                if val >= corr_threshold and (ca, cb) not in seen:
                    insights.high_corr_pairs.append(
                        CorrelationPair(
                            col_a=ca, col_b=cb, corr_value=round(float(val), 4)
                        )
                    )
                    seen.add((ca, cb))

    insights.has_high_correlation = len(insights.high_corr_pairs) > 0

    # ── 3. Outliers (IQR on numeric features) ──────────────────────────
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - outlier_iqr_factor * iqr
        upper = q3 + outlier_iqr_factor * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        frac = n_outliers / len(series)
        if frac >= outlier_min_frac and n_outliers >= 3:
            insights.outlier_columns.append(
                OutlierInfo(
                    column=col,
                    outlier_count=n_outliers,
                    outlier_frac=round(frac, 4),
                    lower_bound=round(float(lower), 4),
                    upper_bound=round(float(upper), 4),
                )
            )

    insights.has_outliers = len(insights.outlier_columns) > 0

    # ── 4. Duplicates ──────────────────────────────────────────────────
    dup_count = int(df[feature_cols].duplicated().sum())
    insights.duplicate_count = dup_count
    insights.has_duplicates = dup_count > 0

    # ── 5. Skewness (numeric features, |skew| > 2 = highly skewed) ────
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        skew_val = float(series.skew())
        if abs(skew_val) > 2.0:
            insights.skewed_columns.append(
                SkewInfo(column=col, skewness=round(skew_val, 4))
            )
    insights.has_high_skew = len(insights.skewed_columns) > 0

    # ── 6. Datetime-like columns ───────────────────────────────────────
    for col in cat_cols:
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(sample, errors="raise")
            insights.datetime_like_columns.append(col)
        except (ValueError, TypeError):
            pass

    # ── 7. Scale range (max std / min std among numeric) ───────────────
    if len(num_cols) >= 2:
        stds = df[num_cols].std().replace(0, np.nan).dropna()
        if len(stds) >= 2:
            insights.scale_range_ratio = round(float(stds.max() / stds.min()), 2)

    # ── 8. Class imbalance (classification only) ───────────────────────
    if target_column in df.columns:
        vc = df[target_column].value_counts()
        if len(vc) >= 2:
            insights.class_imbalance_ratio = round(
                float(vc.iloc[-1]) / float(vc.iloc[0]), 4
            )

    return insights
