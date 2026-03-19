"""Rule-based scoring utilities for organ matching."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compatibility_score(donor_bg: str, recipient_bg: str) -> float:
    """Apply the required compatibility logic using blood groups."""
    if donor_bg == recipient_bg:
        return 1.0
    if donor_bg == "O":
        return 0.9
    return 0.4


def normalize_series(series: pd.Series) -> pd.Series:
    """Scale a numeric series into the 0-1 range."""
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(min_value, max_value):
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return (series - min_value) / (max_value - min_value)


def build_success_probability(
    compatibility: pd.Series,
    urgency_score: pd.Series,
    health_score: pd.Series,
    distance: pd.Series,
) -> pd.Series:
    """Create the target score and normalize it between 0 and 1."""
    raw_score = (
        (0.5 * compatibility)
        + (0.3 * (urgency_score / 10.0))
        + (0.2 * health_score)
        - (0.1 * distance)
    )
    return normalize_series(raw_score).round(4)
