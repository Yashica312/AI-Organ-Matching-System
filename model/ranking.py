"""Two-stage donor ranking utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import MODEL_FEATURES, prepare_feature_frame


MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"


def load_ranker(model_path: Path | None = None):
    """Load the persisted XGBoost ranking model."""
    return joblib.load(model_path or MODEL_PATH)


def rank_recipients(df: pd.DataFrame, top_n: int = 5, model=None) -> pd.DataFrame:
    """Hard-filter incompatible candidates, score the rest, and return the top ranked set."""
    prepared_df = prepare_feature_frame(df)
    filtered = prepared_df[prepared_df["blood_compat_score"] == 1].copy()

    if filtered.empty:
        return filtered

    ranker = model or load_ranker()
    filtered["predicted_score"] = ranker.predict(filtered[MODEL_FEATURES])
    filtered["final_score"] = filtered["predicted_score"]
    filtered = filtered.sort_values(by="predicted_score", ascending=False)
    return filtered.head(top_n)
