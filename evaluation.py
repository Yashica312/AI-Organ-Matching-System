"""Evaluation helpers for in-domain and cross-organ testing."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import MODEL_FEATURES, TARGET_COLUMN


def evaluate_dataset(model, df: pd.DataFrame, label: str) -> dict[str, float]:
    """Run regression metrics for one dataset."""
    X = df[MODEL_FEATURES].copy()
    y_true = df[TARGET_COLUMN].copy()
    predictions = model.predict(X)

    mae = mean_absolute_error(y_true, predictions)
    rmse = math.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    return {
        "label": label,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def print_metrics_comparison(kidney_metrics: dict[str, float], heart_metrics: dict[str, float]) -> None:
    """Print the required performance comparison output."""
    print("Performance on Kidney dataset")
    print(f"MAE: {kidney_metrics['mae']}")
    print(f"RMSE: {kidney_metrics['rmse']}")
    print(f"R^2 score: {kidney_metrics['r2']}")
    print()
    print("Performance on Heart dataset")
    print(f"MAE: {heart_metrics['mae']}")
    print(f"RMSE: {heart_metrics['rmse']}")
    print(f"R^2 score: {heart_metrics['r2']}")
    print()
    print("Note: The heart dataset is used for cross-organ generalization testing.")
