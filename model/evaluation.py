"""Evaluation helpers for model quality, ranking quality, and explainability."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score

from preprocessing import MODEL_FEATURES, TARGET_COLUMN, prepare_feature_frame


def precision_at_k(y_true, y_pred, k: int = 5, threshold: float = 0.5) -> float:
    """Compute precision within the top-k predictions above a relevance threshold."""
    ranking_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values("y_pred", ascending=False).head(k)
    if ranking_df.empty:
        return 0.0
    return float((ranking_df["y_true"] >= threshold).sum() / len(ranking_df))


def create_shap_summary_plot(model, X: pd.DataFrame, output_path: str | Path) -> Path:
    """Generate and save a SHAP summary plot for tree-based models."""
    import shap

    output_path = Path(output_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def evaluate_dataset(model, df: pd.DataFrame, label: str) -> dict[str, float]:
    """Run regression and ranking-aware metrics for one dataset."""
    prepared_df = prepare_feature_frame(df)
    X = prepared_df[MODEL_FEATURES].copy()
    y_true = pd.to_numeric(prepared_df[TARGET_COLUMN], errors="coerce").fillna(0.5)
    predictions = model.predict(X)

    mae = mean_absolute_error(y_true, predictions)
    rmse = math.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    ndcg = ndcg_score([y_true.to_numpy()], [predictions]) if len(y_true) > 1 else 1.0
    p_at_5 = precision_at_k(y_true.to_numpy(), predictions, k=5)

    return {
        "label": label,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "ndcg": round(float(ndcg), 4),
        "precision_at_5": round(p_at_5, 4),
    }


def print_metrics_comparison(kidney_metrics: dict[str, float], heart_metrics: dict[str, float]) -> None:
    """Print the required performance comparison output."""
    print("Performance on Kidney dataset")
    print(f"MAE: {kidney_metrics['mae']}")
    print(f"RMSE: {kidney_metrics['rmse']}")
    print(f"R^2 score: {kidney_metrics['r2']}")
    print(f"NDCG: {kidney_metrics['ndcg']}")
    print(f"Precision@5: {kidney_metrics['precision_at_5']}")
    print()
    print("Performance on Heart dataset")
    print(f"MAE: {heart_metrics['mae']}")
    print(f"RMSE: {heart_metrics['rmse']}")
    print(f"R^2 score: {heart_metrics['r2']}")
    print(f"NDCG: {heart_metrics['ndcg']}")
    print(f"Precision@5: {heart_metrics['precision_at_5']}")
    print()
    print("Note: The heart dataset is used for cross-organ generalization testing.")
